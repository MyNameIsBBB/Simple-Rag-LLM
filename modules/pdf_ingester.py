import os
import hashlib
from typing import List, Dict, Any

import fitz


class PDFIngester:
    def __init__(self, collection, config):
        """ตัวจัดการ ingest PDF

        Args:
            collection: คอลเลกชัน ChromaDB
            config: โมดูล config
        """
        self._collection = collection
        self._config = config
        self._llm_manager = None

    def _debug(self, message: str) -> None:
        if self._config.DEBUG:
            print(f"[DEBUG] {message}")

    def load_pdf(self, filepath: str) -> List[Dict[str, Any]]:
        """อ่าน PDF เป็นรายการหน้า

        Args:
            filepath: พาธไฟล์ PDF

        Returns:
            รายการหน้าและข้อความ
        """
        pages = []
        doc_name = os.path.basename(filepath)

        with fitz.open(filepath) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "page": page_num,
                        "text": text,
                        "source": doc_name,
                    })

        self._debug(f"loaded_pages={len(pages)} file={doc_name}")
        print(f"📄 โหลด '{doc_name}' สำเร็จ: {len(pages)} หน้า")
        return pages

    def split_into_chunks(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """แยกข้อความเป็นชิ้น

        Args:
            pages: รายการหน้าและข้อความ

        Returns:
            รายการชิ้นข้อความ
        """
        chunks = []
        chunk_size = self._config.CHUNK_SIZE
        overlap = self._config.CHUNK_OVERLAP

        for page_data in pages:
            text = page_data["text"]
            start = 0

            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                chunks.append({
                    "text": chunk_text,
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_index": len(chunks),
                })

                start += chunk_size - overlap

        self._debug(f"chunks={len(chunks)}")
        print(f"✂️  แบ่งเป็น {len(chunks)} chunks")
        return chunks

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """สร้างเวกเตอร์จากข้อความ

        Args:
            texts: รายการข้อความ

        Returns:
            รายการเวกเตอร์
        """
        embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._llm_manager.embed_text(batch)
            embeddings.extend(batch_embeddings)
            print(f"   🔢 Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

        self._debug(f"embeddings={len(embeddings)}")
        return embeddings

    def store_in_chromadb(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """บันทึกชิ้นข้อมูลลงฐานเวกเตอร์

        Args:
            chunks: รายการชิ้นข้อความ
            embeddings: รายการเวกเตอร์
        """
        ids = []
        documents = []
        metadatas = []
        vecs = []

        for chunk, embedding in zip(chunks, embeddings):
            uid = hashlib.md5(
                f"{chunk['source']}_{chunk['page']}_{chunk['chunk_index']}".encode()
            ).hexdigest()

            ids.append(uid)
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
            })
            vecs.append(embedding)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vecs,
        )
        self._debug(f"upserted={len(ids)}")
        print(f"✅ บันทึก {len(ids)} chunks ลงใน ChromaDB สำเร็จ")

    def ingest_pdf(self, filepath: str) -> int:
        """ประมวลผล PDF ทั้งชุด

        Args:
            filepath: พาธไฟล์ PDF

        Returns:
            จำนวนชิ้นที่บันทึก
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ไม่พบไฟล์: {filepath}")

        if not filepath.lower().endswith(".pdf"):
            raise ValueError("รองรับเฉพาะไฟล์ .pdf เท่านั้น")

        print(f"\n{'='*50}")
        print(f"🚀 เริ่ม Ingest: {os.path.basename(filepath)}")
        print(f"{'='*50}")

        pages = self.load_pdf(filepath)
        if not pages:
            print("⚠️  PDF ไม่มีข้อความ (อาจเป็นไฟล์ภาพ/scan)")
            return 0

        chunks = self.split_into_chunks(pages)

        print(f"🔢 กำลัง embed {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)

        self.store_in_chromadb(chunks, embeddings)

        print(f"\n🎉 Ingest สำเร็จ! {len(chunks)} chunks จากไฟล์ '{os.path.basename(filepath)}'")
        return len(chunks)

    def list_ingested_docs(self) -> List[str]:
        """คืนรายชื่อไฟล์ที่บันทึก"""
        results = self._collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            sources.add(meta["source"])
        return sorted(sources)

    def get_collection_count(self) -> int:
        """คืนจำนวนชิ้นทั้งหมด"""
        return self._collection.count()
