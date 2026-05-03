from typing import List, Dict, Any


class RAGEngine:
    def __init__(self, collection, config):
        """ตัวจัดการ RAG

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

    def embed_query(self, query: str) -> List[float]:
        """แปลงคำถามเป็นเวกเตอร์

        Args:
            query: ข้อความคำถาม

        Returns:
            เวกเตอร์คำถาม
        """
        return self._llm_manager.embed_text(query)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """ค้นหาชิ้นข้อมูลที่เกี่ยวข้อง

        Args:
            query: ข้อความคำถาม
            top_k: จำนวนผลลัพธ์สูงสุด

        Returns:
            รายการชิ้นข้อมูลพร้อมแหล่งที่มา
        """
        if self._collection.count() == 0:
            return []

        limit = top_k or self._config.TOP_K_RESULTS
        query_embedding = self.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),
            })

        self._debug(f"retrieved={len(chunks)}")
        return chunks

    def build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """สร้างพรอมพ์จากคำถามและบริบท

        Args:
            query: ข้อความคำถาม
            context_chunks: รายการชิ้นข้อมูล

        Returns:
            ข้อความพรอมพ์
        """
        context_text = ""
        for i, chunk in enumerate(context_chunks, start=1):
            context_text += (
                f"\n--- ข้อความที่ {i} "
                f"(ไฟล์: {chunk['source']}, หน้า: {chunk['page']}, "
                f"ความเกี่ยวข้อง: {chunk['score']:.2%}) ---\n"
                f"{chunk['text']}\n"
            )

        prompt = f"""คุณเป็นผู้ช่วย AI ที่ตอบคำถามจากเอกสาร PDF
ตอบคำถามโดยอิงจากข้อมูลที่ให้เท่านั้น ถ้าข้อมูลไม่เพียงพอให้บอกตรงๆ
ตอบเป็นภาษาเดียวกับคำถาม (ไทยหรืออังกฤษ)

=== ข้อมูลจากเอกสาร ===
{context_text}

=== คำถาม ===
{query}

=== คำตอบ ==="""

        return prompt

    def generate_answer(self, prompt: str) -> str:
        """สร้างคำตอบจากพรอมพ์

        Args:
            prompt: ข้อความพรอมพ์

        Returns:
            คำตอบจากโมเดล
        """
        return self._llm_manager.generate_text(
            prompt=prompt,
            max_tokens=1024,
            stop=["=== คำถาม ==="]
        )

    def query(self, question: str) -> Dict[str, Any]:
        """ตอบคำถามด้วย RAG

        Args:
            question: ข้อความคำถาม

        Returns:
            คำตอบและแหล่งอ้างอิง
        """
        if self._collection.count() == 0:
            return {
                "answer": "⚠️ ยังไม่มีเอกสารใน database กรุณา ingest PDF ก่อนถามคำถาม",
                "sources": [],
                "context_chunks": [],
            }

        context_chunks = self.retrieve(question)

        if not context_chunks:
            return {
                "answer": "❌ ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร",
                "sources": [],
                "context_chunks": [],
            }

        prompt = self.build_prompt(question, context_chunks)
        answer = self.generate_answer(prompt)

        sources = [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in context_chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "context_chunks": context_chunks,
        }
