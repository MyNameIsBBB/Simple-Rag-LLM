import sys

import chromadb

import config
from modules import PDFIngester, RAGEngine


class CLIApp:
    def __init__(self):
        self._client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._ingester = PDFIngester(self._collection, config)
        self._engine = RAGEngine(self._collection, config)

    def _debug(self, message: str) -> None:
        if config.DEBUG:
            print(f"[DEBUG] {message}")

    def _print_banner(self) -> None:
        print("""
╔══════════════════════════════════════════════╗
║              AI RAG — CLI Demo               ║
║        Gemini + ChromaDB + PDF               ║
╚══════════════════════════════════════════════╝
""")

    def _print_help(self) -> None:
        print("""
คำสั่งที่ใช้ได้:
  ingest <path>   — ingest ไฟล์ PDF เข้า ChromaDB
  list            — ดูรายชื่อเอกสารทั้งหมด
  ask             — ถามคำถาม (กด Enter เพื่อถาม)
  clear           — ล้าง ChromaDB ทั้งหมด
  help            — แสดงคำสั่ง
  quit / exit     — ออกจากโปรแกรม
""")

    def _print_status(self) -> None:
        print("✅ เชื่อมต่อ Gemini API และ ChromaDB สำเร็จ")
        print(f"📊 มีข้อมูลอยู่ใน ChromaDB: {self._ingester.get_collection_count()} chunks")

        ingested = self._ingester.list_ingested_docs()
        if ingested:
            print(f"📚 เอกสารในระบบ: {', '.join(ingested)}")
        else:
            print("📭 ยังไม่มีเอกสาร กรุณา ingest PDF ก่อน")

    def _handle_ingest(self, parts: list[str]) -> None:
        if len(parts) < 2:
            print("⚠️  ระบุ path ไฟล์ด้วย เช่น: ingest sample_docs/file.pdf")
            return
        filepath = parts[1].strip('"').strip("'")
        try:
            count = self._ingester.ingest_pdf(filepath)
            print(f"\n✅ Ingest สำเร็จ: {count} chunks")
        except FileNotFoundError:
            print(f"❌ ไม่พบไฟล์: {filepath}")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")

    def _handle_list(self) -> None:
        docs = self._ingester.list_ingested_docs()
        count = self._ingester.get_collection_count()
        if docs:
            print(f"\n📚 เอกสารในระบบ ({len(docs)} ไฟล์, {count} chunks รวม):")
            for doc in docs:
                print(f"   • {doc}")
        else:
            print("📭 ยังไม่มีเอกสาร")

    def _handle_ask(self) -> None:
        print("💬 พิมพ์คำถาม (กด Ctrl+C เพื่อยกเลิก):")
        try:
            question = input("   คำถาม: ").strip()
            if not question:
                return

            self._answer_question(question)
        except KeyboardInterrupt:
            print("\n(ยกเลิก)")

    def _handle_clear(self) -> None:
        confirm = input("⚠️  ลบข้อมูลทั้งหมดใน ChromaDB? (yes/no): ").strip().lower()
        if confirm == "yes":
            self._client.delete_collection(config.CHROMA_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._ingester = PDFIngester(self._collection, config)
            self._engine = RAGEngine(self._collection, config)
            print("✅ ล้างข้อมูลแล้ว")
        else:
            print("ยกเลิก")

    def _answer_question(self, question: str) -> None:
        print("\n🔍 กำลังค้นหาและสร้างคำตอบ...")
        try:
            result = self._engine.query(question)
            print(f"\n{'─'*50}")
            print(f"🤖 คำตอบ:\n{result['answer']}")
            if result["sources"]:
                print("\n📎 อ้างอิง:")
                seen = set()
                for s in result["sources"]:
                    key = (s["source"], s["page"])
                    if key not in seen:
                        seen.add(key)
                        print(f"   • {s['source']} — หน้า {s['page']} ({s['score']:.0%})")
            print(f"{'─'*50}")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")

    def run(self) -> None:
        self._print_banner()
        self._print_status()
        self._print_help()

        while True:
            try:
                user_input = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 ออกจากโปรแกรม")
                break

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "ingest":
                self._handle_ingest(parts)
            elif cmd == "list":
                self._handle_list()
            elif cmd == "ask":
                self._handle_ask()
            elif cmd == "clear":
                self._handle_clear()
            elif cmd == "help":
                self._print_help()
            elif cmd in ("quit", "exit", "q"):
                print("\n👋 ออกจากโปรแกรม")
                break
            else:
                self._answer_question(user_input)


def main() -> None:
    try:
        app = CLIApp()
    except ValueError as e:
        print(f"\n❌ ข้อผิดพลาด: {e}\n")
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
