# 🤖 AI RAG — PDF Chat (CLI)

โปรเจกต์ RAG แบบ CLI สำหรับถามตอบจากไฟล์ PDF ด้วย Gemini และ ChromaDB

---

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **LLM & Embedding:** Google Gemini (google-genai)
- **Vector Database:** ChromaDB (local)
- **PDF Processing:** PyMuPDF

---

## 🚀 Workflow
1. โหลด PDF
2. แบ่งข้อความเป็นชิ้น (chunks)
3. สร้าง embeddings
4. เก็บลง ChromaDB
5. ถามคำถามและดึงประโยคที่เกี่ยวข้องมาให้โมเดลตอบ

---

## 📦 ติดตั้ง
pip install -r requirements.txt

## 🔑 ตั้งค่า API Key
สร้างไฟล์ .env จาก .env.example แล้วใส่ค่า:
GEMINI_API_KEY=your_key_here

---

## ▶️ ใช้งาน CLI
python main.py

คำสั่ง:
- ingest <path>
- list
- ask
- clear
- help
- quit
