import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ ไม่พบ GEMINI_API_KEY\n"
        "   กรุณา copy .env.example เป็น .env แล้วใส่ API Key ของคุณ\n"
        "   รับ key ได้ที่: https://aistudio.google.com/api-keys"
    )

EMBED_MODEL: str = "gemini-embedding-001"

GEN_MODEL: str = "gemini-2.0-flash"

CHROMA_PERSIST_DIR: str = "./chroma_db"

CHROMA_COLLECTION_NAME: str = "pdf_documents"

CHUNK_SIZE: int = 800

CHUNK_OVERLAP: int = 100

TOP_K_RESULTS: int = 5

DEBUG: bool = True

GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)