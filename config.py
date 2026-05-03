import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR: str = "./chroma_db"

CHROMA_COLLECTION_NAME: str = "pdf_documents"

CHUNK_SIZE: int = 800

CHUNK_OVERLAP: int = 100

TOP_K_RESULTS: int = 5

DEBUG: bool = True