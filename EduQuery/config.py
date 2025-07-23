import os
from multiprocessing import cpu_count

class Config:
    CHROMA_DB_PATH = "./data/chroma_db"
    EMBEDDING_MODEL_NAME = "bge-m3:latest"
    FASTAPI_PORT = 8006
    PDF_DIR = "./books"
    BATCH_SIZE = 128 
    MAX_WORKERS = max(1, cpu_count() - 1)
    @classmethod
    def initialize(cls):
        os.makedirs(cls.PDF_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.CHROMA_DB_PATH), exist_ok=True)
        