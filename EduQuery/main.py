from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chromadb import PersistentClient
from EduQuery.config import Config
from EduQuery.api_routes import setup_routes
from dotenv import load_dotenv
import os
import logging

load_dotenv(dotenv_path="/home/server/Eduquery/try1/src/backend/.env")

Config.initialize()
app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = PersistentClient(path=Config.CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="pdf_collection")

setup_routes(app, collection)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.FASTAPI_PORT) 