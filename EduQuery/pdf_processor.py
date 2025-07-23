import os
import logging
import time
from fastapi import HTTPException
from multiprocessing import Pool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from EduQuery.embedder import OllamaEmbedder
from EduQuery.config import Config

def process_pdf_page(args):
    pdf_path, page_num = args
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        if page_num < len(pages):
            page = pages[page_num]
            text = page.page_content.strip()
            if text:
                return {
                    'content': text,
                    'page_number': page_num,
                    'source': os.path.basename(pdf_path)
                }
    except Exception as e:
        logging.error(f"Error processing page {page_num} of {pdf_path}: {e}")
    return None

class PDFProcessor:
    def __init__(self, collection):
        self.collection = collection
        self.embedder = OllamaEmbedder()
    
    def extract_pdf_text_parallel(self, pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            num_pages = len(pages)
            logging.info(f"Processing {num_pages} pages in parallel for {pdf_path}")
            args = [(pdf_path, i) for i in range(num_pages)]
            with Pool(processes=min(Config.MAX_WORKERS, num_pages)) as pool:
                results = pool.map(process_pdf_page, args)
            processed_pages = [page for page in results if page is not None]
            logging.info(f"Successfully processed {len(processed_pages)} pages from {pdf_path}")
            return processed_pages
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")

    def split_into_chunks(self, pages):
        logging.info("Starting chunking of extracted pages...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_texts = [page['content'] for page in pages]
        documents = [Document(page_content=text) for text in all_texts]
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Chunking complete: {len(chunks)} chunks created.")
        return chunks

    def generate_embeddings(self, chunks):
        chunk_texts = [chunk.page_content for chunk in chunks]
        logging.info(f"Starting embedding for {len(chunk_texts)} chunks...")
        chunk_embeddings = self.embedder.embed_documents_batch(chunk_texts)
        logging.info(f"Embedding complete: {len(chunk_embeddings)} embeddings generated.")
        return chunk_texts, chunk_embeddings

    def store_embeddings(self, chunk_texts, chunk_embeddings, metadatas, ids):
        self.collection.add(
            documents=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully added {len(chunk_texts)} chunks to ChromaDB")
        count = self.collection.count()
        logging.info(f"Total chunks in collection: {count}")

    def process_pdf(self, pdf_path):
        pdf_name = os.path.basename(pdf_path)
        logging.info(f"=== STARTING PDF PROCESSING: {pdf_name} ===")
        start_time = time.time()
        try:
            pages = self.extract_pdf_text_parallel(pdf_path)
            logging.info(f"Extracted {len(pages)} pages from PDF")
            chunks = self.split_into_chunks(pages)
            chunk_texts, chunk_embeddings = self.generate_embeddings(chunks)
            ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunk_texts))]
            metadatas = [{
                "source": pdf_name,
                "content_type": "text",
                "page_number": pages[i % len(pages)]['page_number'] if pages else 0
            } for i in range(len(chunk_texts))]
            self.store_embeddings(chunk_texts, chunk_embeddings, metadatas, ids)
            processing_time = time.time() - start_time
            logging.info(f"=== COMPLETED PDF PROCESSING: {pdf_name} in {processing_time:.2f}s ===")
            return True
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}") 