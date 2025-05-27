import os
import requests
import logging
import pytesseract
from pdf2image import convert_from_path
from fastapi import FastAPI, Form, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import PersistentClient
from uvicorn import run
import shutil

class Config:
    CHROMA_DB_PATH = "./data1/chroma_db"
    EMBEDDING_MODEL_NAME = "nomic-embed-text"
    LLAMA_MODEL_NAME = "llama3"
    FASTAPI_PORT = 8005
    PDF_DIR = "./books"
    OLLAMA_ENDPOINT = "http://localhost:11434/api/embeddings"

    @classmethod
    def initialize(cls):
        os.makedirs(cls.PDF_DIR, exist_ok=True)

class OllamaEmbedder:
    def __init__(self, model_name="nomic-embed-text", endpoint="http://localhost:11434/api/embeddings"):
        self.model_name = model_name
        self.endpoint = endpoint

    def embed_query(self, text: str):
        response = requests.post(self.endpoint, json={
            "model": self.model_name,
            "prompt": text
        })
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: list[str]):
        return [self.embed_query(text) for text in texts]

class OCRProcessor:
    @staticmethod
    def ocr_image(image, page_number, source):
        text = pytesseract.image_to_string(image)
        image_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        has_table = any(conf > 80 for conf in image_data['conf'] if conf != -1) and \
                    len(set(image_data['left'])) > 3 and len(set(image_data['top'])) > 3
        
        width, height = image.size
        layout_type = "table" if has_table else "text"
        if width > height * 1.5:
            layout_type = "diagram_wide"
        elif height > width * 1.5:
            layout_type = "diagram_tall"
        
        metadata = {
            "page_number": page_number,
            "source": source,
            "content_type": "image",
            "layout_type": layout_type,
            "width": width,
            "height": height,
            "has_table": has_table,
            "confidence": sum(c for c in image_data['conf'] if c != -1) / len([c for c in image_data['conf'] if c != -1]) if image_data['conf'] else 0
        }
        
        formatted_text = f"""[Image Content - {layout_type.replace('_', ' ').title()}]
{text}

[Image Description]
This is a {layout_type.replace('_', ' ')} {width}x{height} pixels in size.
{'It appears to contain tabular data.' if has_table else ''}
"""
        
        return Document(page_content=formatted_text, metadata=metadata)

class PDFProcessor:
    def __init__(self, collection):
        self.collection = collection

    def extract_pdf_with_per_page_ocr(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        images = convert_from_path(pdf_path)
        processed_pages = []
        ocr_used = False

        for i, page in enumerate(pages):
            text = page.page_content.strip()
            if text:
                processed_pages.append(Document(
                    page_content=text,
                    metadata={
                        "page_number": i,
                        "source": os.path.basename(pdf_path),
                        "content_type": "text"
                    }
                ))
            else:
                ocr_used = True
                ocr_doc = OCRProcessor.ocr_image(images[i], i, os.path.basename(pdf_path))
                processed_pages.append(ocr_doc)

        if ocr_used:
            logging.info(f"🔍 Images detected and OCR applied in: {os.path.basename(pdf_path)}")

        return processed_pages

    def is_pdf_already_processed(self, pdf_name):
        chunk_id = f"{pdf_name}_chunk_0"
        results = self.collection.get(ids=[chunk_id], include=["metadatas"])
        return bool(results["ids"])

    def process_pdf(self, pdf_path):
        pdf_name = os.path.basename(pdf_path)

        if self.is_pdf_already_processed(pdf_name):
            logging.info(f"⏭️ Skipping already processed PDF: {pdf_name}")
            return False

        pages = self.extract_pdf_with_per_page_ocr(pdf_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500 if any(p.metadata.get('content_type') == 'image' for p in pages) else 800,
            chunk_overlap=150 if any(p.metadata.get('content_type') == 'image' for p in pages) else 100
        )
        chunks = text_splitter.split_documents(pages)

        chunk_texts = [chunk.page_content for chunk in chunks]
        embedder = OllamaEmbedder(model_name=Config.EMBEDDING_MODEL_NAME)
        chunk_embeddings = embedder.embed_documents(chunk_texts)

        ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunk_texts))]

        self.collection.add(
            documents=chunk_texts,
            embeddings=chunk_embeddings,
            metadatas=[{
                "source": pdf_name,
                "content_type": chunk.metadata.get('content_type', 'text'),
                "layout_type": chunk.metadata.get('layout_type', 'text'),
                "page_number": chunk.metadata.get('page_number', 0)
            } for chunk in chunks],
            ids=ids,
        )
        logging.info(f"✅ Processed: {pdf_name}")
        return True

class PromptGenerator:
    @staticmethod
    def get_enhanced_prompt(query, history_text, previous_question, previous_answer, retrieved_texts):
        has_images = "[Image Content" in retrieved_texts

        image_handling_instructions = """
       - For image content:
         • Describe what the image contains based on the OCR text
         • Explain diagrams and figures in a clear, structured way
         • For tables, present the information in a well-formatted manner
         • Reference visual elements when they support your explanation
         • Indicate when you're referring to information from an image
        """ if has_images else ""

        return f"""
You are an expert educational AI assistant with deep knowledge across various subjects. Your goal is to provide clear, accurate, and well-structured responses to questions about the content from PDF documents.

Context:
1. Previous Conversation:
{history_text}

2. Last Question: {previous_question}
Last Answer: {previous_answer}

3. Current Question: {query}

4. Retrieved Content from PDF:
{retrieved_texts}

Instructions for Answering:

1. RESPONSE STRUCTURE:
   - Start with a clear, direct answer to the question
   - Break down complex information into organized sections
   - Use appropriate formatting (bold, lists, etc.) for better readability
   - Maintain a professional, educational tone
   - For follow-up questions (identified by patterns below), acknowledge the connection to previous discussion:
     • Short queries: "why", "how", "examples", "key points", "explain more"
     • Clarification requests: "what do you mean by X", "can you clarify"
     • Extension requests: "tell me more", "what about X", "continue"
     • Detail requests: "give examples", "explain in detail", "elaborate"
     • Point-specific queries: "what are the key points", "list the main ideas"
     • Comparison queries: "how does X compare to Y", "what's the difference"

2. CONTENT GUIDELINES:
   - Primary Source: Use the retrieved PDF content as your main reference
   - Accuracy: If information is not clearly found in the PDF, acknowledge this and provide a general response based on the context
   - Depth: Match the detail level to the question's complexity
   - Citations: Reference specific sections or pages when relevant
   - For follow-ups: Reference relevant information from previous responses before adding new details{image_handling_instructions}

3. FORMATTING BEST PRACTICES:
   - Use **bold** for key terms and important concepts
   - Create organized lists with • or numbers for multiple points
   - Add subheadings for different sections in complex answers
   - Use tables when comparing multiple items
   - Include examples when they help clarify concepts

4. RESPONSE TYPES:
   A. For Simple Questions:
      - Provide a concise, direct answer
      - Add 1-2 supporting points if relevant
      - Keep formatting minimal

   B. For Complex Questions:
      Structure your response as:
      1. Brief Overview
      2. Detailed Explanation
      3. Key Points
      4. Examples (if relevant)
      5. Summary (for long answers)

5. SPECIAL INSTRUCTIONS - EDUCATIONAL ASSESSMENTS:
   ONLY IF EXPLICITLY REQUESTED by phrases like "create questions", "make a quiz", "test my knowledge", "practice problems", etc., include ONE or MORE of:
   
   - Multiple Choice Questions (5 questions, 4 options each)
   - Fill in the Blanks
   - True/False Statements
   - Short Answer Questions
   - Match the Following
   
   Otherwise, DO NOT include any assessment materials in your response.

6. QUALITY CHECKS:
   - Ensure response directly addresses the user's question
   - Verify all information against the provided PDF content
   - Maintain consistent formatting throughout
   - Keep language clear and accessible
   - Avoid unnecessary jargon unless specifically relevant
   - For follow-ups: Ensure smooth connection with previous context

Remember: Focus on being helpful, clear, and accurate while maintaining a professional educational tone. Structure your response to make complex information easily digestible.
"""

class ChatManager:
    def __init__(self):
        self.chat_histories = {}

    def get_history(self, chat_key):
        return self.chat_histories.get(chat_key, [])

    def update_history(self, chat_key, question, answer):
        if chat_key not in self.chat_histories:
            self.chat_histories[chat_key] = []
        self.chat_histories[chat_key].append({"question": question, "answer": answer})

    def get_previous_qa(self, chat_key):
        history = self.get_history(chat_key)
        if not history:
            return "", ""
        last_entry = history[-1]
        return last_entry["question"], last_entry["answer"]

class QueryAnalyzer:
    @staticmethod
    def analyze_query(query, previous_question=""):
        normalized_query = query.lower().strip()
        
        follow_up_patterns = {
            'short_queries': ['why', 'how', 'examples', 'explain', 'elaborate', 'continue', 'points'],
            'clarification': ['what do you mean', 'can you clarify', 'what is', 'define'],
            'extension': ['tell me more', 'what about', 'go on', 'and'],
            'detail': ['give examples', 'explain in detail', 'list', 'enumerate'],
            'comparison': ['how does', 'what is the difference', 'compare', 'versus', 'vs']
        }

        new_question_indicators = {
            'question_starters': ['what', 'who', 'where', 'when', 'why', 'how', 'can', 'could'],
            'topic_shifts': ['another', 'different', 'new', 'next', 'instead', 'rather'],
            'complete_sentence': lambda q: len(q.split()) > 4 and q.endswith(('?', '.', '!')),
            'has_subject_verb': lambda q: len(q.split()) >= 3
        }

        is_new_question = (
            (any(normalized_query.startswith(starter + ' ') for starter in new_question_indicators['question_starters']) 
             and new_question_indicators['complete_sentence'](normalized_query))
            or
            any(shift in normalized_query for shift in new_question_indicators['topic_shifts'])
            or
            (new_question_indicators['has_subject_verb'](normalized_query) 
             and not any(pattern in normalized_query for patterns in follow_up_patterns.values() for pattern in patterns))
        )

        is_follow_up = (
            any(pattern in normalized_query for patterns in follow_up_patterns.values() for pattern in patterns) or
            len(normalized_query.split()) <= 4 or
            normalized_query.endswith('?') or
            not normalized_query.endswith(('.', '!', '?'))
        ) if not is_new_question else False

        return is_new_question, is_follow_up

class EduQueryAPI:
    def __init__(self):
        Config.initialize()
        self.app = FastAPI()
        self.setup_middleware()
        self.setup_routes()
        self.client = PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name="pdf_collection")
        self.chat_manager = ChatManager()
        self.pdf_processor = PDFProcessor(self.collection)
        
        logging.basicConfig(level=logging.DEBUG)

    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        @self.app.get("/books/")
        async def list_books():
            existing_books = []
            for book in os.listdir(Config.PDF_DIR):
                if book.endswith('.pdf') and os.path.isfile(os.path.join(Config.PDF_DIR, book)):
                    existing_books.append(book)
            return {"books": existing_books}

        @self.app.get("/book")
        async def get_pdf(subject: str = Query(...)):
            file_path = os.path.join(Config.PDF_DIR, subject)
            if os.path.isfile(file_path):
                return FileResponse(
                    file_path,
                    media_type='application/pdf',
                    filename=subject,
                    headers={"Content-Disposition": "inline"}
                )
            return {"error": "PDF not found"}

        @self.app.post("/query/")
        async def query_rag(query: str = Form(...), session_id: str = Form(...), 
                          book_name: str = Form(...)):
            return await self._handle_query(query, session_id, book_name)

        @self.app.post("/upload/")
        async def upload_pdf(file: UploadFile = File(...)):
            return await self._handle_upload(file)

    async def _handle_query(self, query: str, session_id: str, book_name: str):
        chat_key = f"{session_id}:{book_name}"
        previous_question, previous_answer = self.chat_manager.get_previous_qa(chat_key)
        
        is_new_question, is_follow_up = QueryAnalyzer.analyze_query(query, previous_question)
        
        # Handle casual queries
        if self._is_casual_query(query):
            return await self._handle_casual_query(query, chat_key)
        
        # Process regular query
        embedder = OllamaEmbedder(model_name=Config.EMBEDDING_MODEL_NAME)
        query_embedding = embedder.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=3,
            where={"source": book_name}
        )
        
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        sources = [meta["source"] for meta in metadatas] if metadatas else []
        
        retrieved_texts = "\n\n".join(documents) if documents else "No relevant content found."
        
        history = self.chat_manager.get_history(chat_key)
        history_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" 
                                 for h in history[-5:]])
        
        prompt = PromptGenerator.get_enhanced_prompt(
            query, history_text, previous_question, previous_answer, retrieved_texts
        )
        
        llama_model = ChatOllama(model=Config.LLAMA_MODEL_NAME)
        response = llama_model.invoke(prompt).content
        
        self.chat_manager.update_history(
            chat_key, 
            query if is_new_question else f"{previous_question} -> {query}", 
            response
        )
        
        return {"answer": response, "sources": sources}

    async def _handle_upload(self, file: UploadFile):
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        file_path = None
        try:
            file_path = os.path.join(Config.PDF_DIR, file.filename)
            
            if os.path.exists(file_path):
                raise HTTPException(status_code=400, detail="A file with this name already exists")
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            success = self.pdf_processor.process_pdf(file_path)
            
            if success:
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "PDF uploaded and processed successfully",
                        "filename": file.filename
                    }
                )
            else:
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "PDF uploaded but skipped processing (already exists in database)",
                        "filename": file.filename
                    }
                )
                
        except Exception as e:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    def _is_casual_query(self, query: str) -> bool:
        casual_keywords = [
            "hi", "hello", "hey", "hii", "heyy", "yo", "what's up", "how are you",
            "how's it going", "who are you", "what can you do", "good morning",
            "good evening", "good night", "bye", "see you", "thanks", "thank you"
        ]
        return any(keyword in query.lower() for keyword in casual_keywords)

    async def _handle_casual_query(self, query: str, chat_key: str):
        history = self.chat_manager.get_history(chat_key)
        history_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" 
                                 for h in history[-5:]])
        
        casual_prompt = f"""
You are a friendly and helpful assistant. Here's the conversation so far:

{history_text}

User: {query}
Assistant:
"""
        llama_model = ChatOllama(model=Config.LLAMA_MODEL_NAME)
        response = llama_model.invoke(casual_prompt).content
        
        self.chat_manager.update_history(chat_key, query, response)
        return {"answer": response, "sources": []}

# Initialize the API and expose the FastAPI instance
api = EduQueryAPI()
app = api.app

def main():
    run(app, host="0.0.0.0", port=Config.FASTAPI_PORT)

if __name__ == "__main__":
    main()
