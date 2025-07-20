import os
import shutil
import logging
import hashlib
import pickle
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import PersistentClient
import google.generativeai as genai
import ollama 
import re 
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import time
from typing import List, Dict, Any
load_dotenv()
  

# --- Configuration ---
class Config:
    CHROMA_DB_PATH = "./data/chroma_db"
    EMBEDDING_MODEL_NAME = "bge-m3:latest" 
    FASTAPI_PORT = 8006
    PDF_DIR = "./books"
    CACHE_DIR = "./cache"
    MAX_CACHE_SIZE = 1000  # Maximum number of cached responses
    BATCH_SIZE = 10  # Batch size for embeddings
    MAX_WORKERS = max(1, cpu_count() - 1)  # Use all but one CPU core

    @classmethod
    def initialize(cls):
        os.makedirs(cls.PDF_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.CHROMA_DB_PATH), exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

# --- Caching System ---
class ResponseCache:
    def __init__(self, cache_dir: str, max_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache_file = os.path.join(cache_dir, "llm_cache.pkl")
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")
    
    def _get_hash(self, prompt: str) -> str:
        """Generate hash for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> str:
        """Get cached response for prompt"""
        key = self._get_hash(prompt)
        if key in self.cache:
            logging.info(f"Cache hit for prompt hash: {key[:8]}...")
            return self.cache[key]
        return None
    
    def set(self, prompt: str, response: str):
        """Cache response for prompt"""
        key = self._get_hash(prompt)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = response
        self._save_cache()
        logging.info(f"Cached response for prompt hash: {key[:8]}...")

# --- Optimized Embedding Service ---
class OptimizedOllamaEmbedder:
    def __init__(self, model_name=Config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
    
    def embed_query(self, text: str):
        """Generate embedding for a single query using ollama.embeddings()"""
        try:
            result = ollama.embeddings(model=self.model_name, prompt=text)
            return result["embedding"]
        except Exception as e:
            logging.error(f"Error generating query embedding: {e}")
            raise
    
    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        logging.info(f"Generating batch embeddings for {len(texts)} documents")
        all_embeddings = []
        for i in range(0, len(texts), Config.BATCH_SIZE):
            batch = texts[i:i + Config.BATCH_SIZE]
            batch_embeddings = []
            try:
                if hasattr(ollama, 'embeddings_batch'):
                    results = ollama.embeddings_batch(model=self.model_name, prompts=batch)
                    batch_embeddings = [result["embedding"] for result in results]
                else:
                    for j, text in enumerate(batch):
                        try:
                            embedding = self.embed_query(text)
                            batch_embeddings.append(embedding)
                        except Exception as e:
                            logging.error(f"Failed to embed text {i+j}: {e}")
                            batch_embeddings.append([0.0] * 384)
                all_embeddings.extend(batch_embeddings)
                logging.info(f"Processed batch {i//Config.BATCH_SIZE + 1}/{(len(texts) + Config.BATCH_SIZE - 1)//Config.BATCH_SIZE}")
            except Exception as e:
                logging.error(f"Failed to process batch {i//Config.BATCH_SIZE + 1}: {e}")
                all_embeddings.extend([[0.0] * 384] * len(batch))
        logging.info(f"Completed batch embedding generation: {len(all_embeddings)} embeddings")
        return all_embeddings

# --- Multiprocessing PDF Processing ---
def process_pdf_page(args):
    """Process a single PDF page (for multiprocessing)"""
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
        self.embedder = OptimizedOllamaEmbedder()
    
    def extract_pdf_text_parallel(self, pdf_path):
        """Extract text from PDF using multiprocessing"""
        try:
            # First, get the number of pages
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            num_pages = len(pages)
            
            logging.info(f"Processing {num_pages} pages in parallel for {pdf_path}")
            
            # Prepare arguments for multiprocessing
            args = [(pdf_path, i) for i in range(num_pages)]
            
            # Use multiprocessing to process pages in parallel
            with Pool(processes=min(Config.MAX_WORKERS, num_pages)) as pool:
                results = pool.map(process_pdf_page, args)
            
            # Filter out None results and return valid pages
            processed_pages = [page for page in results if page is not None]
            logging.info(f"Successfully processed {len(processed_pages)} pages from {pdf_path}")
            return processed_pages
            
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {str(e)}")

    def split_into_chunks(self, pages):
        """Split extracted pages into chunks using RecursiveCharacterTextSplitter"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_texts = [page['content'] for page in pages]
        documents = [Document(page_content=text) for text in all_texts]
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Created {len(chunks)} chunks from pages")
        return chunks

    def generate_embeddings(self, chunks):
        """Generate embeddings for the given chunks using batch processing"""
        chunk_texts = [chunk.page_content for chunk in chunks]
        logging.info("Starting batch embedding generation...")
        chunk_embeddings = self.embedder.embed_documents_batch(chunk_texts)
        logging.info(f"Generated {len(chunk_embeddings)} embeddings")
        return chunk_texts, chunk_embeddings

    def store_embeddings(self, chunk_texts, chunk_embeddings, metadatas, ids):
        """Store the embeddings and related data in ChromaDB"""
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
        """Process PDF: extract, chunk, embed, and store (optimized)"""
        pdf_name = os.path.basename(pdf_path)
        logging.info(f"=== STARTING OPTIMIZED PDF PROCESSING: {pdf_name} ===")
        start_time = time.time()
        
        try:
            # Extract text using multiprocessing
            pages = self.extract_pdf_text_parallel(pdf_path)
            logging.info(f"Extracted {len(pages)} pages from PDF")

            # Split into chunks
            chunks = self.split_into_chunks(pages)

            # Generate embeddings using batch processing
            chunk_texts, chunk_embeddings = self.generate_embeddings(chunks)

            # Prepare metadata and IDs
            ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunk_texts))]
            metadatas = [{
                "source": pdf_name,
                "content_type": "text",
                "page_number": pages[i % len(pages)]['page_number'] if pages else 0
            } for i in range(len(chunk_texts))]

            # Store in ChromaDB
            self.store_embeddings(chunk_texts, chunk_embeddings, metadatas, ids)

            processing_time = time.time() - start_time
            logging.info(f"=== COMPLETED OPTIMIZED PDF PROCESSING: {pdf_name} in {processing_time:.2f}s ===")
            return True
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


# --- Chat Management ---
class ChatManager:
    def __init__(self):
        self.chat_histories = {}
    
    def get_history(self, session_id: str):
        return self.chat_histories.get(session_id, [])
    
    def add_to_history(self, session_id: str, question: str, answer: str, embedding: list = None):
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        self.chat_histories[session_id].append({
            "question": question,
            "answer": answer,
            "embedding": embedding,
            "qa_text": f"Q: {question}\nA: {answer}"
        })
    
    def get_recent_history(self, session_id: str, limit: int = 5):
        history = self.get_history(session_id)
        return history[-limit:]

    def get_hybrid_history(self, session_id: str, query_embedding, recent_n: int = 5, relevant_k: int = 2):
        import numpy as np
        history = self.get_history(session_id)
        if not history:
            return []
        recent = history[-recent_n:]
        candidates = [h for h in history if h.get("embedding") is not None and h not in recent]
        if not candidates or query_embedding is None:
            return recent
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        scored = [(h, cosine_sim(query_embedding, h["embedding"])) for h in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        relevant = [h[0] for h in scored[:relevant_k] if h[1] > 0.85]
        return recent + relevant

def is_followup_query(query: str) -> bool:
    """Detect if a query is a follow-up to the previous question."""
    # Check for obvious follow-up phrases
    obvious_followup_phrases = [
        'more', 'in short', 'briefly', 'summarize', 'key points', 'uses', 'examples',
        'explain about', 'tell me about', 'describe about', 'what about', 'how about',
        'in detail', 'in depth', 'more details', 'more information', 'give details',
        'explain them', 'tell me them', 'describe them', 'about them', 'them',
        'simple terms', 'simple', 'beginner', 'easy', 'basics'
    ]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in obvious_followup_phrases)

def is_new_topic_query(query: str, recent_history) -> bool:
    """Detect if the query is about a completely new topic compared to recent history."""
    if not recent_history:
        return True
    
    # Extract key terms from current query
    def extract_terms(text):
        # Simple term extraction - you could use more sophisticated NLP here
        terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        # Remove common words
        common_words = {'what', 'is', 'are', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'this', 'that', 'these', 'those', 'how', 'why', 'when', 'where', 'who', 'which', 'tell', 'me', 'more', 'detail', 'explain', 'state', 'all', 'uses', 'use', 'computer', 'network', 'networks', 'simple', 'beginner', 'easy', 'basics', 'short', 'key', 'points', 'summary', 'them', 'they', 'their'}
        return terms - common_words
    
    current_terms = set(extract_terms(query.lower()))
    
    # Extract key terms from recent history (look at more entries for better context)
    history_terms = set()
    for entry in recent_history[-5:]:  # Look at last 5 entries for better context
        question = entry.get('question', '').lower()
        answer = entry.get('answer', '').lower()
        history_terms.update(extract_terms(question))
        history_terms.update(extract_terms(answer))
    
    # Check for obvious new topic indicators
    new_topic_indicators = [
        'what is', 'define', 'explain', 'tell me about', 'describe',
        'how does', 'what are', 'list', 'show me', 'give me',
        'big data', 'analytics', 'sectors', 'industries', 'applications',
        'mapreduce', 'cloud computing', 'grid computing', 'data analysis'
    ]
    
    # If query contains new topic indicators, it might be a new topic
    has_new_topic_indicators = any(indicator in query.lower() for indicator in new_topic_indicators)
    
    # If there's significant overlap, it's likely the same topic
    if current_terms and history_terms:
        overlap = len(current_terms.intersection(history_terms))
        total_unique = len(current_terms.union(history_terms))
        similarity = overlap / total_unique if total_unique > 0 else 0
        
        # More lenient threshold for follow-ups - only treat as new topic if very different
        return similarity < 0.15 and has_new_topic_indicators
    
    return has_new_topic_indicators  # Default based on indicators

# --- Optimized Gemini LLM Integration ---
class OptimizedGeminiLLM:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.cache = ResponseCache(Config.CACHE_DIR, Config.MAX_CACHE_SIZE)

    def invoke(self, prompt: str):
        try:
            # Check cache first
            cached_response = self.cache.get(prompt)
            if cached_response:
                return type('Obj', (object,), {'content': cached_response})
            
            # Generate new response
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Cache the response
            self.cache.set(prompt, response_text)
            
            return type('Obj', (object,), {'content': response_text})
        except Exception as e:
            logging.error(f"Gemini LLM error: {e}")
            return type('Obj', (object,), {'content': f"Error from Gemini: {str(e)}"})

# --- Casual Query Handling ---
CASUAL_RESPONSES = {
    "hello": "Hello! How can I assist you with your PDF content today?",
    "hi": "Hi there! Ready to help with your queries.",
    "hey": "Hey! Ask me anything related to the document.",
    "thanks": "You're welcome!",
    "thank you": "Glad I could help!",
    "who are you": "I'm an educational assistant designed to answer questions from your PDF content.",
    "good morning": "Good morning! Let me know what you'd like to learn today.",
    "good afternoon": "Good afternoon! How can I help you with the document?",
    "good evening": "Good evening! Ready to assist with your PDF queries.",
    "bye": "Goodbye! Feel free to return if you have more questions.",
    "goodbye": "Take care! Come back anytime for more help.",
    "see you": "See you later! Have a great day.",
    "how are you": "I'm doing well, thank you! How can I help you with your PDF content?",
    "what can you do": "I can answer questions about your uploaded PDF documents, help you understand the content, and provide detailed explanations. Just ask me anything related to the document!",
    "help": "I'm here to help! You can ask me questions about your PDF content, request explanations, or ask for summaries. What would you like to know?",
}

def is_casual_query(query: str) -> bool:
    """Detect if a query is casual/greeting using regex for exact keyword match."""
    normalized_query = query.lower().strip()
    for keyword in CASUAL_RESPONSES.keys():
        if re.search(rf'\b{re.escape(keyword)}\b', normalized_query):
            return True
    return False

def handle_casual_query(query: str) -> str:
    """Handle casual queries with predefined responses using regex lookup."""
    normalized_query = query.lower().strip()
    for keyword, response in CASUAL_RESPONSES.items():
        if re.search(rf'\b{re.escape(keyword)}\b', normalized_query):
            return response
    return "Hello! Feel free to ask questions about the PDF content."

# --- Follow-up Query Rewriting ---
def rewrite_followup_query(current_query, chat_history):
    """Rewrite follow-up queries by combining with the last user question if needed."""
    # Check if it's a follow-up query
    if is_followup_query(current_query):
        # Find last user question in chat history (skip casual queries)
        for entry in reversed(chat_history):
            q = entry.get("question", "")
            if q and not is_casual_query(q):
                last_question = q
                break
        else:
            last_question = ""
        
        # For very short follow-ups (1-2 words), combine with previous question
        if last_question and len(current_query.split()) <= 2:
            return f"{last_question} {current_query}".strip()
        # For longer follow-ups, keep as is but they'll get context from history
    
    return current_query

# --- Optimized Query Processing ---
class QueryProcessor:
    def __init__(self, collection, chat_manager):
        self.collection = collection
        self.chat_manager = chat_manager
        self.embedder = OptimizedOllamaEmbedder()
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.llm = OptimizedGeminiLLM(api_key)
        logging.info("Using Optimized Gemini LLM with caching for responses.")

    def embed_query(self, query: str):
        """Generate embedding for the query string"""
        return self.embedder.embed_query(query)

    def semantic_search(self, query_embedding, book_name: str, n_results: int):
        """Perform semantic search in the collection"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"source": book_name}
        )
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        return documents, metadatas

    def handle_list_query_fallback(self, book_name: str):
        """Fallback for list-type queries: get all available chunks for this book"""
        all_results = self.collection.get(
            where={"source": book_name},
            include=["documents", "metadatas"]
        )
        all_docs = all_results.get("documents", [])
        all_metas = all_results.get("metadatas", [])
        return all_docs, all_metas

    def retrieve_relevant_content(self, query: str, book_name: str, n_results: int = 15):
        """Retrieve relevant content using semantic search with enhanced retrieval (optimized)"""
        try:
            LIST_KEYWORDS = ['list', 'all', 'every', 'complete', 'entire', 'phases', 'steps', 'process']
            query_lower = query.lower()
            is_list_query = any(keyword in query_lower for keyword in LIST_KEYWORDS)

            # Generate query embedding
            query_embedding = self.embed_query(query)

            # For list-type queries or detailed explanations, retrieve more content
            if is_list_query:
                n_results = max(20, n_results)

            # Search for relevant chunks
            documents, metadatas = self.semantic_search(query_embedding, book_name, n_results)

            if not documents:
                return [], []

            # If we have documents but they seem incomplete for list queries, try to get more
            if is_list_query:
                all_docs, all_metas = self.handle_list_query_fallback(book_name)
                if len(all_docs) > len(documents):
                    documents = all_docs
                    metadatas = all_metas

            logging.info(f"Retrieved {len(documents)} chunks for query: {query}")
            return documents, metadatas

        except Exception as e:
            logging.error(f"Error retrieving content: {e}")
            return [], []
    
    def build_strict_prompt(self, query: str, retrieved_texts: str, chat_history: str = ""):
        """Build the strict prompt for the LLM with clear separation between history and current question."""
        base_prompt = f"""
IMPORTANT INSTRUCTIONS:

1. Answer using ONLY the information from the PDF content provided below.
2. Do NOT use any outside knowledge, general information, or make any guesses.
3. If the information is not in the PDF, explicitly state \"The answer is not present in the provided PDF content. Can you please ask something from the PDF?\"
4. For list-type questions, extract ONLY the items that are explicitly mentioned in the PDF content. Do NOT add or infer any items.
5. If a source is not directly listed or described in the PDF, do NOT include it in your answer.
6. Preserve the structure (e.g., tables, bullet points) as in the PDF.
7. If the user asks for 'one point', 'key points', or a summary about multiple items (such as topics, entities, etc.), search the entire PDF content for each item and present the most relevant information found for each, if present. If no information is present for an item, state that the answer is not present in the provided PDF content and based on the content present use knowledge to give a accurate information. This instruction applies to all types of PDFs, not just sector-based queries.
8. COMPREHENSIVE RESPONSE: Always provide ALL available information related to the query from the PDF content, even if the user doesn't explicitly ask for \"all\" or \"list\". Give complete, thorough answers that cover all relevant points, aspects, or items found in the document. Do not limit yourself to just a few points unless the user specifically requests brevity.
9. Format the response clearly, if needed:
   - Use headings and subheadings where appropriate.
   - Use bullet points or numbered lists if they help readability or understanding.
   - Keep sections well-organized and easy to read.
   - If the content is very short or simple, avoid unnecessary formatting—keep it concise.
10. If the user asks for an explanation in simple terms, easy method, or mentions being a beginner:  
   - Provide a clear, simplified explanation using ONLY the information from the PDF content provided below.  
   - Use beginner-friendly, easy-to-understand language.  
   - Avoid complex terms, technical jargon, or overly detailed descriptions unless specifically present in the PDF.  
   - Break down concepts step by step based strictly on the structure and explanations in the PDF.  
   - Provide relevant examples or analogies only if they are included in the PDF content. Do not invent or add outside information.  
11. CONTEXT INTELLIGENCE: Analyze the user's question in relation to the conversation context. If it's a follow-up question (asking for more details, explanations, examples, clarifications, or different formats about the previous topic), provide additional information about that topic. If it's a completely new topic, focus exclusively on the new question.

PDF Content:
{retrieved_texts}

"""
        if chat_history:
            prompt = f"""
You are a helpful EDUCATIONAL AI assistant designed to answer questions about PDF content.

CONVERSATION FLOW INTELLIGENCE:
- Analyze the entire conversation flow to understand the current context
- Detect if this is a follow-up question or a completely new topic
- Maintain context across multiple follow-up questions about the same subject
- Only treat as a new topic when the user explicitly asks about something completely different

FOLLOW-UP PATTERN RECOGNITION:
- If the user asks for more details, explanations, examples, or clarifications → FOLLOW-UP
- If the user asks for different formats (headings, summary, key points, simple terms) → FOLLOW-UP
- If the user asks "what about X" or "tell me about X" where X relates to previous topic → FOLLOW-UP
- If the user asks follow-up questions repeatedly about the same topic → CONTINUE FOLLOW-UP
- If the user asks for simpler explanations or beginner-friendly content → FOLLOW-UP
- If the user uses pronouns like "them", "they", "it" referring to previous content → FOLLOW-UP

NEW TOPIC DETECTION:
- Only treat as new topic if user explicitly asks about a completely different subject
- Look for new topic indicators like "what is", "define", "explain", "list" for different subjects
- If the user asks about the same subject but in a different way → STILL FOLLOW-UP

MULTIPLE FOLLOW-UP HANDLING:
- The conversation may contain multiple follow-up questions about the same topic
- Each follow-up should build upon the previous information provided
- Consider the entire conversation flow, not just the last question
- Maintain context until a completely new topic is introduced

RESPONSE GUIDELINES:
- For follow-ups: Provide additional information about the topic from the conversation context
- For new topics: Answer only the current question, do not reference previous topics
- Always base your answer on the PDF content provided below
- Be contextually aware and intelligent about topic transitions
- Build upon previous responses when handling multiple follow-ups
- If user asks for simpler explanations, provide beginner-friendly content

Previous conversation history (for context):
{chat_history}

Now, intelligently analyze and answer the following question based ONLY on the PDF content provided below:

{base_prompt}

Question: {query}

Answer:
"""
        else:
            prompt = f"""
{base_prompt}

Question: {query}

Answer:
"""
        return prompt


    def generate_response(self, query: str, retrieved_texts: str, session_id: str):
        """Generate LLM response based on retrieved content with strict PDF-only context, using hybrid chat recall."""
        try:
            # Get embedding for the current query
            query_embedding = self.embed_query(query)
            
            # Check if this is a follow-up query
            is_followup = is_followup_query(query)
            
            # Get recent history for topic detection (increased to handle multiple follow-ups)
            recent_history = self.chat_manager.get_recent_history(session_id, limit=8)
            
            # Get hybrid Q/A pairs from history (increased context for better follow-up handling)
            hybrid_history = self.chat_manager.get_hybrid_history(session_id, query_embedding, recent_n=3, relevant_k=2)
            
            # Enhanced follow-up logic: Always include context for follow-ups, be more lenient for new topics
            if is_followup:
                # For follow-ups, always include history for context
                pass  # Keep hybrid_history as is
            else:
                # Check if this is a new topic (only if we have recent history)
                is_new_topic = is_new_topic_query(query, recent_history) if recent_history else True
                if is_new_topic:
                    hybrid_history = []
            
            history_text = ""
            if hybrid_history:
                history_text = "\n\n".join([
                    f"Q: {h['question']}\nA: {h['answer']}" for h in hybrid_history
                ])
            
            # Build prompt using helper
            prompt = self.build_strict_prompt(query, retrieved_texts, history_text)
            
            # Generate response (with caching)
            response = self.llm.invoke(prompt).content
            return response, query_embedding
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", None

# --- Optimized FastAPI App ---
class EduQueryAPI:
    def __init__(self):
        Config.initialize()
        self.app = FastAPI(title="Optimized PDF QA API")
        self.setup_middleware()
        self.setup_routes()
        
        # Initialize services
        self.client = PersistentClient(path=Config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name="pdf_collection")
        self.pdf_processor = PDFProcessor(self.collection)
        self.chat_manager = ChatManager()
        self.query_processor = QueryProcessor(self.collection, self.chat_manager)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logging.info("Optimized EduQuery API initialized")
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "Optimized EduQuery API is running", "status": "ok"}
        
        # REMOVED TEST ENDPOINT THAT RETURNS COLLECTION COUNTS
        # REMOVED MAINTENANCE ENDPOINT
        
        @self.app.post("/upload/")
        async def upload_pdf(file: UploadFile = File(...)):
            """Upload and process a PDF file with optimized processing"""
            try:
                if not file.filename.endswith('.pdf'):
                    raise HTTPException(status_code=400, detail="Only PDF files are allowed")
                
                file_path = os.path.join(Config.PDF_DIR, file.filename)
                if os.path.exists(file_path):
                    raise HTTPException(status_code=400, detail="A file with this name already exists")
                
                # Save file
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process PDF with optimized processing
                self.pdf_processor.process_pdf(file_path)
                
                return JSONResponse(
                    status_code=200, 
                    content={"message": "PDF uploaded and processed successfully with optimizations", "filename": file.filename}
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logging.error(f"Error uploading PDF: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to upload PDF: {str(e)}")
        
        @self.app.get("/books/")
        async def list_books():
            """List all uploaded PDFs"""
            try:
                books = [b for b in os.listdir(Config.PDF_DIR) if b.endswith('.pdf')]
                return {"books": books}
            except Exception as e:
                logging.error(f"Error listing books: {e}")
                return {"books": [], "error": str(e)}
        
        @self.app.get("/book")
        async def get_pdf(subject: str = Query(...)):
            """Get a specific PDF file for viewing"""
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
        async def query_pdf(
            query: str = Form(...), 
            book_name: str = Form(...), 
            session_id: str = Form(default="default"),
            debug: bool = Form(default=False)
        ):
            try:
                book_specific_session = f"{session_id}_{book_name}"
                if is_casual_query(query):
                    casual_response = handle_casual_query(query)
                    self.chat_manager.add_to_history(book_specific_session, query, casual_response)
                    return {"answer": casual_response, "sources": []}
                chat_history = self.chat_manager.get_history(book_specific_session)
                query_to_use = rewrite_followup_query(query, chat_history)
                documents, metadatas = self.query_processor.retrieve_relevant_content(query_to_use, book_name)
                if not documents:
                    return {"answer": "No relevant content found in the PDF for your query.", "sources": []}
                retrieved_texts = "\n\n".join(documents)
                response, query_embedding = self.query_processor.generate_response(query_to_use, retrieved_texts, book_specific_session)
                self.chat_manager.add_to_history(book_specific_session, query, response, query_embedding)
                result = {
                    "answer": response, 
                    "sources": [meta.get("source", book_name) for meta in metadatas]
                }
                if debug:
                    result["debug"] = {
                        "chunks_retrieved": len(documents),
                        "total_chunks_in_book": self.collection.count(),
                        "retrieved_content_preview": retrieved_texts[:1000] + "..." if len(retrieved_texts) > 1000 else retrieved_texts,
                        "book_specific_session": book_specific_session,
                        "cache_enabled": True,
                        "batch_processing": True
                    }
                return result
            except Exception as e:
                logging.error(f"Error processing query: {e}")
                return {"answer": f"Error processing query: {str(e)}", "sources": []}
        
        
        @self.app.post("/end_session/")
        async def end_session(session_id: str = Form(...), book_name: str = Form(...)):
            """End a session and optionally remove PDF"""
            try:
                # Remove PDF file if requested
                file_path = os.path.join(Config.PDF_DIR, book_name)
                removed = False
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed = True
                
                # Remove embeddings
                try:
                    # Delete by metadata filter - this is the correct way to remove embeddings for a specific book
                    self.collection.delete(where={"source": book_name})
                    logging.info(f"Successfully removed embeddings for {book_name}")
                except Exception as e:
                    logging.error(f"Failed to remove embeddings for {book_name}: {e}")
                
                # Clear chat history
                if session_id in self.chat_manager.chat_histories:
                    del self.chat_manager.chat_histories[session_id]
                
                return {
                    "removed": removed, 
                    "filename": book_name, 
                    "message": "Session ended successfully"
                }
                
            except Exception as e:
                logging.error(f"Error ending session: {e}")
                return {"error": f"Failed to end session: {str(e)}"}

# --- Initialize and run ---
api = EduQueryAPI()
app = api.app

def main():
    run(app, host="0.0.0.0", port=Config.FASTAPI_PORT)

if __name__ == "__main__":
    main()
