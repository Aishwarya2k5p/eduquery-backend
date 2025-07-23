import logging
import os
from EduQuery.embedder import OllamaEmbedder
from EduQuery.chat_manager import ChatManager
from EduQuery.llm import GeminiLLM
from EduQuery.config import Config
from EduQuery.utils import is_followup_query, is_new_topic_query, rewrite_followup_query

class QueryProcessor:
    def __init__(self, collection, chat_manager: ChatManager):
        self.collection = collection
        self.chat_manager = chat_manager
        self.embedder = OllamaEmbedder()
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.llm = GeminiLLM(api_key)
        logging.info("Using Gemini LLM with caching for responses.")

    def embed_query(self, query: str):
        return self.embedder.embed_query(query)

    def semantic_search(self, query_embedding, book_name: str, n_results: int):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"source": book_name}
        )
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        return documents, metadatas

    def handle_list_query_fallback(self, book_name: str):
        all_results = self.collection.get(
            where={"source": book_name},
            include=["documents", "metadatas"]
        )
        all_docs = all_results.get("documents", [])
        all_metas = all_results.get("metadatas", [])
        return all_docs, all_metas

    def retrieve_relevant_content(self, query: str, book_name: str, n_results: int = 15):
        try:
            LIST_KEYWORDS = ['list', 'all', 'every', 'complete', 'entire', 'phases', 'steps', 'process']
            query_lower = query.lower()
            is_list_query = any(keyword in query_lower for keyword in LIST_KEYWORDS)
            query_embedding = self.embed_query(query)
            if is_list_query:
                n_results = max(20, n_results)
            documents, metadatas = self.semantic_search(query_embedding, book_name, n_results)
            if not documents:
                return [], []
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
        base_prompt = f"""
IMPORTANT INSTRUCTIONS:

1. Answer using ONLY the information from the PDF content provided below.
2. Do NOT use any outside knowledge, general information, or make any guesses.
3. If the information is not in the PDF, explicitly state \"The answer is not present in the provided PDF content. Can you please ask something from the PDF?\"
4. For list-type questions, extract ONLY the items that are explicitly mentioned in the PDF content. Do NOT add or infer any items.
5. If a source is not directly listed or described in the PDF, do NOT include it in your answer.
6. Preserve the structure (e.g., tables, bullet points) as in the PDF.
7. If the user asks for 'one point', 'key points', or a summary about multiple items (such as topics, entities, etc.), search the entire PDF content for each item and present the most relevant information found for each, if present. If no information is present for an item, state that the answer is not present in the provided PDF content and based on the content present, use knowledge to give accurate information. This instruction applies to all types of PDFs.
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

    def generate_response(self, query: str, retrieved_texts: str, session_id: str, book_name: str = None):
        try:
            query_embedding = self.embed_query(query)
            is_followup = is_followup_query(query)
            recent_history = self.chat_manager.get_recent_history(session_id, limit=8)
            hybrid_history = self.chat_manager.get_hybrid_history(session_id, query_embedding, recent_n=3, relevant_k=2)
            if is_followup:
                pass
            else:
                is_new_topic = is_new_topic_query(query, recent_history) if recent_history else True
                if is_new_topic:
                    hybrid_history = []
            history_text = ""
            if hybrid_history:
                history_text = "\n\n".join([
                    f"Q: {h['question']}\nA: {h['answer']}" for h in hybrid_history
                ])
            prompt = self.build_strict_prompt(query, retrieved_texts, history_text)
            response = self.llm.invoke(prompt, query, book_name, retrieved_texts).content
            return response, query_embedding
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}", None 