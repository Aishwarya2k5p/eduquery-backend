import re
import logging
from typing import List, Dict, Any

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

def is_followup_query(query: str) -> bool:
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
    if not recent_history:
        return True
    def extract_terms(text):
        terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        common_words = {'what', 'is', 'are', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'this', 'that', 'these', 'those', 'how', 'why', 'when', 'where', 'who', 'which', 'tell', 'me', 'more', 'detail', 'explain', 'state', 'all', 'uses', 'use', 'simple', 'beginner', 'easy', 'basics', 'short', 'key', 'points', 'summary', 'them', 'they', 'their'}
        return terms - common_words
    current_terms = set(extract_terms(query.lower()))
    history_terms = set()
    for entry in recent_history[-5:]:
        question = entry.get('question', '').lower()
        answer = entry.get('answer', '').lower()
        history_terms.update(extract_terms(question))
        history_terms.update(extract_terms(answer))
    new_topic_indicators = [
        'what is', 'define', 'explain', 'tell me about', 'describe',
        'how does', 'what are', 'list', 'show me', 'give me',
    ]
    has_new_topic_indicators = any(indicator in query.lower() for indicator in new_topic_indicators)
    if current_terms and history_terms:
        overlap = len(current_terms.intersection(history_terms))
        total_unique = len(current_terms.union(history_terms))
        similarity = overlap / total_unique if total_unique > 0 else 0
        return similarity < 0.15 and has_new_topic_indicators
    return has_new_topic_indicators

def is_casual_query(query: str) -> bool:
    normalized_query = query.lower().strip()
    for keyword in CASUAL_RESPONSES.keys():
        if re.search(rf'\b{re.escape(keyword)}\b', normalized_query):
            return True
    return False

def handle_casual_query(query: str) -> str:
    normalized_query = query.lower().strip()
    for keyword, response in CASUAL_RESPONSES.items():
        if re.search(rf'\b{re.escape(keyword)}\b', normalized_query):
            return response
    return "Hello! Feel free to ask questions about the PDF content."

def rewrite_followup_query(current_query, chat_history):
    if is_followup_query(current_query):
        for entry in reversed(chat_history):
            q = entry.get("question", "")
            if q and not is_casual_query(q):
                last_question = q
                break
        else:
            last_question = ""
        if last_question and len(current_query.split()) <= 2:
            return f"{last_question} {current_query}".strip()
    return current_query 