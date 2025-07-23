import logging
import numpy as np
from typing import List, Dict, Any

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