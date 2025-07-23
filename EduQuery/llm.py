import logging
import google.generativeai as genai

class GeminiLLM:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def get_cache_key(self, question, book_name, retrieved_texts):
        return question.strip() + "||" + book_name.strip() + "||" + retrieved_texts.strip()

    def invoke(self, prompt: str, question: str = None, book_name: str = None, retrieved_texts: str = None):
        try:
            cache_key = prompt
            if question is not None and book_name is not None and retrieved_texts is not None:
                cache_key = self.get_cache_key(question, book_name, retrieved_texts)
            response = self.model.generate_content(prompt)
            response_text = response.text
            return type('Obj', (object,), {'content': response_text})
        except Exception as e:
            logging.error(f"Gemini LLM error: {e}")
            return type('Obj', (object,), {'content': f"Error from Gemini: {str(e)}"}) 