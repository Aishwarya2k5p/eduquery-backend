import logging
import ollama
from EduQuery.config import Config
from typing import List

class OllamaEmbedder:
    def __init__(self, model_name=Config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
    
    def embed_query(self, text: str):
        try:
            result = ollama.embeddings(model=self.model_name, prompt=text)
            return result["embedding"]
        except Exception as e:
            logging.error(f"Error generating query embedding: {e}")
            raise
    
    def embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        logging.info(f"Starting batch embedding for {len(texts)} documents.")
        all_embeddings = []
        for i in range(0, len(texts), Config.BATCH_SIZE):
            batch = texts[i:i + Config.BATCH_SIZE]
            logging.info(f"Processing embedding batch {i//Config.BATCH_SIZE + 1} (size: {len(batch)})")
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
                            logging.error(f"Failed to embed text {i+j}: {e}. Using default zero embedding.")
                            batch_embeddings.append([0.0] * 384)
                all_embeddings.extend(batch_embeddings)
                logging.info(f"Finished embedding batch {i//Config.BATCH_SIZE + 1}/{(len(texts) + Config.BATCH_SIZE - 1)//Config.BATCH_SIZE}")
            except Exception as e:
                logging.error(f"Failed to process batch {i//Config.BATCH_SIZE + 1}: {e}")
                all_embeddings.extend([[0.0] * 384] * len(batch))
        logging.info(f"All batches processed. Total embeddings: {len(all_embeddings)}")
        return all_embeddings 