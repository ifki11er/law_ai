from langchain_ollama import OllamaEmbeddings
from config.settings import OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL

class LegalEmbedding:
    """
    Embedding wrapper for Ollama's bge-m3 model.
    """
    
    def __init__(self, model_name: str = OLLAMA_EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        
    def get_embeddings(self) -> OllamaEmbeddings:
        """
        Return the raw LangChain embeddings object.
        """
        return self.embeddings
