import torch
from config.settings import OLLAMA_EMBEDDING_MODEL

class LegalEmbedding:
    """
    [v4+] High-performance Local Embedding using Sentence-Transformers (via GPU if available).
    Falls back to Ollama only if sentence-transformers/HuggingFaceEmbeddings is unavailable.
    """
    
    def __init__(self, model_name: str = OLLAMA_EMBEDDING_MODEL, base_url: str = None):
        # We prefer HuggingFaceEmbeddings locally for 100x faster execution using CUDA
        self.embeddings = None
        
        # Try local HuggingFace first
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[LegalEmbedding] 로컬 HuggingFace Embeddings 로드 중: BAAI/bge-m3 (장치: {device})")
            
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": device, "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True}
            )
            print("[LegalEmbedding] 로컬 HuggingFace Embeddings 로드 완료.")
        except Exception as e:
            print(f"[LegalEmbedding] ⚠️ 로컬 HuggingFace Embeddings 로드 실패 ({e}). Ollama로 Fallback합니다.")
            from langchain_ollama import OllamaEmbeddings
            from config.settings import OLLAMA_BASE_URL
            self.embeddings = OllamaEmbeddings(
                model=model_name,
                base_url=base_url or OLLAMA_BASE_URL
            )
        
    def get_embeddings(self):
        """
        Return the raw LangChain embeddings object.
        """
        return self.embeddings
