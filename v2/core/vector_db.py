from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import CHROMA_DB_PATH
from typing import List, Dict, Any, Tuple

class LegalVectorDB:
    """
    Core vector database manager for Chroma DB.
    """
    
    def __init__(self, embedding_function, db_path: str = CHROMA_DB_PATH):
        self.db_path = db_path
        self.embedding_function = embedding_function
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )
        
    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 2000):
        """
        Add chunks to Chroma DB in batches to prevent SQLite parameter limits.
        Chunks must be dictionaries with 'text' and 'metadata'.
        """
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
            
        total_docs = len(documents)
        print(f"Ingesting {total_docs} chunks to Chroma DB at {self.db_path} (Batch size: {batch_size})...")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            current_batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            print(f" -> [배치 {current_batch_num}/{total_batches}] {i}번부터 {min(i + batch_size, total_docs)}번 청크 적재 중...")
            self.db.add_documents(batch)
            
        print("Ingestion complete.")

    def search_with_scores(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search similarities in Chroma DB and return list of tuples (Document, score).
        Distance score in Chroma represents cosine/L2 distance where lower score = more similar.
        """
        results = self.db.similarity_search_with_score(query, k=k)
        return results
        
    def get_db(self) -> Chroma:
        """
        Return the raw Chroma DB object.
        """
        return self.db
