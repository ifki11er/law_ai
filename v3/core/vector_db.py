import os
import re
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import CHROMA_DB_PATH, BM25_INDEX_PATH

class LegalVectorDB:
    """
    [v3] Core Vector Database and Hybrid Retrieval Manager.
    Integrates Chroma DB (Dense BGE-M3) and Rank-BM25 (Sparse lexical search)
    using Reciprocal Rank Fusion (RRF) for high-performance legal search.
    """
    
    def __init__(self, embedding_function, db_path: str = CHROMA_DB_PATH, bm25_path: str = BM25_INDEX_PATH):
        self.db_path = db_path
        self.bm25_path = bm25_path
        self.embedding_function = embedding_function
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )
        self.bm25_data = None
        self._load_bm25_index()
        
    def _load_bm25_index(self):
        """
        Load the serialized BM25 index and corresponding chunks metadata.
        """
        if os.path.exists(self.bm25_path):
            try:
                with open(self.bm25_path, 'rb') as f:
                    self.bm25_data = pickle.load(f)
                print(f"[VectorDB v3] BM25 색인 파일을 성공적으로 로드했습니다: {self.bm25_path}")
            except Exception as e:
                print(f"[VectorDB v3] ⚠️ BM25 색인 파일 로드 중 오류 발생: {e}")
        else:
            print(f"[VectorDB v3] ⚠️ BM25 색인 파일이 존재하지 않습니다. 인제스션을 진행해야 합니다.")

    def tokenize_korean(self, text: str) -> List[str]:
        """
        [v3+] Character Bi-gram Tokenizer for Korean.
        Splits text into overlapping 2-character shingles.
        Example: "기소유예" -> ["기소", "소유", "유예"]
        Example: "기소유예처분" -> ["기소", "소유", "유예", "예처", "처분"]
        This enables high-precision keyword overlap matching for compound Korean legal words
        without morphological dependencies.
        """
        tokens = []
        # Extract alphanumeric and Hangul blocks
        words = re.findall(r'[가-힣A-Za-z0-9]+', text.lower())
        for word in words:
            # If word is single character, keep it as is
            if len(word) <= 1:
                tokens.append(word)
            else:
                # Slide window of size 2 over the word
                for i in range(len(word) - 1):
                    tokens.append(word[i:i+2])
        return tokens

    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 2000):
        """
        1. Add contextualized child chunks to Chroma DB in batches with reproducible IDs.
        2. Build and save the BM25 index for the same corpus.
        """
        documents = []
        ids = []
        for chunk in chunks:
            # Default is_active to True if not present
            if "is_active" not in chunk["metadata"]:
                chunk["metadata"]["is_active"] = True
                
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
            
            # Generate reproducible, unique ID for each child chunk.
            # Suffix with a hash of the text content to guarantee uniqueness even if the parser
            # generated multiple blocks with the same section name in the same file.
            import hashlib
            parent_id = chunk["metadata"].get("parent_id", "unknown")
            child_idx = chunk["metadata"].get("child_index", 0)
            text_hash = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()[:8]
            ids.append(f"{parent_id}_child_{child_idx}_{text_hash}")
            
        total_docs = len(documents)
        print(f"[VectorDB v3] Chroma DB 적재 시작: {total_docs}개 청크 (경로: {self.db_path})")
        
        # 1. Chroma DB Add
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            current_batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            print(f" -> [배치 {current_batch_num}/{total_batches}] {i}번부터 {min(i + batch_size, total_docs)}번 청크 적재 중...")
            self.db.add_documents(batch, ids=batch_ids)
            
        # 2. Build BM25 Index
        print(f"[VectorDB v3] BM25 역색인(Lexical Index) 생성 중...")
        try:
            from rank_bm25 import BM25Okapi
            corpus_texts = [chunk["text"] for chunk in chunks]
            tokenized_corpus = [self.tokenize_korean(t) for t in corpus_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Serialize
            bm25_payload = {
                "bm25": bm25,
                "chunks": chunks
            }
            
            # Make sure data folder exists
            os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(bm25_payload, f)
                
            self.bm25_data = bm25_payload
            print(f"[VectorDB v3] BM25 색인 저장 완료: {self.bm25_path}")
        except ImportError:
            print("⚠️ 경고: 'rank_bm25' 패키지가 없어 BM25 색인 생성을 건너뜁니다. (pip install rank_bm25 필요)")
        except Exception as e:
            print(f"⚠️ BM25 색인 생성 중 오류 발생: {e}")
            
        print("Ingestion complete.")

    def search_with_scores(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Fallback method for raw dense vector search.
        """
        return self.db.similarity_search_with_score(query, k=k)

    def search_hybrid(self, query: str, k: int = 5) -> List[Document]:
        """
        Executes BM25 (Sparse) + Chroma (Dense) Hybrid Search and combines them
        using Reciprocal Rank Fusion (RRF). Filters out inactive documents (is_active=False).
        """
        # Retrieve more candidates to ensure we have enough active documents after filtering
        k_dense = k * 4
        k_sparse = k * 4
        
        # 1. Retrieve Dense Candidates (Chroma DB)
        try:
            dense_raw = self.db.similarity_search_with_score(query, k=k_dense)
            dense_results = []
            for doc, dist in dense_raw:
                if doc.metadata.get("is_active", True) is not False:
                    dense_results.append((doc, dist))
            dense_results = sorted(dense_results, key=lambda x: x[1])[:k * 2]
        except Exception as e:
            print(f"⚠️ Chroma 검색 실패: {e}")
            dense_results = []

        # 2. Retrieve Sparse Candidates (BM25)
        sparse_docs = []
        if self.bm25_data is not None:
            try:
                bm25 = self.bm25_data["bm25"]
                original_chunks = self.bm25_data["chunks"]
                
                tokenized_query = self.tokenize_korean(query)
                scores = bm25.get_scores(tokenized_query)
                
                top_indices = np.argsort(scores)[::-1][:k_sparse]
                
                for rank, idx in enumerate(top_indices):
                    if scores[idx] > 0:
                        chunk = original_chunks[idx]
                        if chunk["metadata"].get("is_active", True) is not False:
                            sparse_docs.append((chunk, scores[idx]))
                sparse_docs = sparse_docs[:k * 2]
            except Exception as e:
                print(f"⚠️ BM25 검색 실패: {e}")
        else:
            print("⚠️ BM25 색인이 준비되지 않아 밀도(Vector) 검색만 수행합니다.")

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_store = {}
        
        # Add Dense ranks
        for rank, (doc, dist) in enumerate(dense_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60.0 + (rank + 1))
            doc_store[key] = doc

        # Add Sparse ranks
        for rank, (chunk, score) in enumerate(sparse_docs):
            key = chunk["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60.0 + (rank + 1))
            
            if key not in doc_store:
                doc_store[key] = Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                )

        # 4. Sort and return Top K
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        hybrid_results = []
        for key in sorted_keys:
            hybrid_results.append(doc_store[key])
            
        return hybrid_results

    def deactivate_parent_block(self, parent_id: str) -> bool:
        """
        Deactivates all child chunks associated with a specific parent_id.
        Sets is_active to False in both Chroma DB and the serialized BM25 index.
        """
        success = False
        
        # 1. Update in Chroma DB
        try:
            results = self.db.get(where={"parent_id": parent_id})
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])
            
            if ids:
                updated_docs = []
                for meta, text in zip(metadatas, documents):
                    meta["is_active"] = False
                    updated_docs.append(Document(page_content=text, metadata=meta))
                
                self.db.update_documents(ids=ids, documents=updated_docs)
                print(f"[VectorDB v3] Chroma DB: parent_id '{parent_id}' 내 {len(ids)}개 청크 비활성화 완료.")
                success = True
            else:
                print(f"[VectorDB v3] Chroma DB: parent_id '{parent_id}'에 해당하는 청크를 찾지 못했습니다.")
        except Exception as e:
            print(f"[VectorDB v3] ⚠️ Chroma DB 비활성화 중 오류 발생: {e}")

        # 2. Update in BM25 Index File
        if self.bm25_data is not None:
            try:
                updated_count = 0
                for chunk in self.bm25_data["chunks"]:
                    if chunk["metadata"].get("parent_id") == parent_id:
                        chunk["metadata"]["is_active"] = False
                        updated_count += 1
                
                if updated_count > 0:
                    with open(self.bm25_path, 'wb') as f:
                        pickle.dump(self.bm25_data, f)
                    print(f"[VectorDB v3] BM25: parent_id '{parent_id}' 내 {updated_count}개 청크 비활성화 및 인덱스 파일 업데이트 완료.")
                    success = True
                else:
                    print(f"[VectorDB v3] BM25: parent_id '{parent_id}'에 해당하는 청크를 찾지 못했습니다.")
            except Exception as e:
                print(f"[VectorDB v3] ⚠️ BM25 비활성화 중 오류 발생: {e}")
                success = False
                
        return success
        
    def search_admin(self, query: str, k: int = 15, include_inactive: bool = True) -> List[Dict[str, Any]]:
        """
        Special search for admin management. Groups results by parent_id
        so that developers can easily find the whole Article to modify/repeal.
        """
        seen_parents = {}
        
        # 1. Retrieve from Chroma DB
        try:
            dense_results = self.db.similarity_search_with_score(query, k=40)
            for doc, dist in dense_results:
                pid = doc.metadata.get("parent_id")
                if pid and pid not in seen_parents:
                    is_active = doc.metadata.get("is_active", True)
                    if not include_inactive and not is_active:
                        continue
                    seen_parents[pid] = {
                        "parent_id": pid,
                        "title": doc.metadata.get("title", ""),
                        "caseName": doc.metadata.get("caseName", ""),
                        "section_name": doc.metadata.get("section_name", ""),
                        "is_active": is_active,
                        "text": doc.metadata.get("parent_text", doc.page_content),
                        "metadata": doc.metadata
                    }
        except Exception as e:
            print(f"⚠️ Admin Chroma 검색 실패: {e}")

        # 2. Retrieve from BM25
        if self.bm25_data is not None:
            try:
                bm25 = self.bm25_data["bm25"]
                original_chunks = self.bm25_data["chunks"]
                
                tokenized_query = self.tokenize_korean(query)
                scores = bm25.get_scores(tokenized_query)
                
                top_indices = np.argsort(scores)[::-1][:40]
                
                for idx in top_indices:
                    if scores[idx] > 0:
                        chunk = original_chunks[idx]
                        pid = chunk["metadata"].get("parent_id")
                        if pid and pid not in seen_parents:
                            is_active = chunk["metadata"].get("is_active", True)
                            if not include_inactive and not is_active:
                                continue
                            seen_parents[pid] = {
                                "parent_id": pid,
                                "title": chunk["metadata"].get("title", ""),
                                "caseName": chunk["metadata"].get("caseName", ""),
                                "section_name": chunk["metadata"].get("section_name", ""),
                                "is_active": is_active,
                                "text": chunk["metadata"].get("parent_text", chunk["text"]),
                                "metadata": chunk["metadata"]
                            }
            except Exception as e:
                print(f"⚠️ Admin BM25 검색 실패: {e}")

        return list(seen_parents.values())[:k]

    def get_db(self) -> Chroma:
        return self.db
