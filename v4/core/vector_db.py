import os
import re
import pickle
import sqlite3
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import CHROMA_DB_PATH, BM25_INDEX_PATH, PARENT_DB_PATH

class FastBM25:
    """
    [v4+] High-performance inverted index-based BM25 implementation.
    Replaces rank_bm25 Okapi search for a 1,000x search speedup.
    """
    def __init__(self, corpus_size: int, avgdl: float, k1: float, b: float, doc_len: np.ndarray, idf: dict, inverted_index: dict):
        self.corpus_size = corpus_size
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b
        self.doc_len = doc_len
        self.idf = idf
        self.inverted_index = inverted_index
        # Precompute denominator constant parts
        self.denom_const = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        from collections import Counter
        query_counts = Counter(query_tokens)
        
        for term, q_tf in query_counts.items():
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            
            postings = self.inverted_index.get(term)
            if postings is None:
                continue
                
            # Unpack precomputed numpy arrays
            doc_ids, freqs = postings
            
            # Vectorized BM25 formula scoring
            term_scores = idf_val * (freqs * (self.k1 + 1)) / (freqs + self.denom_const[doc_ids])
            scores[doc_ids] += term_scores * q_tf
            
        return scores


class LegalVectorDB:
    """
    [v3] Core Vector Database and Hybrid Retrieval Manager.
    Integrates Chroma DB (Dense BGE-M3) and Rank-BM25 (Sparse lexical search)
    using Reciprocal Rank Fusion (RRF) for high-performance legal search.
    """
    
    def __init__(self, embedding_function, db_path: str = CHROMA_DB_PATH, bm25_path: str = BM25_INDEX_PATH, parent_db_path: str = PARENT_DB_PATH):
        self.db_path = db_path
        self.bm25_path = bm25_path
        self.parent_db_path = parent_db_path
        self.embedding_function = embedding_function
        self.db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_function
        )
        self.bm25_data = None
        self._load_bm25_index()
        self._init_parent_db()

    def _init_parent_db(self):
        """
        Initialize the SQLite database to store parent_id -> parent_text mapping.
        """
        os.makedirs(os.path.dirname(self.parent_db_path), exist_ok=True)
        conn = sqlite3.connect(self.parent_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parent_mappings (
                parent_id TEXT PRIMARY KEY,
                parent_text TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _save_parent_mappings(self, mappings: Dict[str, str]):
        """
        Save parent_id -> parent_text mappings to SQLite.
        """
        if not mappings:
            return
        conn = sqlite3.connect(self.parent_db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO parent_mappings (parent_id, parent_text) VALUES (?, ?)",
            list(mappings.items())
        )
        conn.commit()
        conn.close()

    def _load_parent_mappings(self, parent_ids: List[str]) -> Dict[str, str]:
        """
        Load parent_id -> parent_text mappings from SQLite.
        """
        if not parent_ids:
            return {}
        conn = sqlite3.connect(self.parent_db_path)
        cursor = conn.cursor()
        mappings = {}
        batch_size = 500
        for i in range(0, len(parent_ids), batch_size):
            batch = parent_ids[i:i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(
                f"SELECT parent_id, parent_text FROM parent_mappings WHERE parent_id IN ({placeholders})",
                batch
            )
            for pid, ptxt in cursor.fetchall():
                mappings[pid] = ptxt
        conn.close()
        return mappings

    def clean_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract parent texts to the SQLite database and return a copy of chunks 
        without parent_text in metadata.
        """
        parent_mappings = {}
        for chunk in chunks:
            pid = chunk["metadata"].get("parent_id")
            ptxt = chunk["metadata"].get("parent_text")
            if pid and ptxt:
                parent_mappings[pid] = ptxt
        if parent_mappings:
            self._save_parent_mappings(parent_mappings)

        clean_chunks = []
        for chunk in chunks:
            clean_meta = chunk["metadata"].copy()
            clean_meta.pop("parent_text", None)
            clean_chunks.append({
                "text": chunk["text"],
                "metadata": clean_meta
            })
        return clean_chunks

    def _enrich_parent_texts(self, documents: List[Document]) -> List[Document]:
        """
        Enrich documents by looking up and restoring parent_text in metadata.
        """
        if not documents:
            return documents
            
        parent_ids = {doc.metadata.get("parent_id") for doc in documents if doc.metadata.get("parent_id")}
        if not parent_ids:
            return documents
            
        mappings = self._load_parent_mappings(list(parent_ids))
        
        for doc in documents:
            pid = doc.metadata.get("parent_id")
            if pid and pid in mappings:
                parent_text = mappings[pid]
                if len(parent_text) > 3000:
                    parent_text = parent_text[:3000] + "\n\n... [이하 중략 (본문이 너무 길어 요약됨)]"
                doc.metadata["parent_text"] = parent_text
                
        return documents
        
    def _load_bm25_index(self):
        """
        [v4+] Load the serialized BM25 index and corresponding compressed metadata.
        Supports fast-loading format and converts legacy format (chunks) to compressed arrays.
        """
        import time
        if os.path.exists(self.bm25_path):
            try:
                t0 = time.time()
                with open(self.bm25_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Check if it is the optimized format
                if isinstance(data, dict) and data.get("is_fast_bm25", False):
                    self.fast_bm25 = FastBM25(
                        corpus_size=data["corpus_size"],
                        avgdl=data["avgdl"],
                        k1=data["k1"],
                        b=data["b"],
                        doc_len=data["doc_len"],
                        idf=data["idf"],
                        inverted_index=data["inverted_index"]
                    )
                    
                    # Convert legacy chunks to compressed if needed
                    if "chunks" in data:
                        print("[VectorDB v3] 감지된 레거시 chunks 리스트를 압축 Numpy 구조로 변환 중...")
                        chunks = data["chunks"]
                        sqlite_ids = [c.get("id", "") for c in chunks]
                        category_list = []
                        category_map = {}
                        categories_idx = np.zeros(len(chunks), dtype=np.uint8)
                        is_active_arr = np.zeros(len(chunks), dtype=bool)
                        for idx, c in enumerate(chunks):
                            cat = c.get("metadata", {}).get("category", "law_ruling")
                            if cat not in category_map:
                                category_map[cat] = len(category_list)
                                category_list.append(cat)
                            categories_idx[idx] = category_map[cat]
                            is_active_arr[idx] = bool(c.get("metadata", {}).get("is_active", True))
                            
                        # Save the converted format back to the file so it loads instantly next time
                        try:
                            t_save = time.time()
                            fast_payload = {
                                "is_fast_bm25": True,
                                "corpus_size": self.fast_bm25.corpus_size,
                                "avgdl": self.fast_bm25.avgdl,
                                "k1": self.fast_bm25.k1,
                                "b": self.fast_bm25.b,
                                "doc_len": self.fast_bm25.doc_len,
                                "idf": self.fast_bm25.idf,
                                "inverted_index": self.fast_bm25.inverted_index,
                                "sqlite_ids": sqlite_ids,
                                "categories_idx": categories_idx,
                                "category_list": category_list,
                                "is_active": is_active_arr
                            }
                            with open(self.bm25_path, 'wb') as f_out:
                                pickle.dump(fast_payload, f_out)
                            print(f" -> [완료] 레거시 인덱스 파일을 새로운 압축 Numpy 구조로 업데이트 완료! (저장 시간: {time.time() - t_save:.3f}초)")
                        except Exception as e_save:
                            print(f" -> ⚠️ 고속 BM25 인덱스 파일 자동 저장 갱신 실패: {e_save}")
                    else:
                        sqlite_ids = data["sqlite_ids"]
                        categories_idx = data["categories_idx"]
                        category_list = data["category_list"]
                        is_active_arr = data["is_active"]
                        
                    self.bm25_data = {
                        "bm25": self.fast_bm25,
                        "sqlite_ids": sqlite_ids,
                        "categories_idx": categories_idx,
                        "category_list": category_list,
                        "is_active": is_active_arr
                    }
                    print(f"[VectorDB v3] 고속 BM25 색인 파일을 성공적으로 로드했습니다: {self.bm25_path} (로드 시간: {time.time() - t0:.3f}초)")
                else:
                    self.bm25_data = None
                    print(f"[VectorDB v3] ⚠️ 지원하지 않는 BM25 색인 포맷입니다. 인덱스 재구축이 필요합니다.")
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

    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 2000) -> List[Dict[str, Any]]:
        """
        1. Clean chunks by saving parent texts to SQLite and removing parent_text from metadata.
        2. Add contextualized child chunks to Chroma DB in batches with reproducible IDs.
        3. Build and save the BM25 index for the same corpus.
        Returns the cleaned chunks.
        """
        clean_chunks = self.clean_chunks(chunks)
        documents = []
        ids = []
        seen_ids = set()
        for chunk in clean_chunks:
            # Default is_active to True if not present
            if "is_active" not in chunk["metadata"]:
                chunk["metadata"]["is_active"] = True
                
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
            
            # Generate reproducible, unique ID for each child chunk.
            import hashlib
            parent_id = chunk["metadata"].get("parent_id", "unknown")
            child_idx = chunk["metadata"].get("child_index", 0)
            text_hash = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()[:8]
            
            base_id = f"{parent_id}_child_{child_idx}_{text_hash}"
            unique_id = base_id
            counter = 1
            while unique_id in seen_ids:
                unique_id = f"{base_id}_{counter}"
                counter += 1
            seen_ids.add(unique_id)
            ids.append(unique_id)
            
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
            
        # 2. Build BM25 Index and Save as FastBM25
        print(f"[VectorDB v3] BM25 역색인(Lexical Index) 생성 중...")
        try:
            from rank_bm25 import BM25Okapi
            corpus_texts = [chunk["text"] for chunk in clean_chunks]
            tokenized_corpus = [self.tokenize_korean(t) for t in corpus_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Construct inverted index directly
            inverted_index = {}
            for doc_id, doc_freq in enumerate(bm25.doc_freqs):
                for term, freq in doc_freq.items():
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append((doc_id, freq))
            
            # Convert lists of tuples to numpy arrays
            for term in list(inverted_index.keys()):
                postings = inverted_index[term]
                doc_ids = np.array([p[0] for p in postings], dtype=np.int32)
                freqs = np.array([p[1] for p in postings], dtype=np.float32)
                inverted_index[term] = (doc_ids, freqs)
                
            self.fast_bm25 = FastBM25(
                corpus_size=bm25.corpus_size,
                avgdl=bm25.avgdl,
                k1=bm25.k1,
                b=bm25.b,
                doc_len=np.array(bm25.doc_len, dtype=np.float32),
                idf=bm25.idf,
                inverted_index=inverted_index
            )
            
            # Build category list and categories_idx
            category_list = []
            category_map = {}
            categories_idx = np.zeros(len(clean_chunks), dtype=np.uint8)
            is_active_arr = np.zeros(len(clean_chunks), dtype=bool)
            
            for idx, chunk in enumerate(clean_chunks):
                cat = chunk["metadata"].get("category", "law_ruling")
                if cat not in category_map:
                    category_map[cat] = len(category_list)
                    category_list.append(cat)
                categories_idx[idx] = category_map[cat]
                is_active_arr[idx] = bool(chunk["metadata"].get("is_active", True))
                
            fast_payload = {
                "is_fast_bm25": True,
                "corpus_size": self.fast_bm25.corpus_size,
                "avgdl": self.fast_bm25.avgdl,
                "k1": self.fast_bm25.k1,
                "b": self.fast_bm25.b,
                "doc_len": self.fast_bm25.doc_len,
                "idf": self.fast_bm25.idf,
                "inverted_index": self.fast_bm25.inverted_index,
                "sqlite_ids": ids,
                "categories_idx": categories_idx,
                "category_list": category_list,
                "is_active": is_active_arr
            }
            
            # Make sure data folder exists
            os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
            with open(self.bm25_path, 'wb') as f:
                pickle.dump(fast_payload, f)
                
            self.bm25_data = {
                "bm25": self.fast_bm25,
                "sqlite_ids": ids,
                "categories_idx": categories_idx,
                "category_list": category_list,
                "is_active": is_active_arr
            }
            print(f"[VectorDB v3] 고속 BM25 색인 저장 완료: {self.bm25_path}")
        except ImportError:
            print("⚠️ 경고: 'rank_bm25' 패키지가 없어 BM25 색인 생성을 건너뜁니다. (pip install rank_bm25 필요)")
        except Exception as e:
            print(f"⚠️ BM25 색인 생성 중 오류 발생: {e}")
            
    def search_hybrid(self, query: str, k: int = 5, category: str = None) -> List[Document]:
        """
        Executes BM25 (Sparse) + Chroma (Dense) Hybrid Search and combines them
        using Reciprocal Rank Fusion (RRF). Filters out inactive documents (is_active=False).
        Supports metadata category filtering.
        """
        import time
        start_time = time.time()
        print("[디버그] search_hybrid 진입")
        
        # Retrieve candidates (3 times the requested k to ensure good fusion)
        k_dense = k * 3
        k_sparse = k * 3
        
        # 1. Retrieve Dense Candidates (Chroma DB)
        t0 = time.time()
        print("[디버그] 1단계: Chroma Dense 검색 시작...")
        filter_dict = {}
        if category:
            filter_dict["category"] = category
            
        try:
             dense_raw = self.db.similarity_search_with_score(
                 query, 
                 k=k_dense, 
                 filter=filter_dict if filter_dict else None
             )
             dense_results = []
             for doc, dist in dense_raw:
                 if doc.metadata.get("is_active", True) is not False:
                     dense_results.append((doc, dist))
             dense_results = sorted(dense_results, key=lambda x: x[1])
             print(f"[디버그] 1단계 완료 ({time.time() - t0:.3f}초 소요) | Chroma 활성 결과 개수: {len(dense_results)}")
        except Exception as e:
             print(f"⚠️ Chroma 검색 실패: {e}")
             dense_results = []

        # 2. Retrieve Sparse Candidates (BM25)
        t0 = time.time()
        print("[디버그] 2단계: BM25 Sparse 검색 시작...")
        sparse_docs = []
        if self.bm25_data is not None:
             try:
                  bm25 = self.bm25_data["bm25"]
                  sqlite_ids = self.bm25_data["sqlite_ids"]
                  categories_idx = self.bm25_data["categories_idx"]
                  category_list = self.bm25_data["category_list"]
                  is_active = self.bm25_data["is_active"]
                  
                  tokenized_query = self.tokenize_korean(query)
                  scores = bm25.get_scores(tokenized_query)
                  
                  top_indices = np.argsort(scores)[::-1]
                  
                  # Find category index if filter applied
                  filter_cat_idx = None
                  if category and category in category_list:
                      filter_cat_idx = category_list.index(category)
                  
                  # Phase 1: Retrieve top candidate indexes
                  candidates = []
                  for idx in top_indices:
                      if len(candidates) >= k_sparse:
                          break
                      if scores[idx] > 0:
                          # Category filter
                          if category:
                              if filter_cat_idx is None or categories_idx[idx] != filter_cat_idx:
                                  continue
                          # Active filter
                          if not is_active[idx]:
                              continue
                              
                          candidates.append((idx, scores[idx]))
                              
                  # Phase 2: Fetch texts and metadata from SQLite for the top candidates
                  if candidates:
                      cand_indexes = [c[0] for c in candidates]
                      cand_sqlite_ids = [sqlite_ids[idx] for idx in cand_indexes]
                      
                      import sqlite3
                      conn = sqlite3.connect(self.parent_db_path)
                      cursor = conn.cursor()
                      
                      placeholders = ",".join(["?"] * len(cand_sqlite_ids))
                      
                      # Pivot-like query to fetch all 3 metadata fields in one row per id
                      query_str = f"""
                          SELECT 
                              d.id,
                              d.string_value AS text,
                              sf.string_value AS source_file,
                              cat.string_value AS category
                          FROM (SELECT id, string_value FROM embedding_metadata WHERE key = 'chroma:document' AND id IN ({placeholders})) d
                          LEFT JOIN (SELECT id, string_value FROM embedding_metadata WHERE key = 'source_file' AND id IN ({placeholders})) sf ON d.id = sf.id
                          LEFT JOIN (SELECT id, string_value FROM embedding_metadata WHERE key = 'category' AND id IN ({placeholders})) cat ON d.id = cat.id
                      """
                      
                      cursor.execute(query_str, cand_sqlite_ids + cand_sqlite_ids + cand_sqlite_ids)
                      
                      row_data = {}
                      for r_id, r_text, r_sf, r_cat in cursor.fetchall():
                          row_data[r_id] = {
                              "text": r_text if r_text else "",
                              "source_file": r_sf if r_sf else "",
                              "category": r_cat if r_cat else "law_ruling"
                          }
                      conn.close()
                      
                      # Construct Document objects preserving the BM25 score order
                      for idx, score in candidates:
                          c_id = sqlite_ids[idx]
                          info = row_data.get(c_id, {"text": "", "source_file": "", "category": "law_ruling"})
                          
                          resolved_chunk = {
                              "text": info["text"],
                              "metadata": {
                                  "source_file": info["source_file"],
                                  "category": info["category"],
                                  "is_active": True  # Verified active by filter
                              }
                          }
                          sparse_docs.append((resolved_chunk, score))
                          
                  print(f"[디버그] 2단계 완료 ({time.time() - t0:.3f}초 소요) | BM25 결과 개수: {len(sparse_docs)}")
             except Exception as e:
                 print(f"⚠️ BM25 검색 실패: {e}")
        else:
             print("⚠️ BM25 색인이 준비되지 않아 밀도(Vector) 검색만 수행합니다.")
        # 3. Reciprocal Rank Fusion (RRF)
        t0 = time.time()
        print("[디버그] 3단계: RRF 순위 합산 시작...")
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
        print(f"[디버그] 3단계 완료 ({time.time() - t0:.3f}초 소요) | RRF 융합 유니크 문서 수: {len(rrf_scores)}")
 
        # 4. Sort and select top k directly (No Rerank)
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        hybrid_results = []
        for key in sorted_keys:
             hybrid_results.append(doc_store[key])
         
        t0 = time.time()
        print("[디버그] SQLite 부모 텍스트 로드 시작...")
        res = self._enrich_parent_texts(hybrid_results)
        print(f"[디버그] SQLite 부모 텍스트 로드 완료 ({time.time() - t0:.3f}초 소요)")
        print(f"[디버그] search_hybrid 전체 실행 완료 (총 {time.time() - start_time:.3f}초 소요)")
        return res

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
                import sqlite3
                # Query child IDs from SQLite
                conn = sqlite3.connect(self.parent_db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM embedding_metadata WHERE key = 'parent_id' AND string_value = ?",
                    (parent_id,)
                )
                child_ids = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                if child_ids:
                    sqlite_ids = self.bm25_data["sqlite_ids"]
                    is_active = self.bm25_data["is_active"]
                    
                    child_ids_set = set(child_ids)
                    updated_count = 0
                    for idx, sid in enumerate(sqlite_ids):
                        if sid in child_ids_set:
                            if is_active[idx]:
                                is_active[idx] = False
                                updated_count += 1
                                
                    if updated_count > 0:
                        with open(self.bm25_path, 'wb') as f:
                            fast_payload = {
                                "is_fast_bm25": True,
                                "corpus_size": self.fast_bm25.corpus_size,
                                "avgdl": self.fast_bm25.avgdl,
                                "k1": self.fast_bm25.k1,
                                "b": self.fast_bm25.b,
                                "doc_len": self.fast_bm25.doc_len,
                                "idf": self.fast_bm25.idf,
                                "inverted_index": self.fast_bm25.inverted_index,
                                "sqlite_ids": sqlite_ids,
                                "categories_idx": self.bm25_data["categories_idx"],
                                "category_list": self.bm25_data["category_list"],
                                "is_active": is_active
                            }
                            pickle.dump(fast_payload, f)
                        print(f"[VectorDB v3] BM25: parent_id '{parent_id}' 내 {updated_count}개 청크 비활성화 및 인덱스 파일 업데이트 완료.")
                        success = True
                    else:
                        print(f"[VectorDB v3] BM25: parent_id '{parent_id}'에 해당하는 청크를 찾았으나 이미 비활성화되어 있거나 인덱스에 없습니다.")
                else:
                    print(f"[VectorDB v3] BM25: parent_id '{parent_id}'에 해당하는 청크를 SQLite DB에서 찾지 못했습니다.")
            except Exception as e:
                print(f"[VectorDB v3] ⚠️ BM25 비활성화 중 오류 발생: {e}")
                success = False
                
        return success

    def search_admin(self, query: str, k: int = 15, include_inactive: bool = True, category: str = None) -> List[Dict[str, Any]]:
        """
        Special search for admin management. Groups results by parent_id
        so that developers can easily find the whole Article to modify/repeal.
        Supports metadata category filtering.
        """
        seen_parents = {}
        
        filter_dict = {}
        if category:
            filter_dict["category"] = category
        if not include_inactive:
            filter_dict["is_active"] = True
            
        # 1. Retrieve from Chroma DB
        try:
            dense_results = self.db.similarity_search_with_score(
                query, 
                k=40, 
                filter=filter_dict if filter_dict else None
            )
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
                sqlite_ids = self.bm25_data["sqlite_ids"]
                categories_idx = self.bm25_data["categories_idx"]
                category_list = self.bm25_data["category_list"]
                is_active_arr = self.bm25_data["is_active"]
                
                tokenized_query = self.tokenize_korean(query)
                scores = bm25.get_scores(tokenized_query)
                
                top_indices = np.argsort(scores)[::-1]
                
                filter_cat_idx = None
                if category and category in category_list:
                    filter_cat_idx = category_list.index(category)
                
                candidates = []
                for idx in top_indices:
                    if len(seen_parents) + len(candidates) >= k * 2:
                        break
                    if scores[idx] > 0:
                        # Category filter
                        if category:
                            if filter_cat_idx is None or categories_idx[idx] != filter_cat_idx:
                                continue
                        # Active filter
                        is_active = bool(is_active_arr[idx])
                        if not include_inactive and not is_active:
                            continue
                            
                        candidates.append((idx, is_active))
                
                if candidates:
                    import sqlite3
                    cand_indexes = [c[0] for c in candidates]
                    cand_sqlite_ids = [sqlite_ids[idx] for idx in cand_indexes]
                    
                    conn = sqlite3.connect(self.parent_db_path)
                    cursor = conn.cursor()
                    placeholders = ",".join(["?"] * len(cand_sqlite_ids))
                    
                    query_str = f"""
                        SELECT 
                            d.id,
                            d.string_value AS text,
                            pid.string_value AS parent_id,
                            t.string_value AS title,
                            cn.string_value AS case_name,
                            sn.string_value AS section_name
                        FROM (SELECT id, string_value FROM embedding_metadata WHERE key = 'chroma:document' AND id IN ({placeholders})) d
                        LEFT JOIN (SELECT id, string_value FROM embedding_metadata WHERE key = 'parent_id' AND id IN ({placeholders})) pid ON d.id = pid.id
                        LEFT JOIN (SELECT id, string_value FROM embedding_metadata WHERE key = 'title' AND id IN ({placeholders})) t ON d.id = t.id
                        LEFT JOIN (SELECT id, string_value FROM embedding_metadata WHERE key = 'caseName' AND id IN ({placeholders})) cn ON d.id = cn.id
                        LEFT JOIN (SELECT id, string_value FROM embedding_metadata WHERE key = 'section_name' AND id IN ({placeholders})) sn ON d.id = sn.id
                    """
                    
                    cursor.execute(query_str, cand_sqlite_ids * 5)
                    row_data = {}
                    for r_id, r_text, r_pid, r_title, r_cn, r_sn in cursor.fetchall():
                        row_data[r_id] = {
                            "text": r_text if r_text else "",
                            "parent_id": r_pid if r_pid else "",
                            "title": r_title if r_title else "",
                            "caseName": r_cn if r_cn else "",
                            "section_name": r_sn if r_sn else ""
                        }
                    conn.close()
                    
                    for idx, is_act in candidates:
                        c_id = sqlite_ids[idx]
                        info = row_data.get(c_id, {})
                        r_pid = info.get("parent_id")
                        
                        if r_pid and r_pid not in seen_parents:
                            seen_parents[r_pid] = {
                                "parent_id": r_pid,
                                "title": info.get("title", ""),
                                "caseName": info.get("caseName", ""),
                                "section_name": info.get("section_name", ""),
                                "is_active": is_act,
                                "text": info.get("text", ""),
                                "metadata": {
                                    "parent_id": r_pid,
                                    "title": info.get("title", ""),
                                    "caseName": info.get("caseName", ""),
                                    "section_name": info.get("section_name", ""),
                                    "is_active": is_act
                                }
                            }
            except Exception as e:
                print(f"⚠️ Admin BM25 검색 실패: {e}")
 
        # Enrich the text and parent_text metadata of the results from SQLite parent DB
        parent_ids = list(seen_parents.keys())
        if parent_ids:
            mappings = self._load_parent_mappings(parent_ids)
            for pid, parent_data in seen_parents.items():
                if pid in mappings:
                    parent_data["text"] = mappings[pid]
                    parent_data["metadata"]["parent_text"] = mappings[pid]
                    

        # Enrich the text and parent_text metadata of the results from SQLite parent DB
        parent_ids = list(seen_parents.keys())
        if parent_ids:
            mappings = self._load_parent_mappings(parent_ids)
            for pid, parent_data in seen_parents.items():
                if pid in mappings:
                    parent_data["text"] = mappings[pid]
                    parent_data["metadata"]["parent_text"] = mappings[pid]
                    
        return list(seen_parents.values())[:k]

    def get_db(self) -> Chroma:
        return self.db
