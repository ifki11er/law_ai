import sqlite3
import os
import sys
import time
import re
import numpy as np
import pickle
import math
import gc
from collections import Counter
from typing import List, Dict, Tuple

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import PARENT_DB_PATH, BM25_INDEX_PATH
from core.vector_db import FastBM25

db_path = PARENT_DB_PATH
bm25_path = BM25_INDEX_PATH

def tokenize_korean(text: str) -> List[str]:
    tokens = []
    words = re.findall(r'[가-힣A-Za-z0-9]+', text.lower())
    for word in words:
        if len(word) <= 1:
            tokens.append(word)
        else:
            for i in range(len(word) - 1):
                tokens.append(word[i:i+2])
    return tokens

def rebuild_full_bm25():
    if not os.path.exists(db_path):
        print(f"[rebuild_bm25] ⚠️ SQLite DB 파일이 존재하지 않습니다: {db_path}")
        return
        
    t_start = time.time()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # [Pass 1] Count document lengths and unique term doc-frequencies
    print("[rebuild_bm25] [1단계/Pass 1] 데이터 개수 및 단어 빈도 카운팅 시작...")
    
    cursor.execute("SELECT COUNT(*) FROM embedding_metadata WHERE key = 'chroma:document'")
    corpus_size = cursor.fetchone()[0]
    print(f"[rebuild_bm25] 총 청크 수: {corpus_size:,}개")
    
    if corpus_size == 0:
        print("[rebuild_bm25] ⚠️ 적재된 청크가 없습니다. 재구축을 중단합니다.")
        conn.close()
        return
        
    doc_lengths = np.zeros(corpus_size, dtype=np.float32)
    doc_freqs = {}
    
    cursor.execute("""
        SELECT string_value FROM embedding_metadata WHERE key = 'chroma:document'
    """)
    
    doc_id = 0
    batch_size = 100000
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            text = row[0] if row[0] else ""
            tokens = tokenize_korean(text)
            doc_lengths[doc_id] = len(tokens)
            
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
            doc_id += 1
            
        print(f" -> {doc_id:,} / {corpus_size:,}개 청크 빈도 측정 완료...")
        
    print(f"[rebuild_bm25] 1단계 완료. (소요 시간: {time.time() - t_start:.2f}초)")
    
    avgdl = float(np.mean(doc_lengths))
    print(f"[rebuild_bm25] 평균 문서 길이(avgdl): {avgdl:.2f}")
    
    # Calculate IDF
    t_idf = time.time()
    idf = {}
    for term, df in doc_freqs.items():
        idf[term] = float(math.log(1 + (corpus_size - df + 0.5) / (df + 0.5)))
    print(f"[rebuild_bm25] {len(idf):,}개 단어의 IDF 가중치 계산 완료. (소요 시간: {time.time() - t_idf:.2f}초)")
    
    # [Pass 2] Pre-allocate flat numpy arrays and build inverted index
    print("\n[rebuild_bm25] [2단계/Pass 2] 메모리 효율적 Numpy 역색인 구조 설계 및 배치 기록...")
    total_postings = sum(doc_freqs.values())
    print(f"[rebuild_bm25] 총 색인 포스팅 수: {total_postings:,}개")
    
    doc_ids_flat = np.zeros(total_postings, dtype=np.int32)
    freqs_flat = np.zeros(total_postings, dtype=np.float32)
    
    offsets = {}
    starts = {}
    current_offset = 0
    for term, count in doc_freqs.items():
        offsets[term] = current_offset
        starts[term] = current_offset
        current_offset += count
        
    # Free doc_freqs dictionary
    del doc_freqs
    gc.collect()
    
    t_pass2 = time.time()
    sqlite_ids = []
    
    category_list = []
    category_map = {}
    categories_idx = np.zeros(corpus_size, dtype=np.uint8)
    is_active = np.ones(corpus_size, dtype=bool)
    
    # Execute query to fetch metadata and texts for mapping (no source_file join is needed here)
    cursor.execute("""
        SELECT 
            d.id,
            d.string_value AS text,
            cat.string_value AS category
        FROM (SELECT id, string_value FROM embedding_metadata WHERE key = 'chroma:document') d
        LEFT JOIN (SELECT id, string_value FROM embedding_metadata WHERE key = 'category') cat ON d.id = cat.id
    """)
    
    doc_id = 0
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            doc_sqlite_id = row[0]
            text = row[1] if row[1] else ""
            category = row[2] if row[2] else "law_ruling"
            
            sqlite_ids.append(doc_sqlite_id)
            
            # Map category to index
            if category not in category_map:
                category_map[category] = len(category_list)
                category_list.append(category)
            categories_idx[doc_id] = category_map[category]
            
            # Tokenize and index
            tokens = tokenize_korean(text)
            counts = Counter(tokens)
            for term, freq in counts.items():
                offset = offsets[term]
                doc_ids_flat[offset] = doc_id
                freqs_flat[offset] = freq
                offsets[term] += 1
            doc_id += 1
            
        print(f" -> {doc_id:,} / {corpus_size:,}개 청크 역색인 기록 완료...")
        
    print(f"[rebuild_bm25] 2단계 완료. (소요 시간: {time.time() - t_pass2:.2f}초)")
    
    # Convert flat arrays to sliced arrays for each term (O(1) memory view slicing)
    print("\n[rebuild_bm25] [3단계] 최종 인덱스 슬라이싱 및 메모리 압축화...")
    t_conv = time.time()
    inverted_index = {}
    for term, start_offset in starts.items():
        end_offset = offsets[term]
        inverted_index[term] = (doc_ids_flat[start_offset:end_offset], freqs_flat[start_offset:end_offset])
    print(f"[rebuild_bm25] 슬라이싱 매핑 완료. (소요 시간: {time.time() - t_conv:.2f}초)")
    
    # Clean temporary structures
    del offsets
    del starts
    gc.collect()
    
    print("\n[rebuild_bm25] [4단계] 다이어트 완료된 FastBM25 인덱스 디스크 직렬화...")
    t_save = time.time()
    
    fast_payload = {
        "is_fast_bm25": True,
        "corpus_size": corpus_size,
        "avgdl": avgdl,
        "k1": 1.5,
        "b": 0.75,
        "doc_len": doc_lengths,
        "idf": idf,
        "inverted_index": inverted_index,
        "sqlite_ids": sqlite_ids,
        "categories_idx": categories_idx,
        "category_list": category_list,
        "is_active": is_active
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(bm25_path), exist_ok=True)
    
    temp_bm25_path = bm25_path + ".tmp"
    with open(temp_bm25_path, "wb") as f_out:
        pickle.dump(fast_payload, f_out)
        
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    os.rename(temp_bm25_path, bm25_path)
    
    print(f"[rebuild_bm25] BM25 역색인 재구축 및 저장 성공: {bm25_path}")
    print(f"[rebuild_bm25] 전체 작업 총 소요 시간: {time.time() - t_start:.2f}초")
    
    file_size_mb = os.path.getsize(bm25_path) / (1024 * 1024)
    print(f"[rebuild_bm25] 직렬화 파일 용량: {file_size_mb:.2f} MB")
    
    conn.close()

if __name__ == "__main__":
    rebuild_full_bm25()
