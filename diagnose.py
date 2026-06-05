import os
import sys
import sqlite3
import pickle

# v4 폴더를 path에 추가하여 모듈을 임포트할 수 있게 함
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4"))

from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from config.settings import CHROMA_DB_PATH, BM25_INDEX_PATH, PARENT_DB_PATH

def main():
    print("="*60)
    print("=== [형법 제5조 검색 및 DB 진단 스크립트] ===")
    print("="*60)

    # 1. SQLite 직접 조회 (Parent mappings 및 metadata)
    print("\n[1단계] SQLite(parent_mappings)에서 '형법' & '제5조' 검색 시도...")
    if not os.path.exists(PARENT_DB_PATH):
        print(f"❌ 에러: SQLite DB가 존재하지 않습니다: {PARENT_DB_PATH}")
        return

    conn = sqlite3.connect(PARENT_DB_PATH)
    cursor = conn.cursor()
    
    # parent_mappings 테이블에서 검색
    try:
        cursor.execute("SELECT parent_id, SUBSTR(parent_text, 1, 200) FROM parent_mappings WHERE parent_text LIKE '%형법%' AND parent_text LIKE '%제5조%'")
        rows = cursor.fetchall()
        print(f" -> parent_mappings 테이블 검색 완료: {len(rows)}개 발견")
        for i, row in enumerate(rows[:5]):
            print(f"    [{i+1}] ID: {row[0]}")
            print(f"        Text: {row[1]}...")
    except Exception as e:
        print(f"❌ parent_mappings 조회 중 에러: {e}")

    # embedding_metadata 테이블에서 검색
    print("\n[2단계] SQLite(embedding_metadata)에서 '형법' & '제5조' 키워드 매칭 청크 검색...")
    try:
        # chroma:document 내용에서 검색
        cursor.execute("""
            SELECT id, SUBSTR(string_value, 1, 150) 
            FROM embedding_metadata 
            WHERE key = 'chroma:document' 
              AND string_value LIKE '%형법%' 
              AND string_value LIKE '%제5조%'
            LIMIT 10
        """)
        rows = cursor.fetchall()
        print(f" -> embedding_metadata(chroma:document) 검색 완료: {len(rows)}개 발견 (최대 10개 출력)")
        for i, row in enumerate(rows):
            print(f"    [{i+1}] Chunk ID: {row[0]}")
            print(f"        Snippet: {row[1]}...")
    except Exception as e:
        print(f"❌ embedding_metadata 조회 중 에러: {e}")

    conn.close()

    # 2. RAG 검색 시뮬레이션
    print("\n[3단계] RAG VectorDB 로딩 및 하이브리드 검색 시뮬레이션...")
    try:
        embedding = LegalEmbedding()
        vector_db = LegalVectorDB(
            embedding_function=embedding.get_embeddings(),
            db_path=CHROMA_DB_PATH,
            bm25_path=BM25_INDEX_PATH
        )
        
        query = "형법 5조 내용 알려줘"
        print(f" -> 쿼리: '{query}'")
        
        # Dense 검색만 시뮬레이션
        print("\n --- 1) Chroma Dense Search (Top 5) ---")
        dense_results = vector_db.db.similarity_search_with_score(query, k=5)
        for idx, (doc, score) in enumerate(dense_results):
            print(f"    [{idx+1}등] Score(Distance): {score:.4f}")
            print(f"       Title: {doc.metadata.get('title')}")
            print(f"       Parent ID: {doc.metadata.get('parent_id')}")
            print(f"       Snippet: {doc.page_content[:150].replace('\n', ' ')}...")
            
        # BM25 검색만 시뮬레이션
        print("\n --- 2) BM25 Sparse Search (Top 5) ---")
        if vector_db.bm25_data is not None:
            bm25 = vector_db.bm25_data["bm25"]
            sqlite_ids = vector_db.bm25_data["sqlite_ids"]
            tokenized_query = vector_db.tokenize_korean(query)
            scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:5] if 'np' in globals() else sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
            
            # import numpy if not imported
            import numpy as np
            top_indices = np.argsort(scores)[::-1][:5]
            
            for rank, idx in enumerate(top_indices):
                score = scores[idx]
                if score > 0:
                    sid = sqlite_ids[idx]
                    print(f"    [{rank+1}등] Score: {score:.4f} | SQLite ID: {sid}")
                    # SQLite에서 텍스트 조회
                    conn = sqlite3.connect(PARENT_DB_PATH)
                    c = conn.cursor()
                    c.execute("SELECT string_value FROM embedding_metadata WHERE key = 'chroma:document' AND id = ?", (sid,))
                    row = c.fetchone()
                    text = row[0] if row else "텍스트 없음"
                    c.execute("SELECT string_value FROM embedding_metadata WHERE key = 'title' AND id = ?", (sid,))
                    row_t = c.fetchone()
                    title = row_t[0] if row_t else "제목 없음"
                    conn.close()
                    print(f"       Title: {title}")
                    print(f"       Snippet: {text[:150].replace('\n', ' ')}...")
        else:
            print("    ⚠️ BM25 인덱스가 로드되지 않았습니다.")

        # Hybrid RRF 검색
        print("\n --- 3) Hybrid RRF Search (Top 5) ---")
        hybrid_results = vector_db.search_hybrid(query, k=5)
        for idx, doc in enumerate(hybrid_results):
            print(f"    [{idx+1}등]")
            print(f"       Title: {doc.metadata.get('title')}")
            print(f"       Parent ID: {doc.metadata.get('parent_id')}")
            print(f"       Snippet: {doc.page_content[:150].replace('\n', ' ')}...")
            print(f"       Has Parent Text: {'parent_text' in doc.metadata}")
            if 'parent_text' in doc.metadata:
                print(f"       Parent Text Snippet: {doc.metadata['parent_text'][:150].replace('\n', ' ')}...")

    except Exception as e:
        print(f"❌ 검색 시뮬레이션 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
