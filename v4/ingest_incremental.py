import sys
import os
import glob
import pickle
from typing import Set
from core.parser import LegalParserV2
from core.splitter import LegalSplitterV2
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from config.settings import CHROMA_DB_PATH, BM25_INDEX_PATH, AI_HUB_DATA_PATH

def get_ingested_files() -> Set[str]:
    """
    Chroma DB(sqlite3)의 메타데이터를 직접 조회하여 이미 적재 완료된 파일명 목록을 반환합니다.
    """
    ingested = set()
    import sqlite3
    from config.settings import PARENT_DB_PATH
    if os.path.exists(PARENT_DB_PATH):
        try:
            conn = sqlite3.connect(PARENT_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT string_value FROM embedding_metadata WHERE key = 'source_file'"
            )
            ingested = {row[0] for row in cursor.fetchall() if row[0]}
            conn.close()
            print(f"[체크 완료] SQLite 데이터베이스에서 이미 적재된 파일 {len(ingested)}개를 확인했습니다.")
        except Exception as e:
            print(f"⚠️ 경고: SQLite DB에서 적재 이력을 조회하지 못했습니다 (무시하고 진행): {e}")
    else:
        print("[체크 완료] 기존 데이터베이스가 없어 전체 신규 적재를 진행합니다.")
    return ingested

def main():
    print("\n" + "="*60)
    print("=== [증분 인제스천 파이프라인 v4 가동 (자동 파일 감지)] ===")
    print("="*60)
    
    # 0. 이미 DB에 들어간 파일명 목록 로드
    ingested_files = get_ingested_files()
    
    # 1. 원천 데이터 폴더 스캔
    parser = LegalParserV2()
    new_documents = []
    
    print("\n[Step 1] 신규 추가된 CSV 및 JSON 파일 자동 탐색 중...")
    
    # Walk raw data directories recursively
    for root, dirs, files in os.walk(parser.raw_data_path):
        # Skip validation data to avoid duplicates/unnecessary files in RAG
        if "validation" in root.lower():
            continue
        if "01.원천데이터" in root or "01-1.정식개방데이터" in root:
            # Determine category based on the relative path directory name
            rel_path = os.path.relpath(root, parser.raw_data_path)
            parts = rel_path.split(os.sep)
            top_folder = parts[0] if parts else ""
            
            if any(k in top_folder for k in ["01.민사법", "02.지식재산권법", "03.행정법", "04.형사법"]):
                category = "law_ruling"
            elif any(k in top_folder for k in ["05.계약", "06.계약 외"]):
                category = "contract_form"
            elif any(k in top_folder for k in ["151.금융", "154.의료"]):
                category = "mrc"
            else:
                category = "law_ruling"  # fallback
                
            immediate_folder = os.path.basename(root)
            
            for file in files:
                filename = os.path.basename(file)
                
                # Check if already ingested
                if filename in ingested_files:
                    continue
                    
                file_path = os.path.join(root, file)
                print(f" └─ [신규 파일 감지] {top_folder}/{immediate_folder}/{filename} 파싱 중... (카테고리: {category})")
                
                if file.endswith(".csv"):
                    doc_blocks = parser.parse_csv_file(file_path, immediate_folder)
                    if doc_blocks:
                        new_documents.extend(doc_blocks)
                elif file.endswith(".json"):
                    doc_blocks = parser.parse_json_file(file_path, category)
                    if doc_blocks:
                        new_documents.extend(doc_blocks)
                        
    if not new_documents:
        print("\n[알림] 새로 추가된 법률/계약/기계독해 파일이 없습니다. 작업을 종료합니다.")
        sys.exit(0)
        
    print(f"\n[Step 1 완료] 신규 파일들로부터 총 {len(new_documents)}개의 논리적 부모 조문을 파싱했습니다.")
        
    # 2. 컨텍스트 합성형 부모-자식 분할
    print("\n[Step 2] 브레드크럼 접두사 합성 및 자식 청킹 분할 시작...")
    splitter = LegalSplitterV2()
    new_chunks = splitter.split_documents(new_documents)
    if not new_chunks:
        print("❌ 에러: 분할된 청크가 존재하지 않습니다.")
        sys.exit(1)
    print(f"[Step 2 완료] 신규 자식 청크 {len(new_chunks):,}개 생성 완료.")
        
    # 3. 임베딩 모델 로드
    print("\n[Step 3] Ollama/bge-m3 임베딩 모델 준비 중...")
    embedding = LegalEmbedding()
    print("[Step 3 완료] 임베딩 모델 준비 완료.")
    
    # 4. Chroma DB 로드 및 신규 청크만 적재
    print(f"\n[Step 4] 기존 Chroma DB에 신규 청크 점진적 추가 적재 시작...")
    db = LegalVectorDB(
        embedding_function=embedding.get_embeddings(), 
        db_path=CHROMA_DB_PATH,
        bm25_path=BM25_INDEX_PATH
    )
    clean_new_chunks = db.add_documents(new_chunks)
    print("[Step 4 완료] Chroma DB 적재 성공.")
    
    # 5. BM25 인덱스 전체 재구축
    print("\n[Step 5] 최적화된 BM25 역색인 전체 재구축 시작...")
    try:
        # Add project root to sys.path to resolve rebuild_bm25 import inside execution context
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from rebuild_bm25 import rebuild_full_bm25
        rebuild_full_bm25()
        print("[Step 5 완료] BM25 역색인 전체 재구축 성공.")
    except Exception as e:
        print(f"❌ 에러: BM25 역색인 재구축 실패: {e}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("=== [증분 인제스천 완료] 신규 법률 데이터가 데이터베이스에 성공적으로 병합되었습니다! ===")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
