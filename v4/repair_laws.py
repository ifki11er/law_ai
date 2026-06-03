import os
import sys
import shutil
import sqlite3
import pickle
from core.parser import LegalParserV2
from core.splitter import LegalSplitterV2
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from config.settings import CHROMA_DB_PATH, BM25_INDEX_PATH

def main():
    print("="*60)
    print("=== [법령 데이터 단독 복구 및 재정비 스크립트 실행] ===")
    print("="*60)
    
    # 1. 컴포넌트 로드
    embedding = LegalEmbedding()
    db = LegalVectorDB(
        embedding_function=embedding.get_embeddings(),
        db_path=CHROMA_DB_PATH,
        bm25_path=BM25_INDEX_PATH
    )
    
    # 2. Chroma DB에서 법령(TS_법령) 관련 데이터만 삭제
    print("\n[1단계] 기존 Chroma DB에서 법령(doc_type: 대한민국 법령) 데이터 제거 중...")
    try:
        collection = db.db._collection
        total_deleted = 0
        batch_size = 500
        while True:
            # 500개씩만 페이징해서 ID 조회 (SQLite 변수 한계 극복)
            results = collection.get(where={"doc_type": "대한민국 법령"}, limit=batch_size)
            ids_to_delete = results.get("ids", [])
            if not ids_to_delete:
                break
            
            collection.delete(ids=ids_to_delete)
            total_deleted += len(ids_to_delete)
            print(f" -> 법령 청크 {total_deleted:,}개 삭제 진행 중...")
            
        if total_deleted > 0:
            print(f" -> Chroma DB 법령 데이터 삭제 완료. (총 {total_deleted:,}개)")
        else:
            print(" -> Chroma DB에 삭제할 법령 데이터가 없습니다.")
    except Exception as e:
        print(f"❌ Chroma 데이터 삭제 중 오류 발생: {e}")
        sys.exit(1)
        
    # 3. SQLite parent_mappings에서 법령 매핑 제거
    print("\n[2단계] SQLite parent_mappings에서 법령 매핑 제거 중...")
    try:
        conn = sqlite3.connect(db.parent_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM parent_mappings WHERE parent_id LIKE 'HS_B_%'")
        deleted_rows = cursor.rowcount
        conn.commit()
        conn.close()
        print(f" -> SQLite에서 {deleted_rows:,}개 법령 본문 매핑 제거 완료.")
    except Exception as e:
        print(f"❌ SQLite 데이터 삭제 중 오류 발생: {e}")
        sys.exit(1)

    # 4. BM25 인덱스에서 법령 청크 제거
    print("\n[3단계] BM25 인덱스 제거 중...")
    # SQLite와 Chroma에서 제거되었으므로, 최종 단계에서 전체 재빌드하면 자동으로 동기화됩니다.
    print(" -> SQLite/Chroma에서 법령이 제거되었습니다. 최종 단계에서 BM25 색인이 전체 재구축됩니다.")

    # 5. 법령 데이터만 다시 파싱 및 인제스천
    print("\n[4단계] 대한민국 법령 데이터셋만 선별하여 재파싱 및 신규 인제스트 시작...")
    parser = LegalParserV2()
    law_documents = []
    
    # TS_법령 디렉토리 경로 찾기
    law_dir = None
    for root, dirs, files in os.walk(parser.raw_data_path):
        if "01.원천데이터" in root and "TS_법령" in root:
            law_dir = root
            break
            
    if not law_dir:
        print("❌ 에러: 원천데이터의 'TS_법령' 폴더를 찾을 수 없습니다. 경로를 확인하세요.")
        sys.exit(1)
        
    print(f" -> 법령 원천 데이터 폴더 찾음: {law_dir}")
    files = os.listdir(law_dir)
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(law_dir, file)
            doc_blocks = parser.parse_csv_file(file_path, "TS_법령")
            if doc_blocks:
                law_documents.extend(doc_blocks)
                
    print(f" -> 총 {len(law_documents)}개의 법령 조문(부모 블록) 파싱 완료 (정규식 수정본 반영).")
    
    # 6. 분할 및 적재
    splitter = LegalSplitterV2()
    law_chunks = splitter.split_documents(law_documents)
    if not law_chunks:
        print(" -> 적재할 신규 법령 청크가 없습니다.")
        sys.exit(0)
        
    # Chroma에 적재
    print("\n[5단계] Chroma DB 및 SQLite에 법령 데이터 재적재 중...")
    clean_law_chunks = db.add_documents(law_chunks)
    
    # 7. BM25 인덱스 전체 재구축 및 저장
    print("\n[6단계] 최적화된 BM25 역색인 전체 재구축 및 파일 동기화 중...")
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from rebuild_bm25 import rebuild_full_bm25
        rebuild_full_bm25()
        print(" -> BM25 역색인 업데이트 성공.")
    except Exception as e:
        print(f"❌ BM25 역색인 재구축 실패: {e}")
        sys.exit(1)
    print("="*60)
    print("=== [법령 데이터 단독 복구 완료! 모든 정규식 버그가 수정되어 반영되었습니다] ===")
    print("="*60)

if __name__ == "__main__":
    main()
