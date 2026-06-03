import sys
import os
import shutil
from core.parser import LegalParserV2
from core.splitter import LegalSplitterV2
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from config.settings import CHROMA_DB_PATH, BM25_INDEX_PATH

def main():
    print("\n" + "="*50)
    print("=== [인제스천 파이프라인 v4] 계층구조 & 하이브리드 DB 구축 시작 ===")
    print("="*50)
    
    # 0. 기존 DB 및 BM25 파일 초기화
    if os.path.exists(CHROMA_DB_PATH):
        print(f"\n[초기화] 기존 v4 DB 폴더({CHROMA_DB_PATH})를 초기화(삭제)합니다...")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print("[초기화 완료] 기존 v4 DB 삭제 완료.")
        except Exception as e:
            print(f"⚠️ 경고: 기존 v4 DB 폴더 삭제 실패: {e}")
            
    if os.path.exists(BM25_INDEX_PATH):
        print(f"[초기화] 기존 BM25 인덱스 파일({BM25_INDEX_PATH})을 삭제합니다...")
        try:
            os.remove(BM25_INDEX_PATH)
            print("[초기화 완료] 기존 BM25 파일 삭제 완료.")
        except Exception as e:
            print(f"⚠️ 경고: 기존 BM25 파일 삭제 실패: {e}")

    # 1. 원천 데이터 파싱 및 계층구조 메타데이터 매핑
    print("\n[Step 1] AI Hub CSV/JSON 원천 데이터 로드 및 계층 구조 분할 시작...")
    parser = LegalParserV2()
    documents = parser.load_all_documents()
    if not documents:
        print("에러: 적재할 문서를 찾을 수 없습니다.")
        sys.exit(1)
    print(f"[Step 1 완료] 총 {len(documents)}개의 논리적 부모 단락(Parent Blocks)을 생성했습니다.")
        
    # 2. 컨텍스트 합성형 부모-자식 분할
    print("\n[Step 2] 브레드크럼 접두사 합성 및 자식 청크 분할(v4) 시작...")
    splitter = LegalSplitterV2()
    chunks = splitter.split_documents(documents)
    if not chunks:
        print("에러: 분할된 청크가 존재하지 않습니다.")
        sys.exit(1)
    print(f"[Step 2 완료] 총 {len(chunks)}개의 컨텍스트 합성 자식 청크 생성 완료.")
        
    # 3. 임베딩 모델 로드
    print("\n[Step 3] Ollama/bge-m3 임베딩 모델 준비 중...")
    embedding = LegalEmbedding()
    print("[Step 3 완료] 임베딩 모델 준비 완료.")
    
    # 4. Chroma DB 및 BM25 동시 구축 및 저장
    print(f"\n[Step 4] Chroma DB v4 및 BM25 하이브리드 적재 시작...")
    print(f"[Step 4] 벡터 DB 저장 경로: {CHROMA_DB_PATH}")
    print(f"[Step 4] BM25 색인 경로: {BM25_INDEX_PATH}")
    
    db = LegalVectorDB(
        embedding_function=embedding.get_embeddings(), 
        db_path=CHROMA_DB_PATH,
        bm25_path=BM25_INDEX_PATH
    )
    db.add_documents(chunks)
    
    print("\n" + "="*50)
    print("=== [인제스천 파이프라인 v4] 계층구조 & 하이브리드 RAG DB 구축 성공 완료! ===")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
