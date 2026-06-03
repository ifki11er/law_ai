import sys
import os
import shutil
from core.parser import LegalParserV2
from core.splitter import LegalSplitterV2
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from config.settings import CHROMA_DB_PATH

def main():
    # Define v2 database path
    CHROMA_DB_PATH = CHROMA_DB_PATH

    print("\n" + "="*50)
    print("=== [인제스천 파이프라인 v2] 하이브리드 세만틱 DB 구축 시작 ===")
    print("="*50)
    
    # 0. 기존 v2 DB 디렉토리가 있으면 중복 적재 방지를 위해 초기화
    if os.path.exists(CHROMA_DB_PATH):
        print(f"\n[초기화] 기존 v2 DB 폴더({CHROMA_DB_PATH})가 감지되어 삭제합니다...")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print("[초기화 완료] 기존 v2 DB 초기화 완료.")
        except Exception as e:
            print(f"⚠️ 경고: 기존 v2 DB 폴더 삭제 실패: {e}")

    # 1. 원천 데이터 파싱 및 메타데이터 결합
    print("\n[Step 1] AI Hub CSV 원천 데이터 로드 및 JSON 메타데이터 매핑 시작...")
    parser = LegalParserV2()
    documents = parser.load_all_documents()
    if not documents:
        print("에러: 적재할 문서를 찾을 수 없습니다.")
        sys.exit(1)
    print(f"[Step 1 완료] 총 {len(documents)}개의 법률 문서를 성공적으로 파싱했습니다.")
        
    # 2. 하이브리드 세만틱 청크 분할 (v2 적용)
    print("\n[Step 2] 임베딩 유사도 분석 기반의 의미 단위 2차 청크 분할(v2) 시작...")
    splitter = LegalSplitterV2()
    chunks = splitter.split_documents(documents)
    if not chunks:
        print("에러: 분할된 청크가 존재하지 않습니다.")
        sys.exit(1)
    print(f"[Step 2 완료] 총 {len(chunks)}개의 세만틱 조각으로 분할 완료.")
        
    # 3. 임베딩 모델 로드
    print("\n[Step 3] Ollama/bge-m3 임베딩 모델 준비 중...")
    embedding = LegalEmbedding()
    print("[Step 3 완료] 임베딩 모델 준비 완료.")
    
    # 4. Chroma DB v2 초기화 및 데이터 적재
    print(f"\n[Step 4] Chroma 로컬 벡터 DB v2 초기화 및 데이터 적재 시작...")
    print(f"[Step 4] 벡터 DB 저장 경로: {CHROMA_DB_PATH}")
    db = LegalVectorDB(embedding_function=embedding.get_embeddings(), db_path=CHROMA_DB_PATH)
    db.add_documents(chunks)
    
    print("\n" + "="*50)
    print("=== [인제스천 파이프라인 v2] 하이브리드 세만틱 DB 구축 성공 완료! ===")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
