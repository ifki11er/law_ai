import sys
from core.parser import LegalParser
from core.splitter import LegalSplitter
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB

def main():
    print("\n" + "="*50)
    print("=== [인제스천 파이프라인] 로컬 DB 구축 시작 ===")
    print("="*50)
    
    # 0. 기존 DB 디렉토리가 있으면 중복 적재 방지를 위해 완전히 초기화(삭제)
    import os
    import shutil
    from config.settings import CHROMA_DB_PATH
    if os.path.exists(CHROMA_DB_PATH):
        print(f"\n[초기화] 기존 DB 폴더({CHROMA_DB_PATH})가 감지되어 데이터 중복 방지를 위해 삭제합니다...")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print("[초기화 완료] 기존 DB 초기화 완료.")
        except Exception as e:
            print(f"⚠️ 경고: 기존 DB 폴더 삭제 실패 (해당 DB를 다른 프로그램이 읽는 중일 수 있습니다): {e}")
            print("안전한 초기화를 위해 가급적 다른 터미널/서버를 끄고 수동으로 해당 폴더를 삭제하신 후 재실행을 권장합니다.")

    # 1. 원천 데이터 파싱 및 메타데이터 결합
    print("\n[Step 1] AI Hub CSV 원천 데이터 로드 및 JSON 메타데이터(사건번호 등) 매핑 시작...")
    parser = LegalParser()
    documents = parser.load_all_documents()
    if not documents:
        print("에러: 적재할 문서를 찾을 수 없습니다. 원천데이터 경로를 확인해주세요.")
        sys.exit(1)
    print(f"[Step 1 완료] 총 {len(documents)}개의 법률 문서를 성공적으로 파싱했습니다.")
        
    # 2. 문맥 의미 단위 청크 분할
    print("\n[Step 2] 긴 법률 문서를 LLM이 읽을 수 있는 크기(약 800자)로 쪼개기(청크 분할) 시작...")
    splitter = LegalSplitter()
    chunks = splitter.split_documents(documents)
    if not chunks:
        print("에러: 분할된 청크가 존재하지 않습니다.")
        sys.exit(1)
    print(f"[Step 2 완료] 문서를 총 {len(chunks)}개의 의미 단위 조각으로 쪼갰습니다.")
        
    # 3. 임베딩 모델 로드
    print("\n[Step 3] 글자를 벡터(숫자 좌표)로 번역할 Ollama/bge-m3 임베딩 모델 준비 중...")
    embedding = LegalEmbedding()
    print("[Step 3 완료] 임베딩 모델 준비 완료.")
    
    # 4. Chroma DB 초기화 및 데이터 적재
    print("\n[Step 4] Chroma 로컬 벡터 DB 초기화 및 데이터 적재 시작...")
    print("[Step 4] 각 텍스트 조각을 벡터로 변환하여 Chroma DB 파일(data/chroma_db)에 쓰는 중...")
    db = LegalVectorDB(embedding_function=embedding.get_embeddings())
    db.add_documents(chunks)
    
    print("\n" + "="*50)
    print("=== [인제스천 파이프라인] 로컬 DB 구축 성공 완료! ===")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
