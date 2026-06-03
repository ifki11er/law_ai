import os
import sys
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB

def main():
    print("=== [Chroma DB 적재 데이터 검증기] ===")
    
    # 1. DB 존재 여부 확인
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db")
    if not os.path.exists(db_dir):
        print(f"❌ 에러: DB 디렉토리({db_dir})가 존재하지 않습니다.")
        print("먼저 'python ingest.py'를 통해 데이터 적재를 완료해야 합니다.")
        return
        
    # 2. RAG 컴포넌트 로드
    print("임베딩 모델 및 Chroma DB 로드 중...")
    try:
        embedding = LegalEmbedding()
        vector_db = LegalVectorDB(embedding_function=embedding.get_embeddings())
    except Exception as e:
        print(f"❌ DB 로드 중 오류 발생: {e}")
        print("Ollama가 정상적으로 켜져 있고 bge-m3 모델이 설치되어 있는지 확인하세요.")
        return

    # 3. 데이터 통계 조회
    chroma_collection = vector_db.get_db()
    try:
        total_count = chroma_collection._collection.count()
        print(f"\n✅ DB 로드 성공!")
        print(f"● 현재 Chroma DB에 저장된 총 청크(조각) 개수: {total_count}개")
    except Exception as e:
        print(f"❌ 통계 조회 중 오류 발생: {e}")
        return

    if total_count == 0:
        print("⚠️ DB가 비어 있습니다. ingest.py가 정상적으로 데이터를 저장했는지 확인하세요.")
        return

    # 4. 샘플 데이터 출력 (최대 2개)
    print("\n" + "="*50)
    print("=== [1. 실제 DB 저장 샘플 데이터 (상위 2개)] ===")
    print("="*50)
    
    try:
        # Chroma에서 limit=2로 데이터를 직접 추출
        db_data = chroma_collection.get(limit=2)
        
        ids = db_data.get('ids', [])
        documents = db_data.get('documents', [])
        metadatas = db_data.get('metadatas', [])
        
        for i in range(len(ids)):
            print(f"\n▶ [저장된 데이터 #{i+1}] (ID: {ids[i]})")
            print("-" * 50)
            print(f"출처 타이틀: {metadatas[i].get('title', '정보 없음')}")
            print(f"사건 번호  : {metadatas[i].get('caseNum', '정보 없음')}")
            print(f"사 건 명   : {metadatas[i].get('caseName', '정보 없음')}")
            print(f"문서 타입  : {metadatas[i].get('doc_type', '정보 없음')}")
            print(f"청크 인덱스: {metadatas[i].get('chunk_index', '0')}")
            print(f"\n[저장된 본문 내용]:\n{documents[i]}")
            print("-" * 50)
    except Exception as e:
        print(f"샘플 출력 중 오류 발생: {e}")

    # 5. 유사도 검색 테스트 수행
    print("\n" + "="*50)
    print("=== [2. 검색 작동 테스트 (쿼리: 기소유예처분)] ===")
    print("="*50)
    
    test_query = "기소유예처분"
    print(f"질문 '{test_query}'으로 Chroma DB 유사도 검색(Top 2) 수행 중...")
    
    try:
        results = vector_db.search_with_scores(test_query, k=2)
        print(f"검색 완료 (검색 결과 {len(results)}개 확보):\n")
        
        for idx, (doc, score) in enumerate(results):
            print(f"[{idx+1}등 결과] 유사도 스코어 (Distance): {score:.4f} (낮을수록 좋음)")
            print(f"  └─ 출처: {doc.metadata.get('title')}")
            print(f"  └─ 본문 요약: {doc.page_content[:150].replace('\n', ' ')}...")
            print("-" * 50)
    except Exception as e:
        print(f"검색 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
