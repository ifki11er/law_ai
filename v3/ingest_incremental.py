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
    기존 BM25 인덱스 메타데이터를 확인하여 이미 데이터베이스에 적재 완료된 파일명 목록을 반환합니다.
    """
    ingested = set()
    if os.path.exists(BM25_INDEX_PATH):
        try:
            with open(BM25_INDEX_PATH, 'rb') as f:
                payload = pickle.load(f)
                chunks = payload.get("chunks", [])
                for chunk in chunks:
                    source = chunk.get("metadata", {}).get("source_file")
                    if source:
                        ingested.add(source)
            print(f"[체크 완료] 기존 데이터베이스에서 이미 적재된 파일 {len(ingested)}개를 확인했습니다.")
        except Exception as e:
            print(f"⚠️ 경고: 기존 BM25 색인 파일을 읽어오지 못했습니다 (무시하고 진행): {e}")
    else:
        print("[체크 완료] 기존 적재 이력이 없습니다. 전체 신규 적재를 진행합니다.")
    return ingested

def main():
    print("\n" + "="*60)
    print("=== [증분 인제스천 파이프라인 v3 가동 (자동 파일 감지)] ===")
    print("="*60)
    
    # 0. 이미 DB에 들어간 파일명 목록 로드
    ingested_files = get_ingested_files()
    
    # 1. 원천 데이터 폴더 스캔 (기존 ingest.py와 완벽히 동일한 탐색 메커니즘)
    parser = LegalParserV2()
    raw_data_path = os.path.join(AI_HUB_DATA_PATH, "01.원천데이터")
    subdirs = ["TS_결정례", "TS_법령", "TS_판결문", "TS_해석례"]
    
    new_documents = []
    
    print("\n[Step 1] 신규 추가된 CSV 파일 자동 탐색 중...")
    for subdir in subdirs:
        dir_path = os.path.join(raw_data_path, subdir)
        if not os.path.exists(dir_path):
            continue
            
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # 이미 적재된 파일이라면 스킵!
            if filename in ingested_files:
                # 디버깅 편의를 위해 스킵 내역은 간단히만 출력
                continue
                
            print(f" └─ [신규 파일 감지] {subdir}/{filename} 파싱 중...")
            doc_blocks = parser.parse_csv_file(file_path, subdir)
            if doc_blocks:
                new_documents.extend(doc_blocks)
                
    if not new_documents:
        print("\n[알림] 새로 추가된 법령/판례 CSV 파일이 없습니다. 작업을 종료합니다.")
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
    db.add_documents(new_chunks)
    print("[Step 4 완료] Chroma DB 적재 성공.")
    
    # 5. BM25 인덱스 병합 및 전체 재구축
    print("\n[Step 5] BM25 역색인 병합 및 전체 재구축 시작...")
    try:
        if db.bm25_data is not None:
            existing_chunks = db.bm25_data["chunks"]
            all_chunks = existing_chunks + new_chunks
            print(f" -> 기존 청크 {len(existing_chunks):,}개 + 신규 청크 {len(new_chunks):,}개 병합 완료.")
        else:
            all_chunks = new_chunks
            print(" -> 기존 BM25 인덱스가 존재하지 않아 신규 인덱스로 단독 생성합니다.")
            
        from rank_bm25 import BM25Okapi
        corpus_texts = [chunk["text"] for chunk in all_chunks]
        tokenized_corpus = [db.tokenize_korean(t) for t in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        bm25_payload = {
            "bm25": bm25,
            "chunks": all_chunks
        }
        
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(bm25_payload, f)
            
        db.bm25_data = bm25_payload
        print(f"[Step 5 완료] BM25 역색인 파일 업데이트 성공 (총 청크 수: {len(all_chunks):,}개).")
    except Exception as e:
        print(f"❌ 에러: BM25 역색인 재구축 실패: {e}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("=== [증분 인제스천 완료] 신규 법률 데이터가 데이터베이스에 성공적으로 병합되었습니다! ===")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
