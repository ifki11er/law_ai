import os
import sys
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from core.update_manager import LegalUpdateManager

def run_test():
    print("=== [v3] 법률 개정/폐지/신설 파이프라인 테스트 ===")
    
    # Initialize Embedding and Database
    embedding = LegalEmbedding()
    db = LegalVectorDB(
        embedding_function=embedding.get_embeddings()
    )
    
    # Initialize Update Manager
    manager = LegalUpdateManager(db)
    
    # 1. 신설 테스트를 위한 임의의 법령 메타데이터 정의
    sample_doc_meta = {
        "doc_id": "99999",
        "file_prefix": "test_law_99999",
        "doc_type": "대한민국 법령",
        "caseNum": "제12345호",
        "caseName": "인공지능기본법",
        "source_file": "test_law_99999.csv"
    }
    
    print("\n--- [Step 1] 신설(Insertion) 테스트 ---")
    section_name = "제5조"
    article_text = (
        "제5조(인공지능의 안전성 확보) ① 정부는 인공지능 기술의 발전과 신뢰성 확보를 위하여 "
        "필요한 조치를 강구하여야 한다. ② 인공지능 개발자는 인간의 존엄성을 해치지 않도록 "
        "윤리적 가치를 준수하여 개발하여야 한다."
    )
    
    # 신설 실행 (최초 버전: v1)
    insert_success = manager.insert_article(
        doc_metadata=sample_doc_meta,
        section_name=section_name,
        text=article_text,
        version="v1"
    )
    
    if insert_success:
        print("-> [신설 성공] 데이터베이스에 정상 입력되었습니다.")
        
        # 검색 테스트
        print("\n[검색 테스트] '인공지능 안전성' 검색 결과:")
        results = db.search_hybrid("인공지능 안전성", k=2)
        for idx, doc in enumerate(results):
            print(f"[{idx+1}] {doc.metadata.get('title')} (active: {doc.metadata.get('is_active')})")
            print(f"    - 내용 일부: {doc.page_content[:100]}...")
            
    # 2. 개정 테스트
    print("\n--- [Step 2] 개정(Amendment) 테스트 ---")
    old_id = "test_law_99999_제5조_v1"
    new_article_text = (
        "제5조(인공지능의 안전성 확보 및 제재) ① 정부는 인공지능 기술의 발전과 신뢰성 확보를 위하여 "
        "강력한 모니터링 시스템과 안전성 가이드라인을 시행한다. ② 위반 시 3천만원 이하의 과태료를 부과한다. "
        "③ 인공지능 개발자는 윤리적 가치를 반드시 준수하여 설계해야 한다."
    )
    
    # 개정 실행 (기존 v1은 비활성화하고, v2로 삽입)
    amend_success = manager.amend_article(
        old_parent_id=old_id,
        doc_metadata=sample_doc_meta,
        section_name=section_name,
        new_text=new_article_text,
        new_version="v2"
    )
    
    if amend_success:
        print("-> [개정 성공] 개정본이 적용되었습니다.")
        
        # 재검색 테스트 (개정본이 상위에 올라오고, 구버전은 비활성화되었는지 확인)
        print("\n[재검색 테스트] '인공지능 안전성' 검색 결과:")
        results = db.search_hybrid("인공지능 안전성", k=2)
        for idx, doc in enumerate(results):
            print(f"[{idx+1}] ID: {doc.metadata.get('parent_id')} (active: {doc.metadata.get('is_active')})")
            print(f"    - 내용: {doc.page_content[:150]}...")

    # 3. 폐지 테스트
    print("\n--- [Step 3] 폐지(Repeal) 테스트 ---")
    current_active_id = "test_law_99999_제5조_v2"
    
    # 폐지 실행
    repeal_success = manager.repeal_article(current_active_id)
    if repeal_success:
        print("-> [폐지 성공] 비활성화 처리가 완료되었습니다.")
        
        # 다시 검색 (비활성화 되었으므로 검색 결과가 나오지 않거나 구버전/신버전 모두 매칭에서 걸러져야 함)
        print("\n[검색 테스트] 폐지 후 '인공지능 안전성' 검색 결과:")
        results = db.search_hybrid("인공지능 안전성", k=2)
        print(f"검색 결과 개수: {len(results)}")
        for idx, doc in enumerate(results):
            print(f"[{idx+1}] ID: {doc.metadata.get('parent_id')} (active: {doc.metadata.get('is_active')})")

if __name__ == "__main__":
    run_test()
