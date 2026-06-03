import os
import glob
from core.parser import LegalParser
from core.splitter import LegalSplitter

def main():
    print("=== [파서 및 청커 동작 검증기] ===")
    
    # 파서 초기화
    parser = LegalParser()
    
    # 원천데이터 경로에서 테스트할 첫 번째 CSV 파일 찾기
    subdirs = ["TS_결정례", "TS_법령", "TS_판결문", "TS_해석례"]
    target_csv = None
    selected_subdir = None
    
    for subdir in subdirs:
        dir_path = os.path.join(parser.raw_data_path, subdir)
        if os.path.exists(dir_path):
            csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
            if csv_files:
                target_csv = csv_files[0]
                selected_subdir = subdir
                break
                
    if not target_csv:
        print("에러: 검증할 CSV 파일을 찾을 수 없습니다. 데이터셋 경로를 확인하세요.")
        return
        
    print(f"\n[검증 대상 파일]: {target_csv}")
    
    # 1. 파싱 검증
    doc = parser.parse_csv_file(target_csv, selected_subdir)
    if not doc:
        print("❌ 파싱 실패!")
        return
        
    print("\n" + "="*50)
    print("=== 1. 파싱 및 메타데이터 결합 결과 ===")
    print("="*50)
    print(f"● 원본 파일: {doc['metadata']['source_file']}")
    print(f"● 문서 타입: {doc['metadata']['doc_type']}")
    print(f"● 사건 번호: {doc['metadata']['caseNum']}")
    print(f"● 사 건 명 : {doc['metadata']['caseName']}")
    print(f"● 선고 일자: {doc['metadata']['finalDate']}")
    print(f"● 매핑 타이틀: {doc['metadata']['title']}")
    print(f"● 전체 합쳐진 본문 크기: {len(doc['text'])}자")
    print(f"\n[본문 처음 300자 미리보기]:\n{doc['text'][:300]}...")
    
    # 2. 청킹 검증
    splitter = LegalSplitter()
    chunks = splitter.split_documents([doc])
    
    print("\n" + "="*50)
    print(f"=== 2. 청킹 분할 결과 (총 {len(chunks)}개 조각) ===")
    print("="*50)
    
    # 앞선 3개 청크만 예시로 출력
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[청크 조각 #{i+1}] (글자수: {len(chunk['text'])}자 | 오버랩/인덱스: {chunk['metadata']['chunk_index']})")
        print("-" * 50)
        print(chunk['text'])
        print("-" * 50)
        
    if len(chunks) > 3:
        print(f"\n... (나머지 {len(chunks) - 3}개의 청크 생략) ...")

if __name__ == "__main__":
    main()
