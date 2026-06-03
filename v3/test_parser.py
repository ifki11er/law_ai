import os
import glob
from core.parser import LegalParserV2
from core.splitter import LegalSplitterV2

def main():
    print("=== [v3 파서 및 청커 동작 검증기] ===")
    
    # 파서 초기화
    parser = LegalParserV2()
    
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
    doc_blocks = parser.parse_csv_file(target_csv, selected_subdir)
    if not doc_blocks:
        print("❌ 파싱 실패!")
        return
        
    first_block = doc_blocks[0]
    print("\n" + "="*50)
    print(f"=== 1. v3 파싱 및 계층구조 분할 결과 (총 {len(doc_blocks)}개 부모 블록) ===")
    print("="*50)
    print(f"● 원본 파일: {first_block['metadata']['source_file']}")
    print(f"● 문서 타입: {first_block['metadata']['doc_type']}")
    print(f"● 사건 번호: {first_block['metadata']['caseNum']}")
    print(f"● 사 건 명 : {first_block['metadata']['caseName']}")
    print(f"● 선고 일자: {first_block['metadata']['finalDate']}")
    print(f"● 대표 타이틀: {first_block['metadata']['title']}")
    
    # 각 부모 블록 요약 출력
    print("\n[생성된 부모 블록 목록]:")
    for idx, b in enumerate(doc_blocks):
        print(f"  [{idx+1}] 블록명: {b['metadata']['section_name']} | 글자수: {len(b['text'])}자 | 브레드크럼: {b['metadata']['breadcrumb']}")
        
    print(f"\n[첫 번째 부모 블록 처음 300자 미리보기]:\n{first_block['text'][:300]}...")
    
    # 2. 청킹 검증
    splitter = LegalSplitterV2()
    chunks = splitter.split_documents(doc_blocks)
    
    print("\n" + "="*50)
    print(f"=== 2. v3 컨텍스트 합성 청킹 결과 (총 {len(chunks)}개 자식 조각) ===")
    print("="*50)
    
    # 앞선 3개 자식 청크만 예시로 출력
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[자식 청크 조각 #{i+1}] (글자수: {len(chunk['text'])}자 | 인덱스: {chunk['metadata']['child_index']})")
        print("-" * 50)
        print(f"합성 텍스트:\n{chunk['text']}")
        print(f"\n연결된 부모 조항 본문 크기: {chunk['metadata']['parent_len']}자")
        print("-" * 50)
        
    if len(chunks) > 3:
        print(f"\n... (나머지 {len(chunks) - 3}개의 자식 청크 생략) ...")

if __name__ == "__main__":
    main()
