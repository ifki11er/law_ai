import os
import glob
import pandas as pd
import numpy as np
from config.settings import AI_HUB_DATA_PATH

def analyze_subdir(dir_path, subdir, max_files=50):
    csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
    if not csv_files:
        return None

    sample_size = min(len(csv_files), max_files)
    target_files = csv_files[:sample_size]
    
    total_rows = 0
    total_chars_list = []
    unit_lengths = []
    
    for file_path in target_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')
            
        total_rows += len(df)
        content_col = '내용' if '내용' in df.columns else df.columns[-1]
        
        # 1. 파일별 전체 텍스트 크기 수집
        sentences = df[content_col].dropna().astype(str).tolist()
        file_chars = sum(len(s) for s in sentences)
        total_chars_list.append(file_chars)
        
        # 2. 구조적 단위 크기 수집
        if '구분' in df.columns:
            # 조문(제X조) 또는 구분 기준으로 단락 묶기
            current_unit = []
            for _, row in df.iterrows():
                gubu = str(row['구분'])
                text = str(row[content_col])
                
                # 조문이 나타나거나 주요 구분이 변경될 때 묶음 처리
                if '조문' in gubu or any(keyword in gubu for keyword in ['판결요지', '판시사항', '질의요지', '회답']):
                    if current_unit:
                        unit_lengths.append(sum(len(t) for t in current_unit))
                        current_unit = []
                current_unit.append(text)
            if current_unit:
                unit_lengths.append(sum(len(t) for t in current_unit))
        else:
            # 단순 문장/단락 길이 수집
            unit_lengths.extend([len(s) for s in sentences])

    return {
        "part": subdir,
        "sample_size": sample_size,
        "total_rows": total_rows,
        "file_chars": total_chars_list,
        "unit_lengths": unit_lengths
    }

def print_comparison_table(results):
    print("\n" + "="*85)
    print(" 📊 [ 파트별 법률 데이터셋 규격 비교 분석 성적표 ]")
    print("="*85)
    
    # Table header
    header_format = "{:<18} | {:>8} | {:>14} | {:>15} | {:>15}"
    row_format = "{:<18} | {:>9} | {:>12,}자 | {:>13,}자 | {:>13,}자"
    
    print(header_format.format("데이터 파트명", "분석파일수", "평균 파일크기", "평균 의미단락크기", "80%이하 단락크기"))
    print("-" * 85)
    
    all_file_chars = []
    all_unit_lengths = []
    total_files = 0
    
    for r in results:
        part_name = r["part"]
        files_count = r["sample_size"]
        total_files += files_count
        
        # File sizes stats
        avg_file_char = int(np.mean(r["file_chars"])) if r["file_chars"] else 0
        all_file_chars.extend(r["file_chars"])
        
        # Unit sizes stats
        avg_unit_char = int(np.mean(r["unit_lengths"])) if r["unit_lengths"] else 0
        pct_80_unit = int(np.percentile(r["unit_lengths"], 80)) if r["unit_lengths"] else 0
        all_unit_lengths.extend(r["unit_lengths"])
        
        # Print part row
        print(row_format.format(part_name, f"{files_count}개", avg_file_char, avg_unit_char, pct_80_unit))
        
    print("-" * 85)
    
    # Calculate Overall metrics
    overall_avg_file = int(np.mean(all_file_chars)) if all_file_chars else 0
    overall_avg_unit = int(np.mean(all_unit_lengths)) if all_unit_lengths else 0
    overall_pct_80 = int(np.percentile(all_unit_lengths, 80)) if all_unit_lengths else 0
    
    # Print Overall row
    print(row_format.format("★ 전체 통합 평균 ★", f"{total_files}개", overall_avg_file, overall_avg_unit, overall_pct_80))
    print("="*85)
    
    # Suggestion message
    print("\n💡 [ 청킹 전략 권장 가이드라인 ]")
    print(f"  - 개별 조문/단락의 흐름이 끊기지 않는 안전한 청킹 사이즈: ★ {overall_pct_80:,}자 이상 ★")
    print(f"  - 현재 설정값: CHUNK_SIZE = 1,500자 / OVERLAP = 300자 (통합 80%선인 {overall_pct_80:,}자를 안전하게 커버함)")
    print("="*85 + "\n")

def main():
    base_dir = os.path.join(AI_HUB_DATA_PATH, "01.원천데이터")
    subdirs = ["TS_결정례", "TS_법령", "TS_판결문", "TS_해석례"]
    
    results = []
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if os.path.exists(dir_path):
            res = analyze_subdir(dir_path, subdir, max_files=50)
            if res:
                results.append(res)
                
    if results:
        print_comparison_table(results)
    else:
        print("분석할 법률 데이터가 존재하지 않습니다. 경로를 확인해주세요.")

if __name__ == "__main__":
    main()
