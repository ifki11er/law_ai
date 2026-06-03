import os
import glob
import json
import re
import pandas as pd
from typing import List, Dict, Any
from config.settings import AI_HUB_DATA_PATH

class LegalParserV2:
    """
    [v2] Parser for AI Hub raw CSV data and metadata enrichment from JSON files.
    Includes advanced Hangul word/name restoration and whitespace normalization.
    """
    
    def __init__(self, training_path: str = AI_HUB_DATA_PATH):
        self.raw_data_path = os.path.join(training_path, "01.원천데이터")
        self.labeled_data_path = os.path.join(training_path, "02.라벨링데이터")
        self.json_map = {}
        self._pre_index_json_files()

    def _pre_index_json_files(self):
        """
        Pre-scan all JSON files in 02.라벨링데이터 to build a prefix-to-path map.
        This avoids doing N recursive glob scans on every file and speeds up ingestion from hours to seconds.
        """
        if not os.path.exists(self.labeled_data_path):
            return
        print("라벨링 데이터(JSON) 메타데이터 색인 중... (v2 파서)")
        search_pattern = os.path.join(self.labeled_data_path, "**", "*.json")
        json_files = glob.glob(search_pattern, recursive=True)
        for path in json_files:
            filename = os.path.basename(path)
            # Prefix format: e.g., "HS_K_10026_QA_2.json" -> "HS_K_10026"
            if "_QA_" in filename:
                prefix = filename.split("_QA_")[0]
            else:
                prefix = os.path.splitext(filename)[0]
            self.json_map[prefix] = path
        print(f"총 {len(self.json_map)}개의 JSON 메타데이터 매핑을 로드했습니다.")
        
    def find_metadata_from_json(self, file_prefix: str) -> Dict[str, str]:
        """
        Look up metadata for the prefix in the pre-indexed map.
        """
        metadata = {}
        json_path = self.json_map.get(file_prefix)
        
        if json_path:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info = data.get("info", {})
                    metadata["caseNum"] = info.get("caseNum", "")
                    metadata["caseName"] = info.get("caseName", "")
                    metadata["finalDate"] = info.get("finalDate", "")
                    metadata["courtCode"] = info.get("courtCode", "")
                    metadata["lawClass"] = info.get("lawClass", "")
                    metadata["DocuType"] = info.get("DocuType", "")
            except Exception:
                pass
                
        return metadata

    def parse_csv_file(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """
        Parse a single CSV file, combine sentences, normalize whitespaces, and build metadata.
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='cp949')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        if df.empty or '내용' not in df.columns:
            return None

        # Sort by sentence index if column exists
        if '문장번호' in df.columns:
            df = df.sort_values(by='문장번호')

        # Combine all sentences into a single document text
        full_text = "\n".join(df['내용'].dropna().astype(str).tolist())
        if not full_text.strip():
            return None

        # [v2 핵심 개선] 한 자씩 벌어져 있는 한글 단어/이름 복원 및 연속 공백 압축
        def merge_match(m):
            # 매칭된 단어 내부의 모든 공백 제거
            return re.sub(r'\s+', '', m.group(0))
            
        # 1. 1~3칸의 공백으로 벌어진 단일 한글 문자열(e.g., '재 판 관', '윤 영 철', '헌   법')을 하나로 붙임
        # 단어 사이의 넓은 공백(e.g., 재판장과 재판관 사이의 7칸 공백)은 병합 대상에서 제외됨
        full_text = re.sub(r'\b[가-힣]\b(?:\s{1,3}\b[가-힣]\b)+', merge_match, full_text)
        
        # 2. 남은 가로 공백들(e.g., 단어 사이의 공백)을 1개의 공백으로 압축
        full_text = re.sub(r'[ \t]+', ' ', full_text)
        
        # 3. 연속된 언더바(_), 대시(-), 등호(=) 등 서명선/구분선 데코레이션 노이즈 제거 (3개 이상 연속될 시 삭제)
        full_text = re.sub(r'[_\-=]{3,}', '', full_text)
        
        # 특정 법률 타이틀이나 서명란 부근의 불필요한 줄바꿈 중복(3회 이상) 방지
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)

        # Determine Document ID and File Prefix
        filename = os.path.basename(file_path)
        file_prefix = os.path.splitext(filename)[0]  # e.g., HS_K_10026
        
        # Determine the first column (usually ID)
        id_col = df.columns[0]
        doc_id = str(df[id_col].iloc[0]) if not df.empty else ""

        # Map document type to a friendly Korean name
        type_mapping = {
            "TS_결정례": "헌법재판소 결정례",
            "TS_법령": "대한민국 법령",
            "TS_판결문": "법원 판결문",
            "TS_해석례": "법률 해석례"
        }
        friendly_type = type_mapping.get(doc_type, doc_type)

        # Build base metadata
        meta = {
            "doc_id": doc_id,
            "file_prefix": file_prefix,
            "doc_type": friendly_type,
            "source_file": filename,
            "caseNum": "",
            "caseName": "",
            "finalDate": "",
            "courtCode": ""
        }

        # Enrich with JSON metadata
        json_meta = self.find_metadata_from_json(file_prefix)
        meta.update(json_meta)

        # Create a user-friendly source title
        if meta["caseNum"] and meta["caseName"]:
            title = f"{friendly_type} ({meta['caseNum']} - {meta['caseName']})"
        elif meta["caseNum"]:
            title = f"{friendly_type} ({meta['caseNum']})"
        else:
            title = f"{friendly_type} (ID: {doc_id})"
            
        meta["title"] = title

        return {
            "text": full_text,
            "metadata": meta
        }

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Scan all subdirectories under 01.원천데이터 and parse CSV files.
        """
        documents = []
        subdirs = ["TS_결정례", "TS_법령", "TS_판결문", "TS_해석례"]
        
        for subdir in subdirs:
            dir_path = os.path.join(self.raw_data_path, subdir)
            if not os.path.exists(dir_path):
                print(f"Directory not found: {dir_path}")
                continue
                
            csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
            print(f"Parsing {len(csv_files)} files in {subdir}... (v2 파서)")
            
            for file_path in csv_files:
                doc = self.parse_csv_file(file_path, subdir)
                if doc:
                    documents.append(doc)
                    
        print(f"Successfully parsed total of {len(documents)} documents. (v2 파서)")
        return documents
