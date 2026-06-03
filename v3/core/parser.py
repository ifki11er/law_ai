import os
import glob
import json
import re
import pandas as pd
from typing import List, Dict, Any
from config.settings import AI_HUB_DATA_PATH

class LegalParserV2:
    """
    [v3] Parser for AI Hub raw CSV data.
    Groups sentences statefully into logical Parent Blocks (Articles, Sections)
    and assigns breadcrumb metadata representing the logical hierarchy of the document.
    """
    
    def __init__(self, training_path: str = AI_HUB_DATA_PATH):
        self.raw_data_path = os.path.join(training_path, "01.원천데이터")
        self.labeled_data_path = os.path.join(training_path, "02.라벨링데이터")
        self.json_map = {}
        self._pre_index_json_files()

    def _pre_index_json_files(self):
        """
        Pre-scan all JSON files in 02.라벨링데이터 to build a prefix-to-path map.
        """
        if not os.path.exists(self.labeled_data_path):
            return
        print("라벨링 데이터(JSON) 메타데이터 색인 중... (v3 파서)")
        search_pattern = os.path.join(self.labeled_data_path, "**", "*.json")
        json_files = glob.glob(search_pattern, recursive=True)
        for path in json_files:
            filename = os.path.basename(path)
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

    def _clean_text(self, text: str) -> str:
        """
        Standard text normalization rules.
        """
        if not text:
            return ""
        
        # Merge space-separated Hangul syllables (e.g. '재 판 관' -> '재판관')
        def merge_match(m):
            return re.sub(r'\s+', '', m.group(0))
        text = re.sub(r'\b[가-힣]\b(?:\s{1,3}\b[가-힣]\b)+', merge_match, text)
        
        # Compress whitespaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove divider lines
        text = re.sub(r'[_\-=]{3,}', '', text)
        
        # Limit multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def parse_csv_file(self, file_path: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Parse a single CSV file, group rows statefully into logical parent blocks,
        normalize text, and build breadcrumbs.
        Returns a list of logical parent documents.
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='cp949')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return []
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        if df.empty or '내용' not in df.columns:
            return []

        # Sort by sentence index
        if '문장번호' in df.columns:
            df = df.sort_values(by='문장번호')

        # Determine Document ID and File Prefix
        filename = os.path.basename(file_path)
        file_prefix = os.path.splitext(filename)[0]
        
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
        base_meta = {
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
        base_meta.update(json_meta)

        # Primary name of the document
        doc_name = base_meta["caseName"] or base_meta["caseNum"] or f"ID_{doc_id}"

        # Stateful Grouping
        parent_blocks = []
        current_block_name = ""
        current_rows = []
        
        # Help identify structured bracket headers in judgments (e.g. 【이 유】)
        bracket_pattern = re.compile(r'^【\s*([가-힣\s]+)\s*】')

        is_law = "법령" in friendly_type

        for _, row in df.iterrows():
            content = str(row['내용']).strip()
            gubu = str(row['구분']).strip() if '구분' in df.columns else ""
            
            if not content:
                continue

            cleaned_content = self._clean_text(content)
            if not cleaned_content:
                continue

            # Check if this row starts a new logical block
            is_new_block = False
            new_block_name = ""

            if is_law:
                # For laws, a new article starts if:
                # 1. 구분 is '조문'
                # 2. Or the text starts with '제N조'
                if gubu == "조문" or re.match(r'^제\d+조', cleaned_content):
                    is_new_block = True
                    # Extract article header, e.g. "제1조(목적)"
                    match = re.match(r'^(제\d+조(?:\([^)]+\))?)', cleaned_content)
                    new_block_name = match.group(1) if match else cleaned_content[:20]
                elif gubu in ["부칙", "전문"] and current_block_name != gubu:
                    is_new_block = True
                    new_block_name = gubu
            else:
                # For judgments, decisions, interpretations:
                # 1. Bracket headers like 【주 문】 or 【이 유】
                bracket_match = bracket_pattern.match(cleaned_content)
                if bracket_match:
                    is_new_block = True
                    # Normalize text inside bracket, e.g. "이   유" -> "이유"
                    new_block_name = re.sub(r'\s+', '', bracket_match.group(1))
                # 2. Or if '구분' column changes
                elif gubu and gubu != "판례내용" and gubu != current_block_name:
                    is_new_block = True
                    new_block_name = gubu

            # If it's a new block, flush the previous one
            if is_new_block:
                if current_rows:
                    block_text = "\n".join(current_rows)
                    block_name = current_block_name if current_block_name else "개요"
                    
                    # Create breadcrumb
                    breadcrumb = f"{friendly_type} > {doc_name} > {block_name}"
                    
                    meta = base_meta.copy()
                    meta["section_name"] = block_name
                    meta["breadcrumb"] = breadcrumb
                    meta["title"] = f"{friendly_type} ({base_meta.get('caseNum') or doc_id}) - {block_name}"
                    meta["parent_id"] = f"{file_prefix}_{block_name}"
                    
                    parent_blocks.append({
                        "text": block_text,
                        "metadata": meta
                    })
                    current_rows = []
                current_block_name = new_block_name

            current_rows.append(cleaned_content)

        # Flush the final block
        if current_rows:
            block_text = "\n".join(current_rows)
            block_name = current_block_name if current_block_name else "본문"
            
            breadcrumb = f"{friendly_type} > {doc_name} > {block_name}"
            
            meta = base_meta.copy()
            meta["section_name"] = block_name
            meta["breadcrumb"] = breadcrumb
            meta["title"] = f"{friendly_type} ({base_meta.get('caseNum') or doc_id}) - {block_name}"
            meta["parent_id"] = f"{file_prefix}_{block_name}"
            
            parent_blocks.append({
                "text": block_text,
                "metadata": meta
            })

        return parent_blocks

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
            print(f"Parsing {len(csv_files)} files in {subdir}... (v3 파서)")
            
            for file_path in csv_files:
                doc_blocks = self.parse_csv_file(file_path, subdir)
                if doc_blocks:
                    documents.extend(doc_blocks)
                    
        print(f"Successfully parsed total of {len(documents)} parent blocks. (v3 파서)")
        return documents
