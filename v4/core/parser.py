import os
import glob
import json
import re
import pandas as pd
from typing import List, Dict, Any
from config.settings import AI_HUB_DATA_PATH

class LegalParserV2:
    """
    [v4] Polymorphic Parser for multi-domain AI Hub legal data.
    Automatically routes files (.csv and .json) from different folders
    (Laws, Judgments, Contract templates, MRC corpora) to their specific parsers
    and standardizes them into structured Parent Blocks with breadcrumbs and category metadata.
    """
    
    def __init__(self, training_path: str = AI_HUB_DATA_PATH):
        self.raw_data_path = training_path
        self.json_map = {}
        self._pre_index_json_files()

    def _pre_index_json_files(self):
        """
        Recursively scan all subdirectories under AI_HUB_DATA_PATH containing '02.라벨링데이터'
        to index metadata files (useful for enrichment of original files).
        """
        print("[Parser v4] 라벨링 데이터(JSON) 메타데이터 색인 중...")
        count = 0
        for root, dirs, files in os.walk(self.raw_data_path):
            # Skip validation data to avoid duplicates/unnecessary files in RAG
            if "validation" in root.lower():
                continue
            if "02.라벨링데이터" in root:
                for file in files:
                    if file.endswith(".json"):
                        path = os.path.join(root, file)
                        filename = os.path.basename(path)
                        # Extract prefix
                        if "_QA_" in filename:
                            prefix = filename.split("_QA_")[0]
                        else:
                            prefix = os.path.splitext(filename)[0]
                        self.json_map[prefix] = path
                        count += 1
        print(f"[Parser v4] 총 {count}개의 JSON 메타데이터 매핑을 로드했습니다.")
        
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
        
        # Merge space-separated Hangul syllables
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
        [v3/v4 Backward Compatible] Parse a single CSV file, group rows statefully,
        and assign 'law_ruling' category.
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

        if '문장번호' in df.columns:
            df = df.sort_values(by='문장번호')

        filename = os.path.basename(file_path)
        file_prefix = os.path.splitext(filename)[0]
        
        id_col = df.columns[0]
        doc_id = str(df[id_col].iloc[0]) if not df.empty else ""

        type_mapping = {
            "TS_결정례": "헌법재판소 결정례",
            "TS_법령": "대한민국 법령",
            "TS_판결문": "법원 판결문",
            "TS_해석례": "법률 해석례"
        }
        friendly_type = type_mapping.get(doc_type, doc_type)

        base_meta = {
            "doc_id": doc_id,
            "file_prefix": file_prefix,
            "doc_type": friendly_type,
            "category": "law_ruling",  # CSV data from criminal law is routed to law_ruling
            "source_file": filename,
            "caseNum": "",
            "caseName": "",
            "finalDate": "",
            "courtCode": "",
            "is_active": True
        }

        json_meta = self.find_metadata_from_json(file_prefix)
        base_meta.update(json_meta)

        doc_name = base_meta["caseName"] or base_meta["caseNum"] or f"ID_{doc_id}"

        parent_blocks = []
        current_block_name = ""
        current_rows = []
        
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

            is_new_block = False
            new_block_name = ""

            if is_law:
                if gubu == "조문" or re.match(r'^제\d+조(?:의\d+)?', cleaned_content):
                    is_new_block = True
                    match = re.match(r'^(제\d+조(?:의\d+)?(?:\([^)]+\))?)', cleaned_content)
                    new_block_name = match.group(1) if match else cleaned_content[:20]
                elif gubu in ["부칙", "전문"] and current_block_name != gubu:
                    is_new_block = True
                    new_block_name = gubu
            else:
                bracket_match = bracket_pattern.match(cleaned_content)
                if bracket_match:
                    is_new_block = True
                    new_block_name = re.sub(r'\s+', '', bracket_match.group(1))
                elif gubu and gubu != "판례내용" and gubu != current_block_name:
                    is_new_block = True
                    new_block_name = gubu

            if is_new_block:
                if current_rows:
                    block_text = "\n".join(current_rows)
                    block_name = current_block_name if current_block_name else "개요"
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

    def parse_json_file(self, file_path: str, category: str) -> List[Dict[str, Any]]:
        """
        Loads a JSON file and routes it to the specific JSON parser.
        """
        filename = os.path.basename(file_path)
        file_prefix = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ JSON 읽기 실패 ({filename}): {e}")
            return []
            
        if category == "law_ruling":
            return self._parse_law_and_judgment_json(data, file_prefix, filename)
        elif category == "contract_form":
            return self._parse_contract_form_json(data, file_prefix, filename)
        elif category == "mrc":
            return self._parse_mrc_json(data, file_prefix, filename)
        else:
            return []

    def _parse_law_and_judgment_json(self, data: Dict[str, Any], file_prefix: str, filename: str) -> List[Dict[str, Any]]:
        """
        [v4 Parser Sub-module] Parses statutory (법령) or case law (판결문) JSON objects.
        """
        parent_blocks = []
        
        # 1. Check if it is a Statute (법령) JSON
        if "statute_name" in data:
            doc_id = data.get("statute_abbrv") or data.get("statute_abbrv") or file_prefix
            statute_name = data.get("statute_name", "")
            effective_date = data.get("effective_date", "")
            statute_category = data.get("statute_category", "법령")
            friendly_type = f"{statute_category} 법령"
            
            base_meta = {
                "doc_id": doc_id,
                "file_prefix": file_prefix,
                "doc_type": friendly_type,
                "category": "law_ruling",
                "source_file": filename,
                "caseNum": "",
                "caseName": statute_name,
                "finalDate": effective_date,
                "courtCode": data.get("statute_type", ""),
                "is_active": True
            }
            
            # Stateful Article (제N조) grouping
            sentences = data.get("sentences", [])
            current_block_name = "전문"
            current_rows = []
            
            for s in sentences:
                cleaned = self._clean_text(s)
                if not cleaned:
                    continue
                
                # Check for new article trigger
                if re.match(r'^제\d+조(?:의\d+)?', cleaned) or cleaned.startswith("부칙"):
                    if current_rows:
                        block_text = "\n".join(current_rows)
                        meta = base_meta.copy()
                        meta["section_name"] = current_block_name
                        meta["breadcrumb"] = f"{friendly_type} > {statute_name} > {current_block_name}"
                        meta["title"] = f"{friendly_type} ({statute_name}) - {current_block_name}"
                        meta["parent_id"] = f"{file_prefix}_{current_block_name}"
                        
                        parent_blocks.append({"text": block_text, "metadata": meta})
                        current_rows = []
                    
                    if cleaned.startswith("부칙"):
                        current_block_name = "부칙"
                    else:
                        match = re.match(r'^(제\d+조(?:의\d+)?(?:\([^)]+\))?)', cleaned)
                        current_block_name = match.group(1) if match else cleaned[:20]
                
                current_rows.append(cleaned)
                
            if current_rows:
                block_text = "\n".join(current_rows)
                meta = base_meta.copy()
                meta["section_name"] = current_block_name
                meta["breadcrumb"] = f"{friendly_type} > {statute_name} > {current_block_name}"
                meta["title"] = f"{friendly_type} ({statute_name}) - {current_block_name}"
                meta["parent_id"] = f"{file_prefix}_{current_block_name}"
                parent_blocks.append({"text": block_text, "metadata": meta})

        # 2. Check if it is a Ruling (판결문) JSON
        elif "normalized_court" in data or "casenames" in data:
            doc_id = data.get("doc_id") or file_prefix
            casename = data.get("casenames", "판결서")
            court = data.get("normalized_court", "")
            announce_date = data.get("announce_date") or data.get("announce_date") or ""
            friendly_type = f"{court or '법원'} 판결문"
            
            base_meta = {
                "doc_id": doc_id,
                "file_prefix": file_prefix,
                "doc_type": friendly_type,
                "category": "law_ruling",
                "source_file": filename,
                "caseNum": doc_id,
                "caseName": casename,
                "finalDate": announce_date,
                "courtCode": court,
                "is_active": True
            }
            
            sentences = data.get("sentences", [])
            current_block_name = "판결사항"
            current_rows = []
            bracket_pattern = re.compile(r'^【\s*([가-힣\s]+)\s*】')
            numbered_header = re.compile(r'^\d+\.\s+([가-힣A-Za-z0-9\s]+)')
            
            for s in sentences:
                cleaned = self._clean_text(s)
                if not cleaned:
                    continue
                
                is_new = False
                new_name = ""
                
                # Group by bracket or numbered headers (e.g. 1. 기초사실)
                bracket_match = bracket_pattern.match(cleaned)
                num_match = numbered_header.match(cleaned)
                
                if bracket_match:
                    is_new = True
                    new_name = re.sub(r'\s+', '', bracket_match.group(1))
                elif num_match and len(cleaned) < 50:  # Avoid matching normal short sentences
                    is_new = True
                    new_name = cleaned.split('\n')[0][:30].strip()
                    
                if is_new:
                    if current_rows:
                        block_text = "\n".join(current_rows)
                        meta = base_meta.copy()
                        meta["section_name"] = current_block_name
                        meta["breadcrumb"] = f"{friendly_type} > {casename} > {current_block_name}"
                        meta["title"] = f"{friendly_type} ({doc_id}) - {current_block_name}"
                        meta["parent_id"] = f"{file_prefix}_{current_block_name}"
                        
                        parent_blocks.append({"text": block_text, "metadata": meta})
                        current_rows = []
                    current_block_name = new_name
                    
                current_rows.append(cleaned)
                
            if current_rows:
                block_text = "\n".join(current_rows)
                meta = base_meta.copy()
                meta["section_name"] = current_block_name
                meta["breadcrumb"] = f"{friendly_type} > {casename} > {current_block_name}"
                meta["title"] = f"{friendly_type} ({doc_id}) - {current_block_name}"
                meta["parent_id"] = f"{file_prefix}_{current_block_name}"
                parent_blocks.append({"text": block_text, "metadata": meta})
                
        return parent_blocks

    def _parse_contract_form_json(self, data: Dict[str, Any], file_prefix: str, filename: str) -> List[Dict[str, Any]]:
        """
        [v4 Parser Sub-module] Parses contract templates (계약서 서식) JSON documents.
        Groups contract text segments statefully by article (제N조) markers.
        """
        parent_blocks = []
        doc_obj = data.get("document", {})
        if not doc_obj:
            return []
            
        doc_id = doc_obj.get("doc_type") or file_prefix
        title = doc_obj.get("title", "계약서")
        created_time = doc_obj.get("created_time", "")
        friendly_type = "계약서 서식"
        
        base_meta = {
            "doc_id": doc_id,
            "file_prefix": file_prefix,
            "doc_type": friendly_type,
            "category": "contract_form",
            "source_file": filename,
            "caseNum": "",
            "caseName": title,
            "finalDate": created_time,
            "courtCode": "",
            "is_active": True
        }
        
        sub_docs = doc_obj.get("sub_documents", [])
        current_article = None
        current_block_name = "전문"
        current_rows = []
        
        for sub_doc in sub_docs:
            contents = sub_doc.get("contents", [])
            text = " ".join([c.get("text", "") for c in contents if c.get("text")]).strip()
            if not text:
                continue
                
            art = sub_doc.get("article")
            is_new = False
            
            # Start a new block when the article index changes (not None)
            if art is not None and art != current_article:
                is_new = True
                new_art_name = f"제{art}조"
                if sub_doc.get("is_article_title") or "제" in text:
                    new_art_name = text.split('\n')[0][:40].strip()
                current_article = art
                
            if is_new:
                if current_rows:
                    block_text = "\n".join(current_rows)
                    meta = base_meta.copy()
                    meta["section_name"] = current_block_name
                    meta["breadcrumb"] = f"{friendly_type} > {title} > {current_block_name}"
                    meta["title"] = f"{friendly_type} ({title}) - {current_block_name}"
                    meta["parent_id"] = f"{file_prefix}_{current_block_name}"
                    
                    parent_blocks.append({"text": block_text, "metadata": meta})
                    current_rows = []
                current_block_name = new_art_name
                
            current_rows.append(text)
            
        if current_rows:
            block_text = "\n".join(current_rows)
            meta = base_meta.copy()
            meta["section_name"] = current_block_name
            meta["breadcrumb"] = f"{friendly_type} > {title} > {current_block_name}"
            meta["title"] = f"{friendly_type} ({title}) - {current_block_name}"
            meta["parent_id"] = f"{file_prefix}_{current_block_name}"
            parent_blocks.append({"text": block_text, "metadata": meta})
            
        return parent_blocks

    def _parse_mrc_json(self, data: Dict[str, Any], file_prefix: str, filename: str) -> List[Dict[str, Any]]:
        """
        [v4 Parser Sub-module] Parses machine reading comprehension (기계독해) JSON datasets.
        Each paragraph context is treated as a cohesive parent block.
        """
        parent_blocks = []
        doc_list = data.get("data", [])
        friendly_type = "기계독해 말뭉치"
        
        for doc_item in doc_list:
            doc_id = doc_item.get("doc_id", "")
            doc_title = doc_item.get("doc_title", "학술말뭉치")
            source = doc_item.get("doc_source", "")
            published = str(doc_item.get("doc_published", ""))
            
            paragraphs = doc_item.get("paragraphs", [])
            for p_idx, p in enumerate(paragraphs):
                context = p.get("context", "").strip()
                context_id = p.get("context_id", f"p{p_idx}")
                if not context:
                    continue
                    
                block_name = f"문단_{context_id}"
                breadcrumb = f"{friendly_type} > {doc_title} > {block_name}"
                
                meta = {
                    "doc_id": doc_id or file_prefix,
                    "file_prefix": file_prefix,
                    "doc_type": friendly_type,
                    "category": "mrc",
                    "source_file": filename,
                    "caseNum": doc_id,
                    "caseName": doc_title,
                    "finalDate": published,
                    "courtCode": source,
                    "section_name": block_name,
                    "breadcrumb": breadcrumb,
                    "title": f"{friendly_type} ({doc_title}) - {block_name}",
                    "parent_id": f"{file_prefix}_{context_id}",
                    "is_active": True
                }
                
                parent_blocks.append({
                    "text": context,
                    "metadata": meta
                })
                
        return parent_blocks

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        [v4 Router] Recursively walks AI_HUB_DATA_PATH (docs/학습데이터),
        detects new dataset folders, and automatically routes CSV/JSON files
        to their corresponding parser while keeping track of category metadata.
        """
        documents = []
        print(f"\n[Parser v4] 데이터셋 탐색 시작: {self.raw_data_path}")
        
        # Traverse all directories recursively
        for root, dirs, files in os.walk(self.raw_data_path):
            # Skip validation data to avoid duplicates/unnecessary files in RAG
            if "validation" in root.lower():
                continue
            # We want files located under raw data directories
            if "01.원천데이터" in root or "01-1.정식개방데이터" in root:
                # Determine category based on the relative path directory name
                rel_path = os.path.relpath(root, self.raw_data_path)
                parts = rel_path.split(os.sep)
                top_folder = parts[0] if parts else ""
                
                if any(k in top_folder for k in ["01.민사법", "02.지식재산권법", "03.행정법", "04.형사법"]):
                    category = "law_ruling"
                elif any(k in top_folder for k in ["05.계약", "06.계약 외"]):
                    category = "contract_form"
                elif any(k in top_folder for k in ["151.금융", "154.의료"]):
                    category = "mrc"
                else:
                    category = "law_ruling"  # fallback
                
                immediate_folder = os.path.basename(root)
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Parse based on file type and category
                    if file.endswith(".csv"):
                        doc_blocks = self.parse_csv_file(file_path, immediate_folder)
                        if doc_blocks:
                            documents.extend(doc_blocks)
                    elif file.endswith(".json"):
                        doc_blocks = self.parse_json_file(file_path, category)
                        if doc_blocks:
                            documents.extend(doc_blocks)
                            
        print(f"[Parser v4] 전체 탐색 완료. 파싱된 총 부모 블록 수: {len(documents):,}개")
        return documents
