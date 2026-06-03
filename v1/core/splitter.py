import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class LegalSplitter:
    """
    Regex lookahead-based dynamic splitter for legal texts.
    It splits documents strictly before legal headers without losing any characters,
    profiles logical unit lengths, and falls back to character splitting only when limits are exceeded.
    """
    
    def __init__(self):
        # Default fallback splitter if needed
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )
        
    def _calculate_80th_percentile(self, unit_lengths: List[int]) -> int:
        if not unit_lengths:
            return 1500
        sorted_lengths = sorted(unit_lengths)
        idx = int(len(sorted_lengths) * 0.8)
        return sorted_lengths[min(idx, len(sorted_lengths) - 1)]

    def _get_split_pattern_and_separators(self, doc_type: str):
        """
        문서 유형별 정규식 분할 패턴과 하위 separators 목록 반환
        """
        if "법령" in doc_type:
            # \n제1조, \n제2조 등의 직전 위치 탐색 (구분자 보존)
            return r'\n(?=제\d+조)', ["\n\n", "\n", " ", ""]
        elif "판결" in doc_type:
            # \n【판시사항】, \n【판결요지】 등의 직전 위치 탐색
            return r'\n(?=【)', ["\n\n", "\n", " ", ""]
        elif "해석" in doc_type:
            # \n질의요지, \n회답, \n이유 등의 직전 위치 탐색
            return r'\n(?=질의요지|회답|이유)', ["\n\n", "\n", " ", ""]
        else:
            # 구조가 단순한 경우 줄바꿈 두 번 기준
            return r'\n\n', ["\n", " ", ""]

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups documents by type, determines optimal chunk size dynamically,
        and splits them.
        """
        # 1. Group documents by doc_type
        docs_by_type = {}
        for doc in documents:
            doc_type = doc["metadata"].get("doc_type", "일반 문서")
            docs_by_type.setdefault(doc_type, []).append(doc)
            
        chunks = []
        
        print("\n" + "="*50)
        print("=== [정규식 전방탐색 동적 청킹 엔진 가동] ===")
        print("="*50)
        
        for doc_type, type_docs in docs_by_type.items():
            pattern, sub_separators = self._get_split_pattern_and_separators(doc_type)
            
            # 2. 정규식 패턴을 사용하여 손상 없이 의미 블록 분리 후 통계 측정
            unit_lengths = []
            for doc in type_docs:
                # re.split을 통해 구분자는 보존하며 분할
                blocks = re.split(pattern, doc["text"])
                unit_lengths.extend([len(b) for b in blocks if b.strip()])
                
            # 3. 통계 기반 동적 사이즈 산출
            raw_pct_80 = self._calculate_80th_percentile(unit_lengths)
            dynamic_chunk_size = max(500, min(raw_pct_80, 5000))
            dynamic_overlap = int(dynamic_chunk_size * 0.2)
            
            print(f"● 파트: [{doc_type}]")
            print(f"  - 수집된 단락 표본 수 : {len(unit_lengths):,}개")
            print(f"  - 계산된 80% 단락 크기: {raw_pct_80:,}자")
            print(f"  - 최종 설정된 청크 크기: ★ {dynamic_chunk_size:,}자 (오버랩: {dynamic_overlap:,}자) ★")
            
            # 4. 의미 블록이 설정한 크기 한도를 초과할 경우를 위해 백업 스플리터 생성
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=dynamic_chunk_size,
                chunk_overlap=dynamic_overlap,
                separators=sub_separators,
                keep_separator=True
            )
            
            # 5. 실제 분할 로직 실행
            type_chunks_count = 0
            for doc in type_docs:
                text = doc["text"]
                metadata = doc["metadata"]
                
                # 1차적으로 정규식을 통해 이상적인 조문/단락 단위로 완전 보존 분할
                first_splits = re.split(pattern, text)
                
                for split in first_splits:
                    stripped = split.strip()
                    if not stripped:
                        continue
                    
                    # 의미 단락이 너무 길어 제한 한도를 초과하면 하위 스플리터로 한 번 더 분할
                    if len(stripped) > dynamic_chunk_size:
                        sub_splits = fallback_splitter.split_text(stripped)
                    else:
                        sub_splits = [stripped]
                        
                    for s_split in sub_splits:
                        s_stripped = s_split.strip()
                        if not s_stripped:
                            continue
                            
                        chunk_meta = metadata.copy()
                        chunk_meta["chunk_index"] = type_chunks_count
                        chunk_meta["split_chunk_size"] = dynamic_chunk_size
                        chunk_meta["split_overlap"] = dynamic_overlap
                        
                        chunks.append({
                            "text": s_stripped,
                            "metadata": chunk_meta
                        })
                        type_chunks_count += 1
                        
            print(f"  └─ 생성된 청크 수: {type_chunks_count:,}개")
            print("-" * 50)
            
        print(f"=== [동적 청킹 완료] 총 {len(chunks):,}개 청크 생성 완료 ===\n")
        return chunks
