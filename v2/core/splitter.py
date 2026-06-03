import re
import math
import numpy as np
from typing import List, Dict, Any
from core.embedding import LegalEmbedding

class LegalSplitterV2:
    """
    [v2] Hybrid Semantic Splitter.
    1. First-order: Splits documents into 'Big Logic Blocks' strictly before legal headers (re.split lookahead).
    2. Second-order: Applies Sentence Cosine Similarity chunking only when a logic block exceeds 1,500 characters.
    """
    
    def __init__(self, semantic_threshold: float = 0.80, min_chunk_size: int = 500, max_chunk_size: int = 3000):
        self.threshold = semantic_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        # Load embedding model for sentence similarity calculations
        self.embedding = LegalEmbedding().get_embeddings()
        
    def _get_split_pattern(self, doc_type: str) -> str:
        """
        Get regex positive lookahead split pattern for big logic blocks.
        """
        if "법령" in doc_type:
            return r'\n(?=제\d+조)'
        elif "판결" in doc_type:
            return r'\n(?=【)'
        elif "해석" in doc_type:
            return r'\n(?=질의요지|회답|이유)'
        else:
            return r'\n\n'

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split a block into sentences cleanly.
        """
        # Split by sentence ending punctuation followed by whitespace
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def _semantic_chunk_block(self, block: str) -> List[str]:
        """
        Splits a single big logic block into smaller chunks based on sentence cosine similarity.
        """
        sentences = self._split_sentences(block)
        if len(sentences) <= 1:
            return [block]
            
        # 1. Embed all sentences in batch to speed up ingestion
        try:
            vectors = self.embedding.embed_documents(sentences)
        except Exception as e:
            print(f"⚠️ 문장 임베딩 중 오류 발생, 글자수 기준으로 임시 분할: {e}")
            # Fallback if embedding api fails
            return [block[i:i+1500] for i in range(0, len(block), 1500)]
            
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        # 2. Iterate through sentences and calculate similarity with the next sentence
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(vectors[i], vectors[i+1])
            current_text = " ".join(current_chunk_sentences)
            
            # Condition to split: 
            # 1. Similarity falls below the threshold (Semantic transition)
            # 2. The current accumulated chunk length is already >= min_chunk_size (500 chars)
            # OR 3. The current chunk exceeds safety limit (max_chunk_size)
            if (sim < self.threshold and len(current_text) >= self.min_chunk_size) or (len(current_text) >= self.max_chunk_size):
                chunks.append(current_text)
                current_chunk_sentences = [sentences[i+1]]
            else:
                current_chunk_sentences.append(sentences[i+1])
                
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes hybrid semantic chunking.
        """
        # Group by doc_type
        docs_by_type = {}
        for doc in documents:
            doc_type = doc["metadata"].get("doc_type", "일반 문서")
            docs_by_type.setdefault(doc_type, []).append(doc)
            
        chunks = []
        
        print("\n" + "="*60)
        print("=== [v2 하이브리드 세만틱 청킹 엔진 가동] ===")
        print("="*60)
        
        for doc_type, type_docs in docs_by_type.items():
            pattern = self._get_split_pattern(doc_type)
            type_chunks_count = 0
            type_chunk_lens = []
            
            print(f"● 파트: [{doc_type}] 분석 중...")
            
            for doc in type_docs:
                text = doc["text"]
                metadata = doc["metadata"]
                
                # 1단계: 정규식 전방 탐색을 통해 거대 의미 단위(제X조 등)로 1차 분할
                first_splits = re.split(pattern, text)
                
                for block in first_splits:
                    stripped_block = block.strip()
                    if not stripped_block:
                        continue
                        
                    # 2단계: 크기 검증 후 임베딩 유사도 기반 2차 분할 적용
                    # 1,500자 이하인 경우, 논리 구조 보존을 위해 분할 없이 통과
                    if len(stripped_block) <= 1500:
                        sub_splits = [stripped_block]
                    else:
                        # 1,500자를 초과하는 대형 단락만 유사도 분할 적용
                        sub_splits = self._semantic_chunk_block(stripped_block)
                        
                    for s_split in sub_splits:
                        s_stripped = s_split.strip()
                        if not s_stripped:
                            continue
                            
                        chunk_meta = metadata.copy()
                        chunk_meta["chunk_index"] = type_chunks_count
                        chunk_meta["is_semantic_split"] = len(stripped_block) > 1500
                        
                        chunks.append({
                            "text": s_stripped,
                            "metadata": chunk_meta
                        })
                        type_chunks_count += 1
                        type_chunk_lens.append(len(s_stripped))
                        
            if type_chunk_lens:
                type_min = min(type_chunk_lens)
                type_max = max(type_chunk_lens)
                type_avg = int(sum(type_chunk_lens) / len(type_chunk_lens))
                print(f"  └─ 완료: 생성된 청크 수 {type_chunks_count:,}개")
                print(f"  └─ 크기: 최소 {type_min:,}자 / 최대 {type_max:,}자 / 평균 {type_avg:,}자")
            else:
                print(f"  └─ 완료: 생성된 청크 수 {type_chunks_count:,}개")
            print("-" * 50)
            
        # Calculate overall stats
        if chunks:
            all_lens = [len(c["text"]) for c in chunks]
            overall_min = min(all_lens)
            overall_max = max(all_lens)
            overall_avg = int(sum(all_lens) / len(all_lens))
            
            # 분포 계산
            under_500 = sum(1 for l in all_lens if l < 500)
            range_500_1000 = sum(1 for l in all_lens if 500 <= l < 1000)
            range_1000_1500 = sum(1 for l in all_lens if 1000 <= l < 1500)
            range_1500_2000 = sum(1 for l in all_lens if 1500 <= l < 2000)
            above_2000 = sum(1 for l in all_lens if l >= 2000)
            
            print(f"=== [v2 세만틱 청킹 완료] 총 {len(chunks):,}개 청크 생성 완료 ===")
            print(f"  📊 전체 청크 크기 요약:")
            print(f"    - 최소 크기: {overall_min:,}자")
            print(f"    - 최대 크기: {overall_max:,}자")
            print(f"    - 평균 크기: {overall_avg:,}자")
            print(f"  📈 크기별 분포:")
            print(f"    - 500자 미만:       {under_500:,}개 ({under_500/len(chunks)*100:.1f}%)")
            print(f"    - 500자 ~ 1,000자:  {range_500_1000:,}개 ({range_500_1000/len(chunks)*100:.1f}%)")
            print(f"    - 1,000자 ~ 1,500자: {range_1000_1500:,}개 ({range_1000_1500/len(chunks)*100:.1f}%)")
            print(f"    - 1,500자 ~ 2,000자: {range_1500_2000:,}개 ({range_1500_2000/len(chunks)*100:.1f}%)")
            print(f"    - 2,000자 이상:     {above_2000:,}개 ({above_2000/len(chunks)*100:.1f}%)")
            print("="*60 + "\n")
        else:
            print(f"=== [v2 세만틱 청킹 완료] 생성된 청크가 없습니다. ===\n")
            
        return chunks
