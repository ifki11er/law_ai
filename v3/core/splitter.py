import re
from typing import List, Dict, Any
from config.settings import PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE

class LegalSplitterV2:
    """
    [v3] Parent-Child Splitter with Contextualization.
    Takes pre-grouped Parent Blocks, splits them into small Child Chunks (~300 chars),
    prefixes each Child Chunk with its hierarchical breadcrumb,
    and attaches the parent block's full text in the metadata.
    """
    
    def __init__(self, child_size: int = CHILD_CHUNK_SIZE):
        self.child_size = child_size

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split a block into sentences cleanly.
        """
        # Split by sentence ending punctuation followed by whitespace
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes Contextualized Parent-Child chunking.
        Each document in the input list represents a Parent Block.
        """
        chunks = []
        
        print("\n" + "="*60)
        print("=== [v3 컨텍스트 합성형 부모-자식 청킹 엔진 가동] ===")
        print("="*60)
        
        total_parents = len(documents)
        parent_lengths = []
        child_lengths = []
        
        for idx, doc in enumerate(documents):
            parent_text = doc["text"]
            metadata = doc["metadata"]
            breadcrumb = metadata.get("breadcrumb", "")
            
            parent_lengths.append(len(parent_text))
            
            # Split parent text into sentences
            sentences = self._split_sentences(parent_text)
            if not sentences:
                continue
                
            child_chunks_text = []
            current_chunk_sentences = []
            current_len = 0
            
            for sentence in sentences:
                sent_len = len(sentence)
                # If adding this sentence exceeds the child_size, finalize the current chunk
                if current_chunk_sentences and (current_len + sent_len > self.child_size):
                    child_chunks_text.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [sentence]
                    current_len = sent_len
                else:
                    current_chunk_sentences.append(sentence)
                    current_len += sent_len + 1 # +1 for space
                    
            if current_chunk_sentences:
                child_chunks_text.append(" ".join(current_chunk_sentences))
                
            # Create the final child chunks
            for child_idx, child_text in enumerate(child_chunks_text):
                # Contextualize: prepend breadcrumb to the child chunk text
                contextualized_text = f"[{breadcrumb}] {child_text}"
                child_lengths.append(len(contextualized_text))
                
                chunk_meta = metadata.copy()
                chunk_meta["parent_text"] = parent_text  # Store the parent text in child's metadata
                chunk_meta["parent_len"] = len(parent_text)
                chunk_meta["child_index"] = child_idx
                chunk_meta["child_len"] = len(contextualized_text)
                
                chunks.append({
                    "text": contextualized_text,
                    "metadata": chunk_meta
                })
                
        # Calculate statistics
        avg_parent_size = int(sum(parent_lengths) / len(parent_lengths)) if parent_lengths else 0
        avg_child_size = int(sum(child_lengths) / len(child_lengths)) if child_lengths else 0
        
        print(f"● 완료: {total_parents:,}개 부모 블록 파싱 완료 (평균 {avg_parent_size:,}자)")
        print(f"  └─ 생성된 자식 청크 수: {len(chunks):,}개")
        print(f"  └─ 자식 청크 평균 크기(접두사 포함): {avg_child_size:,}자")
        print("="*60 + "\n")
        
        return chunks
