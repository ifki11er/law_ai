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
        Split a block into sentences cleanly. If a sentence is longer than child_size,
        further split it into smaller pieces of at most child_size to avoid exceeding 
        Ollama/embedding model context length limits.
        """
        # Split by sentence ending punctuation followed by whitespace
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) <= self.child_size:
                sentences.append(s)
            else:
                # Split the long sentence into pieces of length self.child_size
                start = 0
                while start < len(s):
                    end = start + self.child_size
                    if end >= len(s):
                        sentences.append(s[start:])
                        break
                    # Try to split by space to avoid cutting words
                    last_space = s.rfind(' ', start, end)
                    if last_space != -1 and last_space > start + int(self.child_size * 0.5):
                        sentences.append(s[start:last_space].strip())
                        start = last_space + 1
                    else:
                        # Hard split
                        sentences.append(s[start:end].strip())
                        start = end
        return sentences

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
        
        from config.settings import PARENT_CHUNK_SIZE
        
        for idx, doc in enumerate(documents):
            parent_text = doc["text"]
            metadata = doc["metadata"]
            breadcrumb = metadata.get("breadcrumb", "")
            
            parent_lengths.append(len(parent_text))
            
            # Split parent text into sentences
            all_sentences = self._split_sentences(parent_text)
            if not all_sentences:
                continue
                
            # Group sentences into sub-parent blocks of size <= PARENT_CHUNK_SIZE
            sub_parents = []
            current_sub_parent_sentences = []
            current_sub_parent_len = 0
            
            for sentence in all_sentences:
                sent_len = len(sentence)
                if current_sub_parent_sentences and (current_sub_parent_len + sent_len > PARENT_CHUNK_SIZE):
                    sub_parents.append(" ".join(current_sub_parent_sentences))
                    current_sub_parent_sentences = [sentence]
                    current_sub_parent_len = sent_len
                else:
                    current_sub_parent_sentences.append(sentence)
                    current_sub_parent_len += sent_len + 1  # +1 for space
            if current_sub_parent_sentences:
                sub_parents.append(" ".join(current_sub_parent_sentences))
                
            # Process each sub-parent block
            for sub_idx, sub_parent_text in enumerate(sub_parents):
                # Generate unique parent_id for each sub-parent to map correctly
                orig_parent_id = metadata.get("parent_id", f"doc_{idx}")
                sub_parent_id = f"{orig_parent_id}_p{sub_idx}" if len(sub_parents) > 1 else orig_parent_id
                
                sub_sentences = self._split_sentences(sub_parent_text)
                
                child_chunks_text = []
                current_chunk_sentences = []
                current_len = 0
                
                for sentence in sub_sentences:
                    sent_len = len(sentence)
                    if current_chunk_sentences and (current_len + sent_len > self.child_size):
                        child_chunks_text.append(" ".join(current_chunk_sentences))
                        current_chunk_sentences = [sentence]
                        current_len = sent_len
                    else:
                        current_chunk_sentences.append(sentence)
                        current_len += sent_len + 1  # +1 for space
                        
                if current_chunk_sentences:
                    child_chunks_text.append(" ".join(current_chunk_sentences))
                    
                # Create the final child chunks under this sub-parent
                for child_idx, child_text in enumerate(child_chunks_text):
                    contextualized_text = f"[{breadcrumb}] {child_text}"
                    child_lengths.append(len(contextualized_text))
                    
                    chunk_meta = metadata.copy()
                    chunk_meta["parent_id"] = sub_parent_id
                    chunk_meta["parent_text"] = sub_parent_text  # Store the sub-parent text (<= PARENT_CHUNK_SIZE)
                    chunk_meta["parent_len"] = len(sub_parent_text)
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
