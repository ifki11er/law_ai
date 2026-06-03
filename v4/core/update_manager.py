import os
from typing import Dict, Any, List
from core.splitter import LegalSplitterV2
from core.vector_db import LegalVectorDB

class LegalUpdateManager:
    """
    [v3] Manager to handle database updates for Law Amendment, Repeal, and Insertion.
    Preserves historical versions of law articles while keeping the active state updated.
    """
    
    def __init__(self, db: LegalVectorDB):
        self.db = db
        self.splitter = LegalSplitterV2()

    def repeal_article(self, parent_id: str) -> bool:
        """
        [폐지] Deactivates an existing article block and all its child chunks.
        """
        print(f"\n[폐지 작업 시작] parent_id: {parent_id}")
        success = self.db.deactivate_parent_block(parent_id)
        if success:
            print(f"[폐지 완료] parent_id '{parent_id}'가 성공적으로 비활성화(soft delete) 되었습니다.")
        else:
            print(f"[폐지 실패] parent_id '{parent_id}'를 찾지 못했거나 오류가 발생했습니다.")
        return success

    def insert_article(self, 
                       doc_metadata: Dict[str, Any], 
                       section_name: str, 
                       text: str, 
                       version: str = "v1") -> bool:
        """
        [신설] Inserts a brand new article into the database.
        """
        print(f"\n[신설 작업 시작] {section_name} (버전: {version})")
        
        # Build logical parent block metadata
        friendly_type = doc_metadata.get("doc_type", "대한민국 법령")
        doc_name = doc_metadata.get("caseName") or doc_metadata.get("caseNum") or f"ID_{doc_metadata.get('doc_id')}"
        file_prefix = doc_metadata.get("file_prefix", "custom_law")
        
        breadcrumb = f"{friendly_type} > {doc_name} > {section_name}"
        parent_id = f"{file_prefix}_{section_name}_{version}"
        
        meta = doc_metadata.copy()
        
        # [v4] Ensure category is set
        if "category" not in meta:
            if "계약" in friendly_type:
                meta["category"] = "contract_form"
            elif "기계독해" in friendly_type or "mrc" in friendly_type.lower():
                meta["category"] = "mrc"
            else:
                meta["category"] = "law_ruling"
                
        meta["section_name"] = section_name
        meta["breadcrumb"] = breadcrumb
        meta["title"] = f"{friendly_type} ({doc_metadata.get('caseNum') or doc_metadata.get('doc_id')}) - {section_name}"
        meta["parent_id"] = parent_id
        meta["is_active"] = True
        
        parent_block = {
            "text": text.strip(),
            "metadata": meta
        }
        
        # Split into child chunks
        chunks = self.splitter.split_documents([parent_block])
        if not chunks:
            print("[신설 실패] 청크 분할 과정에서 빈 결과가 반환되었습니다.")
            return False
            
        # Add to Vector DB and rebuild BM25 index
        try:
            # Add to Chroma and SQLite
            self.db.add_documents(chunks)
            
            # Rebuild and save BM25 with all chunks from SQLite
            print("[신설 작업] 최적화된 BM25 역색인 전체 재구축 중...")
            import sys
            import os
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_dir not in sys.path:
                sys.path.append(project_dir)
                
            from rebuild_bm25 import rebuild_full_bm25
            rebuild_full_bm25()
            
            # Reload BM25 index in db instance
            self.db._load_bm25_index()
            
            print(f"[신설 완료] {section_name} ({parent_id})의 청크 {len(chunks)}개가 신규 적재되었습니다.")
            return True
        except Exception as e:
            print(f"[신설 실패] 데이터베이스 반영 실패: {e}")
            return False

    def amend_article(self, 
                      old_parent_id: str, 
                      doc_metadata: Dict[str, Any], 
                      section_name: str, 
                      new_text: str, 
                      new_version: str) -> bool:
        """
        [개정] Deactivates the old version of the article and inserts the new version.
        """
        print(f"\n[개정 작업 시작] 기존 ID: {old_parent_id} -> 신규 버전: {new_version}")
        
        # 1. Deactivate old version
        repeal_success = self.repeal_article(old_parent_id)
        if not repeal_success:
            print(f"⚠️ 경고: 기존 버전 {old_parent_id}을 비활성화하는 데 실패했지만, 개정본 삽입을 진행합니다.")
            
        # 2. Insert new version
        insert_success = self.insert_article(
            doc_metadata=doc_metadata,
            section_name=section_name,
            text=new_text,
            version=new_version
        )
        
        if insert_success:
            print(f"[개정 완료] {section_name}이 성공적으로 개정 반영되었습니다.")
            return True
        else:
            print(f"[개정 실패] {section_name}의 개정본 삽입에 실패했습니다.")
            return False
