# [v3] 법률 RAG 데이터베이스 유지보수 및 개정/폐지 관리 가이드

이 가이드는 법률이 개정, 폐지, 또는 신설되었을 때 RAG 시스템의 Vector DB(Chroma)와 BM25 인덱스 파일(`bm25_index.pkl`)을 안전하게 최신화하는 방법과 구체적인 사용법을 다룹니다.

---

## 1. 개정/폐지 대상 조문 검색 및 식별 방법

데이터를 수정하려면 가장 먼저 **수정하려는 조문의 정확한 `parent_id`와 기존 메타데이터**를 찾아야 합니다.

### ① 검색을 통한 `parent_id` 확인 방법
Python 코드를 사용해 찾고자 하는 법령/조문을 검색하여 정확한 키워드와 `parent_id`를 확보합니다.

```python
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB

# 1. DB 로드
embedding = LegalEmbedding()
db = LegalVectorDB(embedding_function=embedding.get_embeddings())

# 2. 검색을 통해 정보 조회 (예: '민법 제3조'를 수정하고 싶을 때)
results = db.search_hybrid("민법 제3조", k=5)

for doc in results:
    meta = doc.metadata
    print(f"■ parent_id   : {meta.get('parent_id')}")
    print(f"  - 문서 종류  : {meta.get('doc_type')}")
    print(f"  - 문서 제목  : {meta.get('caseName')}")
    print(f"  - 조문 번호  : {meta.get('section_name')}")
    print(f"  - 활성화 여부 : {meta.get('is_active')}")
    print(f"  - 본문 일부  : {doc.page_content[:100]}...\n")
```

> [!TIP]
> 출력 결과에서 `parent_id`를 복사하여 아래의 개정/폐지 작업에 입력값으로 사용합니다. (예: `TS_법령_민법_제3조_v1`)

---

## 2. 시나리오별 유지보수 작업 예시 코드

### Scenario A: 조항 폐지 (Repeal)
특정 조항이 삭제/폐지되어 더 이상 검색에 노출되지 않아야 하지만, 과거 이력을 위해 물리적 삭제는 하지 않는 경우입니다.

```python
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from core.update_manager import LegalUpdateManager

# 1. DB 및 매니저 초기화
db = LegalVectorDB(embedding_function=LegalEmbedding().get_embeddings())
manager = LegalUpdateManager(db)

# 2. 폐지 처리 실행 (기존 식별한 parent_id 입력)
target_parent_id = "TS_법령_민법_제3조_v1"
manager.repeal_article(parent_id=target_parent_id)

# 결과: 해당 조문의 모든 자식 청크가 `is_active = False`로 변경되어 하이브리드 검색에서 제외됨.
```

---

### Scenario B: 조항 개정 (Amendment)
특정 조항의 내용이 변경되어, 기존 조항을 비활성화하고 새로운 내용으로 갱신하는 경우입니다.

```python
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from core.update_manager import LegalUpdateManager

# 1. DB 및 매니저 초기화
db = LegalVectorDB(embedding_function=LegalEmbedding().get_embeddings())
manager = LegalUpdateManager(db)

# 2. 기존 식별한 parent_id 및 신규 문서 메타데이터 정보 정의
old_id = "TS_법령_민법_제3조_v1"

# 신규 조문에 할당할 메타데이터 (기존 검색된 메타데이터를 기반으로 생성)
doc_metadata = {
    "doc_id": "100234",          # 문서 ID
    "file_prefix": "TS_법령_민법",  # 파일 접두사
    "doc_type": "대한민국 법령",
    "caseNum": "법률 제18429호",
    "caseName": "민법",
    "source_file": "TS_법령_민법.csv"
}

# 3. 개정된 새로운 텍스트 작성
new_article_text = (
    "제3조(권리능력의 존속기간) 사람은 생존하는 동안 권리와 의무의 주체가 된다. "
    "다만, 태아의 권리능력 취득 시점에 대해서는 특별법이 정하는 바에 의한다."
)

# 4. 개정 실행 (기존 v1 비활성화 -> 신규 v2 조문 생성)
manager.amend_article(
    old_parent_id=old_id,
    doc_metadata=doc_metadata,
    section_name="제3조",
    new_text=new_article_text,
    new_version="v2"
)
```

---

### Scenario C: 신설 조항 삽입 (New Insertion)
기존 법률에 없던 새로운 조항이 새롭게 추가되는 경우입니다.

```python
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from core.update_manager import LegalUpdateManager

# 1. DB 및 매니저 초기화
db = LegalVectorDB(embedding_function=LegalEmbedding().get_embeddings())
manager = LegalUpdateManager(db)

# 2. 신규 조항이 추가될 법률의 메타데이터 정의
doc_metadata = {
    "doc_id": "100234",
    "file_prefix": "TS_법령_민법",
    "doc_type": "대한민국 법령",
    "caseNum": "법률 제18429호",
    "caseName": "민법",
    "source_file": "TS_법령_민법.csv"
}

# 3. 신설할 조항 내용 작성
new_article_text = (
    "제3조의2(인격권) ① 사람은 자신의 성명, 초상, 음성, 서명, 그 밖의 인격적 징표를 "
    "함부로 침해받지 않을 권리를 가진다. ② 타인의 인격적 징표를 상업적으로 이용하는 행위는 "
    "본인의 동의를 얻어야 한다."
)

# 4. 신설 실행
manager.insert_article(
    doc_metadata=doc_metadata,
    section_name="제3조의2",
    text=new_article_text,
    version="v1"
)
```

---

## 3. 업데이트 이후 데이터 무결성 검증 절차

업데이트(개정/폐지/신설) 작업이 DB와 BM25 인덱스 파일에 완벽히 적용되었는지 검증하기 위해 아래의 절차를 수행하는 것을 권장합니다.

1. **상태 필터링 검증**: 수정 후 기존 `parent_id`를 대상으로 데이터베이스 `get` 조회를 하여 `is_active` 필드가 올바른 논리값(`True`/`False`)으로 바뀌었는지 확인합니다.
2. **하이브리드 검색 검증**: 검색 쿼리를 실행하여 비활성화된 이전 버전 조문은 검색 결과 목록에 나타나지 않고, 새로 활성화된 버전만 정상적으로 조회되는지 눈으로 확인합니다.
