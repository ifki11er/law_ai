import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from core.update_manager import LegalUpdateManager
from agents.guardrail_agent import GuardrailAgent
from agents.llm_agent import LLMAgent
from config.settings import CHROMA_DB_PATH

app = FastAPI(title="Local Legal RAG System v3")

# Pydantic request models
class ChatRequest(BaseModel):
    query: str

class AdminSearchRequest(BaseModel):
    query: str
    include_inactive: bool = True

class RepealRequest(BaseModel):
    parent_id: str

class InsertRequest(BaseModel):
    doc_metadata: dict
    section_name: str
    text: str
    version: str = "v1"

class AmendRequest(BaseModel):
    old_parent_id: str
    doc_metadata: dict
    section_name: str
    new_text: str
    new_version: str

# Read the HTML templates
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
ADMIN_TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "admin.html")

# Initialize core components with v3 database path
print(f"초기화 (v3): 임베딩 및 하이브리드 벡터 DB({CHROMA_DB_PATH}) 로드 중...")

embedding = LegalEmbedding()
vector_db = LegalVectorDB(embedding_function=embedding.get_embeddings(), db_path=CHROMA_DB_PATH)
update_manager = LegalUpdateManager(vector_db)
guardrail_agent = GuardrailAgent()
llm_agent = LLMAgent()
print("초기화 완료 (v3): 하이브리드 서버 준비 완료.")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    Serve the chat UI index.html
    """
    if not os.path.exists(TEMPLATE_PATH):
        return HTMLResponse("HTML Template not found.", status_code=404)
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/admin", response_class=HTMLResponse)
async def get_admin():
    """
    Serve the admin maintenance UI admin.html
    """
    if not os.path.exists(ADMIN_TEMPLATE_PATH):
        return HTMLResponse("Admin HTML Template not found.", status_code=404)
    with open(ADMIN_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/api/admin/search")
async def admin_search_endpoint(payload: AdminSearchRequest):
    try:
        results = vector_db.search_admin(payload.query, include_inactive=payload.include_inactive)
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/admin/repeal")
async def admin_repeal_endpoint(payload: RepealRequest):
    success = update_manager.repeal_article(payload.parent_id)
    return {"success": success}

@app.post("/api/admin/insert")
async def admin_insert_endpoint(payload: InsertRequest):
    success = update_manager.insert_article(
        doc_metadata=payload.doc_metadata,
        section_name=payload.section_name,
        text=payload.text,
        version=payload.version
    )
    return {"success": success}

@app.post("/api/admin/amend")
async def admin_amend_endpoint(payload: AmendRequest):
    success = update_manager.amend_article(
        old_parent_id=payload.old_parent_id,
        doc_metadata=payload.doc_metadata,
        section_name=payload.section_name,
        new_text=payload.new_text,
        new_version=payload.new_version
    )
    return {"success": success}

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Main RAG workflow API endpoint with Triple Guardrails (v3 DB enabled).
    """
    query = payload.query.strip()
    print("\n" + "="*50)
    print(f"[RAG v3 파이프라인 가동] 사용자 질문 수신: '{query}'")
    print("="*50)
    
    # [1차 방어막: LLM 기반 질문 의도 분류]
    print("[1단계: 1차 방어막] Ollama/gemma4를 호출하여 질문의 '법률 의도' 판단 중...")
    is_legal_intent = guardrail_agent.check_legal_intent(query)
    
    if not is_legal_intent:
        print("[1단계 결과] ❌ 일반 질문으로 판정되어 RAG 파이프라인 즉시 차단!")
        blocked_msg = "본 서비스는 제공된 법전 및 판례 데이터에 기반한 법률 질의응답 서비스입니다. 입력하신 질문은 법률과 무관하여 답변이 불가능합니다."
        return {
            "answer": blocked_msg,
            "is_legal_intent": False,
            "blocked_by_guardrail": True,
            "sources": []
        }
    print("[1단계 결과]  ✅ 법률 질문으로 판정되어 2단계(검색)로 진입합니다.")
        
    # [검색 최적화: 질문 재작성 및 확장]
    expanded_query = llm_agent.rewrite_query(query)
        
    # [BM25 + Chroma DB v3 하이브리드 조회]
    print("\n[2단계: 지식 검색] BM25와 Chroma를 이용한 v3 하이브리드 검색 시작...")
    raw_results = vector_db.search_hybrid(expanded_query, k=10)
    print(f"[2단계 결과] 하이브리드 검색 완료 (후보군 {len(raw_results)}개 확보). 3단계(필터링)로 진입합니다.")
    
    # [3차 방어막: 관련성 필터링]
    print("\n[3단계: 3차 방어막] 검색 결과 검증 및 필터링 시작...")
    filtered_docs = guardrail_agent.filter_references(raw_results)
    
    # 디버깅 출력
    if filtered_docs:
        print("\n[3단계 결과] 최종 가드레일을 통과해 LLM에 전달되는 법률 문서 청크:")
        for idx, doc in enumerate(filtered_docs):
            snippet = doc.page_content.replace("\n", " ")[:120] + "..." if len(doc.page_content) > 120 else doc.page_content.replace("\n", " ")
            print(f"  └─ [출처 {idx+1}] {doc.metadata.get('title')}")
            print(f"     내용: {snippet}")
    else:
        print("\n[3단계 결과] ⚠️ 가드레일을 통과한 법률 문서가 없습니다. (참고 근거 없음)")

    # [2차 방어막: 시스템 프롬프트 환각 방지 및 대답 생성]
    print("\n[4단계: 2차 방어막 & 답변 생성] 필터링된 근거들을 조립하여 Ollama/gemma4 모델 호출 중...")
    result = llm_agent.answer_question(query, filtered_docs)
    
    # 답변이 거질 문구인지 검증하여 차단 여부 설정
    was_blocked = result["answer"] == llm_agent.standard_refusal
    
    if was_blocked:
        print("\n[4단계 결과] ⚠️ 2차/3차 방어막에 의해 답변 생성 거부")
    else:
        print("\n[4단계 결과] 🎉 출처 및 부모 조항 복원이 적용된 최종 법률 답변 생성 완료 (v3 하이브리드)!")
        
    print("\n" + "="*50)
    print("[RAG v3 파이프라인 실행 종료]")
    print("="*50 + "\n")
    
    return {
        "answer": result["answer"],
        "is_legal_intent": True,
        "blocked_by_guardrail": was_blocked,
        "sources": result["sources"]
    }
