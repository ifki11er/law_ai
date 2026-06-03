import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from core.update_manager import LegalUpdateManager
from agents.guardrail_agent import GuardrailAgent
from agents.llm_agent import LLMAgent
from config.settings import CHROMA_DB_PATH

app = FastAPI(title="Local Legal RAG System v4")

from typing import List, Dict, Optional

# Pydantic request models
class ChatSource(BaseModel):
    index: int
    title: str
    snippet: str

class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[List[ChatSource]] = None

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

class AdminSearchRequest(BaseModel):
    query: str
    include_inactive: bool = True
    category: str = None

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

# Initialize core components with v4 database path
print(f"초기화 (v4): 임베딩 및 하이브리드 벡터 DB({CHROMA_DB_PATH}) 로드 중...")

embedding = LegalEmbedding()
vector_db = LegalVectorDB(embedding_function=embedding.get_embeddings(), db_path=CHROMA_DB_PATH)
update_manager = LegalUpdateManager(vector_db)
guardrail_agent = GuardrailAgent()
llm_agent = LLMAgent()
print("초기화 완료 (v4): 하이브리드 서버 준비 완료.")

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
        results = vector_db.search_admin(
            payload.query, 
            include_inactive=payload.include_inactive, 
            category=payload.category
        )
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
    Main RAG workflow API endpoint with streaming log events and consolidated response.
    """
    query = payload.query.strip()
    history = [msg.dict() for msg in payload.history]
    
    print("\n" + "="*50)
    print(f"[RAG v4 스트리밍 파이프라인 가동] 사용자 질문 수신: '{query}'")
    print("="*50)
    
    async def event_generator():
        # 1. 1차 통합 질문 분석 (의도 분류 및 검색어 확장) 기동
        yield json.dumps({"type": "log", "message": "🛡️ [1단계] 통합 질문 분석 가동: 의도 분류 및 검색어 확장 분석 중..."}) + "\n"
        await asyncio.sleep(0.05)
        
        analysis = guardrail_agent.analyze_query(query, history)
        category = analysis.get("category", "law_ruling")
        expanded_query = analysis.get("expanded_query", query)
        
        yield json.dumps({"type": "log", "message": f"✅ [분석 완료] 카테고리: '{category}' 도메인 검색 및 쿼리 확장이 완료되었습니다."}) + "\n"
        await asyncio.sleep(0.05)
        
        # BM25 + Chroma DB v4 하이브리드 조회
        yield json.dumps({"type": "log", "message": "📊 [2단계] BM25(Sparse) + Chroma(Dense) 하이브리드 검색 기동..."}) + "\n"
        raw_results = vector_db.search_hybrid(expanded_query, k=10, category=category)
        yield json.dumps({"type": "log", "message": f"🔎 [검색 완료] 관련 후보 조항 및 판례 {len(raw_results)}건을 추출했습니다."}) + "\n"
        await asyncio.sleep(0.05)
        
        # 3차 방어막: 관련성 필터링
        yield json.dumps({"type": "log", "message": "🛡️ [3단계] 3차 방어막 가동: 유사도 검증 및 노이즈 필터링 중..."}) + "\n"
        filtered_docs = guardrail_agent.filter_references(raw_results)
        
        if filtered_docs:
            yield json.dumps({"type": "log", "message": f"📝 [필터링 통과] 최종 {len(filtered_docs)}개의 법률 근거를 채택했습니다."}) + "\n"
        else:
            yield json.dumps({"type": "log", "message": "⚠️ [주의] 검색된 내용 중 질문과 연관성이 높은 법률 근거가 존재하지 않습니다."}) + "\n"
        await asyncio.sleep(0.05)
            
        # 2차 방어막: 답변 생성
        yield json.dumps({"type": "log", "message": "✍️ [4단계] 2차 방어막 가동: 인용 출처 매핑 및 AI 법률 답변 생성 중..."}) + "\n"
        result = llm_agent.answer_question(query, filtered_docs, history, is_search_required=True)
        
        was_blocked = result["answer"] == llm_agent.standard_refusal
        
        yield json.dumps({
            "type": "final",
            "answer": result["answer"],
            "is_legal_intent": True,
            "blocked_by_guardrail": was_blocked,
            "sources": result["sources"]
        }) + "\n"

    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print(" 로컬 리걸 RAG v4 웹 서버 기동 중...")
    print(" 주소: http://127.0.0.1:8002")
    print("="*50 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=True)

