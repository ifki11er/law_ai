import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from agents.guardrail_agent import GuardrailAgent
from agents.llm_agent import LLMAgent
from config.settings import CHROMA_DB_PATH

app = FastAPI(title="Local Legal RAG System v2")

# Pydantic request models
class ChatRequest(BaseModel):
    query: str

# Read the HTML template once on load
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")

# Initialize core components with v2 database path
CHROMA_DB_PATH = CHROMA_DB_PATH
print(f"초기화 (v2): 임베딩 및 벡터 데이터베이스({CHROMA_DB_PATH}) 로드 중...")

embedding = LegalEmbedding()
vector_db = LegalVectorDB(embedding_function=embedding.get_embeddings(), db_path=CHROMA_DB_PATH)
guardrail_agent = GuardrailAgent()
llm_agent = LLMAgent()
print("초기화 완료 (v2): 서버 준비 완료.")

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

@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Main RAG workflow API endpoint with Triple Guardrails (v2 DB enabled).
    """
    query = payload.query.strip()
    print("\n" + "="*50)
    print(f"[RAG v2 파이프라인 가동] 사용자 질문 수신: '{query}'")
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
        
    # [Chroma DB v2 유사도 조회]
    print("\n[2단계: 지식 검색] 임베딩된 질문 벡터로 Chroma DB v2 내 상위 10개 유사 법률 문서 검색 중...")
    raw_results = vector_db.search_with_scores(query, k=10)
    print(f"[2단계 결과] 검색 완료 (후보군 {len(raw_results)}개 확보). 3단계(필터링)로 진입합니다.")
    
    # [3차 방어막: 유사도 점수 컷오프 기반 관련성 없는 근거 제외]
    print("\n[3단계: 3차 방어막] Chroma 검색 결과 스코어 임계값(Threshold = 1.1) 검사 시작...")
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
    
    # 답변이 거절 문구인지 검증하여 차단 여부 설정
    was_blocked = result["answer"] == llm_agent.standard_refusal
    
    if was_blocked:
        print("\n[4단계 결과] ⚠️ 2차/3차 방어막에 의해 답변 생성 거부")
    else:
        print("\n[4단계 결과] 🎉 출처가 포함된 최종 법률 답변 생성 완료 (v2 DB 기반)!")
        
    print("\n" + "="*50)
    print("[RAG v2 파이프라인 실행 종료]")
    print("="*50 + "\n")
    
    return {
        "answer": result["answer"],
        "is_legal_intent": True,
        "blocked_by_guardrail": was_blocked,
        "sources": result["sources"]
    }
