from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from config.settings import SIMILARITY_THRESHOLD
from core.llm_factory import get_llm
from typing import List, Tuple, Any

class GuardrailAgent:
    """
    Agent responsible for 1st Guardrail (Intent filtering via LLM) [Bypassed]
    and 3rd Guardrail (Similarity score cutoff filtering).
    Supports Ollama/GPT switching.
    """
    
    def __init__(self, model_name: str = None, base_url: str = None, threshold: float = SIMILARITY_THRESHOLD):
        self.llm = get_llm(temperature=0.0)
        self.json_llm = get_llm(temperature=0.0, format="json")
        self.threshold = threshold

    def filter_references(self, search_results: List[Any]) -> List[Document]:
        """
        3차 방어막: 유사도 점수 컷오프를 사용하여 관련성 없는 근거를 배제(exclude)합니다.
        검색 결과가 Tuple인 경우 Chroma distance score를 임계값(1.1)과 비교하고,
        이미 필터링된 Document 객체 리스트인 경우 그대로 반환합니다.
        """
        filtered_docs = []
        for item in search_results:
            if isinstance(item, tuple):
                doc, score = item
                # Chroma L2/cosine distance score: 작을수록 유사함
                if score <= self.threshold:
                    filtered_docs.append(doc)
                    print(f"[3차 방어막] 통과 - 스코어: {score:.4f} | 출처: {doc.metadata.get('title')}")
                else:
                    print(f"[3차 방어막] 배제(컷오프) - 스코어: {score:.4f} | 출처: {doc.metadata.get('title')}")
            else:
                # Document 객체인 경우 (하이브리드 검색 등에서 이미 필터링 완료된 상태)
                filtered_docs.append(item)
                
        if search_results and not isinstance(search_results[0], tuple):
            print(f"[3차 방어막] 하이브리드 검색 필터링 완료 상태로 통과. 최종 개수: {len(filtered_docs)}")
        else:
            print(f"[3차 방어막] 최종 필터링된 근거 개수: {len(filtered_docs)} / {len(search_results)}")
            
        return filtered_docs

    def classify_query_category(self, query: str, history: list = None) -> str:
        """
        [v4] Classifies the user query into one of three categories:
        - 'law_ruling': Statutory rules, legal concepts, court cases, punishments, lawsuits.
        - 'contract_form': Contract templates, standard terms, lease agreements, drafting clauses.
        - 'mrc': Financial machine reading comprehension, finance/economics definitions, or generic legal text.
        Returns one of these values, or None if it's ambiguous.
        """
        system_prompt = (
            "당신의 임무는 사용자 질문의 유형에 맞는 검색 카테고리를 분류하는 것입니다.\n"
            "세 가지 카테고리 중 가장 어울리는 카테고리 하나를 영문명으로 출력하십시오:\n"
            "1. 'law_ruling': 법조문 해석, 판례(판결문), 형사/민사/행정 등 법령 관련 질문 및 사법 판단 질문.\n"
            "2. 'contract_form': 계약서 양식, 서식, 조항 작성, 임대차 계약서 작성 팁 등 계약서 템플릿과 조항에 관한 질문.\n"
            "3. 'mrc': 금융 법률 기계독해, 경제/금융 분야 지식, 사전적 의미 질문.\n"
            "오직 'law_ruling', 'contract_form', 'mrc' 중 하나의 단어만 대답하십시오. 단어로만 응답하고 다른 설명을 절대 덧붙이지 마십시오. 판단하기 애매하거나 일반적인 경우라면 'law_ruling'을 대답하십시오."
        )
        
        context = ""
        if history:
            context = "이전 대화 맥락:\n"
            for msg in history[-3:]:
                role = "사용자" if msg["role"] == "user" else "AI"
                context += f"{role}: {msg['content']}\n"
            context += "\n"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}질문: {query}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = response.content.strip().lower()
            print(f"[의도 분류] 카테고리 판정 결과: {result} (질문: {query})")
            
            if "contract_form" in result or "contract" in result or "계약" in result:
                return "contract_form"
            elif "mrc" in result or "금융" in result:
                return "mrc"
            elif "law_ruling" in result or "law" in result or "판례" in result or "법령" in result:
                return "law_ruling"
            else:
                return "law_ruling" # Default fallback
        except Exception as e:
            print(f"[의도 분류] 카테고리 판정 오류: {e}")
            return "law_ruling"

    def analyze_query(self, query: str, history: list = None) -> dict:
        """
        [v4+] 통합 질문 분석기: 검색 카테고리 분류, 질문 재작성을
        단 1회의 LLM 호출로 수행하여 JSON 포맷으로 반환합니다.
        (호환성 유지를 위해 is_legal은 항상 True로 반환합니다)
        """
        import json
        import re
        
        system_prompt = (
            "당신은 법률 RAG 시스템의 프론트엔드 질문 분석기입니다.\n"
            "사용자의 질문과 이전 대화 맥락을 엄격히 분석하여 다음 3가지 항목을 판단하고 반드시 지정된 JSON 형식으로만 출력하십시오.\n\n"
            "판단 항목:\n"
            "1. 'is_search_required' (boolean): 사용자의 질문을 해결하기 위해 새로운 외부 법률 문서/판례 데이터베이스 검색이 필요한지 여부.\n"
            "   - 중요: 사용자의 질문이 구체적인 법률 지식, 종류, 기준, 정의, 조항(예: '여권의 종류는?', '관용여권 발급대상은?', '외교관여권 발급대상은?') 등 법적 근거가 필요한 질문이라면, 이전 대화나 이전 답변 출처에 유사한 단어가 나왔더라도 무조건 true로 판정하여 데이터베이스 검색을 거치도록 해야 합니다. 그렇지 않으면 불완전하고 누락된 답변이 생성됩니다.\n"
            "   - 오직 사용자가 이전 답변의 텍스트 자체를 다루는 요약, 번역, 포맷 변환 요청(예: '방금 해준 답변을 영어로 번역해줘', '위 내용을 요약해줘', '표로 만들어줘') 또는 대화 내용 자체를 확인하는 질문(예: '내가 방금 몇주 진단이라고 했지?')인 경우에만 false로 판정하십시오.\n"
            "2. 'category' (string): 검색에 활용할 카테고리. 아래 3개 중 하나를 선택하십시오.\n"
            "   - 'law_ruling': 일반 법조문 해석, 판례(판결문), 형사/민사/행정 등 법령 관련 질문.\n"
            "   - 'contract_form': 계약서 양식, 서식, 계약 조항 작성 등 계약 템플릿과 서식 관련 질문.\n"
            "   - 'mrc': 금융 법률 기계독해, 경제/금융 분야 지식, 사전적 의미 질문.\n"
            "3. 'expanded_query' (string): 검색 정확도를 극대화하기 위해 질문 내용을 핵심 법적 쟁점(예: 사용자책임, 표현대리, 불법행위책임, 부당이득 등)으로 전환하고 관련 법률 전문 용어를 명시적으로 추가한 검색어.\n"
            "   - 예시 1: 이전 대화에서 '5주 진단 상해'를 다루었을 때, 현재 질문 '그럼 유죄판결 나면 몇년 살아?' ➡️ '상해죄 전치 5주 유죄 판결 실형 형량 양형기준'\n"
            "   - 예시 2: 회사 직원의 사기 행위로 인한 대표자 책임 질문 ➡️ '피용자 사기 회사 대표자 책임 사용자책임 민법 제756조 표현대리'\n"
            "   - 예시 3: 이전 대화에서 '5주 진단 상해'를 언급했고, 현재 질문 '내가 방금 몇주 진단이라고 했지?' ➡️ '상해죄 전치 5주 진단서 내용 확인'\n\n"
            "반드시 아래의 JSON 스키마 형식으로만 출력해야 하며, 다른 설명이나 주석은 절대 추가하지 마십시오:\n"
            "{\n"
            '  "is_search_required": true,\n'
            '  "category": "law_ruling",\n'
            '  "expanded_query": "[사건명/법령명] 질문내용 + 확장키워드"\n'
            "}"
        )
        
        context = ""
        if history:
            context = "이전 대화 맥락:\n"
            for msg in history[-3:]:
                role = "사용자" if msg["role"] == "user" else "AI"
                context += f"{role}: {msg['content']}\n"
                msg_sources = msg.get("sources")
                if msg_sources:
                    context += "  [이전 답변의 출처 문서 내용]\n"
                    for src in msg_sources:
                        context += f"  - 제목: {src.get('title')}\n"
                        context += f"    내용: {src.get('snippet')}\n"
            context += "\n"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}사용자 질문: {query}")
        ]
        
        try:
            response = self.json_llm.invoke(messages)
            content = response.content.strip()
            
            # Clean markdown code blocks if any
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            is_search_required = bool(result.get("is_search_required", True))
            category = str(result.get("category", "law_ruling"))
            llm_expanded = str(result.get("expanded_query", ""))
            
            # Combine the original detailed query with the LLM's expanded keywords to prevent loss of critical facts
            if llm_expanded:
                expanded_query = f"{query} {llm_expanded}".strip()
            else:
                expanded_query = query
            
            print(f"[통합 질문 분석] 완료: is_legal=True, is_search_required={is_search_required}, category={category}, query='{expanded_query}'")
            return {
                "is_legal": True,
                "is_search_required": is_search_required,
                "category": category,
                "expanded_query": expanded_query
            }
        except Exception as e:
            print(f"[통합 질문 분석] ⚠️ 분석 실패 (기본값 Fallback): {e}")
            return {
                "is_legal": True,
                "is_search_required": True,
                "category": "law_ruling",
                "expanded_query": query
            }
