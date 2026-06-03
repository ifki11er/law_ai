from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from config.settings import STANDARD_REFUSAL
from core.llm_factory import get_llm
from typing import List, Dict, Any

class LLMAgent:
    """
    Agent responsible for 2nd Guardrail (System prompt hallucination control)
    and generating citation-backed answers. Supports Ollama/GPT switching.
    """
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.llm = get_llm(temperature=0.1)
        self.standard_refusal = STANDARD_REFUSAL

    def build_context_string(self, docs: List[Document]) -> str:
        """
        Format retrieved docs into a labeled context string for the LLM.
        Restores parent_text if present in metadata to provide complete logical context.
        """
        context_parts = []
        for i, doc in enumerate(docs):
            idx = i + 1
            title = doc.metadata.get("title", f"출처 {idx}")
            # Use parent_text if available (Parent Context Restoration), otherwise fallback to child page_content
            content_body = doc.metadata.get("parent_text", doc.page_content)
            context_parts.append(
                f"[출처 {idx}]\n"
                f"내용: {content_body}\n"
                f"상세정보: {title}\n"
                "----------------------------------------"
            )
        return "\n\n".join(context_parts)

    def answer_question(self, query: str, filtered_docs: List[Document], history: List[Dict[str, str]] = None, is_search_required: bool = True) -> Dict[str, Any]:
        """
        Generates the answer with citation numbers or returns the refusal string, incorporating conversation history.
        Supports bypassing search (RAG bypass) for contextual follow-ups.
        """
        # Optimization: If search was required, but no documents passed the 3rd guardrail, return refusal directly
        if is_search_required and not filtered_docs:
            print("[2차 방어막] 참고 가능한 법률 근거가 없어 즉시 답변 거부 결정")
            return {
                "answer": self.standard_refusal,
                "sources": []
            }

        # Format context for LLM if available
        context_str = self.build_context_string(filtered_docs) if filtered_docs else ""
        
        if filtered_docs:
            system_prompt = (
                "너는 엄격하고 객관적인 대한민국 법률 전문 AI 비서이다.\n"
                "반드시 아래 제공된 [참고 법률 문서]의 내용에만 철저히 기반하여 답변하라.\n"
                "또한, 답변의 정밀성과 가독성을 위해 다음 지침들을 엄격히 준수하라:\n"
                "1. [가독성 향상 지침]: 답변을 작성할 때 가독성이 매우 중요하다. 긴 설명글을 한 단락으로 길게 이어 쓰지 말고, 내용의 흐름에 따라 문단을 적절히 나누어라. 핵심 요점, 판단 근거, 혹은 주요 논점은 글머리 기호(마크다운의 '-', '1.', '2.' 등)를 사용하여 보기 쉽게 구조화하라. 중요한 법적 개념이나 결론 등은 마크다운 볼드체(예: **강제가입 규정**, **각하 결정**)로 강조하여 한눈에 들어오게 하라.\n"
                "2. [주장과 판단의 엄격한 구분]: 참고 문서에 나오는 '당사자(청구인, 피고인 등)의 주장'과 '법원(또는 헌법재판소)의 최종 판단 및 결정'을 명확히 구분하라. 답변은 반드시 당사자의 주장이 아닌 '법원의 최종 판단 및 결정 내용'을 기준으로 작성해야 한다.\n"
                "3. [추론 및 회피 금지]: 제공된 참고 문서 내에 확실한 법적 근거가 존재한다면 '전문가의 상담을 받으라'거나 '단정하기 어렵다'와 같은 애매한 회피성 disclaimer 문구를 절대 작성하지 말고, 문서에 적힌 사실을 바탕으로 확정적이고 직접적으로 답변하라.\n"
                "단, 참고 문서에 명시된 법조문이나 판례의 기준(예: 상해의 정의, 폭행의 처벌 규정 등)을 바탕으로 사용자의 구체적 사안(예: 전치 5주의 진단을 받은 피해 상황)에 법률을 대입하여 상식적이고 논리적인 판단을 내릴 수 있는 경우에는 적극적으로 답변을 작성하라. 예를 들어, 문서에 '폭행으로 치료가 필요한 상처를 입힌 경우 상해죄가 성립한다'는 취지의 판례가 있고 질문자가 '모르는 사람에게 머리를 맞아 전치 5주 진단을 받았다'고 한다면, 이를 판례상의 상해 개념에 부합한다고 판단하여 **상해죄**로 고소 가능하다고 답변하는 것은 완벽히 허용되며 적극 권장된다. 없는 조항을 지어내는 거짓 행위(Hallucination)만 금지할 뿐, 주어진 문서의 법리와 사실관계를 논리적으로 대입하는 법률적 판단은 적극적으로 서술하라.\n"
                "사용자의 질문이 [참고 법률 문서]의 내용과 완전히 무관하거나(예: 다른 분야의 법률 문서만 제공된 경우 등), 참고 문서의 내용만으로는 어떠한 법적 실마리도 찾을 수 없는 경우에만 다음 표준 거절 문장만 정확히 출력해라:\n"
                f"'{self.standard_refusal}'\n\n"
                "또한, 답변의 신뢰성을 위해 다음 인용 규칙을 반드시 지켜야 한다:\n"
                "1. 제공된 [참고 법률 문서]에는 각 구절마다 [출처 1], [출처 2] 등 고유 번호가 붙어있다.\n"
                "2. 답변 본문을 작성할 때, 특정 사실이나 근거 조항을 설명하는 각 문장 끝에 반드시 "
                "해당하는 출처의 번호를 인라인 형식(예: ...해당하지 않는다고 판단하였습니다.[출처 1])으로 표기하라.\n"
                "3. 답변과 무관하게 모든 출처를 다 억지로 넣지 말고, 해당 답변 문장의 실제 근거가 되는 출처만 매핑하라."
            )
        else:
            system_prompt = (
                "너는 엄격하고 객관적인 대한민국 법률 전문 AI 비서이다.\n"
                "현재 질문은 사용자가 이전 대화 기록을 확인하거나 AI가 했던 답변 내용을 다시 묻는 단순 대화 맥락 질문이다.\n"
                "새로운 법률 조항이나 판례를 참조(검색)하지 않고 오직 아래 제공되는 이전 대화 기록(history)을 철저히 분석하여 질문에 대한 사실을 확인하고 친절하고 가독성 있게 대답하라.\n"
                "예를 들어, 사용자가 '내가 방금 몇주 진단이라고 했지?'라고 묻는다면, 이전 대화 기록에서 사용자가 말한 진단 주수(예: 전치 5주)를 찾아 정확히 알려주면 된다. 그 외에 상해죄 등의 추가 법적 판단은 간략히 언급하거나 굳이 길게 덧붙이지 않아도 된다."
            )

        messages = []
        messages.append(SystemMessage(content=system_prompt))
        
        # Append history (limit to last 6 messages / 3 turns to prevent context blowup)
        if history:
            for msg in history[-6:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        if filtered_docs:
            human_content = (
                f"[참고 법률 문서]\n{context_str}\n\n"
                f"사용자 질문: {query}\n\n"
                f"답변:"
            )
        else:
            human_content = (
                f"사용자 질문: {query}\n\n"
                f"답변:"
            )
        messages.append(HumanMessage(content=human_content))

        try:
            import time
            import re
            print("[2차 방어막] LLM 추론 시작...")
            t0 = time.time()
            response = self.llm.invoke(messages)
            print(f"[2차 방어막] LLM 추론 완료 ({time.time() - t0:.3f}초 소요)")
            answer = response.content.strip()
            
            # Find actually cited indices in the answer (e.g., [출처 1], [출처 2])
            cited_indices = set(map(int, re.findall(r'\[출처\s*(\d+)\]', answer)))
            
            # Format source list for metadata output, including only actually cited ones
            sources = []
            for i, doc in enumerate(filtered_docs):
                idx = i + 1
                if idx in cited_indices:
                    sources.append({
                        "index": idx,
                        "title": doc.metadata.get("title", f"출처 {idx}"),
                        "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    })

            # Check if LLM generated response but couldn't find answer, double check standard refusal
            # If the output indicates it cannot answer, align it to standard refusal
            refusal_indicators = ["알 수 없", "찾을 수 없", "답변이 불가능", "근거가 존재하지", "명시되어 있지", "언급되어 있지", "확인할 수 없"]
            if any(ind in answer for ind in refusal_indicators) and len(answer) < 120:
                answer = self.standard_refusal
                sources = []

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            print(f"[2차 방어막] LLM 생성 오류: {e}")
            return {
                "answer": f"시스템 처리 중 오류가 발생했습니다. (상세 오류: {str(e)})",
                "sources": []
            }

    def rewrite_query(self, query: str, history: List[Dict[str, str]] = None) -> str:
        """
        [v3+] LLM Query Rewriter (Context-Aware).
        Converts conversational query into professional legal search keywords.
        Returns the original query combined with the legal keywords.
        """
        import re
        system_prompt = (
            "당신은 법률 RAG 시스템의 질문 재작성기(Query Rewriter)입니다.\n"
            "사용자의 이전 대화 내역과 현재 질문을 참고하여, 법조문이나 판례에 수록되었을 만한 전문적인 법률 용어 및 핵심 검색 키워드를 추출 및 확장하십시오.\n"
            "특히 '그럼', '그거', '그 조항' 등 대화 맥락 속 지시어가 지칭하는 대상을 구체적인 법률 용어로 풀어내어 포함하십시오.\n"
            "설명은 일절 하지 말고, 오직 검색 정확도를 극대화할 수 있는 핵심 키워드 리스트(쉼표로 구분)만 대답하십시오.\n"
            "예시:\n"
            "질문: '차 훔쳐 타면 몇년 감옥 가?'\n"
            "답변: 절도죄, 자동차등불법사용죄, 징역형량, 양형기준"
        )
        
        context = ""
        if history:
            context = "이전 대화 내역:\n"
            for msg in history[-3:]:
                role = "사용자" if msg["role"] == "user" else "AI"
                context += f"{role}: {msg['content']}\n"
            context += "\n"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}질문: '{query}'")
        ]
        try:
            import time
            print("[검색 최적화] LLM을 이용해 법률 검색어 확장 중...")
            t0 = time.time()
            response = self.llm.invoke(messages)
            print(f"[검색 최적화] LLM 검색어 확장 완료 ({time.time() - t0:.3f}초 소요)")
            rewritten = response.content.strip()
            # Clean up potential markdown formatting
            rewritten = re.sub(r'[*`]', '', rewritten)
            combined_query = f"{query} {rewritten}"
            print(f"[검색 최적화] 확장된 쿼리: '{combined_query}'")
            return combined_query
        except Exception as e:
            print(f"[검색 최적화] ⚠️ 쿼리 확장 실패(기존 질문 활용): {e}")
            return query

