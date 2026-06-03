from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from config.settings import OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, STANDARD_REFUSAL
from typing import List, Dict, Any

class LLMAgent:
    """
    Agent responsible for 2nd Guardrail (System prompt hallucination control)
    and generating citation-backed answers using gemma4.
    """
    
    def __init__(self, model_name: str = OLLAMA_LLM_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.1
        )
        self.standard_refusal = STANDARD_REFUSAL

    def build_context_string(self, docs: List[Document]) -> str:
        """
        Format retrieved docs into a labeled context string for the LLM.
        """
        context_parts = []
        for i, doc in enumerate(docs):
            idx = i + 1
            title = doc.metadata.get("title", f"출처 {idx}")
            context_parts.append(
                f"[출처 {idx}]\n"
                f"내용: {doc.page_content}\n"
                f"상세정보: {title}\n"
                "----------------------------------------"
            )
        return "\n\n".join(context_parts)

    def answer_question(self, query: str, filtered_docs: List[Document]) -> Dict[str, Any]:
        """
        Generates the answer with citation numbers or returns the refusal string.
        """
        # Optimization: If no documents passed the 3rd guardrail, return refusal directly
        if not filtered_docs:
            print("[2차 방어막] 참고 가능한 법률 근거가 없어 즉시 답변 거부 결정")
            return {
                "answer": self.standard_refusal,
                "sources": []
            }

        # Format context for LLM
        context_str = self.build_context_string(filtered_docs)
        
        system_prompt = (
            "너는 엄격하고 객관적인 대한민국 법률 전문 AI 비서이다.\n"
            "반드시 아래 제공된 [참고 법률 문서]의 내용에만 철저히 기반하여 답변하라.\n"
            "또한, 답변의 정밀성을 위해 다음 두 가지 지침을 엄격히 준수하라:\n"
            "1. [주장과 판단의 엄격한 구분]: 참고 문서에 나오는 '당사자(청구인, 피고인 등)의 주장'과 '법원(또는 헌법재판소)의 최종 판단 및 결정'을 명확히 구분하라. 답변은 반드시 당사자의 주장이 아닌 '법원의 최종 판단 및 결정 내용'을 기준으로 작성해야 한다.\n"
            "2. [추론 및 회피 금지]: 제공된 참고 문서 내에 확실한 법적 근거가 존재한다면 '전문가의 상담을 받으라'거나 '단정하기 어렵다'와 같은 애매한 회피성 disclaimer 문구를 절대 작성하지 말고, 문서에 적힌 사실을 바탕으로 확정적이고 직접적으로 답변하라.\n"
            "사용자의 질문이 [참고 법률 문서]의 내용과 논리적으로 무관하거나, 문서 안에서 확실한 법적 근거를 찾을 수 없다면 "
            "절대로 유추하여 답변하거나 조항을 임의로 지어내지 말고, 오직 다음 표준 거절 문장만 정확히 출력해라:\n"
            f"'{self.standard_refusal}'\n\n"
            "또한, 답변의 신뢰성을 위해 다음 인용 규칙을 반드시 지켜야 한다:\n"
            "1. 제공된 [참고 법률 문서]에는 각 구절마다 [출처 1], [출처 2] 등 고유 번호가 붙어있다.\n"
            "2. 답변 본문을 작성할 때, 특정 사실이나 근거 조항을 설명하는 각 문장 끝에 반드시 "
            "해당하는 출처의 번호를 인라인 형식(예: ...해당하지 않는다고 판단하였습니다.[출처 1])으로 표기하라.\n"
            "3. 답변과 무관하게 모든 출처를 다 억지로 넣지 말고, 해당 답변 문장의 실제 근거가 되는 출처만 매핑하라."
        )

        human_content = (
            f"[참고 법률 문서]\n{context_str}\n\n"
            f"사용자 질문: {query}\n\n"
            f"답변:"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]

        try:
            print("[2차 방어막] LLM 추론 시작...")
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            # Format source list for metadata output
            sources = []
            for i, doc in enumerate(filtered_docs):
                sources.append({
                    "index": i + 1,
                    "title": doc.metadata.get("title", f"출처 {i+1}"),
                    "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })

            # Check if LLM generated response but couldn't find answer, double check standard refusal
            # If the output indicates it cannot answer, align it to standard refusal
            refusal_indicators = ["알 수 없", "찾을 수 없", "답변이 불가능", "근거가 존재하지"]
            if any(ind in answer for ind in refusal_indicators) and len(answer) < 80:
                answer = self.standard_refusal
                sources = []

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            print(f"[2차 방어막] LLM 생성 오류: {e}")
            return {
                "answer": f"시스템 처리 중 오류가 발생했습니다. {self.standard_refusal}",
                "sources": []
            }
