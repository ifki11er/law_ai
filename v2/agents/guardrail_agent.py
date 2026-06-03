from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from config.settings import OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, SIMILARITY_THRESHOLD
from typing import List, Tuple

class GuardrailAgent:
    """
    Agent responsible for 1st Guardrail (Intent filtering via LLM)
    and 3rd Guardrail (Similarity score cutoff filtering).
    """
    
    def __init__(self, model_name: str = OLLAMA_LLM_MODEL, base_url: str = OLLAMA_BASE_URL, threshold: float = SIMILARITY_THRESHOLD):
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.0  # Zero temperature for deterministic classification
        )
        self.threshold = threshold

    def check_legal_intent(self, query: str) -> bool:
        """
        1차 방어막: 사용자의 질문이 법률/판례 관련 질문인지 판단합니다.
        법률과 무관하면 False를 리턴합니다.
        """
        system_prompt = (
            "당신의 임무는 사용자 질문의 의도가 대한민국 법률, 재판, 판례, 법제도, 헌법, 행정 규정 등 법률 분야와 관련된 것인지 판단하는 것입니다.\n"
            "오직 질문이 법률(사법, 공법, 형사법, 민사법, 행정법, 여권/면허/등록 등 공공/행정 절차 및 규정, 헌법재판 등 모든 법적 규정 및 조항 포함)과 직간접적으로 관련되어 있다면 'YES',\n"
            "전혀 무관한 일상 대화, 요리법, 기술 질문, 일반 상식, 외국 날씨 등이라면 'NO'라고만 대답하십시오. 단어로만 응답하고 다른 설명을 덧붙이지 마십시오.\n"
            "주의: 여권 발급/종류, 운전면허 취득/취소, 주민등록, 소득세 신고 등 행정법 및 관련 규정이 배경이 되는 생활 행정 질문은 모두 'YES'로 분류해야 합니다."
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"질문: {query}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = response.content.strip().upper()
            print(f"[1차 방어막] 의도 분류 결과: {result} (질문: {query})")
            
            if "YES" in result or "Y" == result:
                return True
            if "NO" in result or "N" == result:
                return False
                
            # 한국어 폴백 판단
            if any(word in result for word in ["예", "네", "맞습", "해당", "법률"]):
                return True
            if any(word in result for word in ["아니", "없습", "무관", "아님"]):
                return False
                
            return "YES" in result
        except Exception as e:
            print(f"[1차 방어막] 의도 판단 오류: {e}")
            # 에러 발생 시 보수적으로 통과시키거나 차단할 수 있지만, 여기서는 True로 가정하고 2/3차에서 차단하도록 합니다.
            return True

    def filter_references(self, search_results: List[Tuple[Document, float]]) -> List[Document]:
        """
        3차 방어막: 유사도 점수 컷오프를 사용하여 관련성 없는 근거를 배제(exclude)합니다.
        Chroma의 distance score가 threshold(1.1) 이하인 문서들만 통과시킵니다.
        """
        filtered_docs = []
        for doc, score in search_results:
            # Chroma L2/cosine distance score: 작을수록 유사함
            if score <= self.threshold:
                filtered_docs.append(doc)
                print(f"[3차 방어막] 통과 - 스코어: {score:.4f} | 출처: {doc.metadata.get('title')}")
            else:
                print(f"[3차 방어막] 배제(컷오프) - 스코어: {score:.4f} | 출처: {doc.metadata.get('title')}")
                
        print(f"[3차 방어막] 최종 필터링된 근거 개수: {len(filtered_docs)} / {len(search_results)}")
        return filtered_docs
