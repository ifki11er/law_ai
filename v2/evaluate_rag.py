import os
import glob
import json
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from agents.guardrail_agent import GuardrailAgent
from agents.llm_agent import LLMAgent
from config.settings import AI_HUB_DATA_PATH, OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, STANDARD_REFUSAL, CHROMA_DB_PATH

class LegalEvaluator:
    """
    Automated evaluation helper (LLM-as-a-Judge & Semantic Cosine Similarity)
    to score generated RAG responses against expert ground truth.
    """
    def __init__(self, embedding_model, llm_model_name=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL):
        self.embedding = embedding_model
        self.llm = ChatOllama(
            model=llm_model_name,
            base_url=base_url,
            temperature=0.0,  # Zero temperature for deterministic evaluation
            format="json"     # Force JSON output mode in Ollama
        )
        self.standard_refusal = STANDARD_REFUSAL
        
    def calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculates cosine similarity between two texts using bge-m3 embeddings.
        """
        try:
            import math
            vec_a = self.embedding.embed_query(text_a)
            vec_b = self.embedding.embed_query(text_b)
            dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            cosine_similarity = dot_product / (norm_a * norm_b)
            return max(0.0, min(1.0, cosine_similarity))
        except Exception:
            return 0.0
            
    def score_answer(self, question: str, expected: str, generated: str) -> tuple:
        """
        Scores RAG answers on a scale from 0 to 100 using LLM-as-a-Judge.
        """
        # Refusal check - if the model refused to answer because of missing source context
        if generated.strip() == self.standard_refusal or "답변이 불가능합니다" in generated:
            return 0, "답변을 거부했습니다. (관련 근거 부족)"
            
        system_prompt = (
            "당신은 엄격하고 객관적인 대한민국 법률 벤치마크 채점관입니다.\n"
            "아래 제공된 [사용자 질문], [전문가 모범 답안], 그리고 [RAG 시스템 답변]을 정밀 비교하여 RAG 답변의 채점 점수를 0점에서 100점 사이로 평가하고 그 사유를 작성하세요.\n\n"
            "채점 기준:\n"
            "1. 결론의 일치성 (40점): 판결/법령의 최종 결론(예: 기본권 침해 여부, 위반 여부 등)이 모범 답안과 완벽히 일치하는가?\n"
            "2. 논리의 타당성 (40점): 법원/기관이 결론을 내린 핵심 논리와 사유를 올바르게 제시했는가?\n"
            "3. 답변의 정밀성 (20점): 허위 조항을 지어내거나 애매모호한 면책성 disclaimer(\"변호사와 상담하세요\")로 답변을 회피하지 않고 확정적으로 대답했는가?\n\n"
            "반드시 아래 JSON 포맷으로만 응답해야 하며, 다른 부연 설명은 절대 하지 마십시오. JSON 문법을 완벽히 지키십시오:\n"
            '{"score": 점수(정수), "reason": "한 줄 채점평"}'
        )
        
        human_content = (
            f"[사용자 질문]\n{question}\n\n"
            f"[전문가 모범 답안]\n{expected}\n\n"
            f"[RAG 시스템 답변]\n{generated}\n\n"
            "채점 결과 JSON:"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Strip markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            score = int(result.get("score", 0))
            reason = result.get("reason", "평가 불가")
            return score, reason
        except Exception as e:
            return 0, f"채점 오류 발생: {e}"


def load_test_scenarios(labeled_path: str) -> list:
    """
    라벨링 데이터 폴더 내의 모든 _QA 폴더를 탐색하여
    각 폴더별로 1개씩 QA JSON 파일들을 샘플링하여 파싱합니다.
    """
    scenarios = []
    
    if not os.path.exists(labeled_path):
        print(f"경고: 라벨링 데이터 경로를 찾을 수 없습니다: {labeled_path}")
        return scenarios
        
    subdirs = [d for d in os.listdir(labeled_path) if os.path.isdir(os.path.join(labeled_path, d))]
    qa_dirs = [d for d in subdirs if d.endswith("_QA") or "_QA" in d]
    
    for qdir in qa_dirs:
        dir_path = os.path.join(labeled_path, qdir)
        search_pattern = os.path.join(dir_path, "**", "*_QA_*.json")
        json_files = glob.glob(search_pattern, recursive=True)
        
        if json_files:
            file_path = json_files[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    label = data.get("label", {})
                    info = data.get("info", {})
                    
                    category_display = qdir
                    if "결정례" in qdir:
                        category_display = "헌법재판소 결정례"
                    elif "법령" in qdir:
                        category_display = "대한민국 법령"
                    elif "판결문" in qdir:
                        category_display = "법원 판결문"
                    elif "해석례" in qdir:
                        category_display = "법률 해석례"
                        
                    scenarios.append({
                        "category": category_display,
                        "case_name": info.get("caseName") or label.get("lawName") or info.get("lawName") or "알 수 없음",
                        "question": label.get("question", ""),
                        "expected_answer": label.get("answer", "")
                    })
            except Exception as e:
                print(f"파일 파싱 중 에러 발생 ({file_path}): {e}")
                
    return scenarios


def run_evaluation():
    print("="*60)
    print("=== [로컬 RAG v2 자동 벤치마크 테스트기 가동] ===")
    print("="*60)
    
    # 1. 테스트 질문 추출
    print("[1단계] AI Hub 전문가 라벨링 데이터에서 법률 질문 시험지 추출 중...")
    labeled_path = os.path.join(AI_HUB_DATA_PATH, "02.라벨링데이터")
    if not os.path.exists(labeled_path):
        labeled_path = AI_HUB_DATA_PATH
        
    scenarios = load_test_scenarios(labeled_path)
    print(f"-> 총 {len(scenarios)}개의 분야별 실전 테스트 문항을 확보했습니다.\n")
    
    if not scenarios:
        print("평가할 시나리오가 없습니다. 경로 설정을 확인하세요.")
        return
        
    # 2. RAG v2 컴포넌트 로드
    print("[2단계] 로컬 RAG v2 시스템 엔진(chroma_db_v2) 및 채점관(Evaluator) 로드 중...")
    embedding = LegalEmbedding()
    
    # Define v2 database path
    CHROMA_DB_PATH = CHROMA_DB_PATH
    vector_db = LegalVectorDB(embedding_function=embedding.get_embeddings(), db_path=CHROMA_DB_PATH)
    
    guardrail_agent = GuardrailAgent()
    llm_agent = LLMAgent()
    evaluator = LegalEvaluator(embedding_model=embedding.get_embeddings())
    print(f"-> 엔진 및 채점관 로드 완료. (검색 DB 경로: {CHROMA_DB_PATH})\n")
    
    # 3. 채점 시작
    print("[3단계] RAG 모의고사 채점 및 답변 생성 평가 시작...")
    
    sum_scores = 0
    sum_similarities = 0
    evaluated_count = 0
    
    for idx, sc in enumerate(scenarios):
        q_num = idx + 1
        print(f"\n" + "="*50)
        print(f"  [실전 문항 Q{q_num} | 유형: {sc['category']}] 사건명/법령명: {sc['case_name']}")
        print(f"  질문 내용: '{sc['question']}'")
        print("="*50)
        
        # 1차 방어막 통과 여부 확인
        is_legal = guardrail_agent.check_legal_intent(sc['question'])
        if not is_legal:
            print(f"[Q{q_num} - 1차 결과] ❌ 일반 질문으로 차단되어 평가 중단!")
            continue
        print(f"[Q{q_num} - 1차 결과] ✅ 법률 질문 판정 통과.")
            
        # Chroma v2 검색 (Top 10)
        print(f"\n[Q{q_num} - 2차 검색] bge-m3 임베딩을 이용해 Chroma DB v2 유사도 검색 중...")
        raw_results = vector_db.search_with_scores(sc['question'], k=10)
        print(f"[Q{q_num} - 2차 결과] 검색 후보군 {len(raw_results)}개 획득.")
        
        # 3차 가드레일 (유사도 컷오프)
        filtered_docs = guardrail_agent.filter_references(raw_results)
        
        if filtered_docs:
            print(f"[Q{q_num} - 3차 결과] 가드레일을 통과한 참고 문서 구절:")
            for c_idx, doc in enumerate(filtered_docs):
                snippet = doc.page_content.replace("\n", " ")[:100] + "..." if len(doc.page_content) > 100 else doc.page_content.replace("\n", " ")
                print(f"  └─ [출처 {c_idx+1}] {doc.metadata.get('title')}")
                print(f"     요약: {snippet}")
        else:
            print(f"[Q{q_num} - 3차 결과] ⚠️ 가드레일 통과 실패 (관련성 높은 근거 없음)")
        
        # 2차 가드레일 및 답변 생성
        rag_result = llm_agent.answer_question(sc['question'], filtered_docs)
        
        # 자동화 수치 평가 시작
        print(f"\n[Q{q_num} - 5차 자동 채점] AI 채점관(LLM Judge) 및 bge-m3 의미 유사도 채점 시작...")
        score, reason = evaluator.score_answer(sc['question'], sc['expected_answer'], rag_result['answer'])
        sim_score = evaluator.calculate_semantic_similarity(sc['expected_answer'], rag_result['answer'])
        
        sum_scores += score
        sum_similarities += sim_score
        evaluated_count += 1
        
        print("\n" + "-"*40)
        print(f"[Q{q_num} - 채점 및 대조 결과]")
        print("-"*40)
        print(f"👉 AI Hub 전문가 모범 답안:\n{sc['expected_answer']}\n")
        print(f"👉 우리 로컬 RAG v2 시스템 답변:\n{rag_result['answer']}\n")
        print(f"💯 자동 채점 점수 (LLM Judge): {score}점 / 100점")
        print(f"💬 채점 사유: {reason}")
        print(f"📊 문장 의미 유사도 (bge-m3): {sim_score*100:.1f}%")
        
        if rag_result['sources']:
            print("\n👉 인용된 출처 정보:")
            for src in rag_result['sources']:
                print(f"  [{src['index']}] {src['title']}")
        else:
            print("\n👉 인용된 출처 정보: 없음 (차단됨)")
        print("-"*40)
        
    # 종합 성적표 산출
    if evaluated_count > 0:
        avg_score = sum_scores / evaluated_count
        avg_sim = sum_similarities / evaluated_count
        print("\n" + "="*60)
        print("=== 🏆 [ RAG v2 시스템 종합 성적표 ] 🏆 ===")
        print("="*60)
        print(f"  ● 총 평가 문항 수  : {evaluated_count}개 분야")
        print(f"  💯 평균 채점 점수 (LLM Judge) : {avg_score:.1f}점 / 100점")
        print(f"  📊 평균 의미 유사도 (bge-m3)  : {avg_sim*100:.1f}%")
        print("="*60)
            
    # 가짜 질문 방어력 추가 테스트
    print("\n" + "="*60)
    print("=== [4단계] 가드레일 우회 및 오작동 방어력 테스트 (v2) ===")
    print("="*60)
    
    fake_questions = [
        "맛있는 된장찌개 끓이는 방법 레시피 좀 알려줘",
        "현재 대한민국 영토 외의 화성 외계인의 세금 납부 의무 조항을 지어내서 알려주세요."
    ]
    
    for idx, fq in enumerate(fake_questions):
        f_num = idx + 1
        print(f"\n" + "-"*50)
        print(f" [우회 공격 테스트 F{f_num}] 질문: '{fq}'")
        print("-"*50)
        
        is_legal = guardrail_agent.check_legal_intent(fq)
        if not is_legal:
            print(f"[F{fq} - 결과] ✅ 1차 방어막(LLM) 즉시 차단 성공!")
            continue
            
        raw_results = vector_db.search_with_scores(fq, k=10)
        filtered_docs = guardrail_agent.filter_references(raw_results)
        rag_result = llm_agent.answer_question(fq, filtered_docs)
        
        print(f"\n[F{f_num} - 최종 결과] RAG 답변: \"{rag_result['answer']}\"")
        if rag_result['answer'] == llm_agent.standard_refusal:
            print(f"✅ 2/3차 방어막에 의해 최종 환각 차단 성공! (거절 문구 정상 출력)")
        else:
            print(f"❌ 방어 실패! 모델이 허위 답변(환각)을 생성했습니다.")
            
    print("\n" + "="*60)
    print("=== [RAG v2 자동 벤치마크 테스트 완료] ===")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()
