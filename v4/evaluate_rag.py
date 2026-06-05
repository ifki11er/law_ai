import os
import glob
import json
import sys
import unicodedata
from langchain_core.messages import SystemMessage, HumanMessage
from core.embedding import LegalEmbedding
from core.vector_db import LegalVectorDB
from agents.guardrail_agent import GuardrailAgent
from agents.llm_agent import LLMAgent
from config.settings import AI_HUB_DATA_PATH, STANDARD_REFUSAL, CHROMA_DB_PATH, LLM_PROVIDER, OPENAI_LLM_MODEL, OLLAMA_LLM_MODEL
from core.llm_factory import get_llm

def get_display_width(s: str) -> int:
    """
    Calculates the display width of a string.
    East Asian characters are counted as width 2, others as 1.
    """
    width = 0
    for char in s:
        status = unicodedata.east_asian_width(char)
        if status in ('W', 'F', 'A'):
            width += 2
        else:
            width += 1
    return width

def truncate_string(s: str, max_width: int) -> str:
    """
    Truncates a string to fit within max_width display width, appending '..' if truncated.
    """
    current_width = get_display_width(s)
    if current_width <= max_width:
        return s
    
    res = ""
    width = 0
    limit = max_width - 2
    for char in s:
        char_width = 2 if unicodedata.east_asian_width(char) in ('W', 'F', 'A') else 1
        if width + char_width > limit:
            break
        res += char
        width += char_width
    return res + ".."

def pad_string(s: str, target_width: int, align: str = 'left') -> str:
    """
    Pads a string to the target display width, handling East Asian character widths.
    """
    current_width = get_display_width(s)
    pad_len = target_width - current_width
    if pad_len <= 0:
        return s
    
    if align == 'right':
        return ' ' * pad_len + s
    elif align == 'center':
        left_pad = pad_len // 2
        right_pad = pad_len - left_pad
        return ' ' * left_pad + s + ' ' * right_pad
    else: # left
        return s + ' ' * pad_len

def format_row(cols, widths, alignments):
    """
    Formats a row of cells with specified widths and alignments.
    """
    formatted_cols = []
    for col, width, align in zip(cols, widths, alignments):
        truncated = truncate_string(col, width)
        padded = pad_string(truncated, width, align)
        formatted_cols.append(padded)
    return "| " + " | ".join(formatted_cols) + " |"

class LegalEvaluator:
    """
    Automated evaluation helper (LLM-as-a-Judge & Semantic Cosine Similarity)
    to score generated RAG responses against expert ground truth.
    """
    def __init__(self, embedding_model, llm_model_name=None, base_url=None):
        self.embedding = embedding_model
        self.llm = get_llm(temperature=0.0, format="json")
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
            "1. 결론의 일치성 (40점 만점): 판결/법령의 최종 결론(예: 기본권 침해 여부, 위반 여부 등)이 모범 답안과 완벽히 일치하는가?\n"
            "2. 논리의 타당성 (40점 만점): 법원/기관이 결론을 내린 핵심 논리와 사유를 올바르게 제시했는가?\n"
            "3. 답변의 정밀성 (20점 만점): 허위 조항을 지어내거나 애매모호한 면책성 disclaimer(\"변호사와 상담하세요\")로 답변을 회피하지 않고 확정적으로 대답했는가?\n\n"
            "💡 중요 채점 가이드라인:\n"
            "- 10점, 20점 단위로 뭉뚱그려 채점하지 마시고, 부분적인 결함이나 문맥의 차이를 반영하여 1점 단위로 정밀하게 감점하여 평가해 주세요. (예: 87점, 93점, 76점 등)\n"
            "- 결론이 일치하더라도 사유나 논리가 약간 미진하면 논리성에서 3~7점을 감점하는 등 상세하게 평가하십시오.\n\n"
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
            import time
            t0 = time.time()
            response = self.llm.invoke(messages)
            print(f" -> AI 채점관 LLM 호출 완료 ({time.time() - t0:.3f}초 소요)")
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


def load_test_scenarios(labeled_path: str, ingested_prefixes: set = None) -> list:
    """
    라벨링 데이터 폴더를 재귀적으로 탐색하여 *_QA_*.json 파일들을 찾아 파싱합니다.
    디바이스 용량 절약 및 일관성을 위해, 데이터베이스에 실제 존재하는 파일만 평가합니다.
    최대 100개의 문항을 로드하며, 다양한 카테고리가 고르게 섞이도록 디렉토리별 라운드로빈 방식으로 샘플링합니다.
    """
    scenarios = []
    
    if not os.path.exists(labeled_path):
        print(f"경고: 라벨링 데이터 경로를 찾을 수 없습니다: {labeled_path}")
        return scenarios
        
    import collections
    qa_files_by_dir = collections.defaultdict(list)
    
    for root, dirs, files in os.walk(labeled_path):
        # Skip validation or wait folders
        if "validation" in root.lower():
            continue
        if "02.라벨링데이터" in root or "_QA" in root:
            for file in files:
                if "_QA_" in file and file.endswith(".json"):
                    prefix = file.split("_QA_")[0]
                    # Filter: only evaluate files that have matching ingested prefixes in the DB
                    if ingested_prefixes is not None and prefix not in ingested_prefixes:
                        continue
                    qa_files_by_dir[root].append(os.path.join(root, file))
                    
    # Collect all available QA files and round-robin sample to get a balanced representation
    all_file_paths = []
    if qa_files_by_dir:
        max_files_in_dir = max(len(f_list) for f_list in qa_files_by_dir.values())
        for idx in range(max_files_in_dir):
            for dir_path, f_list in qa_files_by_dir.items():
                if idx < len(f_list):
                    all_file_paths.append((dir_path, f_list[idx]))
                    
    for dir_path, file_path in all_file_paths:
        if len(scenarios) >= 100:
            break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                label = data.get("label", {})
                info = data.get("info", {})
                
                rel_path = os.path.relpath(dir_path, labeled_path)
                parts = rel_path.split(os.sep)
                category_display = parts[-1] if parts else "법률 질문"
                
                if "결정례" in category_display:
                    category_display = "헌법재판소 결정례"
                elif "법령" in category_display:
                    category_display = "대한민국 법령"
                elif "판결문" in category_display:
                    category_display = "법원 판결문"
                elif "해석례" in category_display:
                    category_display = "법률 해석례"
                    
                question = (label.get("input") or label.get("instruction") or label.get("question") or "").strip()
                expected_answer = (label.get("output") or label.get("answer") or "").strip()
                
                # If question is empty, print warning and some debug info about the JSON structure
                if not question:
                    print(f"⚠️ 경고: '{os.path.basename(file_path)}' 파일의 질문 내용이 비어 있습니다.")
                    print(f"   [JSON 루트 Key]: {list(data.keys())}")
                    if "label" in data:
                        print(f"   [label 내부 Key]: {list(data['label'].keys()) if isinstance(data['label'], dict) else data['label']}")
                    
                scenarios.append({
                    "category": category_display,
                    "case_name": info.get("caseName") or label.get("lawName") or info.get("lawName") or info.get("title") or "알 수 없음",
                    "question": question,
                    "expected_answer": expected_answer
                })
        except Exception as e:
            print(f"파일 파싱 중 에러 발생 ({file_path}): {e}")
                
    return scenarios


def run_evaluation():
    print("="*60)
    print("=== [로컬 RAG v4 자동 벤치마크 테스트기 가동] ===")
    print("="*60)
    
    # 1. RAG v4 컴포넌트 및 DB 로드
    print("[1단계] 로컬 RAG v4 시스템 엔진 및 채점관(Evaluator) 로드 중...")
    embedding = LegalEmbedding()
    vector_db = LegalVectorDB(embedding_function=embedding.get_embeddings(), db_path=CHROMA_DB_PATH)
    
    # Extract ingested prefixes from SQLite DB to filter test scenarios
    ingested_prefixes = set()
    try:
        import sqlite3
        conn = sqlite3.connect(vector_db.parent_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT string_value FROM embedding_metadata WHERE key = 'file_prefix'")
        ingested_prefixes = {row[0] for row in cursor.fetchall() if row[0]}
        conn.close()
    except Exception as e:
        print(f"⚠️ ingested_prefixes 로드 중 오류 발생: {e}")
    print(f" -> DB에 적재 완료된 소스 파일 prefix 수: {len(ingested_prefixes):,}개")
    
    guardrail_agent = GuardrailAgent()
    llm_agent = LLMAgent()
    evaluator = LegalEvaluator(embedding_model=embedding.get_embeddings())
    print(f"-> 엔진 및 채점관 로드 완료. (검색 DB 경로: {CHROMA_DB_PATH})\n")
    
    # [v4+] 시스템 엔진 구성표 출력
    llm_provider_display = "OpenAI (GPT API)" if LLM_PROVIDER.lower() == "openai" else "Ollama (로컬)"
    llm_model_display = OPENAI_LLM_MODEL if LLM_PROVIDER.lower() == "openai" else OLLAMA_LLM_MODEL
    
    import torch
    emb_class_name = embedding.embeddings.__class__.__name__
    if "HuggingFace" in emb_class_name:
        device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        emb_display = f"BAAI/bge-m3 (로컬 {device_name})"
    else:
        emb_display = "BAAI/bge-m3 (Ollama Fallback)"
        
    print("="*60)
    print(" 🛠️  [RAG v4 시스템 엔진 구성 정보]")
    print("+----------------------+--------------------------------------------+")
    print("| 구성 구분            | 설정 및 모델 정보                          |")
    print("+----------------------+--------------------------------------------+")
    print(f"| LLM 공급자 (Engine)  | {llm_provider_display:<42} |")
    print(f"| LLM 모델명 (Model)   | {llm_model_display:<42} |")
    print(f"| 임베딩 모델 (Vector) | {emb_display:<42} |")
    print("+----------------------+--------------------------------------------+")
    print("="*60 + "\n")
    
    # 2. 테스트 질문 추출
    print("[2단계] AI Hub 전문가 라벨링 데이터에서 법률 질문 시험지 추출 중...")
    labeled_path = os.path.join(AI_HUB_DATA_PATH, "02.라벨링데이터")
    if not os.path.exists(labeled_path):
        labeled_path = AI_HUB_DATA_PATH
        
    scenarios = load_test_scenarios(labeled_path, ingested_prefixes)
    print(f"-> 총 {len(scenarios)}개의 분야별 실전 테스트 문항을 확보했습니다.\n")
    
    if not scenarios:
        print("평가할 시나리오가 없습니다. DB에 적재된 문서에 해당하는 라벨링 데이터가 존재하지 않거나 경로를 확인하세요.")
        return
        
    # 3. 채점 시작
    print("[3단계] RAG 모의고사 채점 및 답변 생성 평가 시작...")
    
    sum_scores = 0
    sum_similarities = 0
    evaluated_count = 0
    evaluation_results = []
    
    for idx, sc in enumerate(scenarios):
        q_num = idx + 1
        print(f"\n" + "="*50)
        print(f"  [실전 문항 Q{q_num} | 유형: {sc['category']}] 사건명/법령명: {sc['case_name']}")
        print(f"  질문 내용: '{sc['question']}'")
        print("="*50)
        
        # 다중 문서 DB에서 질문 속 '이 사건/이 조항' 등의 대상을 특정하기 위해 사건명/법령명을 쿼리에 합성
        search_query = sc['question']
        if sc['case_name'] and sc['case_name'] != "알 수 없음":
            search_query = f"[{sc['case_name']}] {sc['question']}"
            
        # 1차 통합 질문 분석 (카테고리 분류 및 쿼리 확장) 기동
        print(f"[Q{q_num} - 1차 결과] 통합 질문 분석 기동 중...")
        analysis = guardrail_agent.analyze_query(search_query)
        print(f"[Q{q_num} - 1차 결과] 분석 완료.")
        
        category = analysis.get("category", "law_ruling")
        expanded_query = analysis.get("expanded_query", search_query)
        
        # Chroma v4 하이브리드 검색 (Top 10)
        print(f"\n[Q{q_num} - 2차 검색] BM25와 Chroma를 이용한 v4 하이브리드 검색 중... (카테고리 필터: {category})")
        raw_results = vector_db.search_hybrid(expanded_query, k=10, category=category)
        print(f"[Q{q_num} - 2차 결과] 검색 완료 (후보군 {len(raw_results)}개 확보). 3단계(필터링)로 진입합니다.")
        
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
        
        # Save result for the final summary table
        evaluation_results.append({
            "num": f"Q{q_num}",
            "category": sc['category'],
            "case_name": sc['case_name'],
            "score": f"{score}점",
            "similarity": f"{sim_score*100:.1f}%"
        })
        
        print("\n" + "-"*40)
        print(f"[Q{q_num} - 채점 및 대조 결과]")
        print("-"*40)
        print(f"👉 AI Hub 전문가 모범 답안:\n{sc['expected_answer']}\n")
        print(f"👉 우리 로컬 RAG v4 시스템 답변:\n{rag_result['answer']}\n")
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
        print("\n" + "="*80)
        print("📊 [ 문항별 상세 채점 결과 ]")
        print("="*80)
        
        widths = [5, 16, 32, 10, 10]
        headers = ["No", "유형", "사건명/법령명", "LLM 점수", "유사도"]
        alignments = ["center", "left", "left", "center", "center"]
        
        sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
        print(sep)
        print(format_row(headers, widths, alignments))
        print(sep)
        
        for res in evaluation_results:
            row_data = [
                res["num"],
                res["category"],
                res["case_name"],
                res["score"],
                res["similarity"]
            ]
            print(format_row(row_data, widths, alignments))
            
        print(sep)

        avg_score = sum_scores / evaluated_count
        avg_sim = sum_similarities / evaluated_count

        # Print average row at the bottom of the table
        avg_row_data = ["평균", "-", "-", f"{avg_score:.1f}점", f"{avg_sim*100:.1f}%"]
        print(format_row(avg_row_data, widths, alignments))
        print(sep)

        print("\n" + "="*60)
        print("=== 🏆 [ RAG v4 시스템 종합 성적표 ] 🏆 ===")
        print("="*60)
        print(f"  ● 총 평가 문항 수  : {evaluated_count}개 분야")
        print(f"  💯 평균 채점 점수 (LLM Judge) : {avg_score:.1f}점 / 100점")
        print(f"  📊 평균 의미 유사도 (bge-m3)  : {avg_sim*100:.1f}%")
        print("="*60)
            
    # 가짜 질문 방어력 추가 테스트
    print("\n" + "="*60)
    print("=== [4단계] 가드레일 우회 및 오작동 방어력 테스트 (v4) ===")
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
        
        # 1차 방어막(check_legal_intent)이 제거되었으므로, 모든 질문은 하이브리드 검색 및 3차 유사도 가드레일로 방어합니다.
            
        # [검색 최적화: 질문 재작성 및 확장]
        expanded_fq = llm_agent.rewrite_query(fq)
            
        raw_results = vector_db.search_hybrid(expanded_fq, k=10)
        filtered_docs = guardrail_agent.filter_references(raw_results)
        rag_result = llm_agent.answer_question(fq, filtered_docs)
        
        print(f"\n[F{f_num} - 최종 결과] RAG 답변: \"{rag_result['answer']}\"")
        if rag_result['answer'] == llm_agent.standard_refusal:
            print(f"✅ 2/3차 방어막에 의해 최종 환각 차단 성공! (거절 문구 정상 출력)")
        else:
            print(f"❌ 방어 실패! 모델이 허위 답변(환각)을 생성했습니다.")
            
    print("\n" + "="*60)
    print("=== [RAG v4 자동 벤치마크 테스트 완료] ===")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()
