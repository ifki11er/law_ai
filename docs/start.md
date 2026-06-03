검증 및 사용 요약
Ollama 모델 다운로드: ollama pull bge-m3 및 ollama pull gemma4 실행.
의존성 설치: pip install -r requirements.txt 실행.
지식 데이터베이스 구축: python ingest.py 실행 (AI Hub 데이터 자동 정제 및 크로마 적재).
서버 구동: python -m uvicorn app:app --reload 실행 후 http://localhost:8000 접속.
검증 : python evaluate_rag.py