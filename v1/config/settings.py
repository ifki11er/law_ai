import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Chroma DB Path
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")

# Ollama Settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "gemma4:e2b"
OLLAMA_EMBEDDING_MODEL = "bge-m3"

# Legal Dual Guardrails Threshold
SIMILARITY_THRESHOLD = 1.1

# Standard Refusal Message for out-of-domain queries
STANDARD_REFUSAL = "제공된 법률 데이터 내에서는 관련 내용을 찾을 수 없어 답변이 불가능합니다."

# Chunking Settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


# AI Hub Dataset Path
AI_HUB_DATA_PATH = os.path.join(
    PROJECT_ROOT, 
    "docs", 
    "학습데이터",
    "04.형사법 LLM 사전학습 및 Instruction Tuning 데이터", 
    "3.개방데이터", 
    "1.데이터", 
    "Training"
)
