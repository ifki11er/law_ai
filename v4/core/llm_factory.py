import os
from langchain_ollama import ChatOllama
from config.settings import LLM_PROVIDER, OLLAMA_LLM_MODEL, OLLAMA_BASE_URL, OPENAI_API_KEY, OPENAI_LLM_MODEL

def get_llm(temperature: float = 0.1, format: str = None):
    """
    Factory function to get either ChatOllama or ChatOpenAI based on settings.py
    """
    # Try to load API key from environment variable if settings.py has it empty
    api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    provider = LLM_PROVIDER.lower()
    
    if provider == "openai":
        if not api_key:
            print("⚠️ 경고: LLM_PROVIDER가 'openai'로 설정되었으나 OPENAI_API_KEY가 존재하지 않습니다.")
            print(" -> 환경 변수 'OPENAI_API_KEY'를 설정하거나 settings.py를 수정하세요.")
            print(" -> 임시로 Ollama 모델로 Fallback합니다.")
            provider = "ollama"
        else:
            try:
                from langchain_openai import ChatOpenAI
                kwargs = {
                    "model": OPENAI_LLM_MODEL,
                    "api_key": api_key,
                    "temperature": temperature,
                }
                if format == "json":
                    kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
                return ChatOpenAI(**kwargs)
            except ImportError:
                print("⚠️ 경고: 'langchain-openai' 패키지가 설치되지 않아 OpenAI를 사용할 수 없습니다.")
                print(" -> 'pip install langchain-openai'를 실행하세요.")
                print(" -> 임시로 Ollama 모델로 Fallback합니다.")
                provider = "ollama"
                
    if provider == "ollama":
        kwargs = {
            "model": OLLAMA_LLM_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "temperature": temperature,
        }
        if format == "json":
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)
        
    raise ValueError(f"지원하지 않는 LLM_PROVIDER입니다: {LLM_PROVIDER}")
