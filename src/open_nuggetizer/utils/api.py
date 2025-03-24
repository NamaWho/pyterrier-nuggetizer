import os
from dotenv import load_dotenv

def get_vllm_api_key() -> str:
    load_dotenv(dotenv_path=f".env")
    vllm_api_key = os.getenv("VLLM_API_KEY")
    return vllm_api_key