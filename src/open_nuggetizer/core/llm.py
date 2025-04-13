import os
import time
from typing import Dict, List, Optional, Union, Tuple
import tiktoken
from src.open_nuggetizer.utils.api import get_vllm_api_key
from openai import OpenAI

class LLMHandler:
    def __init__(
        self,
        model: str,
    ):
        self.model = model    
        self.api_key = get_vllm_api_key()
        self.client = self._initialize_client()
         
    def _initialize_client(self):
        try:    
            return OpenAI(
                base_url="http://localhost:8000/v1",
                api_key=self.api_key
            )
        except Exception as e:
            raise Exception(f"Error initializing client: {str(e)}")

    def run(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0
    ) -> Tuple[str, int]:
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=2048,
                    timeout=30
                )
                response = completion.choices[0].message.content

                # Get encoding length, fallback to default encoding if not available
                try:
                    encoding = tiktoken.get_encoding(self.model)
                except:
                    encoding = tiktoken.get_encoding("cl100k_base")
                return response, len(encoding.encode(response))
            except Exception as e:
                time.sleep(0.1)