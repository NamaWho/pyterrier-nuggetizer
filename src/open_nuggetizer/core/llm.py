import os
import time
from typing import Dict, List, Optional, Union, Tuple
import tiktoken
from ..utils.api import get_vllm_api_key
from openai import OpenAI
from vllm import LLM 

class LLMHandler:
    def __init__(
        self,
        model: str,
        api_keys: Optional[Union[str, List[str]]] = None,
    ):
        self.model = model    
        self.current_key_idx = 0
        api_keys = api_keys or get_vllm_api_key()
        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.client = self._initialize_client()
         
    def _initialize_client(self):
        try:    
            return OpenAI(
                base_url="http://localhost:8080/v1",
                api_key=self.api_keys[0]
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
                print(f"Error: {str(e)}")
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                self.client.api_key = self.api_keys[self.current_key_idx]
                time.sleep(0.1)