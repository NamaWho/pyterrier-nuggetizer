import os
import time
from typing import Dict, List, Optional, Union, Tuple
import tiktoken
from vllm import LLM, SamplingParams

class VLLMHandler:
    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 2048
    ):
        self.model = model
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens   
        self.llm = self._initialize_llm()
         
    def _initialize_llm(self):
        try:    
            self.llm = LLM(
                model=self.model,
                tokenizer=self.model,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            raise Exception(f"Error initializing llm: {str(e)}")

    def run_batch(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0,
        top_p: float = 0.95
    ) -> List[Tuple[str, int]]:

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_tokens
        )

        results = self.llm.generate(
            messages=messages,
            sampling_params=sampling_params
        )

        return [(res.outputs[0].text.strip(), len(res.outputs[0].token_ids)) for res in results]
