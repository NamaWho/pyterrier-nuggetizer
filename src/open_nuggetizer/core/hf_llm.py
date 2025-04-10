from vllm.sampling_params import SamplingParams
from outlines import models, generate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.open_nuggetizer.core.types import Response
import torch

_LLM_MODELS = {}

class HFHandler:
    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 2048
    ):
        self.model_name = model
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens   
        self._initialize_llm()
         
    def _initialize_llm(self):
        if self.model_name in _LLM_MODELS:
            self.llm = _LLM_MODELS[self.model_name]
            return
        
        try:    
            # self.llm = models.vllm(model_name=self.model, tokenizer=self.model, dtype=self.dtype, gpu_memory_utilization=self.gpu_memory_utilization)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
            
            self.llm = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", quantization_config=quantization_config, torch_dtype=self.dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = models.Transformers(self.llm, self.tokenizer)
            self.generator = generate.json(self.model, Response)

            _LLM_MODELS[self.model] = self.llm
        except Exception as e:
            raise Exception(f"Error initializing llm: {str(e)}")

    def run (
        self, 
        prompt: str,
        temperature: float = 0,
        top_p: float = 0.95
    # ) -> List[Tuple[str, int]]:
    ) -> Response:

        # results = self.llm.generate(
        #     prompts=prompts,
        #     sampling_params=sampling_params
        # )

        response = self.generator(prompt)
        return response