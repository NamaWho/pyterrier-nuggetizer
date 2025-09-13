# PyTerrier Nuggetizer

**PyTerrier Nuggetizer** is an open-source library for creating and scoring *nuggets* (atomic, verifiable information units) used in **retrieval-augmented generation (RAG)** and **evaluation**.  
It is inspired by the original [Nuggetizer framework](https://arxiv.org/pdf/2411.09607) but provides simpler APIs and direct integration with PyTerrier.

---

## ✨ Features

- Automatic nugget creation from LLM-generated responses  
- Nugget scoring and assignment with multiple modes (`SUPPORT_GRADE_3`, etc.)  
- Integration with PyTerrier and IR datasets (e.g., MS MARCO, TREC RAG 2024)  
- Compatible with **OpenAI API-like backends** and HuggingFace models (e.g., LLaMA, Qwen)  
- Easily extensible pipelines for RAG evaluation  

---

## 🚀 Installation

```bash
git clone https://github.com/NamaWho/pyterrier-nuggetizer.git
cd pyterrier-nuggetizer
pip install -e .
```

## 🔧 Quickstart
### 1. Define the backend

```python
from pyterrier_rag.backend import OpenAIBackend
from transformers import AutoTokenizer
import os

model_name = "llama-3.3-70b-instruct"
tokenizer = AutoTokenizer.from_pretrained("casperhansen/llama-3.3-70b-instruct-awq")

backend = OpenAIBackend(
    model_name,
    api_key=<YOUR_API_KEY>,
    base_url=<<YOUR_BASE_URL>,
    generation_args={"temperature": 0.6, "max_tokens": 256},
    verbose=True,
    parallel=64,
)
```

### 2. Initialize the Nuggetizer
```python
from pyterrier_nuggetizer.nuggetizer import Nuggetizer
from pyterrier_nuggetizer._types import NuggetAssignMode
from fastchat.model import get_conversation_template

conv_template = get_conversation_template("meta-llama-3.1-sp")

nuggetizer = Nuggetizer(
    backend=backend,
    conversation_template=conv_template,
    verbose=True,
    assigner_mode=NuggetAssignMode.SUPPORT_GRADE_3
)
```

### 3. Create and score nuggets
```python
# Nugget creation
nuggets = nuggetizer.create(df_responses)   # df_responses = DataFrame with [qid, query, docno, text]

# Nugget scoring
scored_nuggets = nuggetizer.score(nuggets)

# Save results
scored_nuggets.to_csv("scored_nuggets.csv", index=False)
```

### 4. Use in a PyTerrier RAG pipeline
```python
import pyterrier as pt
from pyterrier_rag.prompt import Concatenator, PromptTransformer
from pyterrier_rag.readers import Reader
from jinja2 import Template

prompt = PromptTransformer(
    instruction=lambda **kwargs: Template(
        "Use the context to answer:\n Context: {{ context }}\n Question: {{ query }}\n Answer:"
    ).render(**kwargs),
    system_message="You are a helpful assistant.",
    conversation_template=conv_template,
    input_fields=["qcontext", "query"],
)

reader = Reader(backend, prompt)
rag_pipeline = (retrieval_stage >> Concatenator() >> reader)
results = rag_pipeline(df_queries)
```

---

**Contributing:**

We welcome contributions from the community to enhance OpenNuggetizer's capabilities. Please refer to our contribution guidelines for more information.

**License:**

This project is licensed under the Apache License v2.0.

