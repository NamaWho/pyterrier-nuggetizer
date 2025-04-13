import re
import ast
from typing import List

def extract_list(text: str) -> List[str]:
    try:
        # Use regex to find the first occurrence of a list-like structure
        match = re.search(r"\[[^\[\]]+\]", text, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
    except Exception as e:
        pass
    raise ValueError("No valid Python list found in response.")
