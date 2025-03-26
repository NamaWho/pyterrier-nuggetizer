import re
import ast
from typing import List

def extract_list(text: str) -> List[str]:
    """
    Extracts a Python list from a string.
    
    Args:
        text (str): The text to extract the list from.
    
    Returns:
        list[str]: The extracted Python list.
    """
    try:
        # Cerca la prima lista Python nel testo
        match = re.search(r"\[[^\[\]]+\]", text, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
    except Exception as e:
        pass
    raise ValueError("No valid Python list found in response.")
