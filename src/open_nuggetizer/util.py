from tqdm import tqdm
import re
import pandas as pd
import ast
from typing import List


def extract_list(text: str) -> List[str]:
    try:
        # Use regex to find the first occurrence of a list-like structure
        match = re.search(r"\[[^\[\]]+\]", text, re.DOTALL)
        if match:
            return ast.literal_eval(match.group(0))
    except Exception as _:
        raise ValueError("No valid Python list found in response.")


def iter_windows(n: int,
                 window_size: int,
                 stride: int,
                 verbose : bool = False,
                 desc: str = None):
    for start_idx in tqdm(range((n // stride) * stride, -1, -stride, desc=desc, disable=verbose), unit='window'):
        end_idx = start_idx + window_size
        if end_idx > n:
            end_idx = n
        window_len = end_idx - start_idx
        if start_idx == 0 or window_len > stride:
            yield start_idx, end_idx, window_len


def save_nuggets(nuggets: pd.DataFrame, file: str) -> None:
    essential = ["qid", "nugget_id", "nugget"]
    optional = ["importance", "assignment"]
    columns = nuggets.columns
    if any([x not in columns for x in essential]):
        raise ValueError("Require at least {essential} columns to save, found {columns}")

    for c in optional:
        if c not in columns:
            nuggets[c] = -1

    nuggets = nuggets[essential+optional]
    nuggets.to_csv(file, sep='\t', index=False, header=False)


def load_nuggets(file: str) -> pd.DataFrame:
    essential = ["qid", "nugget_id", "nugget", "importance", "assignment"]
    nuggets = pd.read_csv(file, sep='\t', index_col=False, names=essential)

    return nuggets


__all__ = ["extract_list", "iter_windows", "save_nuggets", "load_nuggets"]
