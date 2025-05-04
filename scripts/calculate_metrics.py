#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

from open_nuggetizer.util import load_nuggets
from open_nuggetizer import Nuggetizer
from pyterrier_rag import VLLMBackend


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for nugget assignments')
    parser.add_argument('--input_file', type=str, help='Path to input TSV file with assignments')
    parser.add_argument('--nugget_file', type=str, help='Path to input JSONL file with assignments')
    parser.add_argument('--output_file', type=str, help='Path to output TSV file')
    parser.add_argument('--model_name_or_path', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='Model to use for all operations')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for processing')
    parser.add_argument('--max_nuggets', type=int, default=30, help='Maximum number of nuggets to extract')
    args = parser.parse_args()

    nuggets = load_nuggets(args.nugget_file)
    answers = load_answers(args.input_file)
    backend = VLLMBackend(args.model_name_or_path)
    nuggetizer = Nuggetizer(
        backend,
        window_size=args.window_size,
        max_nuggets=args.max_nuggets
    )

if __name__ == '__main__':
    main() 