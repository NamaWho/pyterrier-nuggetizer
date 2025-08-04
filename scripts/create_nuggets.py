import logging
import argparse

import pyterrier as pt
from open_nuggetizer import Nuggetizer
from open_nuggetizer.util import save_nuggets
from pyterrier_rag import VLLMBackend


def setup_logging(log_level: int) -> None:
    """Configure logging based on verbosity level."""
    logging_level = logging.WARNING
    if log_level >= 2:
        logging_level = logging.DEBUG
    elif log_level >= 1:
        logging_level = logging.INFO

    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description='Extract and score nuggets from input JSONL file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input run file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output nugget file')
    parser.add_argument('--model_name_or_path', type=str, default='mistralai/Mistral-7B-Instruct-v0.3', help='Model to use for all operations')
    parser.add_argument('--window_size', type=int, default=10, help='Window size for processing')
    parser.add_argument('--max_nuggets', type=int, default=30, help='Maximum number of nuggets to extract')
    parser.add_argument('--log_level', type=int, default=0, choices=[0, 1, 2],
                      help='Logging level: 0=warnings only, 1=info, 2=debug')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    run_file = pt.io.read_results(args.input_file)

    # Initialize nuggetizer with all configurations
    logger.info("Initializing Nuggetizer")
    backend = VLLMBackend(args.model_name_or_path)
    nuggetizer = Nuggetizer(
        backend,
        window_size=args.window_size,
        max_nuggets=args.max_nuggets
    )

    nuggets = nuggetizer(run_file)
    save_nuggets(nuggets, args.output_file)

    logger.info("Processing complete")


if __name__ == '__main__':
    main()
