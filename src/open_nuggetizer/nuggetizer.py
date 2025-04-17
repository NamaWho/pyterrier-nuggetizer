from typing import Optional, Iterable, List
import logging

import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_rag.prompt import PromptTransformer
from pyterrier_rag.llm import LLM
import pandas as pd

from open_nuggetizer._types import NuggetMode, NuggetScoreMode
from open_nuggetizer.prompts import (
    CREATOR_PROMPT_STRING,
    SCORER_PROMPT_STRING,
)
from open_nuggetizer.util import iter_windows, extract_list


class Nuggetizer(pt.Transformer):
    """
    A transformer pipeline component that generates and scores information nuggets
    relevant to a query using a large language model (LLM).

    Parameters:
        llm (LLM): Language model instance with generate capability
        creator_mode (NuggetMode): Strategy for nugget creation (atomic/noun-phrase/question)
        scorer_mode (NuggetScoreMode): Grading of nugget storing (2 / 3)
        window_size (int, optional): Global window size for document/nugget processing
        creator_window_size (int): Documents per window for creation phase
        scorer_window_size (int): Nuggets per window for scoring phase
        max_nuggets (int): Maximum nuggets to generate per query
        query_field (str): DataFrame column containing queries
        document_field (str): DataFrame column containing documents
        nugget_field (str): Output column for generated nuggets
        score_field (str): Output column for importance scores
        verbose (bool): Enable verbose logging
    """

    def __init__(
        self,
        llm: LLM,
        creator_mode: NuggetMode = NuggetMode.ATOMIC,
        scorer_mode: NuggetScoreMode = NuggetScoreMode.VITAL_OKAY,
        window_size: Optional[int] = None,
        creator_window_size: Optional[int] = 10,
        scorer_window_size: Optional[int] = 10,
        max_nuggets: Optional[int] = 30,
        query_field: str = "query",
        document_field: str = "document",
        nugget_field: str = "nugget",
        score_field: str = "importance",
        verbose: bool = False,
    ):
        assert hasattr(llm, "generate"), "llm must have a generate method"
        assert window_size is None or isinstance(
            window_size, int
        ), "window_size must be an integer"
        assert creator_window_size is None or isinstance(
            creator_window_size, int
        ), "creator_window_size must be an integer"
        assert scorer_window_size is None or isinstance(
            scorer_window_size, int
        ), "scorer_window_size must be an integer"

        self.llm = llm
        self.creator_mode = creator_mode
        self.scorer_mode = scorer_mode
        self.window_size = window_size
        self.creator_window_size = creator_window_size
        self.scorer_window_size = scorer_window_size
        self.max_nuggets = max_nuggets
        self.query_field = query_field
        self.document_field = document_field
        self.nugget_field = nugget_field
        self.score_field = score_field
        self.verbose = verbose

    def __post_init__(self):
        if self.window_size:
            self.creator_window_size = self.window_size
            self.scorer_window_size = self.window_size

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    def generate(self, inp: Iterable[str]):
        return self.llm.generate(inp)

    def create(self, inp: pd.DataFrame) -> pd.DataFrame:
        return NuggetCreator(self)(inp)

    def score(self, inp: pd.DataFrame) -> pd.DataFrame:
        return NuggetScorer(self)(inp)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        columns = inp.columns
        if any([x not in columns for x in [self.query_field, self.document_field]]):
            raise ValueError(
                "DataFrame appears to be malformatted, minimum expected columns [{self.query_field}, {self.document_field}], got {columns}"
            )

        if self.nugget_field not in columns:
            inp = self.create(inp)
            return self.score(inp)
        else:
            return self.score(inp)


class NuggetCreator(pt.Transformer):
    """
    Transformer component that generates query-relevant information nuggets from documents.

    Parameters:
        nuggetizer (Nuggetizer): Parent nuggetizer instance
        mode (NuggetMode): Override for nugget creation strategy
        window_size (int, optional): Override for document processing window size
        verbose (bool, optional): Override for verbose logging

    Attributes:
        system_message (str): The provided system message to the LLM
        prompt (PromptTransformer): Configured prompt transformation pipeline
    """

    system_message: str = (
        "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
    )

    def __init__(
        self,
        nuggetizer: Nuggetizer,
        mode: NuggetMode = None,
        window_size: Optional[int] = None,
        verbose: bool = None,
    ):
        assert nuggetizer is not None, "nuggetizer must be provided"
        assert isinstance(
            nuggetizer, Nuggetizer
        ), "nuggetizer must be an instance of Nuggetizer"
        assert window_size is None or isinstance(
            window_size, int
        ), "window_size must be an integer"

        self.nuggetizer = nuggetizer
        self.mode = mode if mode else nuggetizer.creator_mode
        self.window_size = (
            window_size if window_size else nuggetizer.creator_window_size
        )
        self.query_field = nuggetizer.query_field
        self.document_field = nuggetizer.document_field
        self.nugget_field = nuggetizer.nugget_field
        self.max_nuggets = nuggetizer.max_nuggets
        self.verbose = verbose if verbose is not None else nuggetizer.verbose

        self.prompt = PromptTransformer(
            instruction=CREATOR_PROMPT_STRING,
            system_message=self.system_message,
            model_name_or_path=self.nuggetizer.llm.model_name_or_path,
            answer_extraction=extract_list,
            output_field=self.nugget_field,
            input_fields=[
                self.query_field,
                "context_documents",
                self.nugget_field,
                "max_nuggets",
            ],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0].get("qid", None)
        query = inp[0][self.query_field]
        documents = inp[self.document_field]

        nuggets: List[str] = []

        for start, end, _ in iter_windows(
            len(documents), self.window_size, self.window_size, verbose=self.verbose
        ):
            current_documents = documents[start:end]
            context_string = "\n".join(
                [f"[{i+1}] {doc}" for i, doc in enumerate(current_documents)]
            )
            context = {
                self.query_field: query,
                "context_documents": context_string,
                self.nugget_field: nuggets,
                "max_nuggets": self.max_nuggets,
            }
            prompt = [self.prompt.create_prompt(**context)]
            output = self.nuggetizer.generate(prompt)[0]
            nuggets = self.prompt.answer_extraction(output)[: self.max_nuggets]

        return [
            {
                "qid": qid,
                self.query_field: query,
                f"{self.nugget_field}_id": [
                    f"{qid}_{i+1}" for i in range(len(nuggets))
                ],
                self.nugget_field: nuggets,
            }
        ]


class NuggetScorer(pt.Transformer):
    """
    Transformer component that scores nuggets based on their query importance.

    Parameters:
        nuggetizer (Nuggetizer): Parent nuggetizer instance
        mode (NuggetScoreMode): Override for scoring strategy
        window_size (int, optional): Override for nugget processing window size
        max_nuggets (int, optional): Override for maximum nuggets to process
        verbose (bool, optional): Override for verbose logging

    Attributes:
        system_message (str): The provided system message to the LLM
        prompt (PromptTransformer): Configured prompt transformation pipeline
    """

    system_message: str = (
        "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."
    )

    def __init__(
        self,
        nuggetizer: Nuggetizer,
        mode: NuggetScoreMode = None,
        window_size: Optional[int] = None,
        max_nuggets: Optional[int] = None,
        verbose: bool = None,
    ):
        assert nuggetizer is not None, "nuggetizer must be provided"
        assert isinstance(
            nuggetizer, Nuggetizer
        ), "nuggetizer must be an instance of Nuggetizer"
        assert window_size is None or isinstance(
            window_size, int
        ), "window_size must be an integer"
        assert max_nuggets is None or isinstance(
            max_nuggets, int
        ), "max_nuggets must be an integer"

        self.nuggetizer = nuggetizer
        self.mode = mode if mode else nuggetizer.scorer_mode
        self.window_size = window_size if window_size else nuggetizer.scorer_window_size
        self.max_nuggets = max_nuggets if max_nuggets else nuggetizer.max_nuggets
        self.query_field = nuggetizer.query_field
        self.nugget_field = nuggetizer.nugget_field
        self.score_field = nuggetizer.score_field

        self.verbose = verbose if verbose is not None else nuggetizer.verbose

        self.prompt = PromptTransformer(
            instruction=SCORER_PROMPT_STRING,
            system_message=self.system_message,
            model_name_or_path=self.nuggetizer.llm.model_name_or_path,
            answer_extraction=extract_list,
            output_field=self.score_field,
            input_fields=[self.query_field, self.nugget_field],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0].get("qid", None)
        query = inp[0][self.query_field]
        nugget_ids = inp[f"{self.nugget_field}_id"]
        nuggets = inp[self.nugget_field]

        scores: List[str] = []

        for start, end, _ in iter_windows(
            len(nuggets), self.window_size, self.window_size, verbose=self.verbose
        ):
            current_nuggets = nuggets[start:end]
            context_string = "\n".join(
                [f"[{i+1}] {nug}" for i, nug in enumerate(current_nuggets)]
            )
            context = {
                self.query_field: query,
                self.nugget_field: context_string,
            }
            prompt = [self.prompt.create_prompt(**context)]
            output = self.nuggetizer.generate(prompt)[0]
            scores.extend(self.prompt.answer_extraction(output))

        return [
            {
                "qid": qid,
                self.query_field: query,
                f"{self.nugget_field}_id": nugget_ids,
                self.nugget_field: nuggets,
                self.score_field: scores,
            }
        ]


__all__ = ["Nuggetizer", "NuggetCreator", "NuggetScorer"]
