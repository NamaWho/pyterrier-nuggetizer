from typing import Optional, Iterable, List, Set, Dict
import logging

import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_rag.prompt import PromptTransformer
from pyterrier_rag.backend import Backend
import pandas as pd

from open_nuggetizer._types import NuggetAssignMode
from open_nuggetizer.prompts import (
    CREATOR_PROMPT_STRING,
    SCORER_PROMPT_STRING,
    ASSIGNER_GRADE_2_PROMPT_STRING,
    ASSIGNER_GRADE_3_PROMPT_STRING,
    make_callable_template,
)
from open_nuggetizer._measure import measure_factory
from open_nuggetizer.util import iter_windows, extract_list


class Nuggetizer(pt.Transformer):
    """
    A pipeline component that generates and scores information nuggets
    relevant to a query using a large language model (LLM).

    Parameters:
        backend (Backend): The LLM backend to use for nugget generation and scoring.
        assigner_mode (NuggetAssignMode): Mode for nugget assignment strategy.
        window_size (int, optional): Size of the document processing window.
        creator_window_size (int, optional): Size of the nugget creation window.
        scorer_window_size (int, optional): Size of the nugget scoring window.
        assigner_window_size (int, optional): Size of the nugget assignment window.
        max_nuggets (int, optional): Maximum number of nuggets to generate.
        query_field (str, optional): Name of the query field in input DataFrame.
        document_field (str, optional): Name of the document field in input DataFrame.
        answer_field (str, optional): Name of the answer field in input DataFrame.
        nugget_field (str, optional): Name of the nugget field in output DataFrame.
        importance_field (str, optional): Name of the score field in output DataFrame.
        assignment_field (str, optional): Name of the assignment field in output DataFrame.
        verbose (bool, optional): Whether to enable verbose logging.
    """

    def __init__(
        self,
        backend: Backend,
        # creator_mode: NuggetMode = NuggetMode.ATOMIC,
        # scorer_mode: NuggetScoreMode = NuggetScoreMode.VITAL_OKAY,
        assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_2,
        window_size: Optional[int] = None,
        creator_window_size: Optional[int] = 10,
        scorer_window_size: Optional[int] = 10,
        assigner_window_size: Optional[int] = 10,
        max_nuggets: Optional[int] = 30,
        query_field: Optional[str] = "query",
        document_field: Optional[str] = "text",
        answer_field: Optional[str] = "qanswer",
        nugget_field: Optional[str] = "nugget",
        importance_field: Optional[str] = "importance",
        assignment_field: Optional[str] = "assignment",
        verbose: Optional[bool] = False,
    ):
        assert hasattr(backend, "generate"), "backend must have a generate method"
        assert window_size is None or isinstance(
            window_size, int
        ), "window_size must be an integer"
        assert creator_window_size is None or isinstance(
            creator_window_size, int
        ), "creator_window_size must be an integer"
        assert scorer_window_size is None or isinstance(
            scorer_window_size, int
        ), "scorer_window_size must be an integer"
        assert assigner_window_size is None or isinstance(
            assigner_window_size, int
        ), "assigner_window_size must be an integer"

        self.backend = backend
        self.assigner_mode = assigner_mode
        self.window_size = window_size
        self.creator_window_size = creator_window_size
        self.scorer_window_size = scorer_window_size
        self.assigner_window_size = assigner_window_size
        self.max_nuggets = max_nuggets
        self.query_field = query_field
        self.document_field = document_field
        self.answer_field = answer_field
        self.nugget_field = nugget_field
        self.importance_field = importance_field
        self.assignment_field = assignment_field
        self.verbose = verbose

        self.__post_init__()

    def __post_init__(self):
        if self.window_size:
            self.creator_window_size = self.window_size
            self.scorer_window_size = self.window_size
            self.assigner_window_size = self.window_size

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    def __repr__(self):
        return f"Nuggetizer(backend={self.backend}, assigner_mode={self.assigner_mode}, window_size={self.window_size}, max_nuggets={self.max_nuggets})"

    def generate(self, inp: Iterable[str]):
        return self.backend.generate(inp)

    def create(self, inp: pd.DataFrame) -> pd.DataFrame:
        return NuggetCreator(self)(inp)

    def score(self, inp: pd.DataFrame) -> pd.DataFrame:
        return NuggetScorer(self)(inp)

    def assign(self, inp: pd.DataFrame) -> pd.DataFrame:
        return NuggetAssigner(self)(inp)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        columns = inp.columns
        if any([x not in columns for x in [self.query_field, self.document_field]]):
            raise ValueError(
                f"DataFrame appears to be malformatted, minimum expected columns [{self.query_field}, {self.document_field}], got {columns}"
            )

        if self.nugget_field not in columns:
            inp = self.create(inp)
            return self.score(inp)
        if self.answer_field in columns:
            return self.assign(inp)
        else:
            return self.score(inp)

    def __getattr__(self, attr: str):
        measure = measure_factory(attr, self)
        if measure is not None:
            return measure
        return self.__getattribute__(attr)

    def make_qrels(self, run: pd.DataFrame, nuggets: pd.DataFrame) -> pd.DataFrame:
        # 1) validate inputs
        required_run: Set[str] = {self.query_field, self.answer_field}
        if not required_run.issubset(run.columns):
            raise ValueError(
                f"run must contain columns {required_run}, got {list(run.columns)}"
            )
        required_nuggets: Set[str] = {
            self.query_field,
            f"{self.nugget_field}_id",
            self.nugget_field,
            self.importance_field,
        }
        if not required_nuggets.issubset(nuggets.columns):
            raise ValueError(
                f"nuggets must contain columns {required_nuggets}, got {list(nuggets.columns)}"
            )

        df = pd.merge(
            run[[self.query_field, self.answer_field]],
            nuggets,
            on=self.query_field,
            how="inner",
        )

        assigned = NuggetAssigner(self).transform(df)

        list_cols = [
            self.answer_field,
            f"{self.nugget_field}_id",
            self.nugget_field,
            self.importance_field,
            self.assignment_field,
        ]
        for c in list_cols:
            if assigned[c].apply(lambda x: isinstance(x, list)).any():
                assigned = assigned.explode(c)

        # 5) map textual vitalness → binary relevance
        def to_rel(x: str) -> int:
            xl = str(x).strip().lower()
            # adjust these tests to match your LLM’s exact outputs
            return (
                1
                if ("vital" in xl or "support" in xl or xl in {"1", "true", "yes"})
                else 0
            )

        assigned["relevance"] = assigned[self.assignment_field].map(to_rel)
        assigned = assigned.rename(columns={'nugget_id': 'doc_id'})

        # 6) produce standard qrels: query_id, doc_id, relevance
        qrels = assigned.rename(
            columns={self.query_field: "query_id", f"{self.nugget_field}_id": "doc_id"}
        )[["query_id", "doc_id", "relevance"]].reset_index(drop=True)
        return qrels


class NuggetCreator(pt.Transformer):
    """
    Component that generates query-relevant information nuggets from documents.

    Parameters:
        nuggetizer (Nuggetizer): Parent nuggetizer instance
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
        self.window_size = (
            window_size if window_size else nuggetizer.creator_window_size
        )
        self.query_field = nuggetizer.query_field
        self.document_field = nuggetizer.document_field
        self.nugget_field = nuggetizer.nugget_field
        self.max_nuggets = nuggetizer.max_nuggets
        self.verbose = verbose if verbose is not None else nuggetizer.verbose

        self.__post_init__()

    def __post_init__(self):
        self.prompt = PromptTransformer(
            instruction=make_callable_template(CREATOR_PROMPT_STRING),
            system_message=self.system_message,
            model_name_or_path=self.nuggetizer.backend.model_name_or_path,
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
        documents = [i[self.document_field] for i in inp]

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
            prompt = [self.prompt.create_prompt(context)]
            output = self.nuggetizer.generate(prompt)[0]
            nuggets = self.prompt.answer_extraction(output.text)[: self.max_nuggets]

        if len(nuggets) == 0:
            logging.warning("No Nuggets Generated")

        return [
            {
                "qid": qid,
                self.query_field: query,
                f"{self.nugget_field}_id": f"{qid}_{i+1}",
                self.nugget_field: nugget,
            } for i, nugget in enumerate(nuggets)
        ]


class NuggetScorer(pt.Transformer):
    """
    Component that scores nuggets based on their query importance.

    Parameters:
        nuggetizer (Nuggetizer): Parent nuggetizer instance
        window_size (int, optional): Override for nugget processing window size
        max_nuggets (int, optional): Override for maximum nuggets to process
        verbose (bool, optional): Override for verbose logging

    Attributes:
        system_message (str): The provided system message to the LLM
        mapping (dict): Mapping of labels to scores
        prompt (PromptTransformer): Configured prompt transformation pipeline
    """

    system_message: str = (
        "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."
    )

    mapping: Dict[str, int] = {
        "vital": 1,
        "okay": 0,
    }

    def __init__(
        self,
        nuggetizer: Nuggetizer,
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
        self.window_size = window_size if window_size else nuggetizer.scorer_window_size
        self.max_nuggets = max_nuggets if max_nuggets else nuggetizer.max_nuggets
        self.query_field = nuggetizer.query_field
        self.nugget_field = nuggetizer.nugget_field
        self.importance_field = nuggetizer.importance_field

        self.verbose = verbose if verbose is not None else nuggetizer.verbose

        self.__post_init__()

    def __post_init__(self):
        self.prompt = PromptTransformer(
            instruction=make_callable_template(SCORER_PROMPT_STRING),
            system_message=self.system_message,
            model_name_or_path=self.nuggetizer.backend.model_name_or_path,
            answer_extraction=extract_list,
            output_field=self.importance_field,
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
        nugget_ids = [i[f"{self.nugget_field}_id"] for i in inp]
        nuggets = [i[self.nugget_field] for i in inp]

        importance_scores: List[str] = []

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
            prompt = [self.prompt.create_prompt(context)]
            output = self.nuggetizer.generate(prompt)[0]
            importance_scores.extend(self.prompt.answer_extraction(output.text))
        importance_scores = [self.mapping.get(x.lower(), 0) for x in importance_scores]

        return [
            {
                "qid": qid,
                self.query_field: query,
                f"{self.nugget_field}_id": idx,
                self.nugget_field: nugget,
                self.importance_field: importance_score,
            } for idx, nugget, importance_score in zip(nugget_ids, nuggets, importance_scores)
        ]


class NuggetAssigner(pt.Transformer):
    """
    Component that assigns nuggets based on their supporting an answer

    Parameters:
        nuggetizer (Nuggetizer): Parent nuggetizer instance
        mode (NuggetAssignMode): Override for assigning strategy
        window_size (int, optional): Override for nugget processing window size
        verbose (bool, optional): Override for verbose logging

    Attributes:
        system_message (str): The provided system message to the LLM
        prompt (PromptTransformer): Configured prompt transformation pipeline
    """

    system_message: str = (
        "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage."
    )

    def __init__(
        self,
        nuggetizer: Nuggetizer,
        mode: NuggetAssignMode = None,
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
        self.mode = mode if mode else nuggetizer.assigner_mode
        self.window_size = window_size if window_size else nuggetizer.assigner_window_size
        self.query_field = nuggetizer.query_field
        self.nugget_field = nuggetizer.nugget_field
        self.importance_field = nuggetizer.importance_field
        self.answer_field = nuggetizer.answer_field
        self.assignment_field = nuggetizer.assignment_field

        self.verbose = verbose if verbose is not None else nuggetizer.verbose

        self.__post_init__()

    def __post_init__(self):
        instruction = (
            ASSIGNER_GRADE_2_PROMPT_STRING
            if self.mode == NuggetAssignMode.SUPPORT_GRADE_2
            else ASSIGNER_GRADE_3_PROMPT_STRING
        )
        self.prompt = PromptTransformer(
            instruction=make_callable_template(instruction),
            system_message=self.system_message,
            model_name_or_path=self.nuggetizer.backend.model_name_or_path,
            answer_extraction=extract_list,
            output_field=self.assignment_field,
            input_fields=[self.query_field, "context", "nuggets"],
        )

        if self.mode == NuggetAssignMode.SUPPORT_GRADE_2:
            self.mapping = {
                "support": 1,
                "not_support": 0,
            }
        else:
            self.mapping = {
                "support": 2,
                "partial_support": 1,
                "not_support": 0,
            }

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0].get("qid", None)
        query = inp[0][self.query_field]
        qanswer = inp[0][self.answer_field]
        nugget_ids = [i[f"{self.nugget_field}_id"] for i in inp]
        nuggets = [i[self.nugget_field] for i in inp]
        importance = [i[self.importance_field] for i in inp]

        assignments: List[str] = []

        for start, end, _ in iter_windows(
            len(nuggets), self.window_size, self.window_size, verbose=self.verbose
        ):
            current_nuggets = nuggets[start:end]
            context = {
                self.query_field: query,
                "nuggets": current_nuggets,
                "context": qanswer,
            }
            prompt = [self.prompt.create_prompt(context)]
            output = self.nuggetizer.generate(prompt)[0]
            assignments.extend(self.prompt.answer_extraction(output))
        assignments = [self.mapping.get(x.lower(), 0) for x in assignments]
    
        return [
            {
                "qid": qid,
                self.query_field: query,
                self.answer_field: qanswer,
                f"{self.nugget_field}_id": idx,
                self.nugget_field: nugget,
                self.importance_field: important,
                self.assignment_field: assignment,
            } for idx, nugget, important, assignment in zip(nugget_ids, nuggets, importance, assignments)
        ]


__all__ = ["Nuggetizer", "NuggetCreator", "NuggetScorer", "NuggetAssigner"]
