import pytest
import pandas as pd
from open_nuggetizer import Nuggetizer
from open_nuggetizer._types import NuggetMode, NuggetScoreMode, NuggetAssignMode


class DummyBackend:
    def __init__(self):
        self.model_name_or_path = "dummy-model"

    def generate(self, prompts):
        text = prompts[0]
        if "NuggetizeLLM" in text:
            return ['["nugget1", "nugget2"]']
        elif "NuggetizeScoreLLM" in text:
            return ['["vital", "okay"]']
        else:
            return ['["support", "not_support"]']


@pytest.fixture
def df_docs():
    return pd.DataFrame(
        [
            {"query": "Q1", "document": "DocA"},
            {"query": "Q1", "document": "DocB"},
        ]
    )


def test_initialization_and_transform_errors():
    backend = DummyBackend()
    nug = Nuggetizer(backend)
    # missing required columns → ValueError
    with pytest.raises(ValueError):
        nug.transform(pd.DataFrame({"foo": [1]}))


def test_full_pipeline(df_docs):
    backend = DummyBackend()
    nug = Nuggetizer(
        backend,
        assigner_mode=NuggetAssignMode.SUPPORT_GRADE_2,
        max_nuggets=2,
    )
    df_out = nug.transform(df_docs)
    # should have the final qrels style output
    assert set(df_out.columns) >= {"query_id", "doc_id", "relevance"}
    # two nuggets → two distinct doc_ids
    assert df_out["doc_id"].nunique() == 2
    # relevance should be 1 or 0
    assert sorted(df_out["relevance"].unique()) == [0, 1]
