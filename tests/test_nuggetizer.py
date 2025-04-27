import pytest
import pandas as pd
from open_nuggetizer import Nuggetizer
from open_nuggetizer._types import NuggetAssignMode

from types import SimpleNamespace

class DummyBackend:
    def __init__(self):
        self.model_name_or_path = "dummy-model"

    def generate(self, prompts):
        text = prompts[0]
        if "NuggetizeLLM" in text:
            return [SimpleNamespace(text='["nugget1", "nugget2"]')]
        elif "NuggetizeScoreLLM" in text:
            return [SimpleNamespace(text='["vital", "okay"]')]
        else:
            return [SimpleNamespace(text='["support", "not_support"]')]


@pytest.fixture
def df_docs():
    return pd.DataFrame(
        [
            {"qid": "1", "query": "Q1", "text": "DocA"},
            {"qid": "1", "query": "Q1", "text": "DocB"},
        ]
    )


def test_initialization_and_transform_errors():
    backend = DummyBackend()
    nug = Nuggetizer(backend, window_size=1)
    # missing required columns → ValueError
    with pytest.raises(ValueError):
        nug.transform(pd.DataFrame({"foo": [1]}))


def test_full_pipeline(df_docs):
    backend = DummyBackend()
    nug = Nuggetizer(
        backend,
        assigner_mode=NuggetAssignMode.SUPPORT_GRADE_2,
        max_nuggets=2,
        window_size=1,
    )
    df_out = nug.transform(df_docs)
    # should have the final qrels style output
    assert len(set(df_out.columns)) >= len({"qid", "docno", "relevance"})
    # two nuggets → two distinct nugget_ids
    assert df_out["nugget_id"].nunique() == 2
    # relevance should be 1 or 0
    assert sorted(df_out["importance"].unique()) == [0, 1]
