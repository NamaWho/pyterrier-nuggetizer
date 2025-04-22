import pytest
import pandas as pd
from open_nuggetizer import Nuggetizer
from open_nuggetizer.nuggetizer import NuggetCreator
from open_nuggetizer._types import NuggetMode


class DummyBackend:
    def __init__(self):
        self.model_name_or_path = "dummy"

    def generate(self, prompts):
        # always produce two nuggets in a Python list
        return ['["alpha", "beta"]']


@pytest.fixture
def simple_df():
    return pd.DataFrame([{"qid": "Q1", "query": "test", "document": "d1"}])


def test_creator_basic(simple_df):
    backend = DummyBackend()
    nug = Nuggetizer(backend, max_nuggets=2)
    creator = NuggetCreator(nug)
    df_out = creator.transform(simple_df)
    # one row per query
    assert df_out.shape[0] == 1
    row = df_out.iloc[0]
    assert row["nugget"] == ["alpha", "beta"]
    assert row["nugget_id"] == ["Q1_1", "Q1_2"]
