import pytest
import pandas as pd
from pyterrier_nuggetizer import Nuggetizer
from pyterrier_nuggetizer.nuggetizer import NuggetCreator

from types import SimpleNamespace


class DummyBackend:
    def __init__(self):
        self.model_name_or_path = "dummy"

    def generate(self, prompts):
        # always produce two nuggets in a Python list
        return [SimpleNamespace(text='["alpha", "beta"]')]

@pytest.fixture
def simple_df():
    return pd.DataFrame([{"qid": "Q1", "query": "test", "text": "d1"}])


def test_creator_basic(simple_df):
    backend = DummyBackend()
    nug = Nuggetizer(backend, max_nuggets=2, window_size=1)
    creator = NuggetCreator(nug)
    df_out = creator.transform(simple_df)
    assert df_out.shape[0] == 2
    row = df_out.iloc[0]
    assert df_out["nugget"].tolist() == ["alpha", "beta"]
    assert df_out["nugget_id"].tolist() == ["Q1_1", "Q1_2"]
