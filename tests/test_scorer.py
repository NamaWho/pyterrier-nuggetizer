import pytest
import pandas as pd
from open_nuggetizer import Nuggetizer
from open_nuggetizer.nuggetizer import NuggetScorer
from open_nuggetizer._types import NuggetScoreMode


class DummyBackend:
    def __init__(self):
        self.model_name_or_path = "dummy"

    def generate(self, prompts):
        # always label two nuggets
        return ['["vital", "okay"]']


@pytest.fixture
def nuggets_df():
    return pd.DataFrame(
        [
            {
                "qid": "Q1",
                "query": "test",
                "nugget_id": ["Q1_1", "Q1_2"],
                "nugget": ["n1", "n2"],
            }
        ]
    )


def test_scorer_basic(nuggets_df):
    backend = DummyBackend()
    nug = Nuggetizer(backend, scorer_mode=NuggetScoreMode.VITAL_OKAY)
    scorer = NuggetScorer(nug)
    df_out = scorer.transform(nuggets_df())
    assert df_out.shape[0] == 1
    row = df_out.iloc[0]
    # mapping: vital→1, okay→0
    assert row["importance"] == [1, 0]
