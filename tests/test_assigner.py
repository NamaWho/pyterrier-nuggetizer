import pytest
import pandas as pd
from open_nuggetizer import Nuggetizer
from open_nuggetizer.nuggetizer import NuggetAssigner
from open_nuggetizer._types import NuggetAssignMode


class DummyBackend:
    def __init__(self):
        self.model_name_or_path = "dummy"

    def generate(self, prompts):
        # always assign two nuggets
        return ['["support", "not_support"]']


@pytest.fixture
def scored_df():
    return pd.DataFrame(
        [
            {
                "qid": "Q1",
                "query": "test",
                "qanswer": "ans",
                "nugget_id": ["Q1_1", "Q1_2"],
                "nugget": ["n1", "n2"],
                "importance": [1, 0],
            }
        ]
    )


def test_assigner_grade2(scored_df):
    backend = DummyBackend()
    nug = Nuggetizer(backend, assigner_mode=NuggetAssignMode.SUPPORT_GRADE_2, window_size=1)
    assigner = NuggetAssigner(nug)
    df_out = assigner.transform(scored_df)
    assert df_out["vital"].tolist() == [1, 0]
