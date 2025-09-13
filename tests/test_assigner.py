import pytest
import pandas as pd
from pyterrier_nuggetizer import Nuggetizer
from pyterrier_nuggetizer.nuggetizer import NuggetAssigner
from pyterrier_nuggetizer._types import NuggetAssignMode

class DummyBackendGrade2:
    def __init__(self):
        self.model_name_or_path = "dummy"

    def generate(self, prompts):
        # always assign two nuggets
        return ['["support", "not_support", "support"]']

class DummyBackendGrade3:
    def __init__(self):
        self.model_name_or_path = "dummy"

    def generate(self, prompts):
        # always assign three nuggets
        return ['["support", "not_support", "partial_support"]']

@pytest.fixture
def scored_df():
    return pd.DataFrame(
        {
            "qid":       ["Q1",    "Q1", "Q1"],
            "query":     ["test",  "test", "test"],
            "qanswer":   ["ans",   "ans", "ans"],
            "nugget_id": ["Q1_1",  "Q1_2", "Q1_3"],
            "nugget":    ["n1",    "n2", "n3"],
            "importance":[1,       0,       0],
        }
    )


def test_assigner_grade2(scored_df):
    backend = DummyBackendGrade2()
    nug = Nuggetizer(backend, assigner_mode=NuggetAssignMode.SUPPORT_GRADE_2, window_size=1)
    assigner = NuggetAssigner(nug)
    df_out = assigner.transform(scored_df)
    assert df_out["assignment"].tolist() == [1, 0, 1]

def test_assigner_grade3(scored_df):
    backend = DummyBackendGrade3()
    nug = Nuggetizer(backend, assigner_mode=NuggetAssignMode.SUPPORT_GRADE_3, window_size=1)
    assigner = NuggetAssigner(nug)
    df_out = assigner.transform(scored_df)
    assert df_out["assignment"].tolist() == [2, 0, 1]
