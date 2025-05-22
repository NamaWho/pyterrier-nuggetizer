import pytest
import pandas as pd
from open_nuggetizer.measure._ir_measures import measure_factory
from open_nuggetizer.nuggetizer import Nuggetizer
from open_nuggetizer.measure._provider import NuggetEvalProvider
from open_nuggetizer._types import NuggetAssignMode
import pdb

class DummyBackend:
    """
    Dummy backend for simulating model inference.
    """
    def __init__(self):
        self.model_name_or_path = "dummy_model"

    def generate(self, prompts):
        # Simulate model output
        return ['["support", "partial_support", "not_support"]']

@pytest.fixture
def dummy_run():
    """
    Simulated run data for testing metrics.
    """
    return pd.DataFrame(
        {
            "qid": ["Q1", "Q1", "Q2", "Q2"],
            "query": ["Query1", "Query1", "Query2", "Query2"],
            "qanswer": ["Answer1", "Answer2", "Answer3", "Answer4"],
        }
    )

@pytest.fixture
def dummy_qrels():
    """
    Simulated qrels data for testing metrics.
    """
    return pd.DataFrame(
        {
            "qid": ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"],
            "nugget_id": ["N1", "N2", "N3", "N1", "N2", "N3"],
            "nugget": ["Nugget1", "Nugget2", "Nugget3", "Nugget4", "Nugget5", "Nugget6"],
            "importance": [1, 0, 1, 0, 1, 0],
        }
    )

# class DummyNuggetizer(Nuggetizer):
#     def __init__(self, assigner_mode = NuggetAssignMode.SUPPORT_GRADE_2, conversation_template = None, window_size = None, creator_window_size = 10, scorer_window_size = 10, assigner_window_size = 10, max_nuggets = 30, query_field = "query", document_field = "text", answer_field = "qanswer", nugget_field = "nugget", importance_field = "importance", assignment_field = "assignment", verbose = False):
#         super().__init__(DummyBackend(), assigner_mode, conversation_template, window_size, creator_window_size, scorer_window_size, assigner_window_size, max_nuggets, query_field, document_field, answer_field, nugget_field, importance_field, assignment_field, verbose)

#     def assign_to_run(self, run, qrels):
#         """
#         Simulate nugget assignment to the run.
#         """
#         # Simulate nugget assignment logic
#         assignments = pd.DataFrame({
#             "qid": run["qid"],
#             "nugget_id": ["N1", "N2", "N3", "N4", "N5", "N6"],
#             "support": [1, 0, 1, 0, 1, 0],
#             "partial_support": [0.5, 0.2, 0.8, 0.1, 0.6, 0.3],
#         })
#         return assignments

# def test_vital_score_metric(dummy_run, dummy_qrels):
#     """
#     Test the VitalScore metric using the updated structure.
#     """
#     backend = DummyBackend()
#     nuggetizer = Nuggetizer(backend=backend)
#     vital_score_metric = measure_factory("VitalScore", nuggetizer)
#     results = list(vital_score_metric.runtime_impl(dummy_qrels, dummy_run))

#     assert isinstance(results, list)
#     assert len(results) == 2  # One result per query
#     for result in results:
#         assert result.query_id in ["Q1", "Q2"]
#         assert 0.0 <= result.value <= 1.0  # VitalScore should be between 0 and 1

def test_vital_score_nuggetizer_metric(dummy_run, dummy_qrels):
    """
    Test the VitalScore metric using the updated structure.
    """
    backend = DummyBackend()
    nuggetizer = Nuggetizer(backend)
    vital_score_metric = nuggetizer.VitalScore()

    #dummy_qrels = dummy_qrels.rename(columns={"qid":"query_id"})
    #pdb.set_trace()
    for result in vital_score_metric.iter_calc(dummy_qrels, dummy_run):
        assert result.query_id in ["Q1", "Q2"]
        assert 0.0 <= result.value <= 1.0  # VitalScore should be between 0 and 1

# def test_weighted_score_metric(dummy_run, dummy_qrels):
#     """
#     Test the WeightedScore metric using the updated structure.
#     """
#     backend = DummyBackend()
#     nuggetizer = Nuggetizer(backend=backend, assigner_mode=NuggetAssignMode.SUPPORT_GRADE_3)
#     weighted_score_metric = measure_factory("WeightedScore", nuggetizer)
#     results = list(weighted_score_metric.runtime_impl(dummy_qrels, dummy_run))

#     assert isinstance(results, list)
#     assert len(results) == 2  # One result per query
#     for result in results:
#         assert result.query_id.startswith("Q")
#         assert 0.0 <= result.value <= 1.0  # WeightedScore should be between 0 and 1

# def test_unsupported_metric():
#     """
#     Test behavior when an unsupported metric is requested.
#     """
#     backend = DummyBackend()
#     nuggetizer = Nuggetizer(backend=backend)
#     with pytest.raises(AttributeError):
#         measure_factory("UnsupportedMetric", nuggetizer)