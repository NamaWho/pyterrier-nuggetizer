import pytest
import pandas as pd
from open_nuggetizer.measure._ir_measures import measure_factory
from open_nuggetizer.nuggetizer import Nuggetizer
from open_nuggetizer.measure._provider import NuggetEvalProvider
from open_nuggetizer._types import NuggetAssignMode

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

def test_vital_score_metric(dummy_run, dummy_qrels):
    """
    Test the VitalScore metric using the updated structure.
    """
    backend = DummyBackend()
    nuggetizer = Nuggetizer(backend=backend)
    vital_score_metric = measure_factory("VitalScore", nuggetizer)
    results = list(vital_score_metric.runtime_impl(dummy_qrels, dummy_run))

    assert isinstance(results, list)
    assert len(results) == 2  # One result per query
    for result in results:
        assert result.query_id in ["Q1", "Q2"]
        assert 0.0 <= result.value <= 1.0  # VitalScore should be between 0 and 1

# def test_strict_vital_score_metric(dummy_run, dummy_qrels):
#     """
#     Test the VitalScore metric with strict mode using the updated structure.
#     """
#     backend = DummyBackend()
#     nuggetizer = Nuggetizer(backend=backend, assigner_mode=NuggetAssignMode.SUPPORT_GRADE_3)
#     vital_score_metric = measure_factory("VitalScore", nuggetizer, strict=True)
#     results = list(vital_score_metric.runtime_impl(dummy_qrels, dummy_run))

#     assert isinstance(results, list)
#     assert len(results) == 2  # One result per query
#     for result in results:
#         assert result.query_id.startswith("Q")
#         assert 0.0 <= result.value <= 1.0  # VitalScore should be between 0 and 1

def test_weighted_score_metric(dummy_run, dummy_qrels):
    """
    Test the WeightedScore metric using the updated structure.
    """
    backend = DummyBackend()
    nuggetizer = Nuggetizer(backend=backend, assigner_mode=NuggetAssignMode.SUPPORT_GRADE_3)
    weighted_score_metric = measure_factory("WeightedScore", nuggetizer)
    results = list(weighted_score_metric.runtime_impl(dummy_qrels, dummy_run))

    assert isinstance(results, list)
    assert len(results) == 2  # One result per query
    for result in results:
        assert result.query_id.startswith("Q")
        assert 0.0 <= result.value <= 1.0  # WeightedScore should be between 0 and 1

def test_unsupported_metric():
    """
    Test behavior when an unsupported metric is requested.
    """
    backend = DummyBackend()
    nuggetizer = Nuggetizer(backend=backend)
    with pytest.raises(ValueError, match="Measure UnsupportedMetric is not supported."):
        measure_factory("UnsupportedMetric", nuggetizer)