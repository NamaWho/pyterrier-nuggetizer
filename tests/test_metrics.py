import pytest
import pandas as pd
from open_nuggetizer._measure import measure_factory
from open_nuggetizer.nuggetizer import Nuggetizer

class DummyNuggetizer:
    """
    Dummy implementation of Nuggetizer for testing metrics.
    """
    def make_qrels(self, run, nuggets):
        # Simulate qrels creation for testing
        return pd.DataFrame(
            {
                "qid": run["query"],
                "doc_id": ["D1", "D2"],
                "relevance": [1, 0],
            }
        )

@pytest.fixture
def dummy_run():
    return pd.DataFrame(
        {
            "query": ["Q1", "Q1"],
            "qanswer": ["ans1", "ans2"],
        }
    )

@pytest.fixture
def dummy_nuggets():
    return pd.DataFrame(
        {
            "qid": ["Q1", "Q1"],
            "nugget_id": ["N1", "N2"],
            "nugget": ["nugget1", "nugget2"],
            "importance": [1, 0],
        }
    )

def test_precision_metric(dummy_run, dummy_nuggets):
    """
    Test the Precision (P) metric.
    """
    nuggetizer = DummyNuggetizer()
    precision_metric = measure_factory("P", nuggetizer)
    results = precision_metric.runtime_impl(dummy_nuggets, dummy_run)
    assert isinstance(results, list)
    assert len(results) == 2  
    assert results[0].value == 1.0
    assert results[1].value == 0.0 

def test_recall_metric(dummy_run, dummy_nuggets):
    """
    Test the Recall (R) metric.
    """
    nuggetizer = DummyNuggetizer()
    recall_metric = measure_factory("R", nuggetizer)
    results = recall_metric.runtime_impl(dummy_nuggets, dummy_run)
    assert isinstance(results, list)
    assert len(results) == 2  
    assert results[0].value == 1.0 
    assert results[1].value == 0.0 

def test_unsupported_metric():
    """
    Test behavior when an unsupported metric is requested.
    """
    nuggetizer = DummyNuggetizer()
    with pytest.raises(ValueError):
        measure_factory("UnsupportedMetric", nuggetizer)