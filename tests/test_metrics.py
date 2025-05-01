import pytest
import pandas as pd
from open_nuggetizer._measure import measure_factory
from open_nuggetizer.nuggetizer import Nuggetizer
import ir_measures

class DummyNuggetizer:
    """
    Dummy implementation of Nuggetizer for testing metrics.
    """
    def make_qrels(self, run, nuggets):
        # Simulate qrels creation for testing
        return pd.DataFrame(
            {
                "query_id": run["query_id"],
                "doc_id": ["D1", "D2"],
                "relevance": [1, 0],
            }
        )
    
    def __getattr__(self, attr: str):
        measure = measure_factory(attr, self)
        if measure is not None:
            return measure
        return self.__getattribute__(attr)

@pytest.fixture
def dummy_run():
    return pd.DataFrame(
        {
            "query_id": ["Q1", "Q1"],
            "doc_id": ["D1", "D2"],
            "score": [1.0, 0.0],
        }
    )

@pytest.fixture
def dummy_nuggets():
    return pd.DataFrame(
        {
            "query_id": ["Q1", "Q1"],
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
    precision = nuggetizer.P
    results = precision.runtime_impl(dummy_nuggets, dummy_run)
    results = list(results)

    print("results: " , results)
    assert isinstance(results, list)
    assert len(results) == 1  
    assert results[0].query_id == "Q1"
    assert results[1].value == 0.0 

def test_unsupported_metric():
    """
    Test behavior when an unsupported metric is requested.
    """
    nuggetizer = DummyNuggetizer()
    with pytest.raises(ValueError):
        measure_factory("UnsupportedMetric", nuggetizer)