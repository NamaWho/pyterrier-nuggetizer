import ir_measures
from open_nuggetizer.measure._provider import NuggetEvalProvider
import open_nuggetizer
from open_nuggetizer.measure._util import RAGRunConverter

SUPPORTED_MEASURES = {'VitalScore', 'WeightedScore', 'AllScore'}


def measure_factory(attr: str, nuggetizer_provider: str):
    if attr in SUPPORTED_MEASURES:
        M = getattr(open_nuggetizer.measure._measures, attr)
        _SUPPORTED_PARAMS = dict(M.SUPPORTED_PARAMS)
        name = repr(nuggetizer_provider) + '.' + M.NAME

        class _RuntimeMeasure(ir_measures.measures.Measure):
            nonlocal _SUPPORTED_PARAMS
            SUPPORTED_PARAMS = _SUPPORTED_PARAMS
            NAME = M.NAME
            __name__ = name
            _nuggetizer_provider = nuggetizer_provider
            _nuggetizer_base_measure = M

            def runtime_impl(self, qrels, run):
                assignments = self._nuggetizer_provider.assign_to_run(run, qrels)
                assignments = assignments.rename(columns={'qid': 'query_id'})
                evaluator = NuggetEvalProvider().evaluator([self], qrels)
                assignments = RAGRunConverter(assignments).as_dict_of_dict()
                return evaluator.iter_calc(assignments)
        Measure = _RuntimeMeasure()
        return Measure
    else:
        raise ValueError(f"Measure {attr} is not supported.")
