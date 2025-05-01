import ir_measures
from ir_measures.providers.pytrec_eval import PytrecEvalEvaluator
from open_nuggetizer.measure._provider import WeightedNuggetPrecisionEvaluator

SUPPORTED_MEASURES = {'P'}


def measure_factory(attr: str, nuggetizer_provider: str):
    if attr in SUPPORTED_MEASURES:
        M = getattr(ir_measures.measures, attr)
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
                inf_qrels = self._nuggetizer_provider.make_qrels(run, qrels)
                evaluator = CwlEvaluator([self], inf_qrels, {(None, 0., 1.): [self]}, verify_gains=False)
                return evaluator.iter_calc(run)
        Measure = _RuntimeMeasure()
        return Measure