import ir_measures
from ir_measures.providers.cwl_eval import CwlEvaluator

SUPPORTED_MEASURES = {"P", "R"}

def measure_factory(attr: str, nuggetizer_provider: 'open_nuggetizer.Nuggetizer'):
    """
    Factory function to create a measure based on the provided attribute and nuggetizer provider.

    Args:
        attr (str): The name of the measure to create.
        nuggetizer_provider (str): The nuggetizer provider to use.

    Returns:
        Measure: An instance of the measure class.
    """
    if attr in SUPPORTED_MEASURES:
        M = getattr(ir_measures.measures, attr)
        _SUPPORTED_PARAMS = dict(M.SUPPORTED_PARAMS)
        name = repr(nuggetizer_provider) + "." + M.NAME

        class _RuntimeMeasure(ir_measures.measures.Measure):
            nonlocal _SUPPORTED_PARAMS
            SUPPORTED_PARAMS = _SUPPORTED_PARAMS
            NAME = M.NAME
            __name__ = name
            _nuggetizer_provider = nuggetizer_provider
            _nuggetizer_base_measure = M

            def runtime_impl(self, nuggets, run):
                inf_qrels = self._nuggetizer_provider.make_qrels(run, nuggets)
                evaluator = CwlEvaluator(
                    [self], inf_qrels, {(None, 0.0, 1.0): [self]}, verify_gains=False
                )
                return evaluator.iter_calc(run)

        Measure = _RuntimeMeasure()
        if "max_rel" in _SUPPORTED_PARAMS:
            Measure = Measure(max_rel=1)
        return Measure
    else:
        raise ValueError(f"Measure {attr} is not supported.")
