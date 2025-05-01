from typing import Iterator
import ir_measures
from ir_measures import providers, Metric
from ir_measures.providers.base import Any
from open_nuggetizer.measure._wnp import _wNP


class WeightedNuggetPrecisionEvaluator(providers.Evaluator):
    def __init__(self, measures, qrels, invocations):
        super().__init__(measures, set(qrels.keys()))
        self.qrels = qrels
        self.invocations = invocations

    def iter_calc(self, run) -> Iterator['Metric']:
        """Compute the metrics for the run, discarding topics with no relevant documents"""

        for measure, (vital, partial, strict) in self.invocations:
            for qid, nuggets in self.qrels.items():
                selected = [(nid, vals[1]) for nid, vals in nuggets.items() if vals[0] >= vital]

                if len(selected) < 0:
                    continue

                partial_support = [n for n in selected if 0 < n[1] <= partial]
                full_support = [n for n in selected if n[1] > partial]

                if (not len(full_support) > 0) or (not len(partial_support) and strict):
                    continue

                value = len(full_support)
                if not strict:
                    value += 0.5 * len(partial_support)

                value = value / len(selected)

                yield Metric(query_id=qid, measure=measure, value=value)


class WeightedNuggetPrecisionProvider(providers.Provider):
    """WeightedNuggetPrecision provider"""
    NAME = "weighted nugget precision"
    SUPPORTED_MEASURES = [
        _wNP(rel=Any()),
    ]

    def _evaluator(self, measures, qrels) -> providers.Evaluator:
        invocations = []
        for measure in measures:
            if measure.NAME == _wNP.NAME:
                invocations.append((measure, measure['vital'], measure['partial'], measure['strict']))
            else:
                raise ValueError(f'unsupported measure {measure}')
        qrels = ir_measures.util.QrelsConverter(qrels).as_dict_of_dict()
 
        return WeightedNuggetPrecisionEvaluator(measures, qrels, invocations)


providers.register(WeightedNuggetPrecisionProvider())
