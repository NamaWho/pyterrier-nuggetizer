from typing import Iterator, List, Tuple
import ir_measures
from ir_measures import providers, Metric
from ir_measures.providers.base import Any
from open_nuggetizer.measure._measures import _AllScore, _VitalScore, _WeightedScore
from open_nuggetizer.measure._util import NuggetQrelsConverter


class NuggetScoreEvaluator(providers.Evaluator):
    def __init__(self, measures, qrels, invocations):
        super().__init__(measures, set(qrels.keys()))
        self.qrels = qrels
        self.invocations = invocations

    def _unweighted(self, nuggets, partial_rel, strict, partial_weight):
        full_support = [n for n in nuggets if n[1] > partial_rel]
        partial_support = [n for n in nuggets if 0 < n[1] <= partial_rel]

        if not len(full_support) > 0:
            return 0.0

        value = len(full_support)
        if not strict:
            value += partial_weight * len(partial_support)

        return value / len(nuggets)

    def _weighted(self, nuggets, rel, partial_rel, strict, partial_weight):
        vital_nuggets = [n for n in nuggets if n[2] > rel]
        okay_nuggets = [n for n in nuggets if 0 < n[2] <= rel]

        vital_score = self._unweighted(vital_nuggets, partial_rel, strict, partial_weight)
        okay_score = self._unweighted(okay_nuggets, partial_rel, strict, partial_weight)

        denominator = len(vital_nuggets) + 0.5 * len(okay_nuggets)
        if denominator == 0:
            return 0.0
        return (vital_score + 0.5 * okay_score) / denominator

    def iter_calc(self, run) -> Iterator['Metric']:
        for measure, rel, partial_rel, strict, partial_weight, weighted in self.invocations:
            for qid, _nuggets in run.items():
                qrels = self.qrels.get(qid, {})
                if len(_nuggets) < 1:
                    continue

                nuggets = [(n[0], n[1], qrels.get(n[0], 0)) for n in _nuggets]

                if strict:
                    nuggets = [n for n in nuggets if n[2] >= rel]

                if len(nuggets) < 1:
                    continue

                if weighted:
                    yield Metric(query_id=qid,
                                 measure=measure,
                                 value=self._weighted(nuggets, rel, partial_rel, strict, partial_weight))
                else:
                    yield Metric(query_id=qid,
                                 measure=measure,
                                 value=self._unweighted(nuggets, rel, partial_rel, strict, partial_weight))


class NuggetEvalProvider(providers.Provider):
    """NuggetEval provider"""
    NAME = "nugget precision"
    SUPPORTED_MEASURES = [
       _AllScore(strict=Any()),
       _VitalScore(rel=Any(), strict=Any()),
       _WeightedScore(rel=Any(), strict=Any(), partial_weight=Any()),
    ]

    def _build_invocations(self, measures) -> List[Tuple[Metric, int, bool, float]]:
        invocations = []
        for measure in measures:
            if measure.NAME in [m.NAME for m in self.SUPPORTED_MEASURES]:
                if measure.NAME == _VitalScore.NAME:
                    invocations.append((measure, measure['rel'], measure['partial_rel'], measure['strict'], 0.5, False))
                elif measure.NAME == _WeightedScore.NAME:
                    invocations.append((measure, measure['rel'], measure['partial_rel'], False, measure['partial_weight'], True))
                elif measure.NAME == _AllScore.NAME:
                    invocations.append((measure, 0, measure['partial_rel'], measure['strict'], 0.5, False))
        return invocations

    def _evaluator(self, measures, qrels) -> providers.Evaluator:
        qrels = NuggetQrelsConverter(qrels).as_dict_of_dict()

        invocations = self._build_invocations(measures)
        return NuggetScoreEvaluator(measures, qrels, invocations)


providers.register(NuggetEvalProvider())
