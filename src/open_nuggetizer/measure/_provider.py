from typing import Iterator
import ir_measures
from ir_measures import providers, Metric
from ir_measures.providers.base import Any
from open_nuggetizer.measure._measures import _AllScore, _VitalScore, _WeightedScore

class AllScoreEvaluator(providers.Evaluator):
    def __init__(self, measures, qrels, invocations):
        super().__init__(measures, set(qrels.keys()))
        self.qrels = qrels
        self.invocations = invocations

    def iter_calc(self, run) -> Iterator['Metric']:
        for measure, strict in self.invocations:
            for qid, nuggets in run.items():

                if len(nuggets) <= 0:
                    yield Metric(query_id=qid, measure=measure, value=0)
                    continue

                partial_support = [n for n in nuggets if 0 < n[1] <= 1]
                full_support = [n for n in nuggets if n[1] > 1]

                if (not len(full_support) > 0) or (not len(partial_support) and strict):
                    yield Metric(query_id=qid, measure=measure, value=0)
                    continue

                value = len(full_support)
                if not strict:
                    value += 0.5 * len(partial_support)

                value = value / len(nuggets)

                yield Metric(query_id=qid, measure=measure, value=value)

class VitalScoreEvaluator(providers.Evaluator):
    def __init__(self, measures, qrels, invocations):
        super().__init__(measures, set(qrels.keys()))
        self.qrels = qrels
        self.invocations = invocations

    def iter_calc(self, run) -> Iterator['Metric']:
        for measure, strict in self.invocations:
            for qid, nuggets in run.items():
                # Select nuggets based on their importance (either vital or okay)
                nuggets = [(nugget_id, attr[1]) for nugget_id, attr in nuggets.items() if attr[0] >= 1]

                if len(nuggets) <= 0:
                    yield Metric(query_id=qid, measure=measure, value=0)
                    continue

                partial_support = [n for n in nuggets if 0 < n[1] <= 1]
                full_support = [n for n in nuggets if n[1] > 1]

                if (not len(full_support) > 0) or (not len(partial_support) and strict):
                    yield Metric(query_id=qid, measure=measure, value=0)
                    continue

                value = len(full_support)
                if not strict:
                    value += 0.5 * len(partial_support)

                value = value / len(nuggets)

                yield Metric(query_id=qid, measure=measure, value=value)

class WeightedScoreEvaluator(providers.Evaluator):
    def __init__(self, measures, qrels, invocations):
        super().__init__(measures, set(qrels.keys()))
        self.qrels = qrels
        self.invocations = invocations

    def iter_calc(self, run) -> Iterator['Metric']:
        for measure, strict in self.invocations:
            for qid, nuggets in run.items():

                if len(nuggets) <= 0:
                    yield Metric(query_id=qid, measure=measure, value=0)
                    continue
                
                partial_support = [n for n in nuggets if 0 < n[1] <= 1]
                partial_support_okay = [n for n in partial_support if n[0] == 0]
                partial_support_vital = [n for n in partial_support if n[0] == 1]
                
                full_support = [n for n in nuggets if n[1] > 1]
                full_support_okay = [n for n in full_support if n[0] == 0]
                full_support_vital = [n for n in full_support if n[0] == 1]

                if (not len(full_support) > 0) or (not len(partial_support) and strict):
                    yield Metric(query_id=qid, measure=measure, value=0)
                    continue

                value = len(full_support_vital) + 0.5 * len(full_support_okay)
                if not strict:
                    value += 0.5 (len(partial_support_vital) + 0.5 * len(partial_support_okay))

                vital = len(full_support_vital) + len(partial_support_vital)
                okay = len(full_support_okay) + len(partial_support_okay)

                value = value / (vital + 0.5 * okay)

                yield Metric(query_id=qid, measure=measure, value=value)


class NuggetEvalProvider(providers.Provider):
    """NuggetEval provider"""
    NAME = "nugget precision"
    SUPPORTED_MEASURES = [
        _AllScore(strict=Any()),
        _VitalScore(strict=Any()),
        _WeightedScore(strict=Any()),
    ]

    def _evaluator(self, measures, qrels) -> providers.Evaluator:
        invocations = []
        for measure in measures:
            if measure.NAME in [m.NAME for m in self.SUPPORTED_MEASURES]:
                invocations.append((measure, measure['strict']))
            else:
                raise ValueError(f'Unsupported measure {measure}')
        qrels = ir_measures.util.QrelsConverter(qrels).as_dict_of_dict()

        # TODO: fix this
        return AllScoreEvaluator(measures, qrels, invocations)

providers.register(NuggetEvalProvider())
