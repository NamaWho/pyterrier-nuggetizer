# import ir_measures
# from ir_measures.providers.pytrec_eval_provider import PytrecEvalEvaluator, PytrecEvalInvoker
# #from open_nuggetizer.measure._provider import WeightedNuggetPrecisionEvaluator

# SUPPORTED_MEASURES = {'P'}

# def measure_factory(attr: str, nuggetizer_provider: str):
#     if attr in SUPPORTED_MEASURES:
#         M = getattr(ir_measures.measures, attr)
#         _SUPPORTED_PARAMS = dict(M.SUPPORTED_PARAMS)
#         name = repr(nuggetizer_provider) + '.' + M.NAME

#         class _RuntimeMeasure(ir_measures.measures.Measure):
#             nonlocal _SUPPORTED_PARAMS
#             SUPPORTED_PARAMS = _SUPPORTED_PARAMS
#             NAME = M.NAME
#             __name__ = name
#             _nuggetizer_provider = nuggetizer_provider
#             _nuggetizer_base_measure = M

#             def runtime_impl(self, qrels, run):
#                 assignments = self._nuggetizer_provider.make_qrels(run, qrels)
#                 # invoker = PytrecEvalInvoker(
#                 #     [self], assignments, verify_gains=False
#                 # )
#                 evaluator = PytrecEvalEvaluator(
#                     [self], invoker, assignments
#                 )
#                 # return evaluator.iter_calc(run)
#                 return assignments
#         Measure = _RuntimeMeasure()
#         return Measure
#     else:
#         raise ValueError(f"Measure {attr} is not supported.")