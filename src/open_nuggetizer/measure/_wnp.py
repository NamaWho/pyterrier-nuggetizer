from ir_measures import measures


class _WeightedNuggetPrecision(measures.Measure):
    """
    Basic measure for that computes the percentage of documents in the top cutoff results
    that are labeled as relevant. cutoff is a required parameter, and can be provided as
    P@cutoff.
    """
    __name__ = 'WeightedNuggetPrecision'
    NAME = __name__
    PRETTY_NAME = 'Weighted Nugget Precision'
    SHORT_DESC = 'The percentage of documents in the top k results that are relevant.'
    SUPPORTED_PARAMS = {
        'vital': measures.ParamInfo(dtype=int, default=1, desc='minimum vital score to be considered vital (inclusive)'),
        'partial': measures.ParamInfo(dtype=int, default=1, desc='maximum assignment score to be considered partial support (inclusive)'),
        'strict': measures.ParamInfo(dtype=bool, default=True, desc='Exclude discounted partial support in measure'),
    }


WeightedNuggetPrecision = _WeightedNuggetPrecision()
measures.register(WeightedNuggetPrecision, ['WeightedNuggetPrecision'])
