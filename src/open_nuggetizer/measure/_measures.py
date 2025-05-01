from ir_measures import measures

class _AllScore(measures.Measure):
    __name__ = 'AllScore'
    NAME = __name__
    PRETTY_NAME = 'All Score'
    SHORT_DESC = "Average of the scores for all nuggets in an answer"
    SUPPORTED_PARAMS = {
        'strict': measures.ParamInfo(dtype=bool, default=True, desc='Exclude nuggets partially supported in measure'),
    }

AllScore = _AllScore()
measures.register(AllScore, ['AllScore'])

class _VitalScore(measures.Measure):
    __name__ = 'VitalScore'
    NAME = __name__
    PRETTY_NAME = 'Vital Score'
    SHORT_DESC = "Average of the scores for only the vital nuggets in an answer"
    SUPPORTED_PARAMS = {
        'strict': measures.ParamInfo(dtype=bool, default=True, desc='Exclude nuggets partially supported in measure'),
    }

VitalScore = _VitalScore()
measures.register(VitalScore, ['VitalScore'])

class _WeightedScore(measures.Measure):
    __name__ = 'WeightedScore'
    NAME = __name__
    PRETTY_NAME = 'Weighted Score'
    SHORT_DESC = "Weighted average of the scores for all nuggets in an answer"
    SUPPORTED_PARAMS = {
        'strict': measures.ParamInfo(dtype=bool, default=True, desc='Exclude nuggets partially supported in measure'),
    }

WeightedScore = _WeightedScore()
measures.register(WeightedScore, ['WeightedScore'])
