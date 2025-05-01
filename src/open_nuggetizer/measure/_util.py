import warnings
import re
import io
import ast
import gzip
import contextlib
import itertools
import tempfile
from typing import Dict, List
from collections import defaultdict
from typing import NamedTuple, Union, Iterable
import ir_measures
if False: # this is to allow type-checking for pandas
    import pandas


class Qrel(NamedTuple):
    query_id: str
    doc_id: str
    relevance: int
    assignment: int
    iteration: str = '0'


class ScoredAnswer(NamedTuple):
    query_id: str
    answer: str
    score: float


class Answer(NamedTuple):
    query_id: str
    answer: str


class NuggetQrelsConverter:
    def __init__(self, qrels, strict=True):
        self.qrels = qrels
        self._predicted_format = None
        self.strict = strict # setting strict to false prevents missing columns from raising an error for DFs

    def tee(self, count):
        t, err = self.predict_type()
        if t == 'namedtuple_iter':
            teed_qrels = itertools.tee(self.qrels, count)
            return [NuggetQrelsConverter(qrels) for qrels in teed_qrels]
        return [self for _ in range(count)]

    def predict_type(self):
        if self._predicted_format:
            return self._predicted_format
        result = 'UNKNOWN'
        error = None
        if isinstance(self.qrels, dict):
            result = 'dict_of_dict'
        elif hasattr(self.qrels, 'itertuples'):
            cols = self.qrels.columns
            missing_cols = [f for f in Qrel._fields if f not in cols and f not in Qrel._field_defaults]
            if missing_cols and self.strict:
                error = f'DataFrame missing columns: {list(missing_cols)} (found {list(cols)})'
            else:
                result = 'pd_dataframe'                
        elif hasattr(self.qrels, '__iter__'):
            # peek
            # TODO: is this an OK approach?
            self.qrels, peek_qrels = itertools.tee(self.qrels, 2)
            sentinal = object()
            item = next(peek_qrels, sentinal)
            if isinstance(item, tuple) and hasattr(item, '_fields'):
                fields = item._fields
                missing_fields = [f for f in Qrel._fields if f not in fields and f not in Qrel._field_defaults]
                if not missing_fields:
                    result = 'namedtuple_iter'
                else:
                    error = f'namedtuple iter missing fields: {list(missing_fields)} (found {list(fields)})'
            elif item is sentinal:
                result = 'namedtuple_iter'
            else:
                error = 'iterable not a namedtuple iterator'
        else:
            required_fields = tuple(f for f in Qrel._fields if f not in Qrel._field_defaults)
            error = f'unexpected format; please provide either: (1) an iterable of namedtuples (fields {required_fields}, e.g., from ir_measures.Qrel); (2) a pandas DataFrame with columns {required_fields}; or (3) a dict-of-dict'
        self._predicted_format = (result, error)
        return result, error

    def as_dict_of_dict(self):
        t, err = self.predict_type()
        if t == 'dict_of_dict':
            return self.qrels
        else:
            result = {}
            for qrel in self.as_namedtuple_iter():
                if qrel.query_id not in result:
                    result[qrel.query_id] = {}
                result[qrel.query_id][qrel.doc_id] = (qrel.relevance, qrel.assignment)
            return result

    def as_namedtuple_iter(self):
        t, err = self.predict_type()
        if t == 'namedtuple_iter':
            yield from self.qrels
        if t == 'dict_of_dict':
            for query_id, docs in self.qrels.items():
                for doc_id, (relevance, assignment) in docs.items():
                    yield Qrel(query_id=query_id, doc_id=doc_id, relevance=relevance, assignment=assignment)
        if t == 'pd_dataframe':
            if 'iteration' in self.qrels.columns:
                yield from (Qrel(qrel.query_id, qrel.doc_id, qrel.relevance, qrel.assignment, qrel.iteration) for qrel in self.qrels.itertuples())
            else:
                yield from (Qrel(qrel.query_id, qrel.doc_id, qrel.assignment, qrel.relevance) for qrel in self.qrels.itertuples())
        if t == 'UNKNOWN':
            raise ValueError(f'unknown qrels format: {err}')

    def as_pd_dataframe(self):
        t, err = self.predict_type()
        if t == 'pd_dataframe':
            if 'iteration' not in self.qrels.columns:
                return self.qrels.assign(iteration=['0'] * len(self.qrels))
            return self.qrels
        else:
            pd = ir_measures.lazylibs.pandas()
            return pd.DataFrame(self.as_namedtuple_iter())

    @contextlib.contextmanager
    def as_tmp_file(self):
        with tempfile.NamedTemporaryFile(mode='w+t') as f:
            for qrel in self.as_namedtuple_iter():
                f.write('{query_id} 0 {doc_id} {relevance} {assignment}\n'.format(**qrel._asdict()))
            f.flush()
            f.seek(0)
            yield f


class RAGRunConverter:
    def __init__(self, run, strict=True):
        self.run = run
        self._predicted_format = None
        self.strict = strict # setting strict to false prevents missing columns from raising an error for DFs

    def tee(self, count):
        t, err = self.predict_type()
        if t == 'namedtuple_iter':
            teed_run = itertools.tee(self.run, count)
            return [RAGRunConverter(run) for run in teed_run]
        return [self for _ in range(count)]

    def predict_type(self):
        if self._predicted_format:
            return self._predicted_format
        result = 'UNKNOWN'
        error = None
        if isinstance(self.run, dict):
            result = 'dict_of_dict'
        elif hasattr(self.run, 'itertuples'):
            cols = self.run.columns
            missing_cols = set(Answer._fields) - set(cols)
            if missing_cols and self.strict:
                error = f'DataFrame missing columns: {list(missing_cols)} (found {list(cols)})'
            else:
                result = 'pd_dataframe'
        elif hasattr(self.run, '__iter__'):
            # peek
            # TODO: is this an OK approach?
            self.run, peek_run = itertools.tee(self.run, 2)
            sentinal = object()
            item = next(peek_run, sentinal)
            if isinstance(item, tuple) and hasattr(item, '_fields'):
                fields = item._fields
                missing_fields = set(Answer._fields) - set(fields)
                if not missing_fields:
                    result = 'namedtuple_iter'
                else:
                    error = f'namedtuple iter missing fields: {list(missing_fields)} (found {list(fields)})'
            elif item is sentinal:
                result = 'namedtuple_iter'
            else:
                error = 'iterable not a namedtuple iterator'
        else:
            error = f'unexpected format; please provide either: (1) an iterable of namedtuples (fields {Answer._fields}, e.g., from pyterrier_nuggetizer.measure.Answer); (2) a pandas DataFrame with columns {Answer._fields}; or (3) a dict-of-dict'
        self._predicted_format = (result, error)
        return result, error

    def as_dict_of_dict(self):
        t, err = self.predict_type()
        if t == 'dict_of_dict':
            return self.run
        else:
            result = {}
            for answer in self.as_namedtuple_iter():
                if answer.query_id not in result:
                    result[answer.query_id] = {}
                result[answer.query_id][answer.doc_id] = answer.answer
            return result

    def as_namedtuple_iter(self):
        t, err = self.predict_type()
        if t == 'namedtuple_iter':
            yield from self.run
        if t == 'dict_of_dict':
            for query_id, docs in self.run.items():
                for _, answer in docs.items():
                    yield Answer(query_id=query_id, answer=answer)
        if t == 'pd_dataframe':
            yield from (Answer(a.query_id, a.answer) for a in self.run.itertuples())
        if t == 'UNKNOWN':
            raise ValueError(f'unknown run format: {err}')

    def as_pd_dataframe(self):
        t, err = self.predict_type()
        if t == 'pd_dataframe':
            return self.run
        else:
            pd = ir_measures.lazylibs.pandas()
            return pd.DataFrame(self.as_namedtuple_iter())

    @contextlib.contextmanager
    def as_tmp_file(self):
        with tempfile.NamedTemporaryFile(mode='w+t') as f:
            ranks = {}
            for answer in self.as_namedtuple_iter():
                key = answer.query_id
                rank = ranks.setdefault(key, 0)
                f.write('{query_id} Q0 {answer} run\n'.format(**answer._asdict(), rank=rank))
                ranks[key] += 1
            f.flush()
            f.seek(0)
            yield f
