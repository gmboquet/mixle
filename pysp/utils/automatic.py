"""Automatic detection of data type for estimators. Note that pysp.bstats is removed. """
import math
import numbers
import numpy as np
from collections import defaultdict, Iterable

from pysp.stats.pdist import ParameterEstimator

from typing import Optional, Any, Sequence, Dict, TypeVar, Union
T = TypeVar('T')

def get_optional_estimator(est: ParameterEstimator, missing_value: Optional[Any] = None):
    from pysp.stats.optional import OptionalEstimator
    return OptionalEstimator(est, missing_value=missing_value)


def get_sequence_estimator(est: ParameterEstimator) -> 'ParameterEstimator':
    from pysp.stats.sequence import SequenceEstimator
    return SequenceEstimator(est)

def get_ignored_estimator() -> 'ParameterEstimator':
    from pysp.stats.ignored import IgnoredEstimator
    return IgnoredEstimator()

def get_composite_estimator(ests: Sequence[ParameterEstimator]) -> 'ParameterEstimator':
    from pysp.stats.composite import CompositeEstimator
    return CompositeEstimator(ests)

def get_categorical_estimator(vdict: Dict[T, float], pseudo_count: Optional[float] = None, emp_suff_stat: bool = True) \
        -> 'ParameterEstimator':
    from pysp.stats.categorical import CategoricalEstimator

    if emp_suff_stat:
        cnt = sum(vdict.values())
        suff_stat = {k: v / cnt for k, v in vdict.items()}
    else:
        suff_stat = None

    return CategoricalEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat)

def get_poisson_estimator(vdict: Dict[int, float], pseudo_count: Optional[float] = None, emp_suff_stat: bool = True) \
        -> 'ParameterEstimator':

    from pysp.stats.poisson import PoissonEstimator

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0

        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k * v

        ss_1 = ss_1 / ss_0

    elif pseudo_count is not None:
        ss_1 = 1.0

    else:
        ss_1 = None

    return PoissonEstimator(pseudo_count=pseudo_count, suff_stat=ss_1)


def get_gaussian_estimator(vdict: Dict[Union[np.floating, float], float], pseudo_count: Optional[float] = None,
                           emp_suff_stat: bool = True) -> 'ParameterEstimator':

    from pysp.stats.gaussian import GaussianEstimator

    if emp_suff_stat:
        ss_0 = 0.0
        ss_1 = 0.0
        ss_2 = 0.0
        for k, v in vdict.items():
            if math.isfinite(k):
                ss_0 += v
                ss_1 += k*v
                ss_2 += k*k*v
        ss_1 = ss_1 / ss_0
        ss_2 = (ss_2 / ss_0) - ss_1*ss_1

    elif pseudo_count is not None:
        ss_1 = 1.0e-6
        ss_2 = 1.0e-6
    else:
        ss_1 = None
        ss_2 = None

    return GaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=(ss_1, ss_2))

class DatumNode(object):

    def __init__(self, parent=None, data=None):
        self.children   = []
        self.parent     = parent
        self.vdict      = defaultdict(int)
        self.count      = 0
        self.none_count = 0
        self.nan_count  = 0
        self.inf_count  = 0
        self.str_count  = 0
        self.float_count = 0
        self.int_count = 0
        self.bool_count = 0
        self.obj_count = 0
        self.neg_count = 0
        self.zero_count = 0

        if data is not None:
            self.add_data(data)

    def add_data(self, x):
        for xx in x:
            self.add_datum(xx)

    def add_datum(self, x):
        self.count += 1

        if isinstance(x, (tuple, list)):
            for i,xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        elif isinstance(x, (Iterable,)) and not isinstance(x, (str,)):
            for i,xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        elif x is None:
            self.none_count += 1
        else:
            self.vdict[x] += 1
            self._analyze_type(x)

    def copy(self):
        rv = DatumNode(self.parent)
        rv.children = [u.copy() for u in self.children]
        rv.vdict = self.vdict.copy()
        return rv

    def merge(self, x):

        self.count += x.count
        self.none_count += x.none_count
        self.nan_count += x.nan_count

        for i in range(len(x.children)):
            temp = self._get_child_node(i).merge(x.children[i])
            self.children[i] = temp
        for k,v in x.vdict.items():
            self.vdict[k] += v

        self.neg_count += x.neg_count
        self.inf_count += x.inf_count
        self.int_count += x.int_count
        self.float_count += x.float_count
        self.obj_count += x.obj_count
        self.str_count += x.str_count
        self.bool_count += x.bool_count
        return self

    def _analyze_type(self, x, v=1):

        if isinstance(x, (float, np.float_)):
            if math.isnan(x):
                self.nan_count += v
            elif math.isinf(x):
                self.inf_count += v
            elif math.floor(x) == x:
                self.int_count += v
            else:
                self.float_count += v
            if x == 0:
                self.zero_count += v
            if math.isfinite(x) and x < 0:
                self.neg_count += v

        elif isinstance(x, (int, np.int_)):
            self.int_count += v
        elif isinstance(x, bool):
            self.bool_count += v
        elif isinstance(x, str):
            self.str_count += v
        else:
            self.obj_count += v

    def get_estimator(self, pseudo_count: Optional[float] = 1.0, emp_suff_stat: bool = True):
        # Value Type Node
        rv = get_ignored_estimator()

        if len(self.children) == 0 and len(self.vdict) > 0:
            if self.obj_count > 0:
                rv = get_ignored_estimator()
            elif self.str_count > 0:
                rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat)
            elif self.float_count > 0:
                rv = get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat)
            elif self.int_count > 0:
                if self.neg_count > 0:
                    rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat)
                else:
                    rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat)
                    # More checking before we use this
                    #rv = get_poisson_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
            else:
                rv = get_ignored_estimator()

        # Lists of Same Size
        elif len(self.children) > 0 and len(set([u.count for u in self.children])) == 1 and \
                all([u.count == self.count for u in self.children]):
            rv = get_composite_estimator([u.get_estimator(pseudo_count, emp_suff_stat) for u in self.children])

        # Lists of Different Size
        elif len(self.children) > 0 and len(set([u.count for u in self.children])) > 1:
            child = self.children[0].copy()
            for u in self.children[1:]:
                child = child.merge(u)
            rv = get_sequence_estimator(child.get_estimator(pseudo_count, emp_suff_stat))

        if self.none_count > 0:
            rv = get_optional_estimator(rv, None)

        if self.nan_count > 0:
            rv = get_optional_estimator(rv, math.nan)

        return rv

    def _get_child_node(self, idx: int):
        while len(self.children) <= idx:
            self.children.append(DatumNode(self))
        return self.children[idx]

def get_estimator(data, pseudo_count: Optional[float] = 1.0, emp_suff_stat: bool = True):
    return DatumNode(data=data).get_estimator(pseudo_count, emp_suff_stat)


