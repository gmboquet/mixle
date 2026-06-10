"""Automatic detection of data type for estimators.

Builds estimators for pysp.stats by default; pass use_bstats=True to build the
Bayesian (conjugate-prior) estimators from pysp.bstats instead. get_dpm_mixture
fits a Dirichlet process mixture over automatically-typed data with
variational inference.
"""
import math
import numbers
import numpy as np
from collections import defaultdict
from collections.abc import Iterable

from pysp.stats.pdist import ParameterEstimator

from typing import Optional, Any, Sequence, Dict, TypeVar, Union
T = TypeVar('T')

# Leaf-typing heuristics: integers with at most this many distinct values (or
# at most this fraction of observations) are modeled as categorical rather
# than Poisson/Gaussian; string fields where nearly every value is unique are
# treated as identifiers and ignored.
MAX_INT_CATEGORICAL_DISTINCT = 20
MAX_INT_CATEGORICAL_FRACTION = 0.05
ID_DISTINCT_FRACTION = 0.95
ID_MIN_COUNT = 100


def get_optional_estimator(est: ParameterEstimator, missing_value: Optional[Any] = None, use_bstats: bool = False):
    if use_bstats:
        from pysp.bstats.optional import OptionalEstimator
        return OptionalEstimator(est, missing_value=missing_value)
    from pysp.stats.optional import OptionalEstimator
    return OptionalEstimator(est, missing_value=missing_value)


def get_length_estimator(len_dict: Dict[int, int], pseudo_count: Optional[float] = None,
                         emp_suff_stat: bool = True, use_bstats: bool = False) -> 'ParameterEstimator':
    """Length model for sequences: categorical for a single observed length,
    Poisson (count) otherwise."""
    if len(len_dict) <= 1:
        return get_categorical_estimator(dict(len_dict), pseudo_count, emp_suff_stat, use_bstats=use_bstats)
    if use_bstats:
        from pysp.bstats.poisson import PoissonEstimator
        return PoissonEstimator()
    return get_poisson_estimator(dict(len_dict), pseudo_count, emp_suff_stat)


def get_sequence_estimator(est: ParameterEstimator, len_dict: Optional[Dict[int, int]] = None,
                           pseudo_count: Optional[float] = None, emp_suff_stat: bool = True,
                           use_bstats: bool = False) -> 'ParameterEstimator':
    len_est = None
    if len_dict:
        len_est = get_length_estimator(len_dict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
    if use_bstats:
        from pysp.bstats.sequence import SequenceEstimator
        return SequenceEstimator(est) if len_est is None else SequenceEstimator(est, len_estimator=len_est)
    from pysp.stats.sequence import SequenceEstimator
    return SequenceEstimator(est) if len_est is None else SequenceEstimator(est, len_estimator=len_est)


def get_set_estimator(member_dict: Dict[Any, int], num_sets: int, pseudo_count: Optional[float] = None,
                      emp_suff_stat: bool = True, use_bstats: bool = False) -> 'ParameterEstimator':
    """Bernoulli set model with membership probabilities from observed sets."""
    if use_bstats:
        from pysp.bstats.setdist import BernoulliSetEstimator
        return BernoulliSetEstimator()
    from pysp.stats.setdist import BernoulliSetEstimator
    suff_stat = None
    if emp_suff_stat and num_sets > 0:
        suff_stat = {k: v / num_sets for k, v in member_dict.items()}
    return BernoulliSetEstimator(pseudo_count=pseudo_count, suff_stat=suff_stat)

def get_ignored_estimator(use_bstats: bool = False) -> 'ParameterEstimator':
    if use_bstats:
        from pysp.bstats.ignored import IgnoredEstimator
        return IgnoredEstimator()
    from pysp.stats.ignored import IgnoredEstimator
    return IgnoredEstimator()

def get_composite_estimator(ests: Sequence[ParameterEstimator], use_bstats: bool = False) -> 'ParameterEstimator':
    if use_bstats:
        from pysp.bstats.composite import CompositeEstimator
        return CompositeEstimator(ests)
    from pysp.stats.composite import CompositeEstimator
    return CompositeEstimator(ests)

def get_categorical_estimator(vdict: Dict[T, float], pseudo_count: Optional[float] = None, emp_suff_stat: bool = True,
                              use_bstats: bool = False) -> 'ParameterEstimator':
    if use_bstats:
        from pysp.bstats.categorical import CategoricalEstimator
        from pysp.bstats.catdirichlet import DictDirichletDistribution
        alpha = 1.0 if pseudo_count is None else pseudo_count
        return CategoricalEstimator(prior=DictDirichletDistribution({k: alpha for k in vdict.keys()}))

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
                           emp_suff_stat: bool = True, use_bstats: bool = False) -> 'ParameterEstimator':

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

    if use_bstats:
        from pysp.bstats.gaussian import GaussianEstimator
        from pysp.bstats.normgamma import NormalGammaDistribution

        # weakly data-informed normal-gamma prior centered on the empirical
        # moments (when available)
        mu0 = ss_1 if ss_1 is not None else 0.0
        v0 = ss_2 if (ss_2 is not None and ss_2 > 0) else 1.0
        a0 = 1.001
        prior = NormalGammaDistribution(mu0, 1.0e-3, a0, a0*v0)
        return GaussianEstimator(prior=prior)

    from pysp.stats.gaussian import GaussianEstimator
    return GaussianEstimator(pseudo_count=(pseudo_count, pseudo_count), suff_stat=(ss_1, ss_2))

class DatumNode(object):
    """Accumulates type/structure evidence for one slot of the data.

    Tuples are treated as fixed-arity records (positional children). Lists,
    arrays, and other sized iterables are positional only if every observation
    has the same length (vector semantics); otherwise they are variable-length
    sequences of a merged element type with a length model. Sets map to a
    Bernoulli set model and dicts are ignored.
    """

    def __init__(self, parent=None, data=None):
        self.children   = []
        self.parent     = parent
        self.vdict      = defaultdict(int)
        self.len_dict   = defaultdict(int)
        self.set_member = defaultdict(int)
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
        self.tuple_count = 0
        self.seq_count = 0
        self.set_count = 0

        if data is not None:
            self.add_data(data)

    def add_data(self, x):
        for xx in x:
            self.add_datum(xx)

    def add_datum(self, x):
        self.count += 1

        if x is None:
            self.none_count += 1
        elif isinstance(x, (str, bytes)):
            self.vdict[x] += 1
            self._analyze_type(x)
        elif isinstance(x, tuple):
            self.tuple_count += 1
            self.len_dict[len(x)] += 1
            for i, xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        elif isinstance(x, (set, frozenset)):
            self.set_count += 1
            self.len_dict[len(x)] += 1
            for xx in x:
                self.set_member[xx] += 1
        elif isinstance(x, dict):
            self.obj_count += 1
        elif isinstance(x, Iterable):
            x = list(x)
            self.seq_count += 1
            self.len_dict[len(x)] += 1
            for i, xx in enumerate(x):
                self._get_child_node(i).add_datum(xx)
        else:
            self._analyze_type(x)
            if not (isinstance(x, (float, np.floating)) and not math.isfinite(x)):
                self.vdict[x] += 1

    _COUNTERS = ('count', 'none_count', 'nan_count', 'inf_count', 'str_count', 'float_count',
                 'int_count', 'bool_count', 'obj_count', 'neg_count', 'zero_count',
                 'tuple_count', 'seq_count', 'set_count')

    def copy(self):
        rv = DatumNode(self.parent)
        rv.children = [u.copy() for u in self.children]
        rv.vdict = self.vdict.copy()
        rv.len_dict = self.len_dict.copy()
        rv.set_member = self.set_member.copy()
        for c in self._COUNTERS:
            setattr(rv, c, getattr(self, c))
        return rv

    def merge(self, x):
        for c in self._COUNTERS:
            setattr(self, c, getattr(self, c) + getattr(x, c))

        for i in range(len(x.children)):
            self.children[i] = self._get_child_node(i).merge(x.children[i])
        for k, v in x.vdict.items():
            self.vdict[k] += v
        for k, v in x.len_dict.items():
            self.len_dict[k] += v
        for k, v in x.set_member.items():
            self.set_member[k] += v

        return self

    def _analyze_type(self, x, v=1):

        if isinstance(x, (bool, np.bool_)):
            self.bool_count += v
        elif isinstance(x, (float, np.floating)):
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
        elif isinstance(x, (int, np.integer)):
            self.int_count += v
            if x == 0:
                self.zero_count += v
            if x < 0:
                self.neg_count += v
        elif isinstance(x, (str, bytes)):
            self.str_count += v
        else:
            self.obj_count += v

    def _leaf_estimator(self, pseudo_count, emp_suff_stat, use_bstats):
        if self.obj_count > 0 or len(self.vdict) == 0:
            return get_ignored_estimator(use_bstats=use_bstats)

        if self.str_count > 0:
            # identifier-like fields (nearly all values distinct) carry no
            # density information; ignore them instead of fitting a
            # one-bucket-per-row categorical
            if self.count >= ID_MIN_COUNT and len(self.vdict) >= ID_DISTINCT_FRACTION * self.count:
                return get_ignored_estimator(use_bstats=use_bstats)
            return get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        if self.bool_count > 0 and self.float_count == 0 and self.int_count == 0:
            return get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        if self.float_count > 0:
            return get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        if self.int_count > 0:
            distinct = len(self.vdict)
            if distinct <= max(MAX_INT_CATEGORICAL_DISTINCT, MAX_INT_CATEGORICAL_FRACTION * self.count):
                return get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            if self.neg_count == 0:
                if use_bstats:
                    from pysp.bstats.poisson import PoissonEstimator
                    return PoissonEstimator()
                return get_poisson_estimator(self.vdict, pseudo_count, emp_suff_stat)
            return get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)

        return get_ignored_estimator(use_bstats=use_bstats)

    def _merged_child(self):
        child = self.children[0].copy()
        for u in self.children[1:]:
            child = child.merge(u)
        return child

    def get_estimator(self, pseudo_count: Optional[float] = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
        structured = self.tuple_count + self.seq_count + self.set_count
        typed = self.count - self.none_count

        if typed == 0:
            rv = get_ignored_estimator(use_bstats=use_bstats)

        elif structured > 0 and (len(self.vdict) > 0 or self.obj_count > 0 or
                                 (self.set_count > 0 and self.set_count < structured)):
            # mixed scalars/containers or mixed container kinds: not modelable
            rv = get_ignored_estimator(use_bstats=use_bstats)

        elif self.set_count > 0:
            rv = get_set_estimator(self.set_member, self.set_count, pseudo_count, emp_suff_stat,
                                   use_bstats=use_bstats)

        elif structured > 0:
            fixed_arity = len(self.len_dict) == 1
            if self.tuple_count > 0 and self.seq_count == 0 and fixed_arity:
                # records: positional composite
                rv = get_composite_estimator(
                    [u.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats) for u in self.children],
                    use_bstats=use_bstats)
            elif fixed_arity and self.tuple_count == 0 and not self._children_homogeneous():
                # fixed-length lists/vectors with positionally distinct types
                rv = get_composite_estimator(
                    [u.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats) for u in self.children],
                    use_bstats=use_bstats)
            else:
                # variable-length (or homogeneous fixed-length) sequences
                child = self._merged_child()
                rv = get_sequence_estimator(
                    child.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats),
                    len_dict=self.len_dict, pseudo_count=pseudo_count, emp_suff_stat=emp_suff_stat,
                    use_bstats=use_bstats)

        else:
            rv = self._leaf_estimator(pseudo_count, emp_suff_stat, use_bstats)

        if self.none_count > 0:
            rv = get_optional_estimator(rv, None, use_bstats=use_bstats)

        if self.nan_count > 0:
            rv = get_optional_estimator(rv, math.nan, use_bstats=use_bstats)

        return rv

    def _children_homogeneous(self):
        """True when all positional children carry the same scalar type profile,
        so a fixed-length list is better modeled as an iid sequence than a
        composite of per-position estimators."""
        if len(self.children) <= 1:
            return True

        def profile(u):
            return (u.str_count > 0, u.bool_count > 0, u.float_count > 0, u.int_count > 0,
                    u.obj_count > 0, len(u.children) > 0)

        profiles = {profile(u) for u in self.children}
        if len(profiles) > 1:
            return False

        # numeric positions with disjoint supports look like distinct dimensions
        p = next(iter(profiles))
        if p[2] or p[3]:
            return False

        return True

    def _get_child_node(self, idx: int):
        while len(self.children) <= idx:
            self.children.append(DatumNode(self))
        return self.children[idx]

def get_estimator(data, pseudo_count: Optional[float] = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
    return DatumNode(data=data).get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats)


def get_dpm_mixture(data, rng=None, max_components: int = 20, max_its: int = 100, delta: float = 1.0e-6,
                    pseudo_count: Optional[float] = 1.0, print_iter: int = 1, out=None):
    """Fit a Dirichlet process mixture to automatically-typed data.

    Component estimators are constructed with get_estimator(use_bstats=True)
    (one independent instance per stick), and the truncated stick-breaking
    posterior is fit with variational inference via pysp.bstats.bestimation.optimize.

    Args:
        data: Sequence of observations of any auto-detectable type.
        rng (Optional[RandomState]): Source of randomness for initialization.
        max_components (int): Truncation level of the stick-breaking representation.
        max_its (int): Maximum number of variational iterations.
        delta (float): Stop when the ELBO improves by less than delta.
        pseudo_count (Optional[float]): Prior strength for the component priors.
        print_iter (int): Progress print frequency.
        out: Output stream for iteration logging (defaults to sys.stdout).

    Returns:
        DirichletProcessMixtureDistribution fit to the data.
    """
    import sys
    from pysp.bstats.dpm import DirichletProcessMixtureEstimator
    from pysp.bstats.bestimation import optimize

    if rng is None:
        rng = np.random.RandomState()
    if out is None:
        out = sys.stdout

    comp_ests = [get_estimator(data, pseudo_count=pseudo_count, use_bstats=True) for _ in range(max_components)]
    est = DirichletProcessMixtureEstimator(comp_ests)

    return optimize(data, est, max_its=max_its, delta=delta, rng=rng, print_iter=print_iter, out=out)


