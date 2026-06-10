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

def get_optional_estimator(est: ParameterEstimator, missing_value: Optional[Any] = None, use_bstats: bool = False):
    if use_bstats:
        from pysp.bstats.optional import OptionalEstimator
        return OptionalEstimator(est, missing_value=missing_value)
    from pysp.stats.optional import OptionalEstimator
    return OptionalEstimator(est, missing_value=missing_value)


def get_sequence_estimator(est: ParameterEstimator, use_bstats: bool = False) -> 'ParameterEstimator':
    if use_bstats:
        from pysp.bstats.sequence import SequenceEstimator
        return SequenceEstimator(est)
    from pysp.stats.sequence import SequenceEstimator
    return SequenceEstimator(est)

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

        if isinstance(x, (float, np.floating)):
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
        elif isinstance(x, bool):
            self.bool_count += v
        elif isinstance(x, str):
            self.str_count += v
        else:
            self.obj_count += v

    def get_estimator(self, pseudo_count: Optional[float] = 1.0, emp_suff_stat: bool = True, use_bstats: bool = False):
        # Value Type Node
        rv = get_ignored_estimator(use_bstats=use_bstats)

        if len(self.children) == 0 and len(self.vdict) > 0:
            if self.obj_count > 0:
                rv = get_ignored_estimator(use_bstats=use_bstats)
            elif self.str_count > 0:
                rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            elif self.float_count > 0:
                rv = get_gaussian_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
            elif self.int_count > 0:
                if self.neg_count > 0:
                    rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
                else:
                    rv = get_categorical_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats=use_bstats)
                    # More checking before we use this
                    #rv = get_poisson_estimator(self.vdict, pseudo_count, emp_suff_stat, use_bstats)
            else:
                rv = get_ignored_estimator(use_bstats=use_bstats)

        # Lists of Same Size
        elif len(self.children) > 0 and len(set([u.count for u in self.children])) == 1 and \
                all([u.count == self.count for u in self.children]):
            rv = get_composite_estimator(
                [u.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats) for u in self.children],
                use_bstats=use_bstats)

        # Lists of Different Size
        elif len(self.children) > 0 and len(set([u.count for u in self.children])) > 1:
            child = self.children[0].copy()
            for u in self.children[1:]:
                child = child.merge(u)
            rv = get_sequence_estimator(child.get_estimator(pseudo_count, emp_suff_stat, use_bstats=use_bstats),
                                        use_bstats=use_bstats)

        if self.none_count > 0:
            rv = get_optional_estimator(rv, None, use_bstats=use_bstats)

        if self.nan_count > 0:
            rv = get_optional_estimator(rv, math.nan, use_bstats=use_bstats)

        return rv

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


