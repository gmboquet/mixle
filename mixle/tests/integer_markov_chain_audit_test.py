"""Regression tests for IntegerMarkovChainAccumulator keyed-parameter sharing.

Audit finding: IntegerMarkovChainAccumulator.key_merge / key_replace only
recursed into self.len_accumulator and never visited self.init_accumulator.
If the init distribution carried its own key, its sufficient statistics were
silently NOT pooled or replaced across the distributed EM reduce, breaking the
keyed-parameter-sharing contract.
"""

import numpy as np

from mixle.stats import IntegerCategoricalEstimator
from mixle.stats.sequences.integer_markov_chain import IntegerMarkovChainAccumulatorFactory


def _make_imc_acc_factory():
    init_f = IntegerCategoricalEstimator(keys="initk").accumulator_factory()
    return IntegerMarkovChainAccumulatorFactory(lag=1, init_factory=init_f)


def test_init_accumulator_key_sharing_pools_and_replaces():
    """Three IMC accumulators sharing an init key over data {2, 5, 9} must all
    carry the POOLED init statistics after merge-then-replace, not their own
    first batch."""
    imc_f = _make_imc_acc_factory()

    a = imc_f.make()
    b = imc_f.make()
    c = imc_f.make()

    # Each sees a distinct init observation.
    a.init_accumulator.update(1, 1.0, None)
    b.init_accumulator.update(3, 1.0, None)
    c.init_accumulator.update(5, 1.0, None)

    # Distributed reduce: merge into a shared dict, then replace from it.
    stats_dict: dict = {}
    for acc in (a, b, c):
        acc.key_merge(stats_dict)

    # The shared init slot must exist and carry the pooled counts (3 obs).
    assert "initk" in stats_dict
    _, pooled = stats_dict["initk"].value()
    assert pooled.sum() == 3.0

    for acc in (a, b, c):
        acc.key_replace(stats_dict)

    # All three init children must now carry the SAME pooled statistics, not
    # just their own first batch (sum == 1.0).
    for acc in (a, b, c):
        _, vals = acc.init_accumulator.value()
        assert vals.sum() == 3.0
        np.testing.assert_array_equal(vals, pooled)


def test_init_accumulator_key_replace_does_not_keep_first_batch():
    """Directly demonstrate the bug is gone: without the fix, b/c would retain
    only their own first batch after replace instead of the pooled value."""
    imc_f = _make_imc_acc_factory()
    a = imc_f.make()
    b = imc_f.make()

    a.init_accumulator.update(0, 1.0, None)
    b.init_accumulator.update(1, 1.0, None)

    stats_dict: dict = {}
    a.key_merge(stats_dict)
    b.key_merge(stats_dict)
    a.key_replace(stats_dict)
    b.key_replace(stats_dict)

    _, av = a.init_accumulator.value()
    _, bv = b.init_accumulator.value()
    # Both should see two counts (one at index 0, one at index 1).
    assert av.sum() == 2.0
    assert bv.sum() == 2.0
    np.testing.assert_array_equal(av, bv)
