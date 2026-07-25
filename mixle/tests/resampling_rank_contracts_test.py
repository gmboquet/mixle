"""Inferential p-values require real resampling work and non-degenerate rank structure."""

import numpy as np
import pytest

from mixle.inference import bootstrap, permutation_test, wild_bootstrap
from mixle.inference.nonparametric import (
    brunner_munzel,
    dunn_test,
    friedman_test,
    kruskal_wallis,
    mann_whitney_u,
    mood_median_test,
    runs_test,
    sign_test,
    wilcoxon_signed_rank,
)


def test_resampling_requires_positive_work_and_valid_labels():
    with pytest.raises(ValueError, match="n_perm"):
        permutation_test(np.array([1.0]), np.array([2.0]), n_perm=0, exact_max=1)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap(np.arange(5.0), np.mean, n_boot=0)
    with pytest.raises(ValueError, match="missing"):
        bootstrap(np.arange(4.0), np.mean, n_boot=10, groups=np.array([0.0, 0.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="missing"):
        permutation_test(
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            stratify=np.array([0.0, np.nan, 0.0, 1.0]),
            n_perm=10,
            exact_max=1,
        )


def test_custom_two_sided_statistic_requires_null_center():
    statistic = lambda x, y: float(np.var(x) / np.var(y))
    with pytest.raises(ValueError, match="null_value"):
        permutation_test(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 5.0]),
            statistic=statistic,
            alternative="two-sided",
            exact_max=1,
            n_perm=20,
        )
    result = permutation_test(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 5.0]),
        statistic=statistic,
        alternative="two-sided",
        null_value=1.0,
        exact_max=1,
        n_perm=20,
    )
    assert 0.0 < result.pvalue <= 1.0


def test_bootstrap_rejects_nonfinite_or_shape_changing_statistics():
    with pytest.raises(ValueError, match="finite"):
        bootstrap(np.arange(5.0), lambda values: np.nan, n_boot=10)
    with pytest.raises(ValueError, match="matching"):
        wild_bootstrap(np.ones(3), np.ones(2), np.mean, n_boot=10)


@pytest.mark.parametrize(
    "call",
    [
        lambda: mann_whitney_u([1.0, 1.0], [1.0, 1.0]),
        lambda: brunner_munzel([1.0, 1.0], [1.0, 1.0]),
        lambda: kruskal_wallis([1.0, 1.0], [1.0, 1.0]),
        lambda: dunn_test([1.0, 1.0], [1.0, 1.0]),
        lambda: mood_median_test([1.0, 1.0], [1.0, 1.0]),
        lambda: wilcoxon_signed_rank([1.0, 1.0], [1.0, 1.0]),
        lambda: sign_test([1.0, 1.0], [1.0, 1.0]),
        lambda: friedman_test([1.0, 1.0], [1.0, 1.0], [1.0, 1.0]),
        lambda: runs_test([1.0, 1.0, 1.0]),
    ],
)
def test_rank_tests_reject_degenerate_reference_distributions(call):
    with pytest.raises(ValueError):
        call()


def test_degenerate_paths_still_validate_method_arguments():
    with pytest.raises(ValueError, match="alternative"):
        mann_whitney_u([1.0], [1.0], alternative="invalid")
    with pytest.raises(ValueError, match="distribution"):
        brunner_munzel([1.0, 2.0], [2.0, 3.0], distribution="invalid")
    with pytest.raises(ValueError, match="zero_method"):
        wilcoxon_signed_rank([0.0], zero_method="invalid")
