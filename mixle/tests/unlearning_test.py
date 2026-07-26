"""P5 commitment-backed exact unlearning tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mixle.experimental.unlearning import (
    certify_unlearning,
    prepare_unlearning,
    retained_records,
    shard_statistic,
    unlearn,
)
from mixle.inference.estimation import optimize
from mixle.stats import CategoricalEstimator, GaussianEstimator, PoissonEstimator


def _shards(rng, kind):
    if kind == "gaussian":
        return [rng.normal(3.0, 2.0, 30).tolist() for _ in range(4)]
    if kind == "categorical":
        return [rng.choice(list("abcd"), 25).tolist() for _ in range(4)]
    return [rng.poisson(4.0, 30).tolist() for _ in range(4)]


def _prepare(estimator, shards, excluded):
    records, manifest = prepare_unlearning(estimator, shards)
    retained = retained_records(records, exclude=excluded)
    return retained, manifest


def test_certificate_is_bitwise_exact_for_audited_closed_form_leaves() -> None:
    for kind, estimator in [
        ("gaussian", GaussianEstimator()),
        ("categorical", CategoricalEstimator()),
        ("poisson", PoissonEstimator()),
    ]:
        shards = _shards(np.random.default_rng(0), kind)
        excluded = {"shard-00000001"}
        retained, manifest = _prepare(estimator, shards, excluded)
        _, certificate = certify_unlearning(
            estimator,
            retained,
            manifest=manifest,
            exclude=excluded,
            expected_manifest_digest=manifest.digest,
        )

        assert certificate.bitwise_exact, f"{kind}: retained re-reduction was not bitwise exact"
        assert certificate.method == "commitment-rereduce-v1"
        assert certificate.n_excluded == 1
        assert certificate.n_retained_shards == 3
        assert certificate.n_shards_total == 4
        assert not certificate.raw_data_accessed
        assert not certificate.excluded_statistics_accessed


def test_unlearn_matches_the_never_saw_it_fit() -> None:
    shards = _shards(np.random.default_rng(1), "gaussian")
    estimator = GaussianEstimator()
    excluded = {"shard-00000002"}
    retained, manifest = _prepare(estimator, shards, excluded)
    unlearned = unlearn(
        estimator,
        retained,
        manifest=manifest,
        exclude=excluded,
        expected_manifest_digest=manifest.digest,
    )

    raw_retained = [value for index, shard in enumerate(shards) if index != 2 for value in shard]
    scratch = optimize(raw_retained, GaussianEstimator(), out=None)
    assert np.isclose(unlearned.mu, scratch.mu)
    assert np.isclose(unlearned.sigma2, scratch.sigma2)


def test_certification_never_rereads_raw_or_excluded_statistics() -> None:
    class OneShotShard:
        def __init__(self, values):
            self.values = values
            self.reads = 0

        def __iter__(self):
            self.reads += 1
            if self.reads > 1:
                raise AssertionError("raw shard was reread after ingestion")
            return iter(self.values)

    raw = {
        "alice": OneShotShard([1.0, 2.0]),
        "bob": OneShotShard([3.0, 4.0]),
        "carol": OneShotShard([5.0, 6.0]),
    }
    estimator = GaussianEstimator()
    records, manifest = prepare_unlearning(estimator, raw)
    retained = retained_records(records, exclude={"bob"})
    del records

    _, certificate = certify_unlearning(
        estimator,
        retained,
        manifest=manifest,
        exclude={"bob"},
        expected_manifest_digest=manifest.digest,
    )

    assert certificate.bitwise_exact
    assert all(shard.reads == 1 for shard in raw.values())
    assert set(retained) == {"alice", "carol"}


def test_unknown_exclusion_ids_and_incomplete_retained_store_are_rejected() -> None:
    estimator = GaussianEstimator()
    records, manifest = prepare_unlearning(estimator, {"a": [1.0], "b": [2.0], "c": [3.0]})

    with pytest.raises(ValueError, match="unknown exclusion IDs"):
        retained_records(records, exclude={"missing"})

    retained = retained_records(records, exclude={"b"})
    del retained["a"]
    with pytest.raises(ValueError, match="missing=.*a"):
        certify_unlearning(
            estimator,
            retained,
            manifest=manifest,
            exclude={"b"},
            expected_manifest_digest=manifest.digest,
        )


def test_tampered_record_or_manifest_anchor_is_rejected() -> None:
    estimator = GaussianEstimator()
    records, manifest = prepare_unlearning(estimator, {"a": [1.0, 2.0], "b": [3.0, 4.0]})
    retained = retained_records(records, exclude={"b"})
    retained["a"] = replace(retained["a"], value=(999.0, 999.0, 2.0, 2.0))

    with pytest.raises(ValueError, match="commitment check"):
        certify_unlearning(
            estimator,
            retained,
            manifest=manifest,
            exclude={"b"},
            expected_manifest_digest=manifest.digest,
        )

    retained = retained_records(records, exclude={"b"})
    with pytest.raises(ValueError, match="external digest anchor"):
        certify_unlearning(
            estimator,
            retained,
            manifest=manifest,
            exclude={"b"},
            expected_manifest_digest="0" * 64,
        )


def test_iterative_or_unregistered_estimator_is_refused_before_ingestion() -> None:
    class IterativeEstimator:
        pass

    with pytest.raises(TypeError, match="not an audited additive single-step estimator"):
        prepare_unlearning(IterativeEstimator(), [[1.0], [2.0]])


def test_subtraction_is_not_the_certified_method_and_can_go_invalid() -> None:
    """Adversarial shard: subtract catastrophically cancels while re-reduction remains well conditioned."""
    rng = np.random.default_rng(7)
    retained = [rng.normal(0.0, 1.0, 200) for _ in range(3)]
    excluded = rng.normal(1e10, 1.0, 200)

    def stats(values):
        values = np.asarray(values, dtype=float)
        return float(values.size), float(values.sum()), float((values * values).sum())

    all_shards = [*retained, excluded]
    count = sum(stats(shard)[0] for shard in all_shards)
    total = sum(stats(shard)[1] for shard in all_shards)
    squares = sum(stats(shard)[2] for shard in all_shards)
    excluded_count, excluded_total, excluded_squares = stats(excluded)
    sub_count = count - excluded_count
    sub_total = total - excluded_total
    sub_squares = squares - excluded_squares
    sub_mean = sub_total / sub_count
    variance_subtract = sub_squares / sub_count - sub_mean**2

    retained_count = sum(stats(shard)[0] for shard in retained)
    retained_total = sum(stats(shard)[1] for shard in retained)
    retained_squares = sum(stats(shard)[2] for shard in retained)
    retained_mean = retained_total / retained_count
    variance_rereduce = retained_squares / retained_count - retained_mean**2

    assert 0.5 < variance_rereduce < 2.0
    assert variance_subtract < 0.0 or abs(variance_subtract - variance_rereduce) > 0.5


def test_determinism_and_public_single_shard_ingestion() -> None:
    estimator = GaussianEstimator()
    record = shard_statistic(estimator, [1.0, 2.0], shard_id="direct")
    assert record.shard_id == "direct"
    assert len(record.commitment) == 64

    shards = _shards(np.random.default_rng(5), "gaussian")
    excluded = {"shard-00000001"}
    retained, manifest = _prepare(estimator, shards, excluded)
    model_one, certificate_one = certify_unlearning(
        estimator,
        retained,
        manifest=manifest,
        exclude=excluded,
        expected_manifest_digest=manifest.digest,
    )
    model_two, certificate_two = certify_unlearning(
        estimator,
        retained,
        manifest=manifest,
        exclude=excluded,
        expected_manifest_digest=manifest.digest,
    )
    assert vars(model_one) == vars(model_two)
    assert certificate_one.as_dict() == certificate_two.as_dict()
