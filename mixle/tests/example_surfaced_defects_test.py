"""Regressions for library defects that only a running example exposed.

Each of these passed the unit suite and failed the first time someone actually ran the documented
snippet. They are grouped because they share a cause: a guard, a default, or a pin that was written
against one configuration and never exercised against the others the library advertises.
"""

from __future__ import annotations

import numpy as np
import pytest

import mixle.stats as st
from mixle.inference import optimize


class ReducedPrecisionMassToleranceTest:
    """MXR-080-2011: the HMM responsibility-mass guard rejected every float32 fit."""

    def test_tolerance_admits_float32_summation_error_and_grows_with_mass(self) -> None:
        from mixle.stats.latent.hidden_markov import _responsibility_mass_tolerance

        # Roundoff accumulates with the number of summed terms, so a fixed bound is wrong at both
        # ends: too loose for a handful of values, too tight for thousands.
        assert _responsibility_mass_tolerance(8000.0) > _responsibility_mass_tolerance(1.0)
        # The measured mismatch on the workload that motivated this: 8000.000276 vs 8000.038925.
        assert _responsibility_mass_tolerance(8000.0) * 8000.0 > abs(8000.038925 - 8000.000276)
        # Still far tighter than any real corruption. A dropped state or double-counted transition
        # moves the identity by a fraction of the total, not by parts per million.
        assert _responsibility_mass_tolerance(8000.0) < 1.0e-3

    def test_a_corrupt_accumulator_is_still_refused(self) -> None:
        from mixle.stats.latent.hidden_markov import _responsibility_mass_tolerance

        for mass in (10.0, 1000.0, 1.0e6):
            # One percent off is not rounding at any supported precision.
            assert not np.isclose(mass * 1.01, mass, rtol=_responsibility_mass_tolerance(mass))

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    def test_an_hmm_fits_at_every_precision_the_engine_offers(self, dtype: str) -> None:
        torch = pytest.importorskip("torch")
        del torch
        from mixle.engines import TorchEngine

        rng = np.random.RandomState(0)
        seqs = [[float(v) for v in rng.normal(mean, 1.0, 40)] for mean in (0.0, 4.0) for _ in range(50)]
        estimator = st.HiddenMarkovEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        seed = optimize(seqs, estimator, max_its=2, out=None)

        # TorchEngine documents float32 as an explicitly opted-in precision, and MPS forces it
        # outright -- so refusing it is refusing a configuration the library itself offers.
        model = optimize(
            seqs,
            estimator,
            max_its=5,
            engine=TorchEngine(device="cpu", dtype=dtype),
            prev_estimate=seed,
            out=None,
        )
        assert model is not None


class SemiSupervisedRowCountTest:
    """MXR-080-2012: the encoder never overrode row_count() for its own payload layout."""

    def test_row_count_reads_the_encoded_observation_count(self) -> None:
        data = [(1.0, None), (5.0, [(1, 1.0)]), (1.2, None), (4.8, None)]
        estimator = st.SemiSupervisedMixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        encoder = optimize(data, estimator, max_its=3, out=None).dist_to_encoder()

        # The base implementation infers a count only from payloads with an unambiguous leading
        # axis. This one is a heterogeneous 4-tuple whose prior arrays are indexed by *labelled*
        # row, so they are shorter than the data whenever anything is unlabelled -- as here.
        assert encoder.row_count(encoder.seq_encode(data)) == len(data)

    def test_a_malformed_payload_is_refused_rather_than_guessed_at(self) -> None:
        estimator = st.SemiSupervisedMixtureEstimator([st.GaussianEstimator(), st.GaussianEstimator()])
        encoder = optimize([(1.0, None), (5.0, None)], estimator, max_its=2, out=None).dist_to_encoder()

        for bad in ((), [], (-1, None, None, None), ("4", None, None, None), (True, None, None, None)):
            with pytest.raises(ValueError):
                encoder.row_count(bad)


class InitializationDrawSelectsSomethingTest:
    """MXR-080-2014: the Bernoulli init mask could select nothing, and leaves disagreed about it."""

    def test_a_random_forest_initializes_at_the_sizes_a_doc_example_uses(self) -> None:
        from mixle.models import RandomForestEstimator

        # p defaults to 0.1, so at two observations an empty draw is the MAJORITY outcome (0.81).
        # This previously failed 10/10 at n=2, 7/10 at n=10, and still 1/10 at n=40 -- and the error
        # blamed "weights", which the caller never supplied.
        for n in (2, 10, 40):
            for trial in range(5):
                rng = np.random.RandomState(trial)
                x = rng.normal(size=(n, 3))
                data = [(list(map(float, row)), int(row[0] > 0)) for row in x]
                model = optimize(
                    data,
                    RandomForestEstimator(task="classification", n_estimators=4, max_depth=3),
                    max_its=2,
                    rng=np.random.RandomState(trial),
                    out=None,
                )
                assert model is not None

    def test_the_empty_draw_is_the_only_outcome_that_is_altered(self) -> None:
        # A forced selection would bias initialization if it fired on ordinary draws. It must fire
        # only when the pass selected nothing at all, so a mask with any positive weight is passed
        # through byte-for-byte.
        seen = []

        class _Recording(st.GaussianEstimator):
            pass

        rng = np.random.RandomState(0)
        data = [float(v) for v in rng.normal(0.0, 1.0, 400)]
        estimator = _Recording()
        accumulator_factory = estimator.accumulator_factory()
        original = type(accumulator_factory.make()).seq_initialize

        def _record(self, x, weights, rng_):  # noqa: ANN001, ANN202
            seen.append(np.asarray(weights, dtype=float).copy())
            return original(self, x, weights, rng_)

        type(accumulator_factory.make()).seq_initialize = _record
        try:
            optimize(data, estimator, max_its=1, rng=np.random.RandomState(1), out=None)
        finally:
            type(accumulator_factory.make()).seq_initialize = original

        assert seen, "the seq_ initialization path did not run"
        total = sum(float(w.sum()) for w in seen)
        assert total > 0.0
        # 400 observations at p=0.1: an empty draw is impossible in practice, so the fallback must
        # not have fired and the mask must still look Bernoulli rather than like a single forced 1.
        assert total > 20.0
