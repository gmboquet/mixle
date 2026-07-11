"""Held-out model-selection benchmark for automatic family inference (worklist I6.4).

``get_estimator(data)`` picks a distribution family automatically. This is the honest receipt for how well
it recovers the *generating* family: draw from several known families, record which family the automatic
selector chooses (a confusion matrix), fit the chosen model and score held-out data, and time it. The bar is
NOT universal correct inference -- some families genuinely nest or overlap, and this test documents those
ambiguity regions as *acceptable* rather than pretending they are errors:

  * an Exponential is exactly a Gamma with shape 1, so selecting Gamma for exponential data is correct;
  * LogNormal and Inverse-Gaussian are both positive, right-skewed families that a finite sample cannot
    always separate.

Where the families are unambiguous (Gaussian, Poisson, Gamma), the selector must recover them; where they
overlap, the selection must land inside the documented acceptable set. The confusion matrix is printed as a
retained result.
"""

import time
import unittest
from collections import Counter

import numpy as np

from mixle.inference.estimation import optimize
from mixle.utils.automatic import get_estimator

_N = 1500  # sample size per trial
_TRIALS = 5  # datasets per family (different seeds)


def _draw(family, rng, n):
    if family == "gaussian":
        return rng.normal(3.0, 1.5, n).tolist()
    if family == "gamma":
        return rng.gamma(3.0, 2.0, n).tolist()
    if family == "poisson":
        return rng.poisson(5.0, n).tolist()
    if family == "exponential":
        return rng.exponential(2.0, n).tolist()
    if family == "lognormal":
        return rng.lognormal(0.0, 0.5, n).tolist()
    raise ValueError(family)


# For each generating family: the set of selected estimator classes that count as a CORRECT recovery,
# including documented nesting/overlap ambiguities.
_ACCEPTABLE = {
    "gaussian": {"GaussianEstimator"},
    "gamma": {"GammaEstimator"},
    "poisson": {"PoissonEstimator"},
    "exponential": {"ExponentialEstimator", "GammaEstimator"},  # Exponential == Gamma(shape=1)
    "lognormal": {"LogGaussianEstimator", "InverseGaussianEstimator"},  # both positive, right-skewed
}
# Families the selector must recover exactly (no nesting/overlap excuse).
_UNAMBIGUOUS = {"gaussian", "gamma", "poisson"}


class ModelSelectionBenchmarkTest(unittest.TestCase):
    def test_recovery_confusion_and_heldout_score(self):
        confusion = {}  # true family -> Counter of selected estimator class
        failures = 0
        total = 0
        t0 = time.time()
        for family in _ACCEPTABLE:
            counter = Counter()
            for seed in range(_TRIALS):
                rng = np.random.RandomState(1000 + seed)
                data = _draw(family, rng, _N)
                total += 1
                try:
                    est = get_estimator(data)
                    selected = type(est).__name__
                    counter[selected] += 1
                    # held-out log score: fit on train, score the held-out half; must be finite.
                    split = _N // 2
                    model = optimize(data[:split], est, max_its=20, out=None)
                    hold = data[split:]
                    ll = float(np.mean([model.log_density(x) for x in hold]))
                    self.assertTrue(np.isfinite(ll), f"{family}: held-out log score not finite ({ll})")
                except Exception:  # noqa: BLE001 - a crash IS a selection failure to count, not to hide
                    failures += 1
            confusion[family] = counter

        elapsed = time.time() - t0

        # Retained result: the confusion matrix and runtime (I6.4 "publish the confusion matrix").
        print("\nmodel-selection confusion (true family -> selected estimator counts):")
        for family, counter in confusion.items():
            print(f"  {family:12s} -> {dict(counter)}")
        print(f"failure rate: {failures}/{total} ; runtime: {elapsed:.2f}s")

        # No unexplained crashes.
        self.assertEqual(failures, 0, "automatic selection crashed on a well-formed sample")

        # Every selection must land in the documented acceptable set for its family.
        for family, counter in confusion.items():
            offenders = set(counter) - _ACCEPTABLE[family]
            self.assertEqual(offenders, set(), f"{family}: selected {offenders}, outside the acceptable/ambiguity set")

        # Unambiguous families must be recovered as the MODAL selection every trial.
        for family in _UNAMBIGUOUS:
            modal, count = confusion[family].most_common(1)[0]
            self.assertEqual(count, _TRIALS, f"{family}: recovered only {count}/{_TRIALS} (modal={modal})")


if __name__ == "__main__":
    unittest.main()
