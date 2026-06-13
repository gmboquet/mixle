import unittest

import numpy as np

from pysp.stats import (
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    KeyValidationError,
    PoissonEstimator,
    seq_encode,
    seq_estimate,
    validate_estimator_keys,
)


class KeyValidationTestCase(unittest.TestCase):

    def test_same_family_same_settings_key_passes(self):
        est = CompositeEstimator((
            GaussianEstimator(keys='shared_gaussian'),
            GaussianEstimator(keys='shared_gaussian'),
        ))
        validate_estimator_keys(est)

    def test_cross_family_key_fails(self):
        est = CompositeEstimator((
            GaussianEstimator(keys='bad_key'),
            PoissonEstimator(keys='bad_key'),
        ))
        with self.assertRaises(KeyValidationError):
            validate_estimator_keys(est)

    def test_same_family_different_settings_key_fails(self):
        est = CompositeEstimator((
            GaussianEstimator(pseudo_count=(1.0, 1.0), suff_stat=(0.0, 1.0), keys='bad_key'),
            GaussianEstimator(pseudo_count=(2.0, 1.0), suff_stat=(0.0, 1.0), keys='bad_key'),
        ))
        with self.assertRaises(KeyValidationError):
            validate_estimator_keys(est)

    def test_seq_estimate_validates_before_combining_stats(self):
        model = GaussianDistribution(0.0, 1.0)
        enc = seq_encode(model.sampler(seed=1).sample(10), model=model)
        est = CompositeEstimator((
            GaussianEstimator(keys='bad_key'),
            PoissonEstimator(keys='bad_key'),
        ))
        with self.assertRaises(KeyValidationError):
            seq_estimate(enc, est, model)


if __name__ == '__main__':
    unittest.main()
