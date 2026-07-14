"""Tests for the D4 leaf hot-swap / analytic roll-up (mixle.inference.leaf_hotswap).

Acceptance criteria under test (see the ConditionalJIT track's D4 item):

1. Held-out density preserved within tolerance -- a plateaued gradient leaf's moment-matched
   surrogate scores genuinely held-out data almost as well as the original gradient leaf did.
2. Faster to same F -- once a leaf has plateaued, updating its (closed-form) surrogate is
   measurably cheaper than continuing to run gradient M-steps on it.
3. Swap-back on misfit -- when the surrogate's fit quality genuinely degrades (a real, computed
   misfit receipt crossing the tolerance), the operator swaps back to the retained original
   gradient leaf, both at the primitive level and end-to-end inside ``run_em_with_hotswap``.
"""

import time
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.inference.leaf_hotswap import (  # noqa: E402
    PlateauMonitor,
    misfit_receipt,
    moment_matched_surrogate,
    run_em_with_hotswap,
    should_swap_back,
    swap_back,
    swap_leaf,
)
from mixle.models import GradLeaf  # noqa: E402
from mixle.models.grad_leaf import GradEstimator  # noqa: E402
from mixle.stats import (  # noqa: E402
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
)


class DiagGauss(torch.nn.Module):
    """The smallest honest density module: a learnable diagonal Gaussian (mirrors grad_leaf_test)."""

    def __init__(self, dim: int = 1, mu0: float = 0.0, log_sigma0: float = 0.0):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.full((dim,), float(mu0)))
        self.log_sigma = torch.nn.Parameter(torch.full((dim,), float(log_sigma0)))

    def _dist(self):
        return torch.distributions.Normal(self.mu, torch.exp(self.log_sigma))

    def log_density(self, x):
        return self._dist().log_prob(x).sum(-1)

    def sample(self, n: int):
        return self._dist().sample((n,))


class BimodalGauss(torch.nn.Module):
    """A genuinely bimodal density module (a learnable mixture-of-2-Gaussians): unlike
    :class:`DiagGauss`, its architecture is NOT constrained to be Gaussian-shaped, so a
    moment-matched Gaussian surrogate cannot trivially match it -- this is the adversarial fixture
    the shipped acceptance test (criterion 1, "held-out density preserved") never actually
    exercised, since :class:`DiagGauss` is itself Gaussian and a Gaussian surrogate matching a
    Gaussian ground truth proves nothing about the general claim."""

    def __init__(self, mu0=(-1.0, 1.0), log_sigma0: float = 0.0):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(list(mu0), dtype=torch.float32))
        self.log_sigma = torch.nn.Parameter(torch.full((2,), float(log_sigma0)))
        self.logit_w = torch.nn.Parameter(torch.zeros(2))

    def log_density(self, x):
        x = x.reshape(-1, 1)
        sigma = torch.exp(self.log_sigma)
        comp_ll = torch.distributions.Normal(self.mu, sigma).log_prob(x)
        log_w = torch.log_softmax(self.logit_w, dim=0)
        return torch.logsumexp(comp_ll + log_w, dim=-1)

    def sample(self, n: int):
        w = torch.softmax(self.logit_w, dim=0)
        comp = torch.multinomial(w, n, replacement=True)
        sigma = torch.exp(self.log_sigma)
        mu = self.mu[comp]
        s = sigma[comp]
        return (mu + s * torch.randn(n)).reshape(-1, 1)


def _data(mu, sigma, n, seed):
    rng = np.random.RandomState(seed)
    return [float(v) for v in rng.normal(mu, sigma, n)]


def _bimodal_data(n, seed, mu=(-3.0, 3.0), sigma=0.5):
    """``n`` draws from an evenly-weighted N(``mu[0]``, ``sigma``) / N(``mu[1]``, ``sigma``) mixture
    -- the adversarial "arbitrary complex leaf" ground truth for :class:`BimodalGauss`."""
    rng = np.random.RandomState(seed)
    n1 = n // 2
    x = np.concatenate([rng.normal(mu[0], sigma, n1), rng.normal(mu[1], sigma, n - n1)])
    rng.shuffle(x)
    return [float(v) for v in x]


def _fit_bimodal_near_convergence(data, seed, m_steps=400, max_its=60):
    """Fit a :class:`BimodalGauss` ``GradLeaf`` close to its optimum on ``data``, mirroring
    :func:`_fit_near_convergence`'s two-stage (EM-driven ``optimize`` then explicit extra gradient
    rounds) convergence discipline."""
    torch.manual_seed(seed)
    fitted = optimize(data, BimodalGauss(mu0=(-1.0, 1.0)), max_its=max_its, out=None, prev_estimate=None)
    estimator = GradEstimator(fitted.module, m_steps=m_steps, lr=5e-3)
    acc = estimator.accumulator_factory().make()
    enc = np.asarray(data)[:, None]
    acc.seq_update(enc, np.ones(len(data)), None)
    return estimator.estimate(float(len(data)), acc.value())


def _fit_near_convergence(mu=3.0, sigma=1.0, n=600, seed=0, m_steps=200, max_its=40):
    """Fit a DiagGauss GradLeaf close to its optimum -- the "plateaued" starting point every test
    in this module needs (Q-gain genuinely near zero, not merely assumed)."""
    torch.manual_seed(seed)  # the torch side of the fixture, like _fit_bimodal_near_convergence
    data = _data(mu, sigma, n, seed)
    fitted = optimize(data, DiagGauss(1, mu0=mu + 1.0), max_its=max_its, out=None, prev_estimate=None)
    # re-run a handful of extra gradient rounds explicitly via GradEstimator so the residual has
    # genuinely stopped moving (optimize()'s own delta-gated EM loop already does this, this just
    # makes the plateau assumption an explicit, checkable part of the fixture rather than implicit).
    estimator = GradEstimator(fitted.module, m_steps=m_steps, lr=5e-3)
    acc = estimator.accumulator_factory().make()
    enc = np.asarray(data)[:, None]
    acc.seq_update(enc, np.ones(len(data)), None)
    fitted = estimator.estimate(float(len(data)), acc.value())
    return fitted, data


class HeldOutDensityPreservedTestCase(unittest.TestCase):
    def test_surrogate_matches_gradient_leaf_on_held_out_data(self):
        fitted, train_data = _fit_near_convergence(mu=3.0, sigma=1.0, n=600, seed=1)
        holdout = np.asarray(_data(3.0, 1.0, 400, seed=99))[:, None]

        surrogate = moment_matched_surrogate(fitted, train_data)

        grad_nll = float(-np.mean(fitted.seq_log_density(holdout)))
        surrogate_nll = float(-np.mean(surrogate.seq_log_density(holdout)))

        print(f"[D4] held-out NLL: gradient leaf={grad_nll:.4f} nats, surrogate={surrogate_nll:.4f} nats")

        # true generating density's held-out NLL is the differential entropy of N(3, 1) -- a lower
        # bound neither fit can beat -- so compare both fits against it, not just against each other.
        true_entropy = 0.5 * np.log(2.0 * np.pi * np.e * 1.0**2)
        self.assertLess(abs(grad_nll - true_entropy), 0.15)
        self.assertLess(abs(surrogate_nll - grad_nll), 0.05, "surrogate held-out NLL should track the gradient leaf's")

    def test_moment_matched_surrogate_recovers_mean_and_variance(self):
        fitted, train_data = _fit_near_convergence(mu=-2.0, sigma=0.5, n=800, seed=2)
        surrogate = moment_matched_surrogate(fitted, train_data)
        self.assertAlmostEqual(float(surrogate.mu[0]), -2.0, delta=0.1)
        self.assertAlmostEqual(float(surrogate.covar[0, 0]), 0.25, delta=0.05)


class FasterToSameFTestCase(unittest.TestCase):
    def test_closed_form_refit_is_faster_than_gradient_m_step(self):
        """Once plateaued, re-fitting the surrogate is a single closed-form pass; re-running the
        gradient M-step pays for ``m_steps`` full SGD iterations every round regardless -- measure
        both directly (wall-clock, consistent with the "or wall-clock" carve-out D2/D3's own
        acceptance criteria use)."""
        fitted, train_data = _fit_near_convergence(mu=1.0, sigma=1.0, n=2000, seed=3, m_steps=200)
        enc = np.asarray(train_data)[:, None]
        weights = np.ones(len(train_data))

        n_rounds = 8

        t0 = time.perf_counter()
        for _ in range(n_rounds):
            moment_matched_surrogate(fitted, train_data, weights=weights)
        closed_form_elapsed = time.perf_counter() - t0

        gradient_estimator = GradEstimator(fitted.module, m_steps=200, lr=5e-3)

        t0 = time.perf_counter()
        for _ in range(n_rounds):
            acc = gradient_estimator.accumulator_factory().make()
            acc.seq_update(enc, weights, None)
            gradient_estimator.estimate(float(weights.sum()), acc.value())
        gradient_elapsed = time.perf_counter() - t0

        speedup = gradient_elapsed / max(closed_form_elapsed, 1e-9)
        print(
            f"[D4] {n_rounds} rounds: closed-form={closed_form_elapsed:.4f}s, "
            f"gradient={gradient_elapsed:.4f}s, speedup={speedup:.1f}x"
        )
        self.assertGreater(speedup, 3.0, "closed-form re-fit should be measurably faster than a gradient M-step")

    def test_run_em_with_hotswap_reduces_gradient_m_steps_after_plateau(self):
        """End to end: a mixture with a pre-plateaued gradient leaf component swaps quickly (low
        patience) and every subsequent round's M-step for that component is closed-form, not
        gradient -- the literal mechanism behind "faster to same F"."""
        rng = np.random.RandomState(5)
        data = [float(v) for v in np.concatenate([rng.normal(-4.0, 1.0, 300), rng.normal(4.0, 1.0, 300)])]

        pretrained, _ = _fit_near_convergence(mu=-4.0, sigma=1.0, n=1000, seed=6, m_steps=300)
        comp0 = GradLeaf(pretrained.module, m_steps=30, lr=5e-3)
        comp1 = GaussianDistribution(3.5, 1.2)
        start = MixtureDistribution([comp0, comp1], [0.5, 0.5])
        estimator = MixtureEstimator([GradEstimator(pretrained.module, m_steps=30, lr=5e-3), GaussianEstimator()])
        enc = seq_encode(data, model=start)

        model, history, swap_records = run_em_with_hotswap(
            enc, estimator, start, max_its=10, delta=1.0e-9, plateau_patience=1
        )

        self.assertIn(0, swap_records, "the plateaued gradient leaf (component 0) should have been swapped")
        swapped_at = swap_records[0].swap_round
        rounds_after_swap = [h for h in history if h.round_index > swapped_at]
        self.assertTrue(rounds_after_swap)
        for h in rounds_after_swap:
            self.assertEqual(h.n_gradient_m_steps, 0, "no gradient M-step should run on the swapped component")
            self.assertGreaterEqual(h.n_closed_form_m_steps, 1)
        print(
            f"[D4] swapped at round {swapped_at}; {len(rounds_after_swap)} subsequent round(s) "
            "used the closed-form M-step instead of gradient descent"
        )


class SwapBackOnMisfitTestCase(unittest.TestCase):
    def test_should_swap_back_fires_on_a_genuine_regime_shift(self):
        fitted, train_data = _fit_near_convergence(mu=0.0, sigma=1.0, n=600, seed=7)
        tree = fitted
        surrogate = moment_matched_surrogate(fitted, train_data)
        new_tree, record = swap_leaf(tree, None, surrogate, round_index=0)
        self.assertIs(new_tree, surrogate)
        self.assertIs(record.original, fitted)

        same_regime_holdout = np.asarray(_data(0.0, 1.0, 300, seed=70))[:, None]
        record.baseline_misfit = misfit_receipt(surrogate, same_regime_holdout)

        # a genuinely shifted holdout regime -- the surrogate (frozen at N(0,1)) should fit this
        # far worse than its own baseline, a real computed misfit, not an assumed one.
        shifted_holdout = np.asarray(_data(6.0, 1.0, 300, seed=71))[:, None]
        shifted_misfit = misfit_receipt(surrogate, shifted_holdout)

        print(f"[D4] baseline misfit={record.baseline_misfit:.4f}, shifted-regime misfit={shifted_misfit:.4f}")
        self.assertGreater(shifted_misfit, record.baseline_misfit)
        self.assertTrue(should_swap_back(record, shifted_misfit))
        self.assertFalse(should_swap_back(record, record.baseline_misfit))

        restored = swap_back(new_tree, record, round_index=1)
        self.assertIs(restored, fitted)
        self.assertTrue(record.swapped_back)
        self.assertEqual(record.swap_back_round, 1)

    def test_run_em_with_hotswap_swaps_back_after_the_data_regime_shifts(self):
        """End to end, in two phases (mirroring the roadmap item's own "the underlying data
        distribution shifts after the swap" construction -- a within-run EM loop's ``enc_data`` is
        fixed for the whole run, exactly like D2/D3's, so a genuine regime shift has to happen
        BETWEEN runs, not through some in-loop mechanism):

        Phase 1 -- run ``run_em_with_hotswap`` for real on stationary data so component 0's
        plateaued gradient leaf genuinely gets swapped for a moment-matched surrogate (this is the
        same swap path :meth:`test_run_em_with_hotswap_reduces_gradient_m_steps_after_plateau`
        exercises, here re-used as the setup for the swap-back test).

        Phase 2 -- the world then shifts: fresh data is drawn from a DIFFERENT distribution than
        component 0 was ever fit on. A real :func:`misfit_receipt` against that shifted data is
        computed for the retained swap record and correctly reads as a genuine degradation versus
        the receipt's own swap-time baseline, and :func:`should_swap_back` /
        :func:`swap_back` -- the exact functions ``run_em_with_hotswap`` calls internally every
        round -- restore the retained original gradient leaf.
        """
        rng = np.random.RandomState(9)
        data = [float(v) for v in np.concatenate([rng.normal(-4.0, 1.0, 300), rng.normal(4.0, 1.0, 300)])]

        pretrained, _ = _fit_near_convergence(mu=-4.0, sigma=1.0, n=1000, seed=10, m_steps=300)
        comp0 = GradLeaf(pretrained.module, m_steps=30, lr=5e-3)
        comp1 = GaussianDistribution(3.5, 1.2)
        start = MixtureDistribution([comp0, comp1], [0.5, 0.5])
        estimator = MixtureEstimator([GradEstimator(pretrained.module, m_steps=30, lr=5e-3), GaussianEstimator()])
        enc = seq_encode(data, model=start)
        matching_holdout = _data(-4.0, 1.0, 200, seed=11)  # same regime as component 0 at swap time

        model, history, swap_records = run_em_with_hotswap(
            enc,
            estimator,
            start,
            max_its=10,
            delta=1.0e-9,
            plateau_patience=1,
            holdout_data=matching_holdout,
            misfit_tol=0.15,
        )
        self.assertIn(0, swap_records)
        record = swap_records[0]
        self.assertFalse(record.swapped_back, "a same-regime holdout should not itself trigger a swap-back")
        self.assertIsNotNone(record.baseline_misfit)

        # phase 2: the world shifts -- fresh data from a completely different regime than
        # component 0 (N(-4, 1)) was ever moment-matched against.
        shifted_holdout = _data(40.0, 1.0, 200, seed=12)
        post_shift_misfit = misfit_receipt(record.surrogate, shifted_holdout)
        print(
            f"[D4] baseline misfit (matching regime)={record.baseline_misfit:.4f}, "
            f"post-shift misfit={post_shift_misfit:.4f}"
        )
        self.assertGreater(post_shift_misfit, record.baseline_misfit)
        self.assertTrue(should_swap_back(record, post_shift_misfit, misfit_tol=0.15))

        restored = swap_back(model, record, round_index=len(history))
        self.assertIs(restored.components[0], record.original)
        self.assertTrue(record.swapped_back)


class MonotoneObjectiveGateCatchesBadSwapTestCase(unittest.TestCase):
    """:func:`moment_matched_surrogate` is, by its own docstring, "a genuine fit... it reads no
    attribute off ``gradient_leaf`` at all" -- a plain MLE Gaussian fit to the raw training data,
    nothing more. On a genuinely bimodal leaf that is a BAD approximation (see
    ``test_moment_matched_surrogate_collapses_a_bimodal_leaf_to_one_gaussian`` below), not merely a
    coarse one. This class is what actually keeps :func:`run_em_with_hotswap` safe on such a leaf:
    its per-round monotone-F accept/reject gate (the same construction D2/D3 already use) must
    reject a swap-plus-refit round whose objective does not improve, and -- the specific bug fixed
    alongside this test -- must roll the SWAP ITSELF back to the retained gradient leaf when it
    does, not just skip the following M-step's parameter update. Before the fix, the swap was
    applied to ``model`` unconditionally and eagerly (before the round's accept/reject check), and
    the ``else`` branch of that check only skipped ``model = candidate`` -- it never undid the
    swap, so a rejected round still silently returned the corrupted surrogate.
    """

    def test_moment_matched_surrogate_collapses_a_bimodal_leaf_to_one_gaussian(self):
        """Reproduces the audit's adversarial finding directly against the primitive: a
        ``BimodalGauss`` GradLeaf correctly learns two well-separated modes, but
        ``moment_matched_surrogate`` -- which never looks at the leaf's own density, only at the
        raw data -- collapses them into one wide Gaussian with a large held-out NLL penalty."""
        train_data = _bimodal_data(2000, seed=1, mu=(-3.0, 3.0), sigma=0.5)
        fitted = _fit_bimodal_near_convergence(train_data, seed=1)
        holdout = np.asarray(_bimodal_data(1000, seed=99, mu=(-3.0, 3.0), sigma=0.5))[:, None]

        learned_mu = sorted(fitted.module.mu.detach().numpy().tolist())
        self.assertAlmostEqual(learned_mu[0], -3.0, delta=0.3)
        self.assertAlmostEqual(learned_mu[1], 3.0, delta=0.3)

        surrogate = moment_matched_surrogate(fitted, train_data)
        grad_nll = float(-np.mean(fitted.seq_log_density(holdout)))
        surrogate_nll = float(-np.mean(surrogate.seq_log_density(holdout)))
        rel_degradation = (surrogate_nll - grad_nll) / grad_nll

        print(
            f"[D4 adversarial] bimodal leaf mu={learned_mu}, held-out NLL: gradient leaf={grad_nll:.4f} "
            f"nats, surrogate={surrogate_nll:.4f} nats ({rel_degradation * 100:.1f}% relative degradation)"
        )
        # this is the module's documented, honest limitation: moment_matched_surrogate ALONE is a
        # coarse, Gaussian-only fit whose quality is not guaranteed on a non-Gaussian leaf.
        self.assertGreater(rel_degradation, 0.5, "a Gaussian surrogate should badly misfit a bimodal leaf")

    def test_run_em_with_hotswap_rejects_and_reverts_a_bad_bimodal_swap(self):
        """End to end: a mixture with a pre-plateaued, genuinely bimodal gradient leaf component
        proposes a moment-matched swap exactly like the well-behaved (Gaussian-shaped) fixture
        does, but here the surrogate is a bad approximation. The per-round monotone-F gate must (1)
        reject that round (``accepted`` reads False), (2) leave NO record of a committed swap
        (``swap_records`` empty -- the swap never survives to be returned), and (3) return a final
        model whose component 0 is still the retained ``GradLeaf``, with held-out fit quality
        essentially unaffected by the rejected swap attempt.
        """
        rng = np.random.RandomState(21)
        comp0_data = _bimodal_data(1200, seed=22, mu=(-3.0, 3.0), sigma=0.5)
        comp1_data = [float(v) for v in rng.normal(20.0, 1.0, 600)]
        full_data = comp0_data + comp1_data
        rng.shuffle(full_data)

        pretrained = _fit_bimodal_near_convergence(comp0_data, seed=22)
        comp0 = GradLeaf(pretrained.module, m_steps=30, lr=5e-3)
        comp1 = GaussianDistribution(19.0, 1.5)
        start = MixtureDistribution([comp0, comp1], [0.66, 0.34])
        estimator = MixtureEstimator([GradEstimator(pretrained.module, m_steps=30, lr=5e-3), GaussianEstimator()])
        enc = seq_encode(full_data, model=start)
        holdout_for_comp0 = _bimodal_data(400, seed=23, mu=(-3.0, 3.0), sigma=0.5)

        pre_swap_nll = float(-np.mean(pretrained.seq_log_density(np.asarray(holdout_for_comp0)[:, None])))

        model, history, swap_records = run_em_with_hotswap(
            enc,
            estimator,
            start,
            max_its=10,
            delta=1.0e-9,
            plateau_patience=1,
            holdout_data=holdout_for_comp0,
            misfit_tol=0.15,
        )

        swap_attempted = any(h.swapped_this_round for h in history)
        self.assertTrue(swap_attempted, "the plateaued bimodal leaf should still trigger a swap attempt")
        rejected_rounds = [h for h in history if h.swapped_this_round and not h.accepted]
        self.assertTrue(rejected_rounds, "the round proposing the bad swap should be rejected by the gate")

        self.assertNotIn(0, swap_records, "a rejected swap must leave no committed SwapRecord for component 0")
        self.assertIsInstance(
            model.components[0], GradLeaf, "the retained gradient leaf must still be in place after rejection"
        )

        final_nll = float(-np.mean(model.components[0].seq_log_density(np.asarray(holdout_for_comp0)[:, None])))
        print(
            f"[D4 adversarial, end-to-end] pre-swap-attempt NLL={pre_swap_nll:.4f}, "
            f"post-rejected-swap-attempt NLL={final_nll:.4f}"
        )
        self.assertLess(
            abs(final_nll - pre_swap_nll),
            0.2,
            "a rejected bad swap must not measurably degrade the returned model's fit quality",
        )


class PlateauMonitorTestCase(unittest.TestCase):
    def test_plateau_requires_sustained_near_zero_q_gain(self):
        monitor = PlateauMonitor(q_gain_tol=1.0e-6, patience=3)
        fitted, _ = _fit_near_convergence(mu=2.0, sigma=1.0, n=500, seed=12)
        # a genuinely converged leaf should read as plateaued after `patience` consecutive
        # convergent Q-gain readings -- the first call has no `prev_residual` yet (q_gain is
        # necessarily None), so it takes `patience + 1` calls total to latch True.
        plateaued_rounds = [monitor.is_plateaued(0, fitted, nobs=1.0) for _ in range(4)]
        self.assertEqual(plateaued_rounds, [False, False, False, True])

    def test_non_gradient_node_never_plateaus(self):
        monitor = PlateauMonitor(patience=1)
        classical = GaussianDistribution(0.0, 1.0)
        self.assertFalse(monitor.is_plateaued(0, classical, nobs=1.0))
        self.assertFalse(monitor.is_plateaued(0, classical, nobs=1.0))


if __name__ == "__main__":
    unittest.main()
