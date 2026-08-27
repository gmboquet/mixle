"""Legitimate input that the library refused: three campaign-four fail-closed findings, one shape.

Each of these raised on data the library models perfectly well by another route, so the refusal was
never a judgement about the data -- it was two parts of one call disagreeing about what the data is.

T4-05 -- a table column with 65 or more distinct STRING values crashed ``optimize()`` with
``ValueError: could not convert string to float: 'user-00000'``, from ``gaussian.py``, for a column
the caller never asked to model as a Gaussian. ``optimize(estimator=None)`` tries the structured model
first (``_maybe_structured_model`` -> ``learn_bayesian_network``), and that learner classified columns
with its own discrete/continuous rule: past ``_is_discrete``'s 64-level cap a string column stopped
qualifying as discrete and was handed a Gaussian template by default, while
``mixle.utils.automatic.get_estimator`` -- the fallback inside the same public call -- routed the
identical column to a categorical at k=65 and an ignored identifier at k=300. Two routing systems
inside one public call, and the crashing one ran first. The 64-level threshold was never the defect
and is unchanged; asking ``_is_discrete`` a question it does not answer was.

T2-01 -- the same disagreement with a different trigger: a numeric column carrying missing values.
``not _is_discrete(col)`` was read as "this is a real number", so a gapped numeric column got a plain
Gaussian and died at ``GaussianDistribution observations contain N NaN entries``. Sharply
size-dependent, because the structured route is only attempted from 40 rows up: a newcomer prototyping
on ``df.head(20)`` saw it work and the full extract explode. ``structure='off'`` fit the same data in
0.1s and recovered the generative truth, so the composite path was always able to model it.

T4-03 -- an all-empty HMM corpus, a designed no-evidence state, died with a bare
``IndexError: list index out of range`` at ``hidden_markov.py`` ``seq_log_density``, on an unguarded
``idx_bands[0]``: with every sequence empty the encoder lays out no time bands at all. The answer was
never undefined -- ``log_density([])`` returns the length term, and the same vectorized pass already
returned 0.0 for an empty sequence whenever a non-empty one shared the batch. Worse than the all-empty
corpus: because this is the chunked scoring path, the measured rule was *crash iff (non-empty
sequences) < num_chunks*, so a corpus holding real evidence fit at ``num_chunks=1`` and raised at
``num_chunks=2`` -- an execution-plan knob documented to change only float summation order deciding
whether a fit happens at all. Note this reproduces only on the pure-Python encoder (``use_numba=False``,
which is what a numpy+scipy install gets); the numba payload takes a different scoring branch, so these
tests pin ``use_numba`` explicitly rather than inheriting whatever the environment has.

Also here, because it is the same defect one level up and was *unmasked* by the T2-01 repair: a
mixture of Bayesian networks fits every component on a SLICE and then scores it against the WHOLE
dataset. A slice holding no gap fit a missingness rate of exactly zero, and a slice missing a rare
level fit a categorical / multinomial-GLM support without it; both then called an ordinary record
impossible, and when every component did so at once the fit died in ``responsibilities`` with
"mixture assigns zero probability to observations at rows [...]". Measured on 60-row tables, the
default hard EM refused 8 of 25 seeds for gaps and 1 of 25 for a rare level, while soft EM -- which
fits every component on the whole dataset with responsibilities -- never did. The shared routing now
carries the family, the discrete support, and a one-pseudo-observation regularization for the sliced
caller only, so a slice-fitted component agrees with soft EM about what values exist.

The overreach direction is tested too: the single-fit path must NOT be regularized (it sees the whole
dataset, so nothing is ever unseen and the smoothing would only perturb the fit that was asked for),
and the empty-corpus guard must not re-break the no-evidence state that ``seq_encode`` deliberately
still encodes (``represent_learned_segment_test``).
"""

import unittest

import numpy as np
import pytest

import mixle.stats as st
from mixle.inference import optimize
from mixle.inference.bayesian_network import (
    _column_routing,
    _GLMFactor,
    learn_bayesian_network,
    learn_mixture_bayesian_network,
    select_mixture_components,
)
from mixle.inference.structure import _columns
from mixle.stats.combinator.optional import OptionalDistribution, OptionalEstimator
from mixle.stats.latent.hidden_markov import HAS_NUMBA, HiddenMarkovEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalEstimator


def _ll(model, data):
    return float(np.sum(model.seq_log_density(model.dist_to_encoder().seq_encode(data))))


def _id_rows(n_rows, n_ids, seed=7):
    """(string id, real value, plan) records, with ``n_ids`` distinct ids spread over ``n_rows``."""
    rng = np.random.RandomState(seed)
    return [
        (f"user-{i % n_ids:05d}", round(float(rng.normal(50.0, 10.0)), 4), "paid" if i % 2 else "free")
        for i in range(n_rows)
    ]


def _gap_rows(n_rows, n_gaps=1, seed=11, gap=float("nan")):
    """(real value with ``n_gaps`` missing entries, category) records."""
    rng = np.random.RandomState(seed)
    return [
        (gap if i < n_gaps else round(float(rng.normal(10.0, 3.0)), 4), "a" if i % 3 else "b") for i in range(n_rows)
    ]


def _two_gap_columns(seed, n_rows=60):
    """Two numeric columns whose gaps sit at OPPOSITE ends, so no single slice holds both."""
    rng = np.random.RandomState(seed)
    left = [round(float(rng.normal(10.0, 3.0)), 4) for _ in range(n_rows)]
    right = [round(float(rng.normal(-4.0, 2.0)), 4) for _ in range(n_rows)]
    nan = float("nan")
    return [
        (nan if i < 2 else left[i], nan if i > n_rows - 3 else right[i], "a" if i % 3 else "b") for i in range(n_rows)
    ]


def _rare_level_rows(seed, n_rows=60):
    """A categorical column with four singleton levels among two common ones."""
    rng = np.random.RandomState(seed)
    levels = ["a"] * 25 + ["b"] * 25 + ["c"] * 6 + ["rare1", "rare2", "rare3", "rare4"]
    rng.shuffle(levels)
    return [
        (round(float(rng.normal(10.0, 3.0)), 4), round(float(rng.normal(-4.0, 2.0)), 4), levels[i])
        for i in range(n_rows)
    ]


def _hmm_estimator(n_states=2, use_numba=False, len_estimator=None):
    kwargs = {} if len_estimator is None else {"len_estimator": len_estimator}
    return HiddenMarkovEstimator([st.GaussianEstimator() for _ in range(n_states)], use_numba=use_numba, **kwargs)


class HighCardinalityStringColumnTest(unittest.TestCase):
    """T4-05: a string column past the 64-level cap is routed, not coerced to float."""

    def test_table_with_three_hundred_distinct_ids_fits(self):
        rows = _id_rows(300, 300)
        model = optimize(rows, out=None)  # raised: could not convert string to float: 'user-00000'
        self.assertTrue(np.isfinite(_ll(model, rows)))

    def test_the_sixty_four_level_boundary_no_longer_decides_fit_or_crash(self):
        """64 always fit and 65 always raised. The threshold survives; the crash does not."""
        for n_ids in (64, 65, 66, 128):
            with self.subTest(n_ids=n_ids):
                rows = _id_rows(300, n_ids)
                self.assertTrue(np.isfinite(_ll(optimize(rows, out=None), rows)))

    def test_a_realistic_name_column_fits(self):
        """Not an artifact of synthetic ids: 90 place-like names over 300 rows."""
        rng = np.random.RandomState(3)
        names = [f"Burgburg{i:02d}" for i in range(90)]
        rows = [
            (names[i % 90], round(float(rng.normal(50.0, 10.0)), 4), "paid" if i % 2 else "free") for i in range(300)
        ]
        self.assertTrue(np.isfinite(_ll(optimize(rows, out=None), rows)))

    def test_the_network_learner_does_not_hand_a_string_column_a_gaussian(self):
        """The cause, stated directly: the routing decision itself, not just the absence of a crash."""
        cols = _columns(_id_rows(300, 300))
        vec_dims, discrete, opaque, templates, levels = _column_routing(cols)
        self.assertFalse(discrete[0])  # 300 levels is past _is_discrete's cap -- unchanged
        self.assertTrue(opaque[0])  # ... and "not discrete" no longer means "a real number"
        self.assertNotIsInstance(templates[0], st.GaussianEstimator)
        self.assertNotIn(0, levels)  # an opaque field declares no discrete support
        self.assertIsInstance(templates[1], st.GaussianEstimator)  # the genuinely real column still is one

    def test_the_high_cardinality_column_takes_no_part_in_the_edge_search(self):
        """An opaque field is a marginal, so a BIC comparison against the composite stays like-for-like."""
        rows = _id_rows(300, 300)
        net = learn_bayesian_network(rows)
        self.assertEqual(list(net.factors[0].parents), [])
        self.assertTrue(all(0 not in f.parents for f in net.factors))


class MissingValueStructureTest(unittest.TestCase):
    """T2-01: a numeric column with gaps is modeled, not refused, at every row count."""

    def test_gapped_column_fits_at_and_above_the_forty_row_gate(self):
        """39 fit only because structure learning is skipped below 40; 40 and up raised."""
        for n_rows in (39, 40, 41, 60, 200):
            with self.subTest(n_rows=n_rows):
                rows = _gap_rows(n_rows)
                self.assertTrue(np.isfinite(_ll(optimize(rows, out=None), rows)))

    def test_auto_and_off_both_model_the_gaps(self):
        """``structure='off'`` was the documented one-flag escape hatch; the default now works too."""
        rows = _gap_rows(200, n_gaps=6)
        auto = optimize(rows, out=None)
        off = optimize(rows, out=None, structure="off")
        self.assertTrue(np.isfinite(_ll(auto, rows)))
        self.assertTrue(np.isfinite(_ll(off, rows)))

    def test_none_and_nan_gap_flavours_are_both_accepted(self):
        """In raw tuples the flavours used to diverge: None was tolerated, float('nan') raised."""
        for gap in (float("nan"), None):
            with self.subTest(gap=repr(gap)):
                rows = _gap_rows(200, n_gaps=6, gap=gap)
                self.assertTrue(np.isfinite(_ll(optimize(rows, out=None), rows)))

    def test_the_missingness_rate_is_fitted_not_dropped(self):
        """The error's first suggestion was "drop the incomplete rows". Nothing is dropped."""
        rows = _gap_rows(200, n_gaps=8)
        net = learn_bayesian_network(rows)
        leaf = net.factors[0].dist
        self.assertIsInstance(leaf, OptionalDistribution)
        self.assertAlmostEqual(leaf.p, 8.0 / 200.0, places=12)

    def test_an_all_gap_column_is_still_fittable(self):
        """The degenerate end of the same axis: every value missing is a rate of one, not a crash."""
        rows = [(float("nan"), "a" if i % 3 else "b") for i in range(60)]
        self.assertTrue(np.isfinite(_ll(optimize(rows, out=None), rows)))

    def test_gapped_dataframe_fits_through_the_default_path(self):
        """The reported shape: a pandas frame with NaN in a continuous column."""
        pd = pytest.importorskip("pandas")
        rng = np.random.RandomState(5)
        tenure = rng.normal(24.0, 8.0, 200)
        spend = rng.normal(80.0, 20.0, 200)
        tenure[:9] = np.nan
        spend[190:] = np.nan
        frame = pd.DataFrame(
            {
                "tenure_months": tenure,
                "monthly_spend": spend,
                "plan": ["paid" if i % 2 else "free" for i in range(200)],
            }
        )
        model = optimize(frame, out=None)
        self.assertIsNotNone(model)


class SlicedMixtureSupportTest(unittest.TestCase):
    """The same disagreement one level up: a component fitted on a slice, scored on the whole dataset."""

    def test_hard_em_fits_records_with_gaps(self):
        """8 of 25 seeds raised "mixture assigns zero probability to observations at rows [...]"."""
        for seed in range(6):
            with self.subTest(seed=seed):
                rows = _two_gap_columns(seed)
                model = learn_mixture_bayesian_network(rows, 2, restarts=1, max_iter=3, max_its=3, seed=seed)
                self.assertTrue(np.isfinite(_ll(model, rows)))

    def test_every_training_row_keeps_a_finite_density(self):
        """The real contract behind the guard: no component may call a training record impossible."""
        rows = _two_gap_columns(1)
        model = learn_mixture_bayesian_network(rows, 2, restarts=1, max_iter=3, max_its=3, seed=1)
        per_row = model.seq_log_density(model.dist_to_encoder().seq_encode(rows))
        self.assertTrue(np.all(np.isfinite(per_row)))
        self.assertEqual(model.responsibilities(rows).shape, (len(rows), 2))

    def test_hard_em_fits_a_rare_categorical_level(self):
        """Same mechanism without any missing value: a singleton level absent from a slice."""
        rows = _rare_level_rows(3)
        model = learn_mixture_bayesian_network(rows, 2, restarts=1, max_iter=3, max_its=3, seed=3)
        self.assertTrue(np.isfinite(_ll(model, rows)))

    def test_model_selection_over_k_survives_gaps(self):
        """select_mixture_components inherits the mixture learner, so it inherited the refusal."""
        rows = _two_gap_columns(1)
        model, report = select_mixture_components(rows, (1, 2), seed=1, restarts=1, max_iter=3, max_its=3)
        self.assertIn(report["k"], (1, 2))
        self.assertTrue(np.isfinite(_ll(model, rows)))

    def test_the_single_fit_path_is_not_regularized(self):
        """Overreach guard: on the whole dataset nothing is unseen, so nothing may be smoothed."""
        cols = _columns(_two_gap_columns(1))
        plain = _column_routing(cols)[3]
        sliced = _column_routing(cols, sliced=True)[3]
        self.assertIsInstance(plain[0], OptionalEstimator)
        self.assertIsNone(plain[0].pseudo_count)
        self.assertIsInstance(sliced[0], OptionalEstimator)
        self.assertEqual(sliced[0].pseudo_count, 0.5)

        # The discrete half needs no template change and must not get one: the automatic detector
        # already declares a categorical's support over the whole column, so a slice that missed a
        # level still has a probability for it. (The one place that re-derived a support from the
        # slice was the GLM edge -- see the ``child_levels`` test below.)
        cat_cols = _columns(_rare_level_rows(3))
        plain_cat = _column_routing(cat_cols)[3][2]
        sliced_cat = _column_routing(cat_cols, sliced=True)[3][2]
        self.assertIsInstance(plain_cat, CategoricalEstimator)
        self.assertIsInstance(sliced_cat, CategoricalEstimator)
        self.assertEqual(set(plain_cat.suff_stat), set(cat_cols[2]))
        self.assertEqual(set(sliced_cat.suff_stat), set(cat_cols[2]))

    def test_the_shared_routing_declares_the_whole_data_support(self):
        """A component's discrete support is the dataset's, not its slice's."""
        rows = _rare_level_rows(3)
        levels = _column_routing(_columns(rows), sliced=True)[4]
        self.assertEqual(set(levels[2]), {row[2] for row in rows})

    def test_a_glm_edge_scores_a_level_its_slice_never_saw(self):
        """The remaining slice-derived support: a multinomial GLM's class list IS its support."""
        rows = _rare_level_rows(3)
        cols = _columns(rows)
        slice_cols = _columns([row for row in rows if row[2] != "rare1"])
        full_levels = _column_routing(cols)[4][2]
        edge = _GLMFactor.fit(2, [0], slice_cols, {}, None, child_levels=full_levels)
        self.assertIn("rare1", edge.levels)
        unseen = [row for row in rows if row[2] == "rare1"]
        self.assertTrue(np.all(np.isfinite(edge.seq_log_density(_columns(unseen)))))

        blind = _GLMFactor.fit(2, [0], slice_cols, {}, None)  # the pre-repair behavior, for contrast
        self.assertNotIn("rare1", blind.levels)
        self.assertTrue(np.all(np.isneginf(blind.seq_log_density(_columns(unseen)))))


class EmptyHiddenMarkovCorpusTest(unittest.TestCase):
    """T4-03: a batch in which every sequence is empty scores; it does not IndexError."""

    def test_all_empty_corpus_fits(self):
        for n_sequences in (1, 8):
            with self.subTest(n_sequences=n_sequences):
                model = optimize([[] for _ in range(n_sequences)], _hmm_estimator(), out=None)
                self.assertEqual(model.n_states, 2)

    def test_all_empty_corpus_fits_with_categorical_emissions(self):
        est = HiddenMarkovEstimator([st.CategoricalEstimator(), st.CategoricalEstimator()], use_numba=False)
        self.assertIsNotNone(optimize([[] for _ in range(8)], est, out=None))

    def test_the_empty_batch_scores_what_the_scalar_path_scores(self):
        """The answer was never undefined: the scalar path already returned it."""
        rng = np.random.RandomState(3)
        corpus = [[float(rng.normal()) for _ in range(4)] for _ in range(12)]
        model = optimize(corpus, _hmm_estimator(), max_its=6, delta=None, out=None)
        encoder = model.dist_to_encoder()
        empty_batch = model.seq_log_density(encoder.seq_encode([[] for _ in range(8)]))
        self.assertEqual(empty_batch.shape, (8,))
        for value in empty_batch:
            self.assertAlmostEqual(float(value), model.log_density([]), places=12)

    def test_an_empty_sequence_scores_the_same_alone_and_beside_a_real_one(self):
        """The mixed batch already worked, which is what proved the all-empty batch had an answer."""
        rng = np.random.RandomState(3)
        corpus = [[float(rng.normal()) for _ in range(4)] for _ in range(12)]
        model = optimize(corpus, _hmm_estimator(), max_its=6, delta=None, out=None)
        encoder = model.dist_to_encoder()
        alone = model.seq_log_density(encoder.seq_encode([[]]))
        mixed = model.seq_log_density(encoder.seq_encode([[], [0.1, 0.2]]))
        self.assertAlmostEqual(float(alone[0]), float(mixed[0]), places=12)

    def test_the_guard_reports_the_model_not_a_convenient_zero(self):
        """Overreach guard, the other direction: a length law that rules out length 0 must still say so.

        Fitted on length-3 sequences only, ``len_dist`` gives ``log P(0) = -inf``. Returning ``0.0``
        for the empty batch would be a *fail-open* fix -- a made-up probability of one for an outcome
        the model calls impossible. All three spellings have to agree on ``-inf``.
        """
        rng = np.random.RandomState(0)
        corpus = [[float(rng.normal()) for _ in range(3)] for _ in range(10)]
        model = optimize(
            corpus, _hmm_estimator(len_estimator=st.CategoricalEstimator()), max_its=6, delta=None, out=None
        )
        encoder = model.dist_to_encoder()
        self.assertEqual(model.log_density([]), float("-inf"))
        self.assertTrue(np.all(np.isneginf(model.seq_log_density(encoder.seq_encode([[]] * 3)))))
        mixed = model.seq_log_density(encoder.seq_encode([[], [0.1, 0.2, 0.3]]))
        self.assertTrue(np.isneginf(mixed[0]))
        self.assertTrue(np.isfinite(mixed[1]))

    def test_the_length_model_term_survives_the_empty_batch(self):
        """Zeroed, the guard would have silently dropped the one term such a batch DOES carry."""
        rng = np.random.RandomState(0)
        corpus = [[float(rng.normal()) for _ in range(3)] for _ in range(10)] + [[] for _ in range(4)]
        model = optimize(
            corpus,
            _hmm_estimator(len_estimator=st.CategoricalEstimator()),
            max_its=6,
            delta=None,
            out=None,
        )
        encoder = model.dist_to_encoder()
        empty_only = model.seq_log_density(encoder.seq_encode([[], []]))
        expected = float(model.len_dist.log_density(0))
        self.assertLess(expected, 0.0)  # a real, non-zero term -- not vacuously satisfied
        for value in empty_only:
            self.assertAlmostEqual(float(value), expected, places=12)
        self.assertAlmostEqual(float(empty_only[0]), model.log_density([]), places=12)

    def test_chunking_no_longer_decides_fit_or_crash(self):
        """Measured rule was: crash iff non-empty sequences < num_chunks. One real sequence, 15 empty."""
        rng = np.random.RandomState(5)
        data = [[float(rng.normal()) for _ in range(4)]] + [[] for _ in range(15)]
        for num_chunks in (1, 2, 4, 8):
            with self.subTest(num_chunks=num_chunks):
                model = optimize(data, _hmm_estimator(), max_its=3, delta=None, out=None, num_chunks=num_chunks)
                self.assertEqual(model.n_states, 2)

    def test_the_estep_credits_an_all_empty_batch_with_its_length_evidence_only(self):
        """The accumulator half of the same guard: no emission, no transition, no initial state."""
        rng = np.random.RandomState(1)
        corpus = [[float(rng.normal()) for _ in range(3)] for _ in range(10)] + [[] for _ in range(4)]
        model = optimize(
            corpus,
            _hmm_estimator(len_estimator=st.CategoricalEstimator()),
            max_its=6,
            delta=None,
            out=None,
        )
        refit = optimize([[] for _ in range(5)], _hmm_estimator(), max_its=2, delta=None, out=None)
        self.assertEqual(refit.n_states, 2)
        self.assertTrue(np.all(np.isfinite(np.asarray(model.transitions, dtype=float))))

    @unittest.skipUnless(HAS_NUMBA, "the numba scoring branch needs numba installed")
    def test_the_numba_and_python_paths_agree_on_the_empty_batch(self):
        """The defect only ever reproduced on the pure-Python encoder; the two must now agree."""
        rng = np.random.RandomState(3)
        corpus = [[float(rng.normal()) for _ in range(4)] for _ in range(12)]
        model = optimize(corpus, _hmm_estimator(), max_its=6, delta=None, out=None)
        python_encoder = model.dist_to_encoder()
        python_encoder.use_numba = False
        numba_encoder = model.dist_to_encoder()
        numba_encoder.use_numba = True
        batch = [[] for _ in range(4)]
        np.testing.assert_allclose(
            model.seq_log_density(python_encoder.seq_encode(batch)),
            model.seq_log_density(numba_encoder.seq_encode(batch)),
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
