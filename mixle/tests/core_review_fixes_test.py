"""Regression tests for the 0.8.0 core-module review findings C-1..C-11 (audit/CODEBASE_REVIEW_LEDGER.md II.A).

Each test pins the corrected behavior of a verified defect: the quantize support bracket (C-1), k-best
Viterbi admissibility (C-2), the split-conformal small-n threshold and certificate coverage (C-3..C-5),
registry id uniqueness and tier ordering (C-6/C-7), and the relations/fault edge-case guards (C-8..C-11).

Also carries the later external-review findings against the same ``mixle.scientist`` functions
(docs/audits/0.8.0-exhaustive-code-review.md MXR-080-0017/0018): ``study``'s alpha/cal_frac domain
and ``distill_to_edge``'s one-shot-iterable handling. They live here rather than in
``scientist_test.py`` because that file is torch+transformers+datasets-gated (optional/slow) for its
real-encoder fixtures, while these are pure input-validation/data-flow checks that need none of that
and belong in the fast default gate -- same rationale that already put the C-3..C-5 `study` tests here.
"""

import itertools
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from mixle.relations import ViterbiPath, branch_and_bound_milp, cardinality_constrained_milp, tsp_held_karp


# --------------------------------------------------------------- C-1: quantize brackets the SPATIAL quantiles
def test_quantize_gaussian_preserves_mean_variance_and_both_tails():
    from mixle import ops
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    g = GaussianDistribution(mu=0.0, sigma2=1.0)
    q = ops.quantize(g, bits=6)
    vals = np.array(list(q.pmap.keys()), dtype=np.float64)
    probs = np.array([q.pmap[v] for v in q.pmap], dtype=np.float64)
    mean = float((vals * probs).sum())
    var = float((probs * (vals - mean) ** 2).sum())
    # the density_quantile bracket discarded the whole left half (support [0.026, 3.20], mean 0.795)
    assert vals.min() < 0.0 < vals.max()  # support spans both signs
    assert abs(mean) < 0.05
    assert var == pytest.approx(1.0, abs=0.1)


# --------------------------------------------------------------- C-2: ViterbiPath with positive log-densities
def _hmm_path_score(log_init, log_trans, log_obs, path):
    score = log_init[path[0]] + log_obs[0][path[0]]
    for t in range(1, len(log_obs)):
        score += log_trans[path[t - 1]][path[t]] + log_obs[t][path[t]]
    return float(score)


def test_viterbi_top1_matches_brute_force_when_emission_log_densities_are_positive():
    # continuous-emission log-densities are routinely > 0; the old zero heuristic was inadmissible there
    n_states, n_steps = 3, 4
    for seed in range(20):
        rng = np.random.RandomState(seed)
        log_init = np.log(rng.dirichlet(np.ones(n_states)))
        log_trans = np.log([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
        log_obs = rng.uniform(-1.0, 2.0, size=(n_steps, n_states))
        best = max(
            _hmm_path_score(log_init, log_trans, log_obs, path)
            for path in itertools.product(range(n_states), repeat=n_steps)
        )
        sol = ViterbiPath(log_init, log_trans, log_obs).solve()
        assert _hmm_path_score(log_init, log_trans, log_obs, sol.value) == pytest.approx(sol.objective, abs=1e-9)
        assert sol.objective == pytest.approx(best, abs=1e-9), f"suboptimal top-1 at seed {seed}"


def test_viterbi_enumeration_is_nonincreasing_and_k_limits():
    rng = np.random.RandomState(7)
    n_states, n_steps = 3, 4
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
    log_obs = rng.uniform(-1.0, 2.0, size=(n_steps, n_states))
    rel = ViterbiPath(log_init, log_trans, log_obs)
    scores = [sol.objective for sol in rel.enumerator()]
    assert len(scores) == n_states**n_steps  # k=None enumerates every path
    assert all(scores[i] >= scores[i + 1] - 1e-12 for i in range(len(scores) - 1))
    top = rel.top(5)
    assert len(top) == 5
    brute = sorted(
        (
            _hmm_path_score(log_init, log_trans, log_obs, path)
            for path in itertools.product(range(n_states), repeat=n_steps)
        ),
        reverse=True,
    )
    assert [sol.objective for sol in top] == pytest.approx(brute[:5], abs=1e-9)
    assert all(isinstance(sol.value, list) and len(sol.value) == n_steps for sol in top)


# --------------------------------------------------------------- C-3/C-4/C-5: scientist.study edge regimes
def _two_class_latents(n_per_class=20, dim=3, seed=0):
    rng = np.random.RandomState(seed)
    z = np.concatenate([rng.normal(0.0, 1.0, size=(n_per_class, dim)), rng.normal(5.0, 1.0, size=(n_per_class, dim))])
    y = ["a"] * n_per_class + ["b"] * n_per_class
    return z, y


def test_study_small_calibration_split_yields_infinite_qhat_not_undercoverage():
    from mixle.scientist import study

    z, y = _two_class_latents()
    # n_cal = 10 and alpha = 0.01: ceil((10+1) * 0.99) = 11 > 10, so no calibration score certifies the
    # level -- the finite-sample threshold is +inf (all labels / abstain), not the max score
    model = study(z, y, alpha=0.01, cal_frac=0.25, seed=0)
    assert model.provenance["n_cal"] == 10  # the regime is visible in provenance
    assert np.isinf(model.qhat)
    sets = model.prediction_sets(z[:5])
    assert all(s == model.classes for s in sets)  # every label returned: abstention, not silent under-coverage
    assert model.abstains(z[:5]).all()


def test_study_reachable_conformal_level_keeps_a_finite_threshold():
    from mixle.scientist import study

    z, y = _two_class_latents()
    model = study(z, y, alpha=0.1, cal_frac=0.25, seed=0)  # ceil(11 * 0.9) = 10 <= n_cal = 10
    assert np.isfinite(model.qhat)


def test_study_certificate_covers_every_class_head():
    from mixle.scientist import study

    z, y = _two_class_latents()
    model = study(z, y, alpha=0.1, cal_frac=0.25, seed=0)
    assert len(model.head) == len(model.classes) == 2
    block_names = [b.name for b in model.certificate.blocks]
    for c in model.classes:
        assert any(f"head[{c!r}]" in name for name in block_names), f"class {c!r} missing from the certificate"
    assert model.certificate.guarantee.name == "GLOBAL_UNIQUE"  # closed-form Gaussian heads, K of them
    assert len(model.certificate.gradient_blocks) == 0


def test_study_raises_clearly_when_a_class_misses_the_fit_split():
    from mixle.scientist import study

    rng = np.random.RandomState(0)
    z = np.concatenate([rng.normal(0.0, 1.0, size=(8, 2)), rng.normal(5.0, 1.0, size=(1, 2))])
    y = ["common"] * 8 + ["rare"] * 1
    n_cal = max(1, int(round(0.25 * len(z))))
    seed = next(s for s in range(100) if 8 in np.random.RandomState(s).permutation(len(z))[:n_cal])
    with pytest.raises(ValueError, match="'rare' has no examples in the fit split"):
        study(z, y, alpha=0.1, cal_frac=0.25, seed=seed)


# --------------------------------------------------------------- MXR-080-0017: scientist.study alpha/cal_frac domain
def test_study_rejects_out_of_domain_alpha_and_cal_frac():
    from mixle.scientist import study

    z, y = _two_class_latents()
    # negative, and > 1: alpha=2.0 used to crash with an out-of-range negative index (IndexError) instead
    # of a clear ValueError -- pytest.raises(ValueError, ...) fails the test if that regresses.
    for bad_alpha in (-1.0, -1e-9, 1.5, 2.0):
        with pytest.raises(ValueError, match=r"alpha must be in \[0\.0, 1\.0\]"):
            study(z, y, alpha=bad_alpha, cal_frac=0.25, seed=0)
    # cal_frac=0.0/-0.5 used to be silently coerced to n_cal=1 (max(1, ...)); cal_frac=1.0 used to surface
    # only as an indirect "class has no examples in the fit split" error with no mention of cal_frac.
    for bad_cal_frac in (0.0, -0.5, 1.0, 1.2):
        with pytest.raises(ValueError, match=r"cal_frac must be in \(0\.0, 1\.0\)"):
            study(z, y, alpha=0.1, cal_frac=bad_cal_frac, seed=0)


def test_study_alpha_boundaries_are_valid_and_qhat_is_the_tightest_not_loosest_score():
    from mixle.scientist import study

    z, y = _two_class_latents()
    # alpha=0.0 (100% coverage requested): unreachable from any finite calibration set -> +inf, the same
    # regime as test_study_small_calibration_split_yields_infinite_qhat_not_undercoverage above.
    m0 = study(z, y, alpha=0.0, cal_frac=0.25, seed=0)
    assert np.isinf(m0.qhat)

    # alpha=1.0 (0% coverage requested) is the other legitimate boundary: the TIGHTEST threshold, i.e.
    # the MINIMUM calibration nonconformity score. The old hand-rolled ceil((n_cal+1)(1-alpha)) index
    # arithmetic reached k=0 here and `sorted_scores[k - 1]` wrapped around via Python's negative
    # indexing to sorted_scores[-1] -- the MAXIMUM score (the loosest threshold) -- instead. Reconstruct
    # the calibration nonconformity scores the same way study() does, from public API only (predict_proba
    # + the documented RandomState(seed).permutation(len(z))[:n_cal] split, already relied on by
    # test_study_raises_clearly_when_a_class_misses_the_fit_split above), and check against the min.
    m1 = study(z, y, alpha=1.0, cal_frac=0.25, seed=0)
    n_cal = m1.provenance["n_cal"]
    cal_idx = np.random.RandomState(0).permutation(len(z))[:n_cal]
    class_of = {c: j for j, c in enumerate(m1.classes)}
    p_cal = m1.predict_proba(z[cal_idx])
    p_true = p_cal[np.arange(len(cal_idx)), [class_of[c] for c in np.asarray(y)[cal_idx]]]
    scores = 1.0 - p_true
    assert m1.qhat == pytest.approx(float(scores.min()))
    assert scores.max() > scores.min()  # the fixture's scores do vary, so this is a real, non-vacuous check
    assert m1.qhat < float(scores.max())  # NOT the old (buggy) wrap-to-maximum behavior


def test_study_validates_latents_shape_and_label_alignment():
    from mixle.scientist import study

    z, y = _two_class_latents()
    with pytest.raises(ValueError, match="two-dimensional"):
        study(z[:, 0], y, alpha=0.1, cal_frac=0.25, seed=0)  # 1-D latents used to crash: z.shape[1] IndexError
    with pytest.raises(ValueError, match="non-empty"):
        study(np.zeros((0, z.shape[1])), [], alpha=0.1, cal_frac=0.25, seed=0)  # used to crash: min() of []
    with pytest.raises(ValueError, match="aligned"):
        study(z, y[:-1], alpha=0.1, cal_frac=0.25, seed=0)  # one label short of latents
    with pytest.raises(ValueError, match="aligned"):
        study(z, y + ["a"], alpha=0.1, cal_frac=0.25, seed=0)  # one label more than latents
    # negative control: a well-formed call is unaffected by the new guards
    model = study(z, y, alpha=0.1, cal_frac=0.25, seed=0)
    assert len(model.classes) == 2


# --------------------------------------------------------------- MXR-080-0018: distill_to_edge one-shot iterables
_EDGE_TRAIN = ["alpha zero", "alpha one", "beta zero", "beta one"]
_EDGE_VAL = ["alpha two", "beta two"]


def _edge_teacher(x):
    return "alpha" if "alpha" in x else "beta"


def _distill_to_edge_probe(train_inputs, val_inputs, val_truth=None):
    """Call distill_to_edge with mixle.task.edge.distill_for_edge mocked out, and return the resulting
    EdgeArtifact plus exactly what distill_for_edge was called with. Mocked (rather than exercising the
    real DOE/torch training stack, per edge_distill_test.py's own `pytest.importorskip("torch")` gate)
    because the bug and its fix are entirely in distill_to_edge's OWN iteration over its arguments,
    before any training happens -- isolating that from the heavy dependency keeps this deterministic,
    fast, and independent of whether torch is installed."""
    import mixle.task.edge as edge_mod
    from mixle.scientist import distill_to_edge

    calls = []

    def fake_distill_for_edge(teacher, train_data, val_data, device, **kw):
        calls.append({"train_data": list(train_data), "val_data": list(val_data), **kw})
        return SimpleNamespace(
            model=_edge_teacher, footprint=SimpleNamespace(bytes=123, torch_free=True), family="structured"
        )

    with mock.patch.object(edge_mod, "distill_for_edge", fake_distill_for_edge):
        art = distill_to_edge(
            _edge_teacher,
            train_inputs,
            val_inputs,
            val_truth if val_truth is not None else ["alpha", "beta"],
            n_init=1,
            n_iter=0,
            seed=0,
        )
    return art, calls[0]


def test_distill_to_edge_materializes_generator_inputs_exactly_once():
    # a one-shot generator (unlike a list) is silently exhausted by a second/third pass; the old code
    # iterated train_inputs/val_inputs three times total (teacher labeling, the list(...) handed to
    # distill_for_edge, and student-metrics scoring), so a generator here used to crash with
    # `IndexError: list index out of range` deep inside distill_for_edge (`_is_record_data` indexing an
    # empty list) -- reproduced against the unfixed code before this test was written.
    art, call = _distill_to_edge_probe((x for x in _EDGE_TRAIN), (x for x in _EDGE_VAL))
    assert call["train_data"] == _EDGE_TRAIN  # the FULL data, not emptied by an earlier pass
    assert call["val_data"] == _EDGE_VAL
    assert call["train_labels"] == ["alpha", "alpha", "beta", "beta"]
    assert call["val_labels"] == ["alpha", "beta"]
    assert art.provenance["n_train"] == len(_EDGE_TRAIN)
    assert art.teacher_accuracy == 1.0
    assert art.student_accuracy == 1.0
    assert art.agreement == 1.0


def test_distill_to_edge_generator_matches_list_input():
    # negative control: a plain (re-iterable) list is the case that "worked by accident" before this fix
    # -- it must still produce exactly the result the generator now produces, one-shot or not.
    gen_art, gen_call = _distill_to_edge_probe((x for x in _EDGE_TRAIN), (x for x in _EDGE_VAL))
    list_art, list_call = _distill_to_edge_probe(list(_EDGE_TRAIN), list(_EDGE_VAL))
    assert gen_call == list_call
    metrics = lambda a: (a.teacher_accuracy, a.student_accuracy, a.agreement, a.provenance)  # noqa: E731
    assert metrics(gen_art) == metrics(list_art)


def test_distill_to_edge_rejects_empty_or_misaligned_inputs():
    # these all raise before distill_for_edge would ever be called, so no mock is needed here.
    from mixle.scientist import distill_to_edge

    with pytest.raises(ValueError, match="at least one training input"):
        distill_to_edge(_edge_teacher, [], _EDGE_VAL, ["alpha", "beta"])
    with pytest.raises(ValueError, match="at least one training input"):
        distill_to_edge(_edge_teacher, (x for x in []), _EDGE_VAL, ["alpha", "beta"])  # exhausted generator
    with pytest.raises(ValueError, match="at least one validation input"):
        distill_to_edge(_edge_teacher, _EDGE_TRAIN, [], [])
    with pytest.raises(ValueError, match="matching length"):
        distill_to_edge(_edge_teacher, _EDGE_TRAIN, _EDGE_VAL, ["alpha"])  # val_truth shorter than val_inputs


# --------------------------------------------------------------- MXR-080-1704: learn() one doc vs a corpus
def test_learn_treats_one_document_string_as_one_document():
    # Scientist.learn iterated its unconstrained `docs` argument directly, so the ordinary
    # single-document call learn("hello world") ingested ELEVEN documents -- one per character -- and
    # stored ["h", "e", ..., "d"] as separate citable knowledge items.
    from mixle.scientist import Scientist

    sci = Scientist()
    assert sci.learn("hello world") == 1
    stored = [item.text for item in sci.knowledge.all()]
    assert stored == ["hello world"]


def test_learn_still_accepts_a_collection_and_consumes_it_once():
    from mixle.scientist import Scientist

    docs = ["first document", "second document"]
    sci = Scientist()
    assert sci.learn(docs) == 2
    assert sorted(item.text for item in sci.knowledge.all()) == sorted(docs)

    from_generator = Scientist()
    assert from_generator.learn(d for d in docs) == 2  # one-shot iterable materialized once
    assert sorted(item.text for item in from_generator.knowledge.all()) == sorted(docs)


def test_learn_rejects_a_non_document_item_by_position():
    from mixle.scientist import Scientist

    with pytest.raises(TypeError, match="document 1"):
        Scientist().learn(["a real document", {"not": "a document"}])


# --------------------------------------------------------------- MXR-080-1703: two latent spaces, not one
def test_image_and_text_latent_spaces_are_declared_separate():
    # perceive() returns 512-d CLIP image features and read() returns 384-d MiniLM embeddings from an
    # independently trained model, with no projection, paired objective, common schema or bridge
    # between them -- yet both docstrings claimed they encoded into "the shared scientific latent
    # space" and StudiedModel called itself cross-modal. Their arrays cannot even be stacked.
    from mixle.scientist import LATENT_SPACES, Scientist, StudiedModel, latent_space

    image, text = latent_space("image"), latent_space("text")
    assert image["space_id"] != text["space_id"]
    assert image["dim"] != text["dim"]
    assert image["encoder"] != text["encoder"]
    assert image["aligned_with"] == () and text["aligned_with"] == ()  # nothing bridges them
    assert set(LATENT_SPACES) == {"image", "text"}
    with pytest.raises(KeyError):
        latent_space("audio")

    for doc in (Scientist.perceive.__doc__, Scientist.read.__doc__, StudiedModel.__doc__):
        assert "shared scientific latent space" not in doc


def test_study_records_which_latent_space_its_head_was_fit_in():
    from mixle.scientist import study

    z, y = _two_class_latents(dim=3)
    model = study(z, y, alpha=0.1, cal_frac=0.25, seed=0)
    assert model.provenance["latent_dim"] == 3


# --------------------------------------------------------------- MXR-080-1706: one label equivalence relation
def _int_edge_teacher(x):
    return 1


def test_distillation_metrics_share_one_label_equivalence_relation():
    # teacher accuracy used NumPy's raw label equality while student accuracy and agreement coerced
    # both sides with .astype(str), so teacher label 1, student label 1 and reference label "1"
    # reported teacher 0.0, student 1.0, agreement 1.0 and retention 0.0 -- a student that matched its
    # teacher on every case scored as having retained nothing of it.
    import mixle.task.edge as edge_mod
    from mixle.scientist import distill_to_edge

    def fake_distill_for_edge(teacher, train_data, val_data, device, **kw):
        return SimpleNamespace(
            model=_int_edge_teacher, footprint=SimpleNamespace(bytes=1, torch_free=True), family="structured"
        )

    with mock.patch.object(edge_mod, "distill_for_edge", fake_distill_for_edge):
        art = distill_to_edge(_int_edge_teacher, _EDGE_TRAIN, _EDGE_VAL, ["1", "1"], n_init=1, n_iter=0, seed=0)

    assert art.agreement == 1.0  # student and teacher both say 1
    assert art.teacher_accuracy == art.student_accuracy  # ONE relation, not two
    assert art.student_accuracy == 0.0  # 1 is not "1" under either metric now
    assert np.isnan(art.retention)  # undefined, not a confident 0.0
    assert art.provenance["label_vocabulary_disjoint"] is True
    assert "undefined retained" in art.render()


def test_distillation_retention_is_reported_when_the_teacher_scores():
    import mixle.task.edge as edge_mod
    from mixle.scientist import distill_to_edge

    def fake_distill_for_edge(teacher, train_data, val_data, device, **kw):
        return SimpleNamespace(
            model=_edge_teacher, footprint=SimpleNamespace(bytes=1, torch_free=True), family="structured"
        )

    with mock.patch.object(edge_mod, "distill_for_edge", fake_distill_for_edge):
        art = distill_to_edge(_edge_teacher, _EDGE_TRAIN, _EDGE_VAL, ["alpha", "beta"], n_init=1, n_iter=0, seed=0)

    assert art.teacher_accuracy == 1.0
    assert art.student_accuracy == 1.0
    assert art.retention == 1.0
    assert art.provenance["label_vocabulary_disjoint"] is False


# --------------------------------------------------------------- C-6/C-7: registry ids and tier ordering
def _json_task_model():
    from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
    from mixle.task.model import StructuredClassifierIO, TaskModel

    return TaskModel(
        model=CategoricalDistribution({"x": 0.5, "y": 0.5}),
        adapter=StructuredClassifierIO(field_keys=None, label_index=0, labels=["x", "y"]),
        payload="json",
    )


def test_registry_rejects_duplicate_entry_id_instead_of_overwriting():
    from mixle.system import Registry

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(d)
        reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="dup")
        with pytest.raises(ValueError, match="already has an entry 'dup'"):
            reg.register(_json_task_model(), capabilities=["cap"], cost=0.02, entry_id="dup")
        assert [e.entry_id for e in reg.find_for("cap")] == ["dup"]  # one index row, artifact intact


def test_registry_auto_ids_scan_past_taken_ones():
    from mixle.system import Registry

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(d)
        reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="entry_0001")
        # len()-based naming would now mint 'entry_0001' again and silently overwrite its artifact
        auto = reg.register(_json_task_model(), capabilities=["cap"], cost=0.02)
        ids = [e.entry_id for e in reg.find_for("cap")]
        assert auto.entry_id != "entry_0001"
        assert len(ids) == len(set(ids)) == 2
        reg.load(auto.entry_id)  # both artifacts reload
        reg.load("entry_0001")


def test_tier_stack_reordering_costs_override_yields_ascending_tiers():
    from mixle.system import Registry

    with tempfile.TemporaryDirectory() as d:
        reg = Registry(d)
        first = reg.register(_json_task_model(), capabilities=["cap"], cost=0.01)
        second = reg.register(_json_task_model(), capabilities=["cap"], cost=0.05)

        def frontier(texts):
            return ["x"] * len(texts)

        # positional override [0.09, 0.02, 1.0] swaps the pool's effective order (Router assumes ascending)
        stack = reg.tier_stack("cap", frontier=frontier, costs=[0.09, 0.02, 1.0])
        assert [cost for _name, _model, cost in stack] == [0.02, 0.09, 1.0]
        assert [name for name, _model, _cost in stack] == [second.entry_id, first.entry_id, "frontier"]
        assert stack[-1][1] is frontier
        # without an override the registered-cost order is unchanged
        stack = reg.tier_stack("cap", frontier=frontier)
        assert [name for name, _model, _cost in stack] == [first.entry_id, second.entry_id, "frontier"]


# --------------------------------------------------------------- C-8: tsp_held_karp infeasibility
def test_tsp_raises_clearly_when_no_hamiltonian_cycle_exists():
    inf = np.inf
    disconnected = np.array([[0.0, 1.0, inf], [1.0, 0.0, inf], [inf, inf, 0.0]])
    with pytest.raises(ValueError, match="no Hamiltonian cycle"):
        tsp_held_karp(disconnected)
    with pytest.raises(ValueError, match="no Hamiltonian cycle"):
        tsp_held_karp(np.array([[0.0, inf], [1.0, 0.0]]))
    cost, tour = tsp_held_karp(np.array([[0.0, 1.0, 4.0], [4.0, 0.0, 1.0], [1.0, 4.0, 0.0]]))  # feasible ring
    assert cost == pytest.approx(3.0)
    assert tour == [0, 1, 2]


# --------------------------------------------------------------- C-9: cardinality MILP bound validation
def test_cardinality_milp_rejects_nonfinite_bounds():
    with pytest.raises(ValueError, match="finite bounds"):
        cardinality_constrained_milp(np.array([1.0, 1.0]), None, None, 1, [(0.0, np.inf), (0.0, 1.0)], sense="min")
    res = cardinality_constrained_milp(np.array([-1.0, -2.0]), None, None, 1, [(0.0, 1.0), (0.0, 1.0)], sense="min")
    assert res is not None
    value, x = res
    assert value == pytest.approx(-2.0)
    assert x == pytest.approx([0.0, 1.0])


# --------------------------------------------------------------- C-10: route_past guards
def test_route_past_guards_empty_tiers_and_length_mismatch():
    from mixle.system import route_past

    with pytest.raises(ValueError, match="at least one tier"):
        route_past([])
    with pytest.raises(ValueError, match="must match one-to-one"):
        route_past([lambda: 1, lambda: 2], names=["only_one"])
    ok = route_past([lambda: 1])
    assert (ok.value, ok.degraded) == (1, False)

    def boom():
        raise RuntimeError("tier down")

    routed = route_past([boom, lambda: 2], names=["primary", "backup"])
    assert (routed.value, routed.degraded, routed.mode) == (2, True, "model_error")


# --------------------------------------------------------------- C-11: MILP incumbent exactness
def test_milp_incumbent_integer_coordinates_are_exact_ints():
    # the LP optimum 0.9999994 is within tol of 1 and must be snapped, not stored raw
    res = branch_and_bound_milp(
        np.array([-1.0]), np.array([[2.0]]), np.array([1.9999988]), integer=[0], bounds=[(0.0, 10.0)], sense="min"
    )
    assert res is not None
    _value, x = res
    assert float(x[0]) == 1.0  # exact, so downstream int(x[0]) cannot truncate to 0


def test_milp_accepts_a_generator_for_integer_indices():
    c = np.array([-1.0, -1.0])
    a_ub = np.array([[3.0, 2.0]])
    b_ub = np.array([7.0])
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    from_list = branch_and_bound_milp(c, a_ub, b_ub, integer=[0, 1], bounds=bounds)
    from_gen = branch_and_bound_milp(c, a_ub, b_ub, integer=(i for i in range(2)), bounds=bounds)
    assert from_list is not None and from_gen is not None
    assert from_gen[0] == pytest.approx(from_list[0])
    assert from_gen[1] == pytest.approx(from_list[1])
    assert all(float(v).is_integer() for v in from_gen[1])
