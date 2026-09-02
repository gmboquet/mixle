"""Release-wave regression tests for the propose/Model/deploy/load cluster (B7-B11, t3/t4 findings).

Each test pins a verified 0.8.0 release-candidate defect:

* B9  -- ``propose(DataFrame)`` modeled the COLUMN NAMES (certificate ``n_rows=5`` for an 891-row
         frame) instead of routing rows through the tabular path ``optimize`` uses.
* B10 -- when every candidate failed to score, ``propose`` returned a winner with no disclosure and
         a ``succeeded`` certificate.
* B8  -- a degenerate near-Dirac continuous fit (scale ~ 1e-26 at a repeated value) WON the held-out
         mean-log-density ranking on float-cast count data.
* B7  -- ``Model.fit`` (default ``restarts="auto"``) returned a Gaussian mixture ~58 nats below the
         optimum on Old Faithful: EM was truncated at optimize()'s 10-iteration default, the supplied
         prototype initialization was silently ignored, and nothing disclosed the truncation.
* B11 -- the dtype-driven candidate universe (int -> discrete families, float -> continuous) was
         silent at the propose() level.
* t3  -- every unreadable-manifest state misreported as "is a pickle-format artifact ... Pass
         trust_code=True"; ``trust_code=None``/``0`` were rejected on artifacts needing no trust;
         corrupt model files leaked raw stdlib exceptions; a list-shaped manifest leaked
         ``AttributeError``; the manifest's self-description was not bound to the model file.
* t4  -- ``propose(..., fit=True)`` could return ``Model.spec = None`` (non-reusable proposal), and
         ``Model(SomeDistribution)`` (the class) leaked ``AttributeError`` from deep inside fit.
"""

import json
import shutil

import numpy as np
import pytest

import mixle
import mixle.dist as D
from mixle.inference.em import run_em
from mixle.lifecycle import _degenerate_likelihood_spike, _tabular_records
from mixle.utils.serialization import SerializationError

# R datasets::faithful, waiting column (272 obs; public domain). Verified against the published
# two-component reference fit: sklearn GaussianMixture(2, random_state=0, n_init=20) on this vector
# reproduces means [54.6998, 80.1455], vars [35.3176, 33.7902], weights [0.3635, 0.6365] and total
# log-likelihood -1034.0194 to four decimals.
FAITHFUL_WAITING = [
    79,
    54,
    74,
    62,
    85,
    55,
    88,
    85,
    51,
    85,
    54,
    84,
    78,
    47,
    83,
    52,
    62,
    84,
    52,
    79,
    51,
    47,
    78,
    69,
    74,
    83,
    55,
    76,
    78,
    79,
    73,
    77,
    66,
    80,
    74,
    52,
    48,
    80,
    59,
    90,
    80,
    58,
    84,
    58,
    73,
    83,
    64,
    53,
    82,
    59,
    75,
    90,
    54,
    80,
    54,
    83,
    71,
    64,
    77,
    81,
    59,
    84,
    48,
    82,
    60,
    92,
    78,
    78,
    65,
    73,
    82,
    56,
    79,
    71,
    62,
    76,
    60,
    78,
    76,
    83,
    75,
    82,
    70,
    65,
    73,
    88,
    76,
    80,
    48,
    86,
    60,
    90,
    50,
    78,
    63,
    72,
    84,
    75,
    51,
    82,
    62,
    88,
    49,
    83,
    81,
    47,
    84,
    52,
    86,
    81,
    75,
    59,
    89,
    79,
    59,
    81,
    50,
    85,
    59,
    87,
    53,
    69,
    77,
    56,
    88,
    81,
    45,
    82,
    55,
    90,
    45,
    83,
    56,
    89,
    46,
    82,
    51,
    86,
    53,
    79,
    81,
    60,
    82,
    77,
    76,
    59,
    80,
    49,
    96,
    53,
    77,
    77,
    65,
    81,
    71,
    70,
    81,
    93,
    53,
    89,
    45,
    86,
    58,
    78,
    66,
    76,
    63,
    88,
    52,
    93,
    49,
    57,
    77,
    68,
    81,
    81,
    73,
    50,
    85,
    74,
    55,
    77,
    83,
    83,
    51,
    78,
    84,
    46,
    83,
    55,
    81,
    57,
    76,
    84,
    77,
    81,
    87,
    77,
    51,
    78,
    60,
    82,
    91,
    53,
    78,
    46,
    77,
    84,
    49,
    83,
    71,
    80,
    49,
    75,
    64,
    76,
    53,
    94,
    55,
    76,
    50,
    82,
    54,
    75,
    78,
    79,
    78,
    78,
    70,
    79,
    70,
    54,
    86,
    50,
    90,
    54,
    54,
    77,
    79,
    64,
    75,
    47,
    86,
    63,
    85,
    82,
    57,
    82,
    67,
    74,
    54,
    83,
    73,
    73,
    88,
    80,
    71,
    83,
    56,
    79,
    78,
    84,
    58,
    83,
    43,
    60,
    75,
    81,
    46,
    90,
    46,
    74,
]
# The two-component optimum on the vector above (mixle EM converged from any tested start; slightly
# better than the best-of-20 sklearn reference -1034.0194).
FAITHFUL_OPTIMUM_LL = -1034.0017


# --- B9: DataFrame / dict-of-columns route through the tabular path -------------------------------


def _toy_frame(n=160):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "fare": rng.gamma(2.0, 16.0, size=n),
            "sex": rng.choice(["male", "female"], size=n, p=[0.65, 0.35]),
            "parch": rng.poisson(0.4, size=n),
        }
    )


def test_propose_dataframe_models_rows_not_column_names():
    df = _toy_frame()
    m = mixle.propose(df, fit=True)
    # the defect returned CategoricalDistribution({'fare': ..., 'sex': ..., 'parch': ...})
    assert type(m.fitted).__name__ != "CategoricalDistribution"
    assert m.evidence["certificate"]["n_rows"] == len(df)
    # one field note per column proves per-column modeling
    field_notes = [n for n in m.notes if n.startswith("field ")]
    assert len(field_notes) == df.shape[1]
    # a real row scores finite; a column-name string is not even a valid observation
    row = tuple(df.iloc[0])
    assert np.isfinite(m.fitted.log_density(row))


def test_propose_dict_of_columns_matches_dataframe_rows():
    df = _toy_frame()
    cols = {name: list(df[name]) for name in df.columns}
    m = mixle.propose(cols, fit=True)
    assert type(m.fitted).__name__ != "CategoricalDistribution"
    assert m.evidence["certificate"]["n_rows"] == len(df)


def test_tabular_records_rejects_ragged_and_scalar_column_mappings():
    with pytest.raises(ValueError, match="equal-length columns"):
        _tabular_records({"a": [1, 2, 3], "b": [1, 2]})
    with pytest.raises(ValueError, match="not a sized"):
        _tabular_records({"a": 1.5})
    with pytest.raises(ValueError, match="empty mapping"):
        _tabular_records({})


def test_model_fit_and_evaluate_accept_dataframes():
    df = _toy_frame(80)
    m = mixle.propose(df, fit=True)
    scores = m.evaluate(df)
    assert scores["n"] == len(df)
    assert np.isfinite(scores["mean_log_density"])
    m2 = mixle.Model(m.spec).fit(df)
    assert m2._fit_info["n"] == len(df)


def test_tabular_records_leaves_record_lists_alone():
    rows = [1.0, 2.0, 3.0]
    assert _tabular_records(rows) == rows
    tuples = [(1.0, "a"), (2.0, "b")]
    assert _tabular_records(tuples) == tuples


# --- B10: all-candidates-failed disclosure ---------------------------------------------------------


def _all_fail_rows(n=40):
    """A dataset where every candidate fails held-out scoring but the final fit is viable:
    one category appears only in the (seed=0, holdout=0.25) validation split."""
    order = np.random.RandomState(0).permutation(n)
    rows = ["a" if i % 2 else "b" for i in range(n)]
    rows[order[0]] = "z"  # order[:n_val] is the holdout; plant the unseen category there
    return rows


def test_all_candidates_failed_is_disclosed_and_not_certified_succeeded():
    m = mixle.propose(_all_fail_rows(), fit=True)
    assert all(("error" in f) or ("skipped" in f) for f in m.frontier)
    rollup = [n for n in m.notes if n.startswith("no candidate could be verified:")]
    assert len(rollup) == 1
    assert "unverified" in rollup[0]
    # certify() ran, but 'succeeded' would claim verification that never happened
    assert m.evidence["certificate"]["status"] == "attempted"
    assert "no candidate could be verified" in m.evidence["certificate"]["reason"]


def test_all_candidates_failed_disclosure_without_fit():
    m = mixle.propose(_all_fail_rows())
    assert any(n.startswith("no candidate could be verified:") for n in m.notes)
    assert m.fitted is None  # fit=False path unchanged otherwise


def test_verified_winner_keeps_succeeded_certificate():
    rng = np.random.default_rng(3)
    m = mixle.propose(list(rng.normal(10.0, 2.0, 200)), fit=True)
    assert any("heldout_mean_log_density" in f for f in m.frontier)
    assert m.evidence["certificate"]["status"] == "succeeded"
    assert not any(n.startswith("no candidate could be verified:") for n in m.notes)


def test_budget_skip_disclosure_still_present():
    rng = np.random.default_rng(4)
    m = mixle.propose(list(rng.normal(0.0, 1.0, 60)), max_candidates=0)
    assert any(n.startswith("search budget: skipped") for n in m.notes)


# --- B8: degenerate likelihood spikes are rejected from the frontier -------------------------------


def test_degenerate_near_dirac_fit_does_not_win():
    # Campaign nine (D-0209) closed the root cause this test used to reach via the frontier's
    # admit-then-reject cycle: GeneralizedPareto's own auto-inference detector now refuses this
    # exact near-Dirac shape at _applies() (its moment estimate and its real scipy MLE disagree by
    # 20+ orders of magnitude on scale). GPD is therefore never admitted, fit, or "rejected from
    # the frontier" for this data any more -- the "degenerate fit rejected" frontier entry and
    # note this test used to grep for no longer appear, which is the fix working, not a
    # regression. The next-best candidate, plain ParetoDistribution (alpha~1.45 < 2), wins
    # instead: a legitimate, non-degenerate fit by ITS OWN family's standards, but with a
    # theoretically infinite variance (a Pareto shape<2 property, unrelated to and out of scope
    # for this campaign -- ParetoDistribution's own detector was not touched).
    #
    # Whether that infinite variance itself needed a fix was investigated separately (campaign
    # nine, D-0209) and asserted below rather than left an unexamined gap. It does not: alpha~1.45
    # is exactly what ParetoEstimator's unclamped MLE computed, so unlike the GeneralizedPareto/
    # Gumbel/Student-t/Logistic/LogGaussian "scale-floored"/"variance-floored" numerical_repairs()
    # notes elsewhere in this campaign -- all a returned parameter differing from what a
    # precision-collapsed computation actually produced -- nothing here was repaired, and
    # numerical_repairs() correctly stays empty. A detector-level gate on alpha<2 was also
    # rejected: a low-alpha Pareto is internally consistent and the textbook-honest model for
    # ordinary real heavy-tailed data (word/city/wealth frequencies commonly have alpha in (1,
    # 2)), unlike the near-Dirac GPD case above (two independent estimates disagreeing by 20+
    # orders of magnitude); refusing it would be guard overreach, forcing a worse-fitting family
    # onto legitimate data -- plain Pareto beats the GEV runner-up by ~5.1 held-out bits here. The
    # disclosure already exists, generically: mixle.summarize()'s own "_status" receipt marks
    # variance/std "invalid" (not "available") whenever a HasMoments distribution's own method
    # returns non-finite, asserted directly below (see also
    # summarize_test.py::test_never_raises_on_undefined_moments, predating this campaign).
    rng = np.random.default_rng(0)
    x = rng.geometric(0.4, size=20000).astype(float)  # ~39% exactly 1.0, integer heavy tail
    m = mixle.propose(x, fit=True)
    s = mixle.summarize(m.fitted)
    assert np.isfinite(s["mean"]), s
    # the winning alpha<2 Pareto's infinite variance is real, not a bug, and already disclosed:
    assert s["variance"] == float("inf") and s["_status"]["variance"]["status"] == "invalid", s
    assert m.fitted.numerical_repairs() == (), m.fitted.numerical_repairs()
    draws = np.asarray(m.sample(50, seed=0), dtype=float)
    assert len(np.unique(np.round(draws, 6))) > 1
    # and its mean is in the right ballpark of the data (empirical mean 2.5), not inf
    assert 1.0 < s["mean"] < 5.0


def test_spike_guard_does_not_reject_sound_fits():
    # three legitimate shapes that share ONE spike signal each, never all of them
    gen = np.random.default_rng(1)
    heavy = list(gen.pareto(1.5, 4000) + 1.0)  # heavy tail, huge density RATIO, max logpdf <= ~0.4
    micro = list(np.random.default_rng(2).normal(0.0, 1e-6, 4000))  # max logpdf ~ +12.9, uniformly
    logn = list(np.random.default_rng(3).lognormal(0.0, 1.0, 4000))
    for data in (heavy, micro, logn):
        m = mixle.propose(data, fit=True)
        assert not any("degenerate fit rejected" in n for n in m.notes), m.notes
        assert any("heldout_mean_log_density" in f for f in m.frontier)


def test_spike_guard_unit_criterion():
    # never fires without a positive pointwise log-density...
    d = D.GaussianDistribution(mu=0.0, sigma2=1.0)
    val = [0.0, 0.1, -0.1, 0.2]
    scores = np.array([-0.9, -0.92, -0.91, -0.93])
    assert _degenerate_likelihood_spike(d, val, scores) is None
    # ... nor on a uniformly high but flat score profile (tiny-scale data)
    flat = np.full(64, 12.9)
    assert _degenerate_likelihood_spike(d, val, flat) is None


def test_spike_guard_unit_criterion_fires_on_majority_atom():
    # Direct synthetic (fitted, val, scores) for the "atom carries a majority of held-out rows"
    # branch itself -- flagged as uncovered by test_majority_atom_collapse_is_rejected_despite_
    # median_at_the_spike below, whose data stopped reaching this branch once D-0209 fixed
    # GeneralizedPareto's _applies() to refuse that majority-atom-plus-tail shape upstream.
    # Needs, directly: a positive top score; top == mid (flat spread, so the spread-gated sibling
    # branch resolves first and does NOT fire); a pathological PIT from calibration_report; and a
    # single repeated value in val carrying a strict majority of the rows.
    atom = 0.25
    val = [atom] * 30 + [-50.0] * 5 + [50.0] * 5  # atom is 30/40 = 75% of rows, a strict majority
    fitted = D.GaussianDistribution(mu=atom, sigma2=1.0)  # cdf(atom) == 0.5 exactly: PIT piles up
    scores = np.full(len(val), 5.0)  # top == mid: spread is 0, well under _SPIKE_SPREAD_NATS
    msg = _degenerate_likelihood_spike(fitted, val, scores)
    assert msg is not None
    assert "carried by a MAJORITY of held-out rows" in msg


# --- B7: Model.fit iterates to tolerance, honors inits, discloses truncation ----------------------


def _faithful_rows():
    return [[float(v)] for v in FAITHFUL_WAITING]


def _total_ll(model, rows):
    return float(sum(model.log_density(x) for x in rows))


@pytest.mark.parametrize("init", [[[55.0], [80.0]], [[50.0], [90.0]], [[70.0], [70.1]]])
def test_old_faithful_reaches_optimum_with_default_fit(init):
    rows = _faithful_rows()
    spec = D.GaussianMixtureDistribution(mu=init, sig2=[[36.0], [36.0]], w=[0.5, 0.5])
    m = mixle.Model(spec).fit(rows)  # restarts defaults to "auto"; max_its defaults high here
    ll = _total_ll(m.fitted, rows)
    assert ll >= FAITHFUL_OPTIMUM_LL - 1.0, f"fit stopped {FAITHFUL_OPTIMUM_LL - ll:.2f} nats short"
    means = sorted(float(u[0]) for u in m.fitted.mu)
    assert abs(means[0] - 54.6) < 1.5 and abs(means[1] - 80.1) < 1.5
    assert m._fit_info["converged"] is True
    assert m._fit_info["n_iter"] >= 1


def test_supplied_initialization_is_honored():
    rows = _faithful_rows()
    # start AT the known optimum with a tiny budget: an honored init stays there, the old
    # structure-only coercion re-initialized from a random subsample and landed 58 nats away
    spec = D.GaussianMixtureDistribution(mu=[[54.635], [80.104]], sig2=[[34.3], [34.6]], w=[0.361, 0.639])
    m = mixle.Model(spec).fit(rows, restarts=None, max_its=2)
    ll = _total_ll(m.fitted, rows)
    assert ll >= FAITHFUL_OPTIMUM_LL - 1.0


def test_iteration_cap_truncation_is_disclosed():
    rows = _faithful_rows()
    spec = D.GaussianMixtureDistribution(mu=[[55.0], [80.0]], sig2=[[36.0], [36.0]], w=[0.5, 0.5])
    m = mixle.Model(spec).fit(rows, restarts=None, max_its=3)
    assert m._fit_info["converged"] is False
    assert any("iteration cap" in n for n in m.notes)


def test_restarts_auto_diversifies_after_unconverged_latent_fit():
    rows = _faithful_rows()
    spec = D.GaussianMixtureDistribution(mu=[[55.0], [80.0]], sig2=[[36.0], [36.0]], w=[0.5, 0.5])
    # a deliberately starved budget: the diversified refit must fire (and be disclosed) instead of
    # silently returning the plateau fit with empty notes
    m = mixle.Model(spec).fit(rows, max_its=3)
    assert any(("iteration cap" in n) or ("saddle" in n) for n in m.notes)


def test_run_em_carries_convergence_provenance():
    from mixle.stats.compute.sequence import seq_encode

    rng = np.random.RandomState(0)
    data = [[float(v)] for v in np.concatenate([rng.normal(0, 1, 60), rng.normal(8, 1, 60)])]
    proto = D.GaussianMixtureDistribution(mu=[[0.5], [7.5]], sig2=[[1.0], [1.0]], w=[0.5, 0.5])
    est = proto.estimator()
    enc = seq_encode(data, encoder=proto.dist_to_encoder())
    fitted = run_em(enc, est, proto, max_its=200, delta=1.0e-9)
    prov = fitted.fit_provenance()
    assert prov is not None
    assert prov.converged is True
    assert 1 <= prov.iterations <= 200
    truncated = run_em(enc, est, proto, max_its=1, delta=1.0e-9)
    tprov = truncated.fit_provenance()
    assert tprov is not None
    assert tprov.converged is False
    assert tprov.iterations == 1


# --- B11: dtype-derived candidate universe is disclosed -------------------------------------------


def test_dtype_universe_notes():
    rng = np.random.default_rng(4)
    xi = (rng.poisson(3.0, 800) + 1).astype(np.int64)
    mi = mixle.propose(xi, fit=True)
    assert any(n.startswith("integer dtype: discrete candidate families") for n in mi.notes)
    mf = mixle.propose(xi.astype(float), fit=True)
    assert any(n.startswith("float dtype: continuous candidate families") for n in mf.notes)


def test_dtype_universe_note_absent_for_composite_records():
    df = _toy_frame(60)
    m = mixle.propose(df)
    assert not any("candidate families considered" in n for n in m.notes)


# --- t3: load()/deploy() error paths ---------------------------------------------------------------


@pytest.fixture()
def artifact(tmp_path):
    rng = np.random.RandomState(0)
    m = mixle.Model(D.GammaDistribution(2.0, 2.0)).fit([float(v) for v in rng.gamma(2, 2, 80)])
    out = tmp_path / "art"
    m.deploy(str(out))
    return out


def test_load_names_nonexistent_path(tmp_path):
    with pytest.raises(SerializationError, match="does not exist"):
        mixle.Model.load(str(tmp_path / "nope"))


def test_load_names_plain_file(tmp_path):
    f = tmp_path / "plain.txt"
    f.write_text("hello")
    with pytest.raises(SerializationError, match="not an artifact directory"):
        mixle.Model.load(str(f))


def test_load_names_empty_directory(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(SerializationError, match="no manifest.json and no model file"):
        mixle.Model.load(str(d))


def test_load_names_interrupted_deploy(tmp_path, artifact):
    crashed = tmp_path / "crashed"
    shutil.copytree(artifact, crashed)
    (crashed / "manifest.json").unlink()
    with pytest.raises(SerializationError, match="interrupted deploy"):
        mixle.Model.load(str(crashed))


def test_load_names_corrupt_manifest(tmp_path, artifact):
    trunc = tmp_path / "trunc"
    shutil.copytree(artifact, trunc)
    (trunc / "manifest.json").write_text((artifact / "manifest.json").read_text()[:40])
    with pytest.raises(SerializationError, match="manifest.json is not valid JSON"):
        mixle.Model.load(str(trunc))
    (trunc / "manifest.json").write_text("")
    with pytest.raises(SerializationError, match="manifest.json is not valid JSON"):
        mixle.Model.load(str(trunc))


def test_load_rejects_non_object_manifest(tmp_path, artifact):
    listman = tmp_path / "listman"
    shutil.copytree(artifact, listman)
    (listman / "manifest.json").write_text("[1,2,3]")
    with pytest.raises(SerializationError, match="must hold a JSON object"):
        mixle.Model.load(str(listman))


def test_load_never_claims_pickle_for_unreadable_manifests(tmp_path, artifact):
    """No unreadable-manifest state may resolve to the 'pass trust_code=True' pickle message."""
    crashed = tmp_path / "crashed2"
    shutil.copytree(artifact, crashed)
    (crashed / "manifest.json").unlink()
    for target in (tmp_path / "missing", crashed):
        with pytest.raises(SerializationError) as err:
            mixle.Model.load(str(target))
        assert "pickle-format artifact" not in str(err.value)
        assert "trust_code=True" not in str(err.value)


def test_trust_code_falsy_no_answers_load_pure_json(artifact):
    for no in (False, None, 0):
        m = mixle.Model.load(str(artifact), trust_code=no)
        assert type(m.fitted).__name__ == "GammaDistribution"


def test_trust_code_ambiguous_values_still_rejected(artifact):
    for bad in ("false", "true", 1, [True]):
        with pytest.raises(ValueError, match="exactly True"):
            mixle.Model.load(str(artifact), trust_code=bad)


def test_load_wraps_corrupt_model_file_under_legacy_manifest(tmp_path, artifact):
    legacy = tmp_path / "legacy"
    shutil.copytree(artifact, legacy)
    man = json.loads((legacy / "manifest.json").read_text())
    man.pop("model_sha256")
    (legacy / "manifest.json").write_text(json.dumps(man))
    payload = (legacy / "model.json").read_bytes()
    (legacy / "model.json").write_bytes(payload[:60])  # truncated JSON
    with pytest.raises(SerializationError, match="model.json is not valid JSON"):
        mixle.Model.load(str(legacy))
    (legacy / "model.json").write_bytes(bytes(range(256)))  # binary garbage
    with pytest.raises(SerializationError, match="could not be read as text"):
        mixle.Model.load(str(legacy))


def test_load_names_missing_model_file(tmp_path, artifact):
    gone = tmp_path / "gone"
    shutil.copytree(artifact, gone)
    (gone / "model.json").unlink()
    with pytest.raises(SerializationError, match="names model.json but that file is not"):
        mixle.Model.load(str(gone))


def test_load_still_refuses_digest_mismatch(tmp_path, artifact):
    tampered = tmp_path / "tampered"
    shutil.copytree(artifact, tampered)
    (tampered / "model.json").write_text((tampered / "model.json").read_text() + " ")
    with pytest.raises(SerializationError, match="does not match the digest"):
        mixle.Model.load(str(tampered))


def test_load_binds_manifest_family_to_model_file(tmp_path, artifact):
    import hashlib

    other = mixle.Model(D.PoissonDistribution(4.0)).fit([1, 2, 3, 4, 2, 3])
    other_art = tmp_path / "pois"
    other.deploy(str(other_art))
    swap = tmp_path / "swap"
    shutil.copytree(artifact, swap)
    shutil.copy(other_art / "model.json", swap / "model.json")
    man = json.loads((swap / "manifest.json").read_text())
    man["model_sha256"] = "sha256:" + hashlib.sha256((swap / "model.json").read_bytes()).hexdigest()
    (swap / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SerializationError, match="do not belong to the same"):
        mixle.Model.load(str(swap))


def test_deploy_load_round_trip_still_works(tmp_path):
    rows = _faithful_rows()
    spec = D.GaussianMixtureDistribution(mu=[[55.0], [80.0]], sig2=[[36.0], [36.0]], w=[0.5, 0.5])
    m = mixle.Model(spec).fit(rows)
    out = tmp_path / "gmm"
    m.deploy(str(out))
    m2 = mixle.Model.load(str(out))
    assert type(m2.fitted).__name__ == "GaussianMixtureDistribution"
    x = rows[0]
    assert m2.fitted.log_density(x) == pytest.approx(m.fitted.log_density(x))
    # the fit receipt (incl. the new convergence fields) survives the round trip
    assert m2._fit_info["n"] == len(rows)
    assert m2._fit_info["converged"] is True


# --- t4 / constructor minors -----------------------------------------------------------------------


def test_model_rejects_distribution_class():
    from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

    with pytest.raises(TypeError, match=r"instantiate it first.*GaussianDistribution"):
        mixle.Model(GaussianDistribution)


def test_gaussian_mixture_rejects_1d_means_with_named_error():
    with pytest.raises(ValueError, match=r"shape \(K, d\).*univariate mixture"):
        D.GaussianMixtureDistribution(mu=[55.0, 80.0], sig2=[36.0, 36.0], w=[0.5, 0.5])
    # the documented forms still construct
    d = D.GaussianMixtureDistribution(mu=[[55.0], [80.0]], sig2=[[36.0], [36.0]], w=[0.5, 0.5])
    assert d.dim == 1
    d2 = D.GaussianMixtureDistribution(
        mu=[[0.0, 0.0], [1.0, 1.0]],
        sig2=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        w=[0.5, 0.5],
    )
    assert d2.dim == 2


def test_propose_spec_round_trips_for_structured_winner():
    rng = np.random.default_rng(0)
    x = rng.geometric(0.4, size=20000).astype(float)  # structured candidate wins here (see B8 test)
    m = mixle.propose(x, fit=True)
    assert m.spec is not None, "a fitted proposal must carry a reusable spec"
    refit = mixle.Model(m.spec).fit(list(x)[:4000])
    assert type(refit.fitted).__name__ == type(m.fitted).__name__


def test_budget_skip_certificate_is_attempted_not_succeeded():
    # Both documented unverified routes must carry the same certificate semantics: a winner the
    # budget skipped is exactly as unverified as one whose candidates all failed (wave-3 check).
    rng = np.random.default_rng(4)
    m = mixle.propose(list(rng.normal(0.0, 1.0, 60)), fit=True, max_candidates=0)
    assert m.evidence["certificate"]["status"] == "attempted"
    assert "search budget skipped every candidate" in m.evidence["certificate"]["reason"]


def test_majority_atom_collapse_is_rejected_despite_median_at_the_spike():
    # With >50% of held-out rows ON the atom the median equals the max, so a spread-gated guard
    # never consulted PIT and the MORE degenerate fit sailed through (wave-3 check).
    #
    # Campaign nine (D-0209) closed the root cause this test's data was reaching that guard
    # through: GeneralizedPareto's own auto-inference detector now refuses this exact
    # majority-atom-plus-tail shape at _applies() (its moment estimate and its real scipy MLE
    # disagree by 20+ orders of magnitude on scale -- confirmed directly against
    # mixle.utils.automatic.detectors.generalized_pareto._applies(x) here). GPD is therefore never
    # admitted, fit, or rejected for this data any more; a two-component Gaussian mixture wins
    # instead (one component correctly collapsing onto the atom, the other modeling the Poisson
    # tail), with sensible, finite moments and a clean certificate -- a better outcome than the
    # admit-then-reject cycle this test originally pinned, not merely a different one.
    #
    # This means _degenerate_likelihood_spike's own "atom carries a majority of held-out rows"
    # branch (mixle/lifecycle.py, the message this test used to grep for) is no longer exercised
    # by ANY test: test_spike_guard_unit_criterion only covers the negative (never-fires) cases.
    # Flagged separately (task_3fd274a3's sibling) rather than silently left uncovered -- a direct
    # unit test synthesizing (fitted, val, scores) to hit that branch should be added.
    rng = np.random.default_rng(7)
    x = np.where(rng.random(3000) < 0.65, 1.0, rng.poisson(6.0, 3000) + 1.0)
    m = mixle.propose(x, fit=True)
    s = mixle.summarize(m.fitted)
    assert np.isfinite(s["mean"]) and np.isfinite(s["variance"]), s
    # true mixture mean is 0.65*1 + 0.35*7 = 3.1; the fit should land close to it, not on the atom.
    assert 2.0 < s["mean"] < 4.5, s
    assert m.evidence["certificate"]["status"] == "succeeded"
