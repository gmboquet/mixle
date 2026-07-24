"""PosteriorRetriever: retrieval by what the model believes — with the evidence cap doing its job."""

import numpy as np
import pytest

torch = None
try:
    import torch  # noqa: F401
except ImportError:
    pass

from mixle.inference import optimize
from mixle.stats import (
    CategoricalEstimator,
    CompositeEstimator,
    GaussianDistribution,
    GaussianEstimator,
    MixtureEstimator,
)


def _records(n, seed=0):
    """Two latent regimes over (kind, level, noise-tag): regime decides kind AND level jointly."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        z = rng.randint(0, 2)
        kind = ["alpha", "beta"][z] if rng.rand() < 0.9 else ["alpha", "beta"][1 - z]
        level = float(rng.normal(-3.0 if z == 0 else 3.0, 1.0))
        tag = str(rng.randint(0, 50))  # a sharp, nearly-unique field that should NOT veto similarity
        rows.append((kind, level, tag))
    return rows


def _fit(rows):
    comp = lambda: CompositeEstimator((CategoricalEstimator(), GaussianEstimator(), CategoricalEstimator()))  # noqa: E731
    return optimize(rows, MixtureEstimator([comp(), comp()]), max_its=25, out=None)


def test_retrieves_same_regime_neighbours():
    from mixle.represent import PosteriorRetriever

    rows = _records(200)
    model = _fit(rows)
    r = PosteriorRetriever(model, rows)

    # a fresh regime-0 query retrieves regime-0 records (negative level), despite a never-seen tag
    hits = r.retrieve(("alpha", -3.2, "9999"), k=10)
    levels = [rows[i][1] for i, _ in hits]
    assert np.mean([lv < 0 for lv in levels]) > 0.8

    hits1 = r.retrieve(("beta", 3.1, "8888"), k=10)
    levels1 = [rows[i][1] for i, _ in hits1]
    assert np.mean([lv > 0 for lv in levels1]) > 0.8


def test_evidence_cap_prevents_single_field_veto():
    from mixle.represent import PosteriorRetriever

    rows = _records(200)
    model = _fit(rows)

    capped = PosteriorRetriever(model, rows, evidence_cap=1.0)
    hits = capped.retrieve(("alpha", -2.8, "no-such-tag"), k=5)
    # with the cap, the unique tag can only testify ~1 nat: regime similarity still wins
    assert np.mean([rows[i][1] < 0 for i, _ in hits]) > 0.6
    # scores are finite: no -inf veto from the unseen tag
    assert all(np.isfinite(a) for _, a in hits)


def test_batch_and_matrix_shapes():
    from mixle.represent import PosteriorRetriever

    rows = _records(60)
    r = PosteriorRetriever(_fit(rows), rows)
    out = r.retrieve_batch([("alpha", -3.0, "1"), ("beta", 3.0, "2")], k=3)
    assert len(out) == 2 and all(len(o) == 3 for o in out)
    m = r.affinity_matrix()
    assert m.shape == (60, 60) and np.all(np.diag(m) == -np.inf)


def test_requires_a_mixture():
    from mixle.represent import PosteriorRetriever

    with pytest.raises(TypeError):
        PosteriorRetriever(GaussianDistribution(0.0, 1.0), [(1,), (2,)])


def test_retrieve_rejects_negative_k():
    # Python's [:k] slicing treats a negative k as "all but the last |k| items", not
    # empty/error -- a negative retrieval count has no sensible meaning, so both the
    # single-query and batch entry points must reject it instead of silently returning
    # the wrong-sized result.
    from mixle.represent import PosteriorRetriever

    rows = _records(30)
    r = PosteriorRetriever(_fit(rows), rows)
    with pytest.raises(ValueError):
        r.retrieve(rows[0], k=-1)
    with pytest.raises(ValueError):
        r.retrieve_batch([rows[0], rows[1]], k=-1)


def test_retrieve_rejects_non_integer_k():
    # A fractional k (e.g. 2.7) must not be silently truncated by an int() cast --
    # it must be rejected as a caller error, same as a negative k.
    from mixle.represent import PosteriorRetriever

    rows = _records(30)
    r = PosteriorRetriever(_fit(rows), rows)
    with pytest.raises(ValueError):
        r.retrieve(rows[0], k=2.7)
    with pytest.raises(ValueError):
        r.retrieve_batch([rows[0], rows[1]], k=2.7)


def test_retrieve_k_edge_cases_still_work():
    from mixle.represent import PosteriorRetriever

    rows = _records(30)
    r = PosteriorRetriever(_fit(rows), rows)
    assert r.retrieve(rows[0], k=0) == []  # k=0 means "return nothing", not an error
    hits = r.retrieve(rows[0], k=10_000)  # k beyond the corpus just returns everything
    assert len(hits) == len(rows)
    hits3 = r.retrieve(rows[0], k=3.0)  # a float k with an exact integer value is fine
    assert len(hits3) == 3
