"""Worklist I6.3 -- route selection is explainable AND the explanation is true.

``fit.explain_fit()`` returns ``{route, reason, caveats}`` naming the inference route
mixle chose. An explanation is only useful if it agrees with what the fitter actually
did; a plausible-but-wrong route label is worse than none. This test pins the I6.3
acceptance -- "explanation agrees with the actual executed fitter in all routing tests"
-- by tying each claimed route to an *observable* property of the fitted object:

  * ``conjugate`` -> the fit carries a closed-form ConjugatePosterior (no MCMC draws);
  * ``em`` / ``map`` -> a point estimate, with no posterior-sample result object;
  * ``laplace`` / ``vi`` / ``mcmc`` / ``nuts`` -> a sampling-backed Posterior that
    reports R-hat / ESS diagnostics.

If ``explain_fit`` claimed ``nuts`` but the fit produced no posterior draws (or claimed
``em`` but produced MCMC diagnostics), the invariant below catches the disagreement.
The explanation is also checked for non-staleness: refitting the same model object under
a different route must update the reported route, not echo the previous one.
"""

from __future__ import annotations

import pytest

from mixle.ppl import Mix, Normal, free

DATA = [-2.1, 1.9, -1.8, 2.3, -2.0, 2.1, -1.9, 2.0, -2.2, 1.7, -1.6, 2.4]

# Routes whose executed fitter yields a sampling-backed posterior (R-hat / ESS).
SAMPLING_ROUTES = frozenset({"laplace", "vi", "mcmc", "nuts"})
# Routes whose executed fitter yields a point estimate (no posterior-sample object).
POINT_ROUTES = frozenset({"em", "map"})


def _assert_result_matches_route(fit: object, route: str) -> None:
    """The observable shape of `fit` must match the route `explain_fit` claims."""
    result = getattr(fit, "result", None)

    if route == "conjugate":
        assert result is not None, "conjugate route reported but no posterior result object"
        assert "Conjugate" in type(result).__name__, (
            f"route=conjugate but result is {type(result).__name__}, not a ConjugatePosterior"
        )
        assert not hasattr(result, "rhat"), (
            "route=conjugate but the result carries MCMC diagnostics -- that is a "
            "sampling route, not a closed-form posterior"
        )
    elif route in POINT_ROUTES:
        assert result is None, (
            f"route={route} is a point estimate but produced a {type(result).__name__} "
            f"posterior-sample result object -- explanation disagrees with the fitter"
        )
    elif route in SAMPLING_ROUTES:
        assert result is not None, f"route={route} reported but no posterior result object"
        assert hasattr(result, "rhat") and hasattr(result, "ess"), (
            f"route={route} claims a sampling-backed posterior but the result "
            f"({type(result).__name__}) reports no R-hat/ESS diagnostics"
        )
    else:  # pragma: no cover - guards against an unmapped route leaking in
        pytest.fail(f"unrecognized route {route!r}; extend the invariant map for I6.3")


# (model factory, expected auto-selected route) -- no `how=`, so the route is chosen
# from the model's structure alone.
AUTO_CASES = [
    ("mle-em", lambda: Normal(free, free), "em"),
    ("conjugate", lambda: Normal(Normal(0, 10), 1.0), "conjugate"),
    ("mixture-em", lambda: Mix([Normal(free, free), Normal(free, free)]), "em"),
]


@pytest.mark.parametrize("label, make, expected_route", AUTO_CASES, ids=[c[0] for c in AUTO_CASES])
def test_auto_route_explanation_agrees_with_fitter(label: str, make, expected_route: str) -> None:
    fit = make().fit(DATA)
    ef = fit.explain_fit()
    assert ef["route"] == expected_route, (
        f"{label}: explain_fit says route={ef['route']!r}, expected {expected_route!r}"
    )
    assert ef.get("reason"), f"{label}: explanation has no 'reason' (the why)"
    _assert_result_matches_route(fit, ef["route"])


@pytest.mark.parametrize("how", sorted(SAMPLING_ROUTES | POINT_ROUTES))
def test_forced_route_explanation_agrees_with_fitter(how: str) -> None:
    """Forcing `how=` must be reflected by explain_fit AND by the produced object."""
    model = Normal(Normal(0, 10), free)  # a prior-bearing model every route can fit
    try:
        fit = model.fit(DATA, how=how)
    except (ImportError, NotImplementedError) as exc:
        # A missing optional dependency is a legitimate fallback -- I6.3 asks that such
        # fallbacks be *named*, not that every route be installable everywhere.
        pytest.skip(f"route {how!r} unavailable in this environment: {exc}")

    ef = fit.explain_fit()
    assert ef["route"] == how, f"forced how={how!r} but explain_fit reports route={ef['route']!r}"
    _assert_result_matches_route(fit, ef["route"])


def test_explanation_is_not_stale_after_refit() -> None:
    """Refitting the same model object under a new route must update the explanation."""
    model = Normal(Normal(0, 10), free)

    first = model.fit(DATA, how="nuts")
    assert first.explain_fit()["route"] == "nuts"

    second = model.fit(DATA, how="laplace")
    assert second.explain_fit()["route"] == "laplace", (
        "explain_fit returned a stale route after refitting under a different `how=`"
    )
    # And the first fit's own explanation is unchanged -- each fit owns its explanation.
    assert first.explain_fit()["route"] == "nuts", "refitting the model mutated a previous fit's explanation"


def test_point_and_sampling_routes_are_observably_distinct() -> None:
    """Sanity anchor: em yields no posterior object; nuts yields one with diagnostics."""
    em_fit = Normal(free, free).fit(DATA)
    nuts_fit = Normal(Normal(0, 10), free).fit(DATA, how="nuts")
    assert getattr(em_fit, "result", None) is None
    assert getattr(nuts_fit, "result", None) is not None
    assert hasattr(nuts_fit.result, "rhat")
