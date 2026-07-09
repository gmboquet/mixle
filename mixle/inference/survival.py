"""Time-to-event (survival) estimators and hazard regression.

Survival analysis models the *time until an event* in the presence of right-censoring (subjects who
leave the study before the event -- their time is a lower bound, not a missing value). The toolkit here
covers the estimators and the regression layer:

  * :func:`kaplan_meier` / :func:`nelson_aalen` -- nonparametric survival and cumulative-hazard curves
    with Greenwood / Poisson variance and confidence bands.
  * :func:`cox_ph` -- the Cox proportional-hazards regression: how covariates multiply the hazard,
    estimated from the partial likelihood (Efron or Breslow tie handling), with stratification and
    time-varying covariates (counting-process ``start, stop`` input), Breslow baseline hazard, and the
    concordance index.
  * :func:`discrete_time_hazard` (+ :func:`to_person_period`) -- discrete-time hazard models fit as a
    binary GLM on the person-period array (logit or complementary-log-log), supporting offsets and
    fixed effects through the design matrix.
  * :func:`aalen_johansen` -- competing-risks cumulative incidence functions (cause-specific).
  * :func:`aalen_additive` -- Aalen's additive-hazards regression (cumulative covariate effects).
  * :func:`frailty_cox` -- shared gamma-frailty Cox for clustered survival (random effect per group),
    fit by EM.

Event indicators are 1 for an observed event and 0 for right-censoring (for competing risks, an integer
cause label with 0 = censored).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from mixle.inference.glm import glm

# --------------------------------------------------------------------------- nonparametric


def _event_table(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (distinct event times, #events at each, #at risk just before each)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    order = np.argsort(time)
    time, event = time[order], event[order]
    uniq = np.unique(time[event == 1])
    n = time.shape[0]
    d = np.array([np.sum((time == t) & (event == 1)) for t in uniq], dtype=float)
    at_risk = np.array([np.sum(time >= t) for t in uniq], dtype=float)
    return uniq, d, at_risk


def kaplan_meier(time: np.ndarray, event: np.ndarray | None = None, *, ci_level: float = 0.95) -> dict[str, np.ndarray]:
    """Kaplan--Meier product-limit estimate of the survival function ``S(t)``.

    Args:
        time: ``(n,)`` observed times (event or censoring).
        event: ``(n,)`` 1 = event, 0 = right-censored (defaults to all events).
        ci_level: confidence level for the log--log survival band.

    Returns:
        ``{'time', 'survival', 'se', 'ci_low', 'ci_high', 'at_risk', 'n_events', 'median'}``.
    """
    time = np.asarray(time, dtype=float)
    event = np.ones_like(time) if event is None else np.asarray(event, dtype=float)
    t, d, y = _event_table(time, event)
    surv = np.cumprod(1.0 - d / y)
    # Greenwood variance of S(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum = np.cumsum(np.where(y * (y - d) > 0, d / (y * (y - d)), 0.0))
    se = surv * np.sqrt(cum)
    z = stats.norm.ppf(0.5 + ci_level / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_surv = np.log(surv)
        v = np.where(log_surv != 0, np.sqrt(cum) / np.abs(log_surv), 0.0)
    ci_low = surv ** np.exp(z * v)
    ci_high = surv ** np.exp(-z * v)
    median = float(t[np.searchsorted(-surv, -0.5)]) if np.any(surv <= 0.5) else float("inf")
    return {
        "time": t,
        "survival": surv,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "at_risk": y,
        "n_events": d,
        "median": median,
    }


def nelson_aalen(time: np.ndarray, event: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Nelson--Aalen estimate of the cumulative hazard ``H(t) = sum d_i / Y_i``.

    Returns:
        ``{'time', 'cumhaz', 'se'}`` with the Poisson-type standard error of the cumulative hazard.
    """
    time = np.asarray(time, dtype=float)
    event = np.ones_like(time) if event is None else np.asarray(event, dtype=float)
    t, d, y = _event_table(time, event)
    cumhaz = np.cumsum(d / y)
    se = np.sqrt(np.cumsum(d / y**2))
    return {"time": t, "cumhaz": cumhaz, "se": se}


# --------------------------------------------------------------------------- Cox PH


@dataclass
class CoxResult:
    """Fitted Cox proportional-hazards model.

    Attributes:
        coef: ``(p,)`` log-hazard-ratio coefficients.
        se: ``(p,)`` standard errors (inverse observed information).
        cov: ``(p, p)`` covariance.
        loglik: maximised partial log-likelihood.
        baseline_time / baseline_cumhaz: Breslow baseline cumulative hazard.
        concordance: Harrell's C-index.
        n_iter: Newton iterations.
    """

    coef: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    loglik: float
    baseline_time: np.ndarray
    baseline_cumhaz: np.ndarray
    concordance: float
    n_iter: int

    def hazard_ratios(self) -> np.ndarray:
        """Return exponentiated Cox coefficients."""
        return np.exp(self.coef)

    def z_values(self) -> np.ndarray:
        """Return Wald z statistics for Cox coefficients."""
        return self.coef / self.se

    def p_values(self) -> np.ndarray:
        """Return two-sided normal-approximation p-values for Cox coefficients."""
        return 2.0 * stats.norm.sf(np.abs(self.z_values()))


def _concordance(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index: fraction of comparable pairs ordered correctly by risk score."""
    n = time.shape[0]
    conc = disc = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if time[j] > time[i]:  # j outlives i -> i should be the higher risk
                if risk[i] > risk[j]:
                    conc += 1
                elif risk[i] < risk[j]:
                    disc += 1
                else:
                    conc += 0.5
                    disc += 0.5
    total = conc + disc
    return float(conc / total) if total > 0 else 0.5


def cox_ph(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    start: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    ties: str = "efron",
    max_iter: int = 50,
    tol: float = 1e-9,
) -> CoxResult:
    """Cox proportional-hazards regression by Newton--Raphson on the partial likelihood.

    The hazard is ``h(t | x) = h0(t) exp(x' beta)``; only the *ordering* of event times enters, so the
    baseline ``h0`` is left unspecified (semi-parametric). Time-varying covariates are supported through
    the counting-process form: pass ``start`` so each row is an at-risk interval ``(start, stop]`` (a
    subject contributes several rows), and the risk set at an event time is every interval covering it.

    Args:
        x: ``(n, p)`` covariates (no intercept -- it is absorbed into the baseline).
        time: ``(n,)`` event/censoring times (the interval *stop* times).
        event: ``(n,)`` 1 = event, 0 = censored.
        start: optional ``(n,)`` interval start times for time-varying covariates / left truncation.
        strata: optional ``(n,)`` labels; each stratum gets its own baseline hazard (coefficients shared).
        ties: ``"efron"`` (default, more accurate) or ``"breslow"`` tie handling.
        max_iter, tol: Newton controls.

    Returns:
        A :class:`CoxResult`.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n, p = X.shape
    start = np.full(n, -np.inf) if start is None else np.asarray(start, dtype=float)
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)

    beta = np.zeros(p)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        for s in np.unique(strata):
            sm = strata == s
            Xs, ts, es, sts = X[sm], time[sm], event[sm], start[sm]
            for et in np.unique(ts[es == 1]):
                risk = (sts < et) & (ts >= et)
                tied = (ts == et) & (es == 1)
                if not np.any(risk):
                    continue
                Xr = Xs[risk]
                theta = np.exp(Xr @ beta)
                Xd = Xs[tied]
                d = Xd.shape[0]
                if ties == "breslow" or d == 1:
                    s0 = theta.sum()
                    s1 = theta @ Xr
                    s2 = (Xr * theta[:, None]).T @ Xr
                    grad += Xd.sum(axis=0) - d * s1 / s0
                    hess -= d * (s2 / s0 - np.outer(s1, s1) / s0**2)
                else:  # Efron
                    theta_d = np.exp(Xd @ beta)
                    s0_full = theta.sum()
                    s1_full = theta @ Xr
                    s2_full = (Xr * theta[:, None]).T @ Xr
                    sd0 = theta_d.sum()
                    sd1 = theta_d @ Xd
                    sd2 = (Xd * theta_d[:, None]).T @ Xd
                    grad += Xd.sum(axis=0)
                    for ell in range(d):
                        f = ell / d
                        a0 = s0_full - f * sd0
                        a1 = s1_full - f * sd1
                        a2 = s2_full - f * sd2
                        grad -= a1 / a0
                        hess -= a2 / a0 - np.outer(a1, a1) / a0**2
        step = np.linalg.solve(hess, grad)
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    cov = np.linalg.inv(-hess)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    # partial log-likelihood (Breslow) and Breslow baseline cumulative hazard
    loglik = 0.0
    base_t, base_h = [], []
    for s in np.unique(strata):
        sm = strata == s
        Xs, ts, es, sts = X[sm], time[sm], event[sm], start[sm]
        cum = 0.0
        for et in np.unique(ts[es == 1]):
            risk = (sts < et) & (ts >= et)
            tied = (ts == et) & (es == 1)
            theta = np.exp(Xs[risk] @ beta)
            s0 = theta.sum()
            loglik += float(np.sum(Xs[tied] @ beta)) - tied.sum() * np.log(s0)
            cum += tied.sum() / s0
            base_t.append(et)
            base_h.append(cum)
    order = np.argsort(base_t)
    base_t = np.asarray(base_t)[order]
    base_h = np.asarray(base_h)[order]
    risk_score = X @ beta
    conc = _concordance(risk_score, time, event)
    return CoxResult(beta, se, cov, float(loglik), base_t, base_h, conc, n_iter)


# --------------------------------------------------------------------------- discrete-time hazard


def to_person_period(
    time: np.ndarray, event: np.ndarray, covariates: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Expand right-censored durations into a person-period (long) array for discrete-time models.

    Each subject contributes one row per discrete period they were at risk; the binary outcome is 1 in
    the period the event occurred and 0 otherwise. Integer ``time`` is the number of periods observed.

    Returns:
        ``{'period', 'outcome', 'subject', 'covariates'}`` (``covariates`` repeated per period if given).
    """
    time = np.asarray(time, dtype=int)
    event = np.asarray(event, dtype=int)
    periods, outcomes, subjects, covs = [], [], [], []
    for i, (ti, ei) in enumerate(zip(time, event)):
        for k in range(1, ti + 1):
            periods.append(k)
            outcomes.append(1 if (ei == 1 and k == ti) else 0)
            subjects.append(i)
            if covariates is not None:
                covs.append(np.asarray(covariates)[i])
    out = {
        "period": np.asarray(periods),
        "outcome": np.asarray(outcomes, dtype=float),
        "subject": np.asarray(subjects),
    }
    if covariates is not None:
        out["covariates"] = np.asarray(covs, dtype=float)
    return out


def discrete_time_hazard(
    x: np.ndarray, outcome: np.ndarray, *, link: str = "cloglog", offset: np.ndarray | None = None
):
    """Discrete-time hazard model: a binary GLM on the person-period array.

    Fit on the long-format data from :func:`to_person_period` (the design ``x`` typically holds period
    indicators / a time trend plus covariates). ``cloglog`` gives the grouped-proportional-hazards
    (interval-censored Cox) interpretation; ``logit`` gives the proportional-odds hazard.

    Returns:
        a :class:`mixle.inference.glm.GLMResult` (binomial family with the chosen link).
    """
    return glm(x, outcome, family="binomial", link=link, offset=offset)


# --------------------------------------------------------------------------- competing risks


def aalen_johansen(time: np.ndarray, event: np.ndarray, *, causes: np.ndarray | None = None) -> dict:
    """Aalen--Johansen cumulative incidence functions for competing risks.

    With several mutually exclusive event types, the cause-specific CIF ``F_k(t)`` is the probability of
    failing from cause ``k`` by time ``t`` accounting for the competing causes (it is *not* ``1 - KM``
    on the cause, which overstates incidence).

    Args:
        time: ``(n,)`` event/censoring times.
        event: ``(n,)`` integer cause label, ``0`` = censored, ``1..K`` = causes.
        causes: optional explicit list of cause labels; inferred from ``event`` if None.

    Returns:
        ``{'time', 'cif': {cause: array}, 'overall_survival'}``.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if causes is None:
        causes = np.array(sorted(c for c in np.unique(event) if c != 0))
    uniq = np.unique(time[event != 0])
    n = time.shape[0]
    surv_prev = 1.0
    cif = {int(k): [] for k in causes}
    surv_curve = []
    km = 1.0
    for t in uniq:
        at_risk = float(np.sum(time >= t))
        d_total = float(np.sum((time == t) & (event != 0)))
        for k in causes:
            d_k = float(np.sum((time == t) & (event == k)))
            inc = surv_prev * d_k / at_risk if at_risk > 0 else 0.0
            prev = cif[int(k)][-1] if cif[int(k)] else 0.0
            cif[int(k)].append(prev + inc)
        km *= 1.0 - d_total / at_risk if at_risk > 0 else 1.0
        surv_prev = km
        surv_curve.append(km)
    return {
        "time": uniq,
        "cif": {k: np.asarray(v) for k, v in cif.items()},
        "overall_survival": np.asarray(surv_curve),
    }


# --------------------------------------------------------------------------- Aalen additive


def aalen_additive(x: np.ndarray, time: np.ndarray, event: np.ndarray, *, intercept: bool = True) -> dict:
    """Aalen's additive-hazards regression: cumulative regression functions ``B(t)``.

    Models ``h(t | x) = b0(t) + sum_j x_j b_j(t)`` with *time-varying* additive effects. At each event
    time the increment ``dB`` is the least-squares solution over the risk set; the cumulative ``B(t)``
    (returned) has interpretable slopes -- a rising ``B_j`` means covariate ``j`` adds hazard.

    Returns:
        ``{'time', 'cum_coef'}`` where ``cum_coef`` is ``(n_event_times, p[+1])`` cumulative coefficients
        (the first column is the baseline when ``intercept`` is True).
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    n = X.shape[0]
    if intercept:
        X = np.column_stack([np.ones(n), X])
    p = X.shape[1]
    event_times = np.unique(time[event == 1])
    cum = np.zeros(p)
    out_t, out_b = [], []
    for et in event_times:
        risk = time >= et
        Xr = X[risk]
        dN = ((time == et) & (event == 1)).astype(float)[risk]
        gram = Xr.T @ Xr
        try:
            incr = np.linalg.solve(gram, Xr.T @ dN)
        except np.linalg.LinAlgError:
            incr = np.linalg.lstsq(Xr, dN, rcond=None)[0]
        cum = cum + incr
        out_t.append(et)
        out_b.append(cum.copy())
    return {"time": np.asarray(out_t), "cum_coef": np.asarray(out_b)}


# --------------------------------------------------------------------------- shared frailty


@dataclass
class FrailtyCoxResult:
    """Shared gamma-frailty Cox result.

    Attributes:
        coef / se: fixed-effect log-hazard-ratios and standard errors.
        theta: estimated frailty variance (0 means no clustering signal).
        frailties: posterior mean random effect per group.
        groups: group labels aligned to ``frailties``.
        n_iter: EM iterations.
    """

    coef: np.ndarray
    se: np.ndarray
    theta: float
    frailties: np.ndarray
    groups: np.ndarray
    n_iter: int = field(default=0)


def frailty_cox(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    groups: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-5,
    ties: str = "breslow",
) -> FrailtyCoxResult:
    """Shared gamma-frailty Cox model for clustered survival, by EM.

    Subjects in the same group share an unobserved frailty ``w_g ~ Gamma(1/theta, 1/theta)`` (mean 1,
    variance ``theta``) that multiplies the hazard, capturing within-group correlation. The E-step takes
    the posterior-mean frailties; the M-step refits Cox with ``log w_g`` as an offset and updates
    ``theta``. ``theta -> 0`` indicates no detectable clustering.

    Returns:
        A :class:`FrailtyCoxResult`.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    theta = 0.5
    log_w = np.zeros(X.shape[0])
    res = None
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        res = _cox_offset(X, time, event, log_w, ties=ties)
        risk_score = np.exp(X @ res)
        base = _breslow_cumhaz(X, time, event, res)
        H = base[np.searchsorted(np.unique(time[event == 1]), time, side="right").clip(0, len(base) - 1)]
        w_post = np.empty(len(uniq))
        theta_terms = []
        for gi, g in enumerate(uniq):
            gm = groups == g
            d_g = float(np.sum(event[gm]))
            expected = float(np.sum(H[gm] * risk_score[gm]))
            shape = 1.0 / theta + d_g
            rate = 1.0 / theta + expected
            w_post[gi] = shape / rate
            theta_terms.append((d_g, expected))
        log_w = np.array([np.log(w_post[np.where(uniq == g)[0][0]]) for g in groups])
        # method-of-moments update for theta from posterior frailty variance
        new_theta = max(float(np.var(w_post)), 1e-4)
        if abs(new_theta - theta) < tol:
            theta = new_theta
            break
        theta = new_theta
    cov = _cox_cov(X, time, event, log_w, res, ties=ties)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return FrailtyCoxResult(res, se, float(theta), w_post, uniq, n_iter)


def _cox_offset(X, time, event, offset, *, ties="breslow", max_iter=50, tol=1e-9):
    """Cox coefficient estimate with a fixed per-observation offset (for the frailty M-step)."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        for et in np.unique(time[event == 1]):
            risk = time >= et
            tied = (time == et) & (event == 1)
            Xr = X[risk]
            theta_r = np.exp(Xr @ beta + offset[risk])
            s0 = theta_r.sum()
            s1 = theta_r @ Xr
            s2 = (Xr * theta_r[:, None]).T @ Xr
            d = tied.sum()
            grad += X[tied].sum(axis=0) - d * s1 / s0
            hess -= d * (s2 / s0 - np.outer(s1, s1) / s0**2)
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _cox_cov(X, time, event, offset, beta, *, ties="breslow"):
    hess = np.zeros((X.shape[1], X.shape[1]))
    for et in np.unique(time[event == 1]):
        risk = time >= et
        tied = (time == et) & (event == 1)
        Xr = X[risk]
        theta_r = np.exp(Xr @ beta + offset[risk])
        s0 = theta_r.sum()
        s1 = theta_r @ Xr
        s2 = (Xr * theta_r[:, None]).T @ Xr
        hess -= tied.sum() * (s2 / s0 - np.outer(s1, s1) / s0**2)
    return np.linalg.inv(-hess)


def _breslow_cumhaz(X, time, event, beta):
    cum = 0.0
    out = []
    for et in np.unique(time[event == 1]):
        risk = time >= et
        tied = (time == et) & (event == 1)
        s0 = np.exp(X[risk] @ beta).sum()
        cum += tied.sum() / s0
        out.append(cum)
    return np.asarray(out)


__all__ = [
    "kaplan_meier",
    "nelson_aalen",
    "CoxResult",
    "cox_ph",
    "to_person_period",
    "discrete_time_hazard",
    "aalen_johansen",
    "aalen_additive",
    "FrailtyCoxResult",
    "frailty_cox",
]
