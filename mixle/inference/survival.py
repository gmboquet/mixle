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
from scipy import optimize, special, stats

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
        baseline_time / baseline_cumhaz: Breslow baseline cumulative hazard for
            an unstratified fit. These arrays are empty for a multi-stratum fit.
        baseline_by_stratum: mapping from each stratum label to its own
            ``time`` and ``cumhaz`` arrays.
        concordance: Harrell's C-index, evaluated only within strata. This is
            ``None`` for counting-process input unless subject identifiers are
            supplied.
        n_iter: Newton iterations.
        converged: whether Newton's method reached ``tol`` with a finite, well-identified fit.
            ``False`` means either it hit ``max_iter`` without reaching ``tol``, a step produced a
            non-finite iterate (kept the last finite one instead), or the final standard errors are
            enormous -- the classic symptom of (quasi-)complete separation, where the partial
            likelihood has no finite maximizer and Newton's steps can shrink below ``tol`` while the
            coefficients themselves keep drifting toward infinity.
    """

    coef: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    loglik: float
    baseline_time: np.ndarray
    baseline_cumhaz: np.ndarray
    concordance: float | None
    n_iter: int
    converged: bool
    baseline_by_stratum: dict[object, dict[str, np.ndarray]] = field(default_factory=dict)

    def hazard_ratios(self) -> np.ndarray:
        """Return exponentiated Cox coefficients."""
        return np.exp(self.coef)

    def z_values(self) -> np.ndarray:
        """Return Wald z statistics for Cox coefficients."""
        return self.coef / self.se

    def p_values(self) -> np.ndarray:
        """Return two-sided normal-approximation p-values for Cox coefficients."""
        return 2.0 * stats.norm.sf(np.abs(self.z_values()))


def _concordance(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    start: np.ndarray,
    strata: np.ndarray,
    subject: np.ndarray,
) -> float:
    """Harrell's C-index across comparable subjects within the same stratum."""
    conc = disc = 0.0
    unique_subjects = np.unique(subject)
    terminal_time: dict[object, float] = {}
    subject_stratum: dict[object, object] = {}
    for identifier in unique_subjects:
        rows = subject == identifier
        stratum_values = np.unique(strata[rows])
        if stratum_values.size != 1:
            raise ValueError("every subject must remain in one stratum")
        terminal_time[identifier] = float(np.max(time[rows]))
        subject_stratum[identifier] = stratum_values[0]
    for i in range(time.shape[0]):
        if event[i] != 1:
            continue
        event_subject = subject[i]
        for competitor in unique_subjects:
            if competitor == event_subject or subject_stratum[competitor] != strata[i]:
                continue
            if terminal_time[competitor] > time[i]:
                active = (
                    (subject == competitor)
                    & (strata == strata[i])
                    & (start < time[i])
                    & (time >= time[i])
                )
                active_rows = np.flatnonzero(active)
                if active_rows.size != 1:
                    raise ValueError("counting-process rows must define exactly one active interval per subject")
                competitor_risk = risk[active_rows[0]]
                if risk[i] > competitor_risk:
                    conc += 1
                elif risk[i] < competitor_risk:
                    disc += 1
                else:
                    conc += 0.5
                    disc += 0.5
    total = conc + disc
    return float(conc / total) if total > 0 else 0.5


def _cox_inputs(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    start: np.ndarray | None,
    strata: np.ndarray | None,
    subject: np.ndarray | None,
    ties: str,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    X = np.asarray(x, dtype=float)
    stop = np.asarray(time, dtype=float)
    observed = np.asarray(event, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("x must be a non-empty two-dimensional design matrix")
    n = X.shape[0]
    if stop.ndim != 1 or observed.ndim != 1 or stop.shape[0] != n or observed.shape[0] != n:
        raise ValueError("time and event must be one-dimensional arrays aligned with x")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(stop)) or not np.all(np.isfinite(observed)):
        raise ValueError("x, time, and event must contain only finite values")
    if np.any((observed != 0) & (observed != 1)):
        raise ValueError("event must contain only 0 and 1")
    begin = np.full(n, -np.inf) if start is None else np.asarray(start, dtype=float)
    if begin.ndim != 1 or begin.shape[0] != n or np.any(np.isnan(begin)) or np.any(begin >= stop):
        raise ValueError("start must be aligned with x and strictly less than time")
    strata_values = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    if strata_values.ndim != 1 or strata_values.shape[0] != n:
        raise ValueError("strata must be a one-dimensional array aligned with x")
    if any(value != value for value in strata_values.tolist()):
        raise ValueError("strata must not contain missing labels")
    identifiers = None if subject is None else np.asarray(subject)
    if identifiers is not None:
        if identifiers.ndim != 1 or identifiers.shape[0] != n:
            raise ValueError("subject must be a one-dimensional array aligned with x")
        if any(value != value for value in identifiers.tolist()):
            raise ValueError("subject must not contain missing labels")
    if ties not in {"breslow", "efron"}:
        raise ValueError("ties must be 'breslow' or 'efron'")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")
    return X, stop, observed, begin, strata_values, identifiers


def cox_ph(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    start: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    subject: np.ndarray | None = None,
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
        subject: optional ``(n,)`` subject identifiers. Required to compute
            concordance for counting-process input with multiple rows per subject.
        ties: ``"efron"`` (default, more accurate) or ``"breslow"`` tie handling.
        max_iter, tol: Newton controls.

    Returns:
        A :class:`CoxResult`.
    """
    counting_process = start is not None
    X, time, event, start, strata, subject = _cox_inputs(
        x, time, event, start, strata, subject, ties, max_iter, tol
    )
    n, p = X.shape

    beta = np.zeros(p)
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        grad = np.zeros(p)
        hess = np.zeros((p, p))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
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
        if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
            break  # divergence overflowed the risk-set accumulators (e.g. exp(x @ beta)): stop here
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break  # singular Hessian (e.g. complete separation): keep the last finite iterate
        beta_new = beta - step
        if not np.all(np.isfinite(beta_new)):
            break  # divergence: keep the last finite iterate rather than poisoning beta with NaN/inf
        delta = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if delta < tol:
            converged = True
            break

    if np.all(np.isfinite(hess)):
        # pinv, not inv: a (quasi-)singular Hessian under separation must not crash here
        cov = np.linalg.pinv(-hess)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        cov = np.full((p, p), np.nan)
        se = np.full(p, np.inf)
        converged = False
    if not np.all(np.isfinite(se)) or np.any(se > 1.0e6):
        # An enormous standard error is the classic symptom of (quasi-)complete separation: the
        # partial likelihood has no finite maximizer, so Newton's steps can shrink below `tol`
        # while beta itself keeps drifting -- a step-size "converged" verdict alone would be
        # indistinguishable from a genuine, well-identified fit (observed repro: coef=27.17,
        # se=67e6, every step still satisfying `tol`).
        converged = False

    # partial log-likelihood (under the requested ties handling) and Breslow baseline cumulative hazard
    loglik = 0.0
    baseline_by_stratum: dict[object, dict[str, np.ndarray]] = {}
    for s in np.unique(strata):
        sm = strata == s
        Xs, ts, es, sts = X[sm], time[sm], event[sm], start[sm]
        cum = 0.0
        stratum_time: list[float] = []
        stratum_hazard: list[float] = []
        for et in np.unique(ts[es == 1]):
            risk = (sts < et) & (ts >= et)
            tied = (ts == et) & (es == 1)
            theta = np.exp(Xs[risk] @ beta)
            s0 = theta.sum()
            d = int(tied.sum())
            loglik += float(np.sum(Xs[tied] @ beta))
            if ties == "breslow" or d == 1:
                loglik -= d * np.log(s0)
            else:  # Efron: the denominator sheds a growing fraction of the tied-set mass
                sd0 = np.exp(Xs[tied] @ beta).sum()
                loglik -= float(np.sum(np.log(s0 - np.arange(d) / d * sd0)))
            cum += d / s0
            stratum_time.append(float(et))
            stratum_hazard.append(float(cum))
        key = s.item() if isinstance(s, np.generic) else s
        baseline_by_stratum[key] = {
            "time": np.asarray(stratum_time, dtype=float),
            "cumhaz": np.asarray(stratum_hazard, dtype=float),
        }
    if len(baseline_by_stratum) == 1:
        sole_baseline = next(iter(baseline_by_stratum.values()))
        base_t = sole_baseline["time"].copy()
        base_h = sole_baseline["cumhaz"].copy()
    else:
        base_t = np.array([], dtype=float)
        base_h = np.array([], dtype=float)
    risk_score = X @ beta
    if counting_process and subject is None:
        conc = None
    else:
        identifiers = np.arange(n) if subject is None else subject
        conc = _concordance(
            risk_score,
            time,
            event,
            start=start,
            strata=strata,
            subject=identifiers,
        )
    return CoxResult(
        coef=beta,
        se=se,
        cov=cov,
        loglik=float(loglik),
        baseline_time=base_t,
        baseline_cumhaz=base_h,
        concordance=conc,
        n_iter=n_iter,
        converged=converged,
        baseline_by_stratum=baseline_by_stratum,
    )


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
        frailty_variance: posterior variance of each group random effect.
        frailty_log_mean: posterior expectation of ``log(w_g)``.
        groups: group labels aligned to ``frailties``.
        n_iter: EM iterations.
        converged: whether the EM convergence criterion was met.
    """

    coef: np.ndarray
    se: np.ndarray
    theta: float
    frailties: np.ndarray
    groups: np.ndarray
    frailty_variance: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    frailty_log_mean: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    n_iter: int = field(default=0)
    converged: bool = field(default=False)
    ties: str = field(default="breslow")


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
    variance ``theta``) that multiplies the hazard, capturing within-group correlation. The E-step
    retains ``E[w_g]``, ``Var(w_g)``, and ``E[log(w_g)]`` under the conjugate gamma posterior. The
    coefficient/baseline M-step uses ``E[w_g]`` in the integrated-hazard term, while the dispersion
    M-step maximises the expected gamma-prior log likelihood using both ``E[w_g]`` and
    ``E[log(w_g)]``. ``theta -> 0`` indicates no detectable clustering.

    Returns:
        A :class:`FrailtyCoxResult`.
    """
    X, time, event, _, _, _ = _cox_inputs(
        x,
        time,
        event,
        None,
        None,
        None,
        ties,
        max_iter,
        tol,
    )
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != X.shape[0]:
        raise ValueError("groups must be a one-dimensional array aligned with x")
    if any(value != value for value in groups.tolist()):
        raise ValueError("groups must not contain missing labels")
    uniq, group_index = np.unique(groups, return_inverse=True)
    theta = 0.5
    w_post = np.ones(uniq.size)
    var_post = np.full(uniq.size, theta)
    elog_post = np.full(uniq.size, special.digamma(1.0 / theta) - np.log(1.0 / theta))
    beta = np.zeros(X.shape[1])
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        previous_beta = beta.copy()
        previous_theta = theta
        previous_w = w_post.copy()
        log_mean_w = np.log(w_post[group_index])
        beta = _cox_offset(X, time, event, log_mean_w, ties=ties)
        baseline = _breslow_cumhaz(X, time, event, beta, log_mean_w, ties=ties)
        w_post, var_post, elog_post = _frailty_posterior(
            X,
            time,
            event,
            group_index,
            uniq.size,
            beta,
            baseline,
            theta,
        )
        theta = _gamma_frailty_variance_mstep(w_post, elog_post)
        change = max(
            float(np.max(np.abs(beta - previous_beta))),
            abs(theta - previous_theta) / max(1.0, previous_theta),
            float(np.max(np.abs(w_post - previous_w))),
        )
        if change < tol:
            converged = True
            break
    log_mean_w = np.log(w_post[group_index])
    cov = _cox_cov(X, time, event, log_mean_w, beta, ties=ties)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if not np.all(np.isfinite(se)):
        raise RuntimeError("frailty Cox fit produced non-finite standard errors")
    return FrailtyCoxResult(
        coef=beta,
        se=se,
        theta=float(theta),
        frailties=w_post,
        groups=uniq,
        frailty_variance=var_post,
        frailty_log_mean=elog_post,
        n_iter=n_iter,
        converged=converged,
        ties=ties,
    )


def _cox_offset(X, time, event, offset, *, ties="breslow", max_iter=50, tol=1e-9):
    """Cox coefficient estimate with a fixed per-observation offset (for the frailty M-step)."""
    _, p = X.shape
    beta = np.zeros(p)
    converged = False
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
            d = int(tied.sum())
            Xd = X[tied]
            grad += Xd.sum(axis=0)
            if ties == "breslow" or d == 1:
                grad -= d * s1 / s0
                hess -= d * (s2 / s0 - np.outer(s1, s1) / s0**2)
            else:
                theta_d = np.exp(Xd @ beta + offset[tied])
                sd0 = theta_d.sum()
                sd1 = theta_d @ Xd
                sd2 = (Xd * theta_d[:, None]).T @ Xd
                for ell in range(d):
                    fraction = ell / d
                    a0 = s0 - fraction * sd0
                    a1 = s1 - fraction * sd1
                    a2 = s2 - fraction * sd2
                    grad -= a1 / a0
                    hess -= a2 / a0 - np.outer(a1, a1) / a0**2
        if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
            raise RuntimeError("frailty Cox coefficient M-step produced non-finite derivatives")
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("frailty Cox coefficient M-step has singular information") from exc
        beta = beta - step
        if not np.all(np.isfinite(beta)):
            raise RuntimeError("frailty Cox coefficient M-step diverged")
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"frailty Cox coefficient M-step failed to converge in {max_iter} iterations")
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
        d = int(tied.sum())
        if ties == "breslow" or d == 1:
            hess -= d * (s2 / s0 - np.outer(s1, s1) / s0**2)
        else:
            Xd = X[tied]
            theta_d = np.exp(Xd @ beta + offset[tied])
            sd0 = theta_d.sum()
            sd1 = theta_d @ Xd
            sd2 = (Xd * theta_d[:, None]).T @ Xd
            for ell in range(d):
                fraction = ell / d
                a0 = s0 - fraction * sd0
                a1 = s1 - fraction * sd1
                a2 = s2 - fraction * sd2
                hess -= a2 / a0 - np.outer(a1, a1) / a0**2
    information = -hess
    if not np.all(np.isfinite(information)) or np.linalg.matrix_rank(information) < information.shape[0]:
        raise RuntimeError("frailty Cox covariance is not identified")
    return np.linalg.inv(information)


def _breslow_cumhaz(X, time, event, beta, offset, *, ties="breslow"):
    """Breslow baseline cumulative hazard under a fixed per-observation ``offset`` (frailty log w_g).

    ``_cox_offset``/``_cox_cov`` both include ``offset`` in the risk-set sum ``s0`` -- this must
    match, or the baseline hazard (and the E-step expected-event-count it feeds in :func:`frailty_cox`)
    is computed as if every group's frailty were exactly 1, silently wrong whenever the fitted
    frailties actually differ from 1 (the entire point of fitting a frailty model)."""
    cum = 0.0
    out = []
    for et in np.unique(time[event == 1]):
        risk = time >= et
        tied = (time == et) & (event == 1)
        s0 = np.exp(X[risk] @ beta + offset[risk]).sum()
        d = int(tied.sum())
        if ties == "breslow" or d == 1:
            increment = d / s0
        else:
            tied_weight = np.exp(X[tied] @ beta + offset[tied]).sum()
            increment = float(np.sum(1.0 / (s0 - np.arange(d) / d * tied_weight)))
        cum += increment
        out.append(cum)
    return np.asarray(out)


def _frailty_posterior(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
    beta: np.ndarray,
    baseline: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_times = np.unique(time[event == 1])
    cumulative_baseline = _cumhaz_at(event_times, baseline, time)
    linear_risk = np.exp(X @ beta)
    exposure = np.bincount(group_index, weights=cumulative_baseline * linear_risk, minlength=n_groups)
    event_count = np.bincount(group_index, weights=event, minlength=n_groups)
    working_theta = max(float(theta), 1.0e-10)
    prior_shape = 1.0 / working_theta
    shape = prior_shape + event_count
    rate = prior_shape + exposure
    mean = shape / rate
    variance = shape / rate**2
    expected_log = special.digamma(shape) - np.log(rate)
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(variance))
        or not np.all(np.isfinite(expected_log))
        or np.any(mean <= 0)
        or np.any(variance < 0)
    ):
        raise RuntimeError("frailty E-step produced invalid posterior moments")
    return mean, variance, expected_log


def _gamma_frailty_variance_mstep(mean: np.ndarray, expected_log: np.ndarray) -> float:
    """Maximise the expected Gamma(a, a) prior log likelihood, returning theta=1/a."""
    group_count = mean.size
    sufficient_term = float(np.sum(expected_log - mean))

    def score(shape: float) -> float:
        return group_count * (np.log(shape) + 1.0 - special.digamma(shape)) + sufficient_term

    lower, upper = 1.0e-8, 1.0e8
    lower_score = score(lower)
    upper_score = score(upper)
    if not np.isfinite(lower_score) or not np.isfinite(upper_score):
        raise RuntimeError("frailty dispersion M-step produced non-finite derivatives")
    if upper_score >= 0:
        return 0.0
    if lower_score <= 0:
        return 1.0 / lower
    shape = optimize.brentq(score, lower, upper, xtol=1.0e-10, rtol=1.0e-12)
    return float(1.0 / shape)


def _cumhaz_at(event_times, base, t):
    """Evaluate the cumulative-hazard step function at arbitrary times ``t``.

    ``base[j]`` is the cumulative hazard AT the sorted ``event_times[j]`` (its own increment
    included); the step function is 0 before the first event and right-continuous, so a time
    between events takes the value at the most recent event time ``<= t``.
    """
    step = np.concatenate([[0.0], np.asarray(base, dtype=float)])
    return step[np.searchsorted(np.asarray(event_times, dtype=float), t, side="right")]


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
