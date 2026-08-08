# mixle.inference statistical-validity audit — 2026-08-08

Scope: the 17 claim-bearing `mixle/inference` modules outside the conformal cores, swept after
pass sixteen of the external review closed (the external passes never entered this surface).
Method: three parallel read-only audits over five modules each, restricted to five defect
classes — (a) math does not deliver the stated property, (b) plug-in estimate treated as a
known constant, (c) uncorrected selection/multiplicity, (d) approximate described as
exact/finite-sample, (e) guarantee stated without its assumptions — plus independent
line-reading and live reproduction of the sharpest items in the main session.

RULES OF THIS LEDGER (the repo's standing worklist discipline):
- Status `CONFIRMED` means reproduced in the main session against the shipped tree, or
  verified by direct line-reading where the defect is textual. Status `demonstrated` means the
  auditing pass executed a simulation with the quoted numbers; each such finding must be
  independently reproduced before its fix lands. Status `reported` means argued but not run.
- One fix per finding (or per shared root cause), each with adverse tests in the same commit
  (D-0147). Never quote a fixed/outstanding ratio from this file without per-finding
  reproduction. Guard-overreach applies: measure before adding any new refusal.
- Fixed findings move to the Fixed section with the commit hash; this file is append/annotate,
  entries are never deleted.

## Fixed

| ID | Site | Fix |
|----|------|-----|
| UQ-1 | `inference/uq.py` payload `coverage_cal` | In-sample-by-construction hit rate (computed on the rows that set qhat) surfaced in `report()` unlabeled; renamed `coverage_cal_in_sample_mechanical` with the constraint stated at the definition. |
| UQ-2 | `inference/uq.py` module docstring | "finite-sample coverage" claimed from a surface the claims gate never scanned; scope triad added and `mixle/inference` added to `COVERAGE_SURFACES` (negated-mention suppression added for `select.py`'s disclaimers, pinned by controls; manifest at 34 sites). |
| GATE-5 | `calibration_gate.py` `CalibrationVerdict.score` | Docstring said "1 == at/below the calibrated-null level"; the formula gives 0.5 at the threshold. Docstring now states the actual mapping and warns against fixed-cutoff gating. |
| FC-1 | `forecast.py` → `price_forecast.paths` → `analysis/valuation.py` NPV | `paths` had no true cross-step dependence (per-step marginal multinomial) plus a state-sorted ordering artifact (0.691 measured, −0.03 shuffled; the honest joint value at that start is ~0.70 — the artifact sat near it by coincidence, diverging elsewhere). Now forward-simulates genuine chain trajectories; per-step marginals unchanged in distribution; adverse test pins the empirical lag-1 correlation to the EXACT analytic joint value and rejects monotone (state-sorted) rows. |
| CAL-9 | `calibration.py` `pit_calibration_error` | Summary line said "mean absolute deviation"; the statistic is the SUM (2× total variation, range [0, 2(1−1/bins)]). Docstring corrected; interacts with `calibration_gate`'s `low_power_threshold=1.0`. |

## Confirmed, fix pending (reproduced in the main session)

| ID | Site | Class | Finding |
|----|------|-------|---------|
| NP-1 | `nonparametric.py:414-417` Wilcoxon | a,d | Normal approximation with no exact branch: at n=5 the extreme outcome reports p=0.0431 (<0.05) while the exact p is 0.0625 → guaranteed 6.25% type-I at nominal 5% (same at n=6). Module claims "matching the conventions of SciPy / R"; both use exact nulls here. |
| NP-3 | `nonparametric.py:209-211` ks_2samp | a | One-sample `kstwo` law evaluated at rounded n₁n₂/(n₁+n₂): complete separation at n=3 returns p=0.0 where the exact permutation p is 0.1. Comment calls it "finite-n". |
| RS-1 | `resampling.py:231-245` m-out-of-n | a,e | Percentile orientation applied to Politis–Romano-rescaled replicates is the REFLECTION of the correct (basic) interval: sample-max example has ci_high == sample max < θ, coverage exactly 0. √n rate also hardwired (comment admits it; docstring does not). |
| RS-9 | `resampling.py:209-231` m == n | a | `m == n` permutes the full sample without replacement: every replicate equals the estimate; returns a zero-width "95% CI" (measured width 5.6e-17) silently. |

## Demonstrated by the auditing passes (numbers quoted; reproduce before fixing)

| ID | Site | Class | One-line finding |
|----|------|-------|------------------|
| E-1 | `event_study.py:119-122` | a | CRITICAL: Haldane-corrected Poisson log-rate effect is inconsistent at small counts; with unequal exposures a TRUE NULL reaches p=3.2e-35 at 1000 subjects (DL pooling shrinks the CI around the wrong value); equal-exposure true effect log 2 estimated at 0.274 with CI (0.204, 0.345). |
| S-2 | `survival.py:713-810` frailty_cox | b | SEs are complete-data information with frailties/theta plugged in: se/SD(β̂) = 0.79–0.88, Wald-95 coverage 0.87–0.91 when θ=1 (control θ=0 is clean). Louis-identity correction needed. |
| S-1 | `survival.py:116-121` KM | d,a | Pointwise log–log intervals labeled a "confidence band": simultaneous coverage 0.60/0.51/0.43 at n=50/100/200 vs nominal 0.95. Rename or implement Hall–Wellner. |
| R-1 | `risk.py:96-109` CVaR GPD | a | The GPD "refinement" fires exactly when the tail is sparse and is strictly worse than the raw tail mean it replaces: sd 17.2 vs 2.47, max 293 vs truth 8.56 at n=100. |
| GATE-1 | `calibration_gate.py:124-134` | b,d | Null threshold is a 0.99 sample quantile of 500 fixed-seed draws used as an exact critical value; realized level ≠ 1% deterministically. Valid MC form is the (1+#{T≥t})/(1+B) p-value. |
| GATE-2 | `calibration_gate.py:235-239` | e | i.i.d. null Monte-Carlo vs the documented construction (shared posterior draws across held-out points) → inflated false-failure rate on genuinely calibrated posteriors. |
| GATE-3 | `calibration_gate.py:249-263` | a,b | Coverage direction label inherits finite-m bias: perfectly calibrated posterior at m=8 reads "overconfident" (coverage_at_ref ≈ 0.78 vs 0.90). |
| GATE-4 | `calibration_gate.py:160-172` | a,e | `power_sufficient` computes no power: SD-ratio-0.8 alternative rejected ≲10% of the time while the flag says power is adequate; non-rejections then presented as affirmative. |
| GATE-6 | `calibration_gate.py:400-417` SBC | e | Rank uniformity needs (near-)independent posterior draws; unthinned MCMC from a CORRECT inference fails the gate. Talts et al. prescribe thinning; docstring omits it. |
| GATE-7 | `calibration_gate.py:506-530` | a | Missing/malformed payload labeled `"failed"` (defined by the module as evidence of miscalibration) instead of `"indeterminate"`. |
| CAL-1 | `calibration.py` coverage_curve | a,b | Ensemble-quantile plug-in coverage biased low at small m (0.667 at nominal 0.95, m=5); feeds GATE-3. |
| CAL-2 | `calibration.py` ECE CI | a | Plug-in ECE is biased up by binomial noise (~√(bins/n)); its bootstrap CI covers the true 0 with probability ~0 under perfect calibration. |
| CAL-3 | `calibration.py` MCE | a | Max over bins of plug-in deviations grows with `bins` under the null (0.05→0.6 as bins 10→500); argmax lands on the sparsest bin. |
| CAL-4 | `calibration.py:295-297` | a,e | Top-label reduction stated as THE multiclass check; a forecaster putting 0.4 on a never-occurring class passes with ECE 0. |
| CAL-5 | `calibration.py:538-547` Platt | a | No target smoothing: separable data → a→∞, degenerate 0/1 outputs under a "robust on little data" claim. |
| CAL-6 | `calibration.py:483-531` | e | Isotonic calibrator's guarantee stated unconditionally; in-sample diagnostics pass by construction (interpolates outcomes). |
| CAL-7/8 | `calibration.py:124-172` | c,b | Pointwise-per-bin bootstrap described at module level as a "band" (~30–45% familywise exclusion at nominal 5%); empty-bin resamples silently dropped from quantiles. |
| SCORING-1 | `scoring.py` ensemble CRPS/energy | a | `fair=False` default is improper for finite ensembles (optimal dispersion factor c*=0.686 at m=5). |
| SCORING-2/3 | `scoring.py` brier_decomposition | a | Plug-in decomposition terms carry bins/n bias; a result key named `'brier'` is not the Brier score. |
| NP-2 | `nonparametric.py:590-595` runs_test | a | No continuity correction (module claims "the usual ... corrections"): exact level 20/252 = 7.94% at n₁=n₂=5. |
| NP-4 | `nonparametric.py:89-105` MWU | e | Docstring frames the null as stochastic ordering; the variance is the F=G exchangeability variance → inflated level under unequal shapes (Brunner–Munzel sibling states this correctly). |
| NP-5 | `nonparametric.py:319-336` Dunn | a,e | Global-null pooled variance reused for every pair under partial nulls: predicted ~24% pairwise rejection at nominal 5% with a concentrated third group. |
| NP-6 | `nonparametric.py:552-561` Page | a | No tie correction (hardcoded no-tie variance) while the module claims the usual corrections; conservative, statistic wrong under ties. |
| NP-7 | `nonparametric.py:151-176` BM | a | Satterthwaite-t recommended "for small samples" where it is documented liberal (<~10/group); complete separation raises instead of reporting. |
| NP-8 | `nonparametric.py:283-293` Mood | a | Uncorrected Pearson χ² with both margins fixed; min group size 1 permitted; "common median" interpretation stated unconditionally. |
| NP-10/11 | `nonparametric.py:481-531` JT | c,a | Pre-specified-ordering requirement unstated (sibling Page states it); documented MWU cross-check numbers only match with continuity OFF, defaults differ. |
| RS-2 | `resampling.py:279-289` BCa | e | "Second-order accurate for i.i.d. data" without the smoothness condition; jackknife acceleration inconsistent for quantiles (jack vector takes ~2 distinct values for the median). |
| RS-3 | `resampling.py:276-278` z₀ | b | Strict `<` + clip: lattice statistics with atoms at the estimate get a spurious bias correction; mid-p treatment needed. |
| RS-4 | `resampling.py:149-155,321-323` MBB | a | Non-circular blocks under-weight endpoints (uncorrected O(ℓ/n) centring bias); docstring's block-length rule materially shorter than coverage-optimal. |
| RS-5 | `resampling.py:140-142,215-216` cluster | a | Validity is asymptotic in the NUMBER OF CLUSTERS (G≥2 permitted; coverage ~0.5–0.7 at G=5); unequal cluster sizes inflate replicate variance via variable resample size. |
| RS-6 | `resampling.py:326-387` wild | e | Heteroscedasticity-robustness claimed on raw residuals with no leverage adjustment (Davidson–Flachaire e/(1−h) needed); `estimate` uses fitted+residuals which diverges for nonlinear statistics. |
| RS-7 | `resampling.py:540-551` sign-flip | d,e | `exact=True` requires symmetry of the differences about the null — stated nowhere (two-sample sibling states exchangeability). |
| RS-8 | `resampling.py:532` default stat | e | Raw mean-difference permutation invites equal-means reading; fails level under unequal variances/sizes (studentization needed); exchangeability statement is honest, usage surface is not. |
| U-1 | `uncertainty.py:9,26-29,233-234` | d | "Exact given the members; the only approximation is M" is false for variational/MCMC members (mode-seeking spread, ESS, chain resampling). |
| U-2 | `uncertainty.py:148-166` BALD | a | Plug-in MI is capped at log M (M=2 permitted → max 0.693 nats vs true 2.079 measured); no bias/cap disclosure. |
| U-3 | `uncertainty.py:188` | a,b | `ddof=0` epistemic variance biased by (M−1)/M — 50% at the permitted M=2. |
| U-4 | `uncertainty.py:373-384` | a,d | Plug-in semantic entropy biased −(K−1)/(2n): −0.53 nats at n=10 on true 2.30, the direction that HIDES hallucination. |
| U-5/6/7/8/9 | `uncertainty.py` various | d,e,a | logsumexp marginal renormalized over observed items called "exact"; asymmetric `equivalent` makes clusters order-dependent; density softmax grid-dependence; clamp breaks the asserted total=aleatoric+epistemic identity at float precision; member_vars=None stores epistemic-only in `total`. |
| G-1 | `glm.py:301-303,451,462` | b | Estimated dispersion + normal (not t_{n−rank}) reference: measured p 0.036 vs correct 0.065 at n=12. |
| G-2 | `glm.py:243-295` | a | "Maximised log-likelihood" evaluated at the REML-style phi (not the MLE), AIC/BIC omit the dispersion parameter and use p not rank; distorts model selection. |
| G-3 | `glm.py:445,465` | a,e | Wald p-values reported for non-estimable coefficients under rank deficiency (pinv min-norm split halves coef and SE; collinearity invisible). |
| G-4 | `glm.py:340,358-360` NB | b,e | Placeholder theta=1.0: model-based SEs measured 1.87× the true sampling SD with no warning; robust=True recovers it. |
| G-5/6/7 | `glm.py` | a,e | elastic_net(l1_ratio=0) ≡ ridge(n·alpha) under an "is ridge" sentence; module promises SEs that two result types lack; HC0 sandwich stated without finite-sample/independence caveats. |
| A-1/2/3 | `_advi.py` | e,d,a | "Convergence postcondition" is a finiteness check on fixed-step/fixed-count Adam; reported objective is K-dependent and (α>1) not a bound; exact `alpha == 1.0` float test leaves a catastrophic-cancellation cliff just below 1 (no tolerance band). |
| P-2 | `price_forecast.py:46` | e,c | `coverage_assumptions` names held-out exchangeability — the assumption the construction does NOT establish (model refit on the calibration window is the binding one, and it is absent). |
| P-3/4 | `price_forecast.py:127-138` | c | Per-lead marginal intervals presented jointly with no simultaneity statement; default sizes put the conformal index at the max of ~18 overlapping residuals. |
| D-1 | `decision.py:357-363` | c | Winner's reported expected_loss/CVaR are post-argmin over K noisy estimates: bias −2.5 MC SEs at K=100, invariant in n; no SE surfaced. |
| S-3 | `survival.py:657,870-888` | a,e | "theta → 0 indicates no detectable clustering": no SE/LRT, boundary null, and θ̂ is never 0 under a true θ=0 (mean 0.046 measured). |
| S-4 | `survival.py:290-459` | e | Counting-process/subject interface invites clustered/recurrent data; only naive information variance exists (se/SD = 0.68 measured on clustered design); no independence statement, no PH diagnostic anywhere in the module. |
| S-5 | `survival.py:595-606` Aalen | a | Increments from near-singular risk sets accumulate; tail of B(t) is numerical noise (max grows 1.4 → 19 as risk set → p); silent min-norm substitution on exact singularity. |
| S-6 | `survival.py:476-488` | a | `time=0` subject silently dropped from person-period expansion, event and all. |
| E-2 | `event_study.py:143-146,250-261` | b,d | DL interval treats τ̂² and plug-in v_i as known + normal quantile: coverage 0.876 at k=3 (HKSJ is the standard remedy) under a "calibrated uncertainty" claim. |
| E-3/4/5 | `event_study.py` | b,e,a | k=1 returns τ²=0 and a pooled "population effect" p-value; tipping_drift hardcodes 1.96 whatever alpha built the result; alpha=1e-17 raises math-domain error. |
| S-1c | `select.py:135-139` | b | N=2 confidence flag is data-independent (margin/spread cancels): fires never at α=0.05, always at α=0.20. Module's disclaimers otherwise exemplary. |

## Judged acceptable, no action (recorded so the judgment is auditable)

- `multiple_testing.py`: verified line-by-line clean (all five corrections standard, dependence
  assumptions stated, combiners declare independence, numerically careful Tippett).
- `model_comparison.py`: sign test at an exact constant null; elpd diff ± SE with "within ~2 SE
  is not decisive" — properly hedged.
- `planning.py` Guarantee ladder: EM capped at STATIONARY*, GLOBAL_UNIQUE only for
  strictly-concave closed forms with proof obligations, aggregate = min over blocks.
- `select.py` disclaimers; `decision.py` `_tail_mass_mean`; survival's Efron/Breslow
  derivatives, Greenwood algebra, Aalen–Johansen construction, discrete-time GLM factorization;
  event_study's identification gate and DL algebra; resampling's permutation exactness logic and
  Mammen moments; `risk.py`'s CVaR ≥ VaR invariant (0 violations over the scanned grid).
