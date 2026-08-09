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
| NP-1 | `nonparametric.py` Wilcoxon | Exact enumeration null at n <= 25 with no zeros/ties (the SciPy/R regime switch): the extreme n=5 outcome now reports exactly 0.0625 and no attainable p sits below it, so the structural 6.25%-at-nominal-5% level violation is gone; continuity correction scoped to the normal branch; adverse test enumerates all 32 sign patterns. |
| NP-3 | `nonparametric.py` ks_2samp | P-value delegated to scipy's two-sample method='auto' (exact small-sample, asymptotic large): separation at n=3 now reports the exact 0.1 instead of 0.0; large-sample values match scipy to 1e-9; module conventions sentence corrected (runs/Page carry no continuity correction). |
| RS-1 | `resampling.py` m-out-of-n | Interval now takes the BASIC (pivotal) orientation subsampling theory yields (method overridden in m-mode, documented): the U(0,θ) sample-max example goes from coverage exactly 0 to covering (conservatively, per the documented sqrt-n-rate assumption); symmetric statistics unchanged. |
| RS-9 | `resampling.py` m == n | Refused with an explanatory error: a without-replacement permutation makes every replicate equal the estimate, so the returned interval had width 0 (a silent 0%-coverage CI). |
| RR17-07 | `task/calibrated_generator.py`, `reason/language_bridge.py`, report demo | Certification now selects under the SERVED schedule (self.seed + prompt, exactly what candidate_set derives); calibrate(seed=) refuses mismatches; per-call seed overrides are refused on a certified generator (the demo's override removed); PosteriorDescriber honors a calibrate seed by REBUILDING its generator under it so one policy certifies and serves; receipt discloses seed_schedule and unique_prompt_count with the duplicate-independence assumption named. Negative test pins the reviewer's construction: on a repeated-prompt population, certify-low-then-serve-high is structurally impossible (served answers replay certification or the gate abstains). |
| GATE-1/2/3g/4/6/7 + RR17-08 | `calibration_gate.py` | Decision = exact column-swap randomization p-value (level-exact for shared-draw AND independent ensembles; quantized 1/(m+1)); SBC decided by the level-exact i.i.d. MC p-value with the Talts thinning precondition documented; power_sufficient requires MEASURED power >= 0.5 against the named 0.8-dispersion alternative at (k, m); direction label judged against the finite-m rank-MC coverage expectation; verifier labels unverifiable payloads indeterminate (still fail-closed); per-row swaps rejected by this commit's own adverse test (invariant only for independent rows). GATE-2/6 documented; GATE-3's calibration.py half (coverage_curve bias, CAL-1) remains open. |
| CAL-9 | `calibration.py` `pit_calibration_error` | Summary line said "mean absolute deviation"; the statistic is the SUM (2× total variation, range [0, 2(1−1/bins)]). Docstring corrected; interacts with `calibration_gate`'s `low_power_threshold=1.0`. |

## Confirmed, fix pending (reproduced in the main session)

(all confirmed findings fixed as of 2026-08-08; next reproductions come from the demonstrated list below)

## Demonstrated by the auditing passes (numbers quoted; reproduce before fixing)

| ID | Site | Class | One-line finding |
|----|------|-------|------------------|
| E-1 | `event_study.py:119-122` | a | CRITICAL — REPRODUCED IN-SESSION 2026-08-08 (unequal exposures, true null: p=4.2e-3/1.2e-7/6.2e-36 at n=50/200/1000; true log-2 effect estimated 0.238 with CI (0.167, 0.309)). Fix is DESIGN-SCALE, not a patch: log((k+0.5)/t) is algebraically identical to the empirical-logit form, so the bias is structural to per-subject Gaussianization at these counts — the repair needs a measured information floor on (k_pre, k_post) below which the per-subject effect is refused, plus an exact conditional pooled route (sum-Poisson → binomial on totals) for the sparse regime; place the floor from bias-vs-count curves before coding. Original finding: Haldane-corrected Poisson log-rate effect is inconsistent at small counts; with unequal exposures a TRUE NULL reaches p=3.2e-35 at 1000 subjects (DL pooling shrinks the CI around the wrong value); equal-exposure true effect log 2 estimated at 0.274 with CI (0.204, 0.345). |
| S-2 | `survival.py:713-810` frailty_cox | b | SEs are complete-data information with frailties/theta plugged in: se/SD(β̂) = 0.79–0.88, Wald-95 coverage 0.87–0.91 when θ=1 (control θ=0 is clean). Louis-identity correction needed. |
| S-1 | `survival.py:116-121` KM | d,a | Pointwise log–log intervals labeled a "confidence band": simultaneous coverage 0.60/0.51/0.43 at n=50/100/200 vs nominal 0.95. Rename or implement Hall–Wellner. |
| R-1 | `risk.py:96-109` CVaR GPD | a | The GPD "refinement" fires exactly when the tail is sparse and is strictly worse than the raw tail mean it replaces: sd 17.2 vs 2.47, max 293 vs truth 8.56 at n=100. |
| GATE-3 | `calibration_gate.py:249-263` | a,b | Coverage direction label inherits finite-m bias: perfectly calibrated posterior at m=8 reads "overconfident" (coverage_at_ref ≈ 0.78 vs 0.90). |
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

## Pass-17 external findings (2026-08-08, NO-GO; reports under /private/tmp/mixle-rereview17.5mIyWg/)

External review pass 17 confirmed seven open ledger items (RR17-06=P-2, RR17-08=GATE-1/2/4,
RR17-09=E-1 sparse-Poisson variant at rate .5 exposures 1:2 — p=1.37e-12 true null at n=1000,
RR17-12=RS-7, RR17-14=U-4, RR17-16=NP-2, RR17-17=NP-4) and added the following NEW findings:

| ID | Sev | Site | One-line finding |
|----|-----|------|------------------|
| RR17-07 | High | `task/calibrated_generator.py` | FIXED 2026-08-08 (see Fixed section). Original: calibrate() seeds selection by (row_index, prompt); default serving seeds by prompt only — the certificate covers a policy that never serves. Deterministic repro: certified upper risk 0.0366 from 1/150 calibration errors; 1000/1000 served answers wrong on a repeated-prompt population. Fix: one stochastic policy for both + repeated-prompt negative test. |
| RR17-01 | High | `examples/task_llm_active_example.py` | Call ledger prints 420 pre-serving teacher calls; the cascade section then buys 600 more calibration labels before serving (actual 1,020). "Fewest labels/most informative" headline survives the inconclusive verdict; the 4.14x label-economics citation is one EIG seed vs five unrelated random seeds. |
| RR17-02 | High | `examples/task_cascade_economics_example.py` | Harvested labels called free are DISCARDED (build_cascade re-queries the teacher on train+htexts); setup labels excluded from the cost estimand: measured 600 labels build 1, 1,312 after build 2; first-round all-in cost $6.563 vs the reported $0.563-vs-$3 framing. |
| RR17-03 | High | `examples/task_cascade_economics_example.py` | corpus() fixes 150 rows per class, so rows are NOT i.i.d.; the pooled McNemar exactness claim does not hold for the overall-mean estimand under the stratified design (pooled 29-14 p=.032 vs spam 14-6 p=.115, ham 15-8 p=.210). |
| RR17-04 | High | `examples/heterogeneous_correctness_example.py` | Acceptance oracle checks abs(abs(mu)-3)<.3 and max(prob)>.6 — wrong concentrated categories still print "recovered ... True". |
| RR17-05 | High | `examples/structured_hmm_example.py` | Emissions initialized within Uniform(-1,1) of truth with an error<1 oracle: a no-op optimizer passes; return-prev-estimate printed "recovered". |
| RR17-10 | High | `inference/forecast.py` | Docstring says MC only for emission quantiles, but the reported MEAN is computed from draws (seeds 1/2: 4.3911 vs 4.4256 at n=25); Forecast carries no draw count / MCSE / method flag. |
| RR17-11 | High | MCMC public summary | One-chain fits report rhat=None/ess=None/NaN split-Rhat yet the summary emits parameter numbers without MCSE (raw sampler had MCSE [0.144, 0.033]); route test checks attribute presence only. |
| RR17-13 | Med | `task/multilabel.py`, `task/structured_out.py` | Answered-slice fields accept impossible states on direct construction (evaluated=1, answered=1, correct=2 -> agreement 2.0, CI [NaN, NaN]); enforce 0 <= correct <= answered <= evaluated. |
| RR17-15 | Med | `inference/nonparametric.py` Wilcoxon | Exact one-sided output internally contradictory: all-positive n=5 'greater' gives p=.03125 and rank-biserial +1 but z=-2.0226 (the descriptive z still uses t=min(R+,R-)). |
| RR17-18 | Med | `examples/lookback_hmm_example.py` | "States distinguishable only through dependence" is false: nonstationary start makes lag-0 transient marginals differ (pairwise TV .07). |
| RR17-19 | Med | `examples/task_cascade_economics_example.py` | Synthetic/stand-in data disclosure lives only in the teacher docstring, not the top-level narrative/output. |
| RR17-20 | Low | `scripts/scan_statistical_claims.py` | Manifest --write records path/classes only; the promised "human audit" has no reviewer/date/attestation fields. |

Minimum-closure order from the report: (1) RR17-07 seed-policy unification + negative test;
(2) GATE cluster (valid MC p-value, dependence-aware null, measured power); (3) P-2/RR17-06
price-forecast leakage refusal-or-disclaimer; (4) E-1/RR17-09 sparse-Poisson refusal or exact
conditional route; (5) RR17-01/02/03 example accounting + stratified inference; (6)
RR17-04/05 example oracles that fail on wrong categories/no-op fits; (7) RR17-10/11/14 MC/MCMC
labeling with draws+MCSE and diagnostic refusal; (8) RS-7/RR17-12 sign-flip assumptions.

## Pass-18 status (2026-08-09, NO-GO, 15 High; reports under /private/tmp/mixle-rereview18.4UVgLt/)

Pass 18 CLOSED RR17-07 (seed policy) and found the pass-17 gate fix itself defective. Fixed this
session: RR17-08 round 2 (a17f95de — ONE invariant jitter across all column relabelings, my level
sim 0.1303 at the reviewer's exact k=10/m=4/alpha=.21 regime vs their 0.1298 control; power =
EXECUTED decision under both regimes with adaptive budget, promotion gated on the CALLER-DECLARED
ensemble_dependence regime, fail-safe shared-draws default; suite size-regression moved to an
attainable alpha/m); RR18-04 (bound counts the declared (prompt, outcome) unit:
outcomes='per-prompt' collapses duplicates/refuses disagreeing verdicts, 'per-row' is the
caller's recorded assertion — language_bridge declares per-row). STILL OPEN from pass 18:
RR18-01 propose() outer-holdout leak (candidate families selected on all rows before the split);
RR18-02 explain_fit says MAP on the neural route; RR18-03 empty caveats on
ensemble/sample/VMP routes; plus the carried RR17-01/02/03/04/05/06/09/10/11/12/14 and
mediums RR17-13/15/16/17/18/19/20 (RR17-14 and RR17-17 severity RAISED to High).

### Pass-18 fix wave 2 (2026-08-09, this session)

Closed with adverse tests, one commit each: RR18-01 (propose() splits before candidate
generation; spy test pins the exact train split), RR17-13 (answered-slice count invariant
refuses impossible states at construction), RR17-15 (Wilcoxon descriptive z signed for the
alternative + method field), RR17-16 (runs test exact conditional null at n <= 60; exhaustive
level 4/252 vs the normal approximation's 20/252), RR17-17 (MWU exchangeability null named on
docstring and result), RR17-12 (PermutationResult carries null_hypothesis per mode; 'exact'
scoped to the stated sharp null), RR17-10 (Forecast carries method/n_draws/mean_mcse; docstring
no longer claims the mean is exact). Still open: RR17-06 price leakage, RR17-09 sparse Poisson,
RR17-01/02/03/04/05 examples, RR17-11 MCMC diagnostics/MCSE, RR17-14 semantic entropy receipt,
RR18-02/03 route explanations, RR17-18/19/20.

### Pass-18 fix wave 3 (2026-08-09): the queue is EMPTY

Every remaining pass-17/18 finding closed, one commit each with adverse verification:
RR17-06 (forecast_price requires the model_fit_length declaration; held-out receipt states its
premise, fitting through the window flips to an explicit in-sample-optimistic disclosure);
RR17-09 (zero-count refusal + arm-mean floor of 4.0 measured from bias curves +
poisson_pooled_rate_ratio exact conditional route: level 0.041/0.05 over 2000 replicates of the
reviewer's null, recovers a true ratio 2); RR17-14 (semantic_entropy_receipt with n/K/bias/
Miller-Madow value; confident() decides on the corrected estimate, bias -0.657 -> -0.365);
RR17-11 (posterior_summary publishes mcse with every number; 'ok' requires finite diagnostics
AND multi-chain R-hat; single chain reads mixing-unassessable; NaN reads unusable; the hdi()
1-D bug that silently failed every multi-chain fit is fixed); RR17-01 (active example all-in
ledger: 1,020 pre-serving calls decomposed, setup-vs-serving economics, headline reduced);
RR17-02/03/19 (cascade: counting teacher, all-in $6.563 vs $3.00 with the COSTS-MORE admission,
harvest genuinely reused — 690 total calls vs 1,312 — per-stratum exact McNemar with the
stratified estimand stated, synthetic disclosure at top and first print); RR17-04 (heterogeneous
oracle checks category IDENTITY per matched cluster); RR17-05 (structured-HMM init displaced
beyond the acceptance bound in example AND test: no-op fits structurally fail; measured 0.15
fit error from 1.97 init error); RR18-02/03 (neural route explained as neural with honest
caveats; sample/ensemble/vmp carry real limitations; unknown routes refused like fit());
RR17-18 (uniform first symbol makes the dependence-only claim exactly true, TV 0.07 -> 0.000);
RR17-20 (manifest --write requires --reviewed-by; attestation block recorded). Plus wave-2 and
the round-2 gate fix. ALL 24 pass-17/18 findings now carry fixes; gate = full suite on the
final tree, then push.

### Own-ledger closure wave (2026-08-09, this session): every remaining open item fixed

The full "Demonstrated by the auditing passes" backlog is closed, one commit per module cluster,
each with measured receipts and adverse tests:

- SCORING-1/2/3 (bda80a3c): crps_ensemble/energy_score fair=True default -- the 1/m^2 form's
  dispersion minimizer measured 0.700 of truth vs 1.000 fair at m=5; brier_decomposition returns
  the real Brier score plus a separately-named binned reconstruction; O(bins/n) bias disclosed.
- U-1/2/3/5/6/7/8/9 (d4c653fc): ddof=1 epistemic variance (pinned 0.5034 at the reviewer's
  fixture); kind="variance-epistemic-only"; total entropy recomputed post-clamp; plug-in BALD MI
  capped at log(M) (0.678 vs cap 0.693 measured); duplicate refusal in the log_probs branch;
  symmetric equivalence evaluation; predictive grid precondition.
- R-1/2/3 (65512342): GPD tail extrapolation now opt-in (gpd_tail=True) -- at n=100 the fitted-
  tail q99.9 measured sd 46.4 vs 2.36 empirical with worst draw 663 vs truth 8.54; docstring
  formula matches the mass-weighted computation.
- G-1..G-7 (ddadeaf2): t_{residual_df} reference whenever dispersion was estimated (level 9% ->
  5.2% at n=8, 4000 reps); ll at the MLE dispersion with AIC k = rank + dispersion; rank-deficient
  designs REFUSE per-coefficient Wald (the pinv min-norm split halved a duplicated column's
  coefficient and SE invisibly); NB theta 1.87x SE warning; HC0/elastic-net/module-promise docs.
- S-1..S-6 (f1a4fe5a): frailty_cox se_method="jackknife" (se/SD 0.84 -> 0.98 measured, 60 reps,
  G=20, theta=1); aalen_additive truncates at rank loss with truncated_at instead of accumulating
  min-norm noise (max |B| 1.4 -> 19 before); to_person_period refuses time=0; KM intervals
  renamed POINTWISE with measured 0.60/0.51/0.43 simultaneous coverage; cox_ph independence/PH
  statement with the 0.68 clustered se/SD number; theta->0 claim replaced by the boundary truth.
- A-1/2/3 (b54b5f82): |alpha-1| <= 1e-6 ELBO band (round-off error measured 4e-7 at 1e-10 rising
  to 2.3e-3 at 1e-13; alpha=1-1e-9 now reproduces alpha=1 exactly); objective_n_eval on
  AdviResult with the K-dependence and alpha>1 not-a-bound documented; finiteness postcondition
  wording (no convergence claim from fixed-step Adam).
- CAL-1..8 (3334c4de): calibration_null_expectation() -- measured perfect-calibration ECE/MCE at
  the caller's profile (MCE null 0.057 -> 0.85 as bins 10 -> 500); coverage_curve null_expectation
  column via the gate's finite-m benchmark (0.64 at m=5/level .95), with the "distribution-free by
  ranks" overclaim corrected in BOTH files (measured 0.632/0.644/0.636 family spread at m=5);
  Platt smoothed targets keep separable data interior; isotonic guarantee scoped held-out;
  pointwise-not-band wording with obs_ci_effective_boot exposure of empty-bin conditioning.
- NP-5..8, NP-10/11 (2c9f9a00): Page exact per-block tie variance (reduces to textbook with no
  ties, tied-null level 5.0% measured at nominal 5%); BM separation reports the exact 1/C(n,n1)
  bound instead of refusing, small-sample liberality stated; Mood routes small 2x2 to Fisher with
  min_expected_count on every result; Dunn global-null scope (the ~24% partial-null number) with
  the brunner_munzel confirmatory route; JT pre-specified-ordering + continuity-off cross-check
  (z 2.61116 reproduced both sides).
- RS-2..6 (22ef49b5): BCa degenerate-jackknife fallback labelled "percentile
  (bca-degenerate-jackknife)"; mid-p z0 (atoms at the estimate no longer read as bias); CIRCULAR
  moving blocks (centring gap 0.008 sd after; O(l/n) endpoint bias before) with n^(1/3) block-
  length guidance; cluster G-asymptotics stated; wild_bootstrap leverage= Davidson-Flachaire
  e/(1-h) (slope CI 0.36 -> 0.50 on a high-leverage design) with the response-scale residual
  contract pinned.
- D-1 (06d7c2b7): bayes_action ships expected_loss_mc_se (verified 1/sqrt(n)) and the winner's
  curse is reproduced at -2.52 MC SEs (K=100; audit -2.5); report="fresh-draws" re-estimates the
  winner selection-independently (+0.08 SEs residual) without touching the default path's exact
  loss-invocation counts (MXR-080-1611 preserved by test).
- S-1c + P-3/4 (f9ca5ef0): select_best N=2 flag declared undefined (was alpha-decided: never at
  .05, always at .20); forecast_price receipts state per-lead-marginal (no simultaneous band) and
  overlapping-origin/quantized-level caveats as coverage_assumptions entries.
- E-1: closed by the RR17-09 design fix (b84de41b) -- the measured floor + exact conditional
  pooled route is precisely the repair E-1's entry demanded; recorded here so the "Confirmed, fix
  pending" section reads as resolved.

One stale cross-file pin surfaced and was fixed in-session (06d7c2b7): a test still encoded the
pre-RR17-02 Haldane-at-zero contract for poisson_lograte_effect. Gate before push: the full local
suite on this final tree.
