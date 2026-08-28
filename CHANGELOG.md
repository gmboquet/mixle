# Changelog

All notable changes to mixle are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.8.0] — 2026-08-26

The credibility, stability, and proof release: turning mixle from a broad, fast-moving research
package into one whose supported core, performance, artifacts, and public claims can be independently
trusted. Tracked by the release checklist in `release-checklists/0.8.0.md`. New capability is deferred
to post-0.8 or kept under `mixle.experimental` per the feature freeze.

### Added

- `BackoffDistribution` / `BackoffEstimator` (`mixle.stats.combinator.backoff`): a two-component
  mixture that reserves a pinned share of mass for outcomes a fitted support cannot represent, so an
  automatically inferred model scores held-out values finitely instead of `-inf`. `escape_weight` is a
  floor and `max_escape_weight` a ceiling — zero is an absorbing state EM will otherwise settle into,
  and an unbounded weight turns smoothing into model selection between the two components. The
  fallback must be a normalized law and determines the tail; a mixture of unnormalized factors is a
  factor, not a law. Replaces the unbounded-support `default_value` shortcut on
  `IntegerCategoricalDistribution`, which could not be a proper law and has been removed.
  `CategoricalDistribution.default_value` is unaffected — its label set is finite.

- Stable ``mixle.semantics`` contracts for value roles, units, transforms and
  Jacobians, constraints, priors, observations, posterior/predictive identity,
  uncertainty, calibration, decisions, capability extensions, and trace sinks;
  plus a packaged cross-project Bayesian-inversion fixture whose semantic
  identity excludes backend, job, and storage-location metadata.
- A machine-readable ownership decision for every top-level public module,
  with drift coverage and migration destinations that preserve compatibility
  until replacement and deprecation gates are satisfied.
- A public-API manifest (`manifests/api_manifest.json`) and a drift gate so any change to the exported surface
  is a reviewed diff. Each package entry is tagged with its maturity tier from the `mixle.maturity`
  registry, so the gate blocks on stable/provisional drift while expected `mixle.experimental` churn is
  reported instead of failing the freeze.
- Release-engineering gates: a weighted-estimation contract test, a base-install optional-import guard,
  a tracked benchmark harness, a pull-request template, and the 0.8.0 release checklist.
- A maintained project documentation set covering charter, requirements, architecture, contracts,
  development, testing, security, scientific validity, operations, releases, and migrations.
- `CompiledEM` as a reusable fused full-mixture strategy, automatically selected by `optimize()` for
  eligible partially fusible heterogeneous mixtures; recursive SQUAREM packing for nested
  mixtures/composites; and function-preserving shared-trunk/residual-expert MoE upcycling.
- Heterogeneous-records-in-native-form workflow (worklist F10.1) hardened end to end: disjoint
  train/calibration/test splitting before any model-selection decision; two genuinely different fit
  routes (automatic `optimize()` and an explicit `learn_bayesian_network` inspect/edit/fit route
  selected by held-out calibration log-density); a transparent independent-fields baseline; held-out
  log score plus task-relevant accuracy reporting; a substantive `explain_fit` (fitted regression
  coefficients, GLM weights, and conditional tables); and save/reload verified bit-identical in a
  fresh OS process. (The real-data flagship script that first exercised this was later removed with
  every other direct dataset usage — real-data demonstrations live in notebooks, not this repository.)
  Backing this:
  `HeterogeneousBayesianNetwork` and its factor classes gained a `describe()` method and a JSON
  serialization registration (`mixle.utils.serialization`), so a heterogeneous structured model now
  persists through the same safe artifact path as every other mixle distribution;
  `manifests/serialization_schema_manifest.json` regenerated to record the 7 newly-registered types (M11.1).
- Write-side result egress: `to_dataframe()`/`to_parquet()` on `ParameterPosterior` (posterior
  parameter draws, one row per draw), `CalibrationReport` (its PIT histogram, one row per bin, or a
  one-row summary when there is no scalar predictive CDF), and `MarkovChainLatentPosterior` (an HMM's
  per-position state posterior -- the Viterbi state and every state's smoothing probability). mixle
  had nine read-side data connectors and no supported way to move a result back into
  pandas/Parquet; `pandas` is an optional extra (`pip install mixle[pandas]`), guarded the same way
  as the other optional dependencies and never imported by the base install.
- Executable distributed language-model training contracts: typed DP/HSDP/FSDP2, tensor, pipeline,
  context, expert, and expert-tensor axes; deterministic accumulation and data sharding; complete
  distributed-checkpoint state; native PyTorch execution for supported plans; and explicit adapters
  for Megatron, Ray Train, and Lightning Fabric. Hardware-dependent performance and multi-GPU claims
  remain unverified until retained receipts exist.

### Fixed

- A fifth candidate campaign (four black-box tester sessions plus two clean-install reproduction
  replays against candidate `21df4e4a`, every blocking/major claim adversarially re-verified, with
  a regression watch across all four prior campaigns; D-0205) returned NO-GO, 2 of 4, with every
  filed blocking and major finding surviving independent re-verification -- unlike every prior
  campaign, none were downgraded. Role B replayed clean: GO, 2 of 2, all eight reproduction runs
  verified, identity independently confirmed by both replayers. Ten confirmed findings (four
  blocking, six major) were fixed, and a second adversarial review of the fix diff itself (a
  practice new this campaign) caught two further gaps in two of the fixes before they shipped.
  - `GeneralizedParetoEstimator`'s EM loop crashed on realistic negative-shape (bounded-tail) data:
    the single-shot fit could produce a self-inconsistent model whose implied support excluded its
    own training data's max, and the next EM iteration's validation of that same data against that
    model raised instead of converging (100% crash rate whenever the precondition occurred, ~15%
    of realistic peaks-over-threshold draws at ordinary sample sizes). Fixed by tracking the
    training data's max alongside the accumulated moments and flooring `scale` (never `shape`, the
    moment-matched estimate) to keep the fit self-consistent, disclosed via `numerical_repairs()`.
    The second-pass review found the fix's own accumulator-side relevance pre-check decided
    whether to carry that max from RAW moments alone, invisible to a `pseudo_count` prior blended
    in afterward -- a strong enough prior could still push the blended shape negative and
    reintroduce the identical crash. Closed by threading whether the estimator carries a prior
    through to the accumulator, which now always carries the max in that case rather than guessing
    from data it cannot see the eventual blend from.
  - A model auto-fit from a pandas nullable-extension (`pd.NA`) numeric column could not score or
    re-encode even its own training data afterward (`TypeError` deep inside `float()` conversion);
    the NaN-spelled equivalent worked. The wrapper's missing-value sentinel was normalized to
    `None` regardless of which sentinel the source data actually carried.
  - The "one dirty cell demotes a continuous column to a `-inf`-scoring memorization table" defect
    class (previously closed for a float-typed column with a stray non-numeric cell) reopened for
    the single most common real trigger: pandas coercing an entire numeric column to string dtype
    once any one row fails float parsing. Auto-inference's column-typing now recognizes an
    all-numeric-looking string column as continuous, the same as it already did for a float column
    with one bad cell.
  - `GammaEstimator`'s hard shape-parameter ceiling (used whenever data has near-zero coefficient
    of variation) now discloses via `numerical_repairs()` when it binds, matching the meticulous
    disclosure convention `GaussianEstimator`'s variance floor already followed.
  - `propose()`'s verified frontier used to collapse entirely -- for the WHOLE joint model, not
    just the offending field -- whenever any single field was identifier-like/high-cardinality or
    an unrecognized dtype, on the reasoning that `CategoricalDistribution`'s documented
    `default_value=0.0` legitimately scores an unseen held-out label at `-inf`. That field is now
    excluded from just its own contribution to the aggregate held-out score, disclosed by name,
    while the rest of the candidate is still verified and ranked. The second-pass review found
    this rescue could itself misfire: it excused a field's `-inf` as an unseen label based only on
    the leaf distribution's type, without checking that the `-inf` actually came from the leaf --
    an `OptionalDistribution`-wrapped field fit with `p==0` (no missing rows in training, an
    ordinary outcome) independently scores a genuinely MISSING held-out value at `-inf` too,
    misdiagnosed as "unseen in training" and silently excused, masking a real missing-value
    generalization failure. The rescue now only excuses a field when every `OptionalDistribution`
    wrapper in its chain has `0 < p < 1`, the only condition under which that wrapper can never by
    itself produce `-inf` and any `-inf` is unambiguously the leaf's.
  - `numerical_repairs()` used to silently return empty after a `Model.deploy()`/`Model.load()`
    round trip, even when a repair had genuinely occurred and the repaired value was still in
    effect -- the documented disclosure mechanism for this exact release went silent on exactly
    the artifacts it exists to describe. It now survives the round trip, mirroring the existing
    `_fit_provenance` envelope treatment.
  - README's second Quickstart example (`mixle.task.solve`) crashed with `ImportError` on the
    documented base install, because its default student is a torch MLP and nothing in that part
    of the README said so -- unlike the very next example, which is explicit about needing torch.
    Now disclosed inline, matching that example's own convention.
  - `ProbabilityDistribution.to_json()`/`from_json()` -- documented as a safe serialization route
    -- silently produced a write-only artifact for at least `VonMisesDistribution`: `to_json()`
    succeeded with no warning, and `from_json()` on the result failed in a later process. The
    module-level `dump_models(verify=True)` already read-back-verifies and refuses for exactly
    this case; `to_json()` now does the same.
  - `TreeHiddenMarkovEstimator` silently fit a model with a NaN-poisoned level-transition parameter
    when trained on a corpus that never reached some tree depth (a legitimate, ordinary training
    shape), and that model then silently scored `nan` on an out-of-sample tree that DID reach that
    depth later -- no warning at fit time, though a related guard already caught the same
    corruption when a deeper tree shared the SAME fitting batch. The zero-evidence transition row
    is no longer folded into the poisoned division at its source.
  - `LDAEstimator`'s default alpha solver diverges to infinity for the large majority of realistic
    small-to-moderate topic-modeling corpora, not just adversarial ones, and its
    `LDAConvergenceError` message implied raising `max_alpha_iter` would help when it provably
    cannot (the residual only shrinks because the divergent denominator grows). The message now
    names the estimator's own working escape hatches instead of a remedy that cannot succeed;
    the solver's divergence itself is a known, deliberately fail-closed design decision (D-0052)
    left as recorded, not unilaterally reversed by this fix.
  - Bookkeeping the fix wave itself invalidated: `mixle/lifecycle.py` crossed the large-module-audit
    line threshold (a new audit entry was added, no behavior change), and a new auto-inference
    dispatch type needed registering in both serialization-schema-manifest profiles.

- A fourth candidate campaign (four black-box tester sessions plus two clean-install reproduction
  replays against candidate `468fbaf9`, every blocking/major claim adversarially re-verified, with
  a regression watch across all three prior campaigns; D-0203) found and fixed the following.
  - The shift-anchored moment repair had been propagated one family at a time, each time only to
    the families a tester had just caught -- and twice, a family left as an accepted "stated limit"
    (Gumbel, then StudentT) came back as a confirmed finding in the next campaign. This wave closes
    the class by systematic audit instead: every location-scale family in
    `mixle.stats.univariate.continuous` and `mixle.stats.multivariate`, and every family whose
    M-step differences raw moments, was measured for shift-equivariance across eight offsets on a
    dyadic grid (exact in float64), and every failure was repaired -- Gumbel, StudentT,
    GeneralizedPareto, SkewNormal, ExponentiallyModifiedGaussian, Rician, Nakagami, and
    `DiagonalGaussianEstimator`'s ridge policy (previously priced off the *mean* of all coordinates'
    variances rather than each coordinate's own, inflating the smallest column by up to 3.2e17x on
    heterogeneous-unit data). The five already-repaired families served as positive controls and
    stayed exact throughout. Separately, raw (unanchored) statistics arriving through an engine
    that stacks reduced moments directly -- reachable from the ordinary public API, not only from a
    hand-built tuple -- cannot be corrected at the M-step; every family that accepts raw statistics
    now WARNS in that case rather than returning a silently wrong fit, closing a gap where the
    warning fired only when the raw variance happened to still be positive.
  - Auto-inference's fail-open and fail-closed gaps: one non-numeric cell in an otherwise continuous
    column silently demoted the whole column to a memorization table that scored `-inf` on unseen
    values; `inf` in a numeric column was silently absorbed as a zero-cost missing sentinel; a
    table column with 65+ distinct string values crashed because two independent column-typing
    rules inside one public call disagreed; the documented `structure='auto'` fallback ("on any
    failure, the historical path proceeds untouched") did not actually hold; an all-empty HMM
    corpus -- a designed no-evidence state -- died on an unguarded index into an empty array; a
    fitted model's missing-value sentinel depended on the input pandas dtype family, so a Series
    built from a nullable-extension dtype could not be scored by a model fitted from a plain one
    (and vice versa) -- fixed at the three call sites (auto-inference, encoding, and
    `Model`/`propose`) that all need the same rule to stay consistent with each other.
  - `GaussianEstimator`'s degenerate-data variance floor disagreed with its diagonal-covariance
    sibling's by up to 18 orders of magnitude on identical zero-spread input; both conventions are
    arbitrary regularizations of a quantity with no correct finite value (the true variance of a
    point mass is 0), so this is now stated rather than silently resolved, with the `min_covar=`
    escape hatch named for a caller who needs one specific value.
  - `InverseGaussianEstimator`'s parameter clamp and its DiagonalGaussian sibling's total-loss gate
    now disclose when they bind, matching every other family's `numerical_repairs()` contract.
  - The interpreted fallback used when numba is not installed (including CI's core lane) called
    Python's `math.exp`, which raises `OverflowError` past ~709 where numba's compiled kernels and
    numpy both return `+inf`; a collapsed-scale fit scoring held-out data at extreme magnitude could
    crash with an opaque `OverflowError` naming neither the family nor the remedy instead of the
    same silently-large density every other environment already returns. It now uses `numpy.exp`.

### Fixed

- `LookbackHiddenMarkovModelEstimator` (`lag=0`) crashed fitting an all-empty corpus (every input
  sequence length 0): `IndexError: index -1 is out of bounds for axis 0 with size 0` from
  `seq_initialize`'s transition-adjacency mask, indexing an array with zero positions. D-0203 had
  recorded this as "the same all-empty-corpus defect class as the now-fixed `hidden_markov.py`,"
  found by an agent, not fixed; that undersold the scope -- the sibling's fix was in *scoring*
  (`seq_log_density`), and its `seq_initialize` already tolerated an all-empty corpus without any
  change. This is a distinct crash site, found and fixed before candidate `6203ec15`'s campaign
  (D-0204). `lag > 0` is unaffected: it already raises a clear, designed `ValueError` for an empty
  sequence at encode time, which is not this defect.

### Fixed

- A third candidate campaign (four black-box tester sessions plus two clean-install reproduction
  replays against candidate `dcec5e29`, every blocking/major claim adversarially re-verified;
  D-0202) found and fixed the following.
  - The shift-anchored moment repair that made the univariate Gaussian exact at extreme data
    offsets had not been propagated to its siblings, and auto-inference -- `optimize(data)` with
    no estimator, the flagship path -- reaches them on ordinary data (epoch timestamps, prices in
    minor units, coordinates). `MultivariateGaussianEstimator` and `DiagonalGaussianEstimator` now
    carry the same conditioning-gated anchored track: a diagonal fit that silently returned a
    variance thousands of times too large at offset 1.7e9 now agrees with exact arithmetic to
    3e-14 relative error. `GeneralizedGaussianEstimator`, `GeneralizedExtremeValueEstimator`, and
    `LogisticEstimator` (whose M-steps go through higher-order or iterative moments rather than a
    single variance) received the same repair. The fused numba scoring kernel for diagonal
    Gaussians, reachable only through an explicit low-level engine registration, still used the
    uncentered form after the Python scorer was repaired -- it now agrees with the Python path to
    the bit at every magnitude tested, and is incidentally about 9% faster. The univariate clamp
    that separates genuine spread from cancellation noise was itself too aggressive at extreme
    magnitude (it read real data as constant above roughly mean 1e15); the scatter is now split
    into a data-carrying term and a rounding term, each clamped on its own footing, propagated to
    every family that gained an anchored track this release.
  - `pd.NA` in numeric data was silently modeled as a categorical value (unlike `NaN` in the same
    position, which fit correctly), so any unseen number then scored `-inf`; pandas extension
    dtypes (`Float64`/`Int64`/`boolean`/`string`) carrying `pd.NA` crashed `optimize()`/
    `Model.fit()`/`propose(fit=True)` with an internal protocol error. Both are fixed at the two
    choke points every auto-inference container shape passes through: family selection now treats
    `pd.NA` as missing exactly like `NaN`, and the encode path is normalized to match, so a
    profiler that says "missing" and an encoder can no longer disagree about the same row.
  - The default `optimize()`/`fit()` initialization starved closed-form sequence fits
    (`MarkovChainEstimator`) on small datasets with a misdirecting error; a row with no outgoing
    transitions is now filled uniformly and the choice is disclosed through
    `numerical_repairs()`. The Bernoulli boundary clamp was applied but undisclosed; it is now
    reported the same way the Gaussian variance floor is.
  - A hand-transcribed identity line in the tester handout named a tree hash that existed nowhere,
    caught independently by two testers and a replay. The handout is no longer hand-authored:
    every identity value is read from the built artifacts, and `scripts/verify_candidate_handout.py`
    recomputes and cross-checks all of them (brief, checksum file, attestation) before a candidate
    is ever shown to a tester.

### Fixed

- A second maintainer-executed candidate campaign (four black-box tester sessions plus two
  clean-install reproduction replays against candidate `c9eb9d2e`, every blocking/major claim
  adversarially re-verified and the self-rated minors independently re-rated; D-0201) found and
  fixed the following.
  - `optimize(data)` with no estimator silently switched to a sequence model on tables with any
    uneven-width row -- a short row, an extra field from an unquoted comma, or a blank line at EOF
    -- memorizing every float as a categorical atom and scoring every training row identically. A
    majority-width table with minority ragged rows now refuses with the offending row index (the
    same diagnosis `propose()` already gave); genuinely variable-length sequence data still fits,
    and the ambiguous middle ground states the reading it took.
  - `mixle.reproduction.wheel_provenance()`'s `verified` key was a hardcoded `True`; it now runs
    the wheel's RECORD self-consistency check, returns `verified: not problems` with the problems
    named, and accepts optional `expected_commit=`/`expected_content_sha256=` bindings. Its
    docstring states exactly what is and is not verified.
  - `Model.deploy()` claimed "the common pure-model path never needs an unsafe pickle load" while
    the Bernoulli-set, Thurstone, and Spearman ranking families silently deployed as pickle. All
    three now have real JSON codecs (round-trip bit-identical), any remaining fallback is disclosed
    in the return value, a warning, and the manifest, and the manifest now records producer
    identity and checks its format-version tag on load.
  - The Gaussian variance floor is scale-relative (unit-equivariant) instead of absolute, with
    sub-noise cancellation residues clamped so algebraically equivalent fits agree exactly;
    materially binding floors and covariance ridges -- including exactly-zero eigenvalue lifts and
    heterogeneous-unit inflation -- are reported through `numerical_repairs()`.
  - Mixture/HMM EM: unrecognised `init=` values are refused naming the accepted ones; the default
    initialization reaches the correct fit on ordinary data inside the default budget; a stalled
    rank-deficient multivariate fit and an iteration-capped latent fit are disclosed instead of
    returned silently; `FitProvenance.final_objective` now describes the RETURNED model.
  - `glm()`: `weights=` semantics are now stated and consistent across likelihood, dispersion and
    standard errors; dummy-coded perfect separation is detected (not just the continuous form);
    `robust=` accepts `'HC1'|'HC2'|'HC3'` validated against statsmodels; `n <= p` fits route
    through the reduced-rank path instead of refusing everything.
  - A pandas DataFrame passed to `optimize()` honors `structure='auto'` instead of silently
    bypassing it; a pandas Series with missing values fits through the Optional path; NaN and
    None mean the same missingness at auto-inference; high-cardinality identifier columns get a
    usable per-column error; `backend=""` is refused instead of silently running locally.
  - `describe()` on the `Model` facade and on estimators returns real descriptions instead of "no
    catalogued capability detected"; `Model.posterior()` works on HMMs; deploy error paths name
    paths and remedies; the LogGaussian/LogNormal naming is aliased in both dialects with the
    parameterization mapping documented in both.
  - Build provenance: the wheel and sdist no longer share one `source_content_sha256` key name for
    two different file populations -- each record names the population its digest covers, and an
    env-var release wheel carries the sdist cross-check digest forward.

### Fixed

- A maintainer-executed release-candidate test campaign (five sessions on public data the maintainer
  did not design, every finding independently re-measured; D-0200) found and fixed the following.
  `glm()` computed the coefficient covariance as a pseudo-inverse of the normal-equations matrix,
  whose conditioning is the square of the design's, so on strongly collinear designs standard errors
  came out orders of magnitude too small with `converged=True` and no warning -- covariance, rank and
  the IRLS solve now come from an SVD of the weighted design, agree with reference implementations
  wherever those are well-posed, report the factorization's numerical rank, warn on near-collinearity,
  and the rank-deficiency message's minimum-norm claim is now true. Perfect separation in binomial
  fits is detected and named (`PerfectSeparationError`, a `RuntimeError` subclass, newly public)
  instead of surfacing an internal IRLS symptom.
- The Gaussian sufficient-statistics accumulator lost variance to catastrophic cancellation
  (`E[x^2]-E[x]^2`), so fitted variance was not shift-invariant and collapsed to the internal floor at
  large offsets; it now accumulates shifted statistics and is shift-stable to 1e-9 at offsets up to
  1e9. One pinned reproduction digest moved by one ulp as a result and the bundle is regenerated.
  Masked arrays are refused with the masks named (their masks were silently dropped by every
  array consumer); NaN observations are named as missing data with the `missing='marginalize'` /
  `marginalized()` remedies instead of a support-interval message; all-zero user-supplied observation
  weights are rejected instead of returning a fabricated `mu=0`; `np.random.Generator` is accepted at
  the public fit entries; the two empty-input paths give one consistent message naming the called
  entry point.
- `propose()` -- the automatic entry point -- failed open in four ways, all closed: a DataFrame was
  iterated as its column names and modeled as a categorical over headers (now routed through the same
  tabular path `optimize()` uses); a degenerate likelihood-spike candidate could win the frontier on
  held-out mean log-density alone (candidates are now screened for spike degeneracy and pathological
  calibration, with the rejection recorded in `Model.notes`); when every candidate failed to score, an
  unverified winner was returned silently with a `succeeded` certificate (now disclosed in notes and
  the certificate says so); the dtype-derived candidate universe (integer input proposes discrete
  families) is now stated in `Model.notes` and the docstring. `Model.spec` now carries the winning
  family after `propose(fit=True)`, so the proposal is refittable without re-inferring.
- Mixture EM stopped on its initialization plateau under the default iteration budget and silently
  ignored supplied initializations; `restarts="auto"` did not diversify. EM now runs to its tolerance,
  honors `init_estimator=`/`estimator=` starting points, diversifies restarts, and reaches the known
  optimum on the Old Faithful reference within 1 nat, deterministically.
- Multivariate Gaussian estimation documented and now records its covariance ridge through the same
  repairs channel as the scalar variance floor, and its serialized artifacts are scipy-version
  independent (old artifacts still load).
- `Model.load()`/`deploy()` error taxonomy: unreadable-manifest states each name their actual cause
  instead of uniformly claiming a pickle-format artifact; `trust_code` is demanded only for artifacts
  that actually contain code; corrupt files no longer leak raw stdlib exceptions; a manifest whose
  JSON is not an object is refused cleanly; when a manifest names `model_sha256`, the model file is
  verified against it on load.
- `mixle.task.solve()` produced a student whose live out-of-distribution gate escalated everything
  while `report()` advertised the offline rate and `promoted=True`: the numeric featurization now
  matches between training and serving (legacy artifacts rebuild their original features), the gate's
  threshold is calibrated on the training distribution, promotion reflects the live gate, and
  `answered_slice` is populated. Base installs get named-extra refusals instead of import tracebacks.
- Family error reporting: a failed generalized-extreme-value optimization is surfaced instead of
  presenting as a clean MLE; Gamma fits on data containing zeros fail closed naming the zeros; Weibull,
  Rayleigh and Beta boundary/zero errors name the offending observations; LogGaussian names NaN as
  missing data and no longer emits a raw divide-by-zero warning on zeros.
- Diagnostics and introspection: single-chain fits report finite `ess_bulk`/`ess_tail` (split-chain
  estimators) instead of NaN, with `split_rhat` receipted as unavailable; `convergence_diagnostics()`
  returns its documented availability receipt for one chain instead of raising; `summarize()` returns
  closed-form moments for mixtures and says what is unavailable instead of returning an empty dict;
  `compare()` rows carry distinguishable model labels; `describe()` recognizes the default
  `optimize()` result; `supports()` accepts the capability strings the library itself emits;
  `ks_1samp` documents that its p-value is asymptotic; `AutoregressiveEnumerable.unrank`'s quantized
  ordering is documented accurately and the README example matches its own output.
- The v0.7 compatibility fixtures now run against the installed wheel and sdist in CI's
  clean-artifact job, not only against the source tree.

- Model selection works on regression and mixed-effects fits. `aic()`, `bic()`, `log_likelihood()`,
  `plugin_log_likelihood()` and `compare()` previously raised
  `AttributeError: 'NoneType' object has no attribute 'dist_to_encoder'` for any model containing a
  `Field(...)` term. A fitted conditional model carries its estimates on `.result` rather than in a
  single distribution object, and the internal lowering returned `None` for it. Linear mixed models
  now report the exact marginal log-likelihood (random effects integrated out), and Gaussian
  identity-link regressions their exact log-likelihood, each with the matching parameter count --
  verified against `statsmodels` to within 3e-12. Quantities that are genuinely undefined are refused
  with a reason rather than approximated: a mixed model's marginal likelihood factors per group, not
  per observation, so `plugin_log_likelihood()`, `waic()` and `loo()` say so instead of recommending
  calls that used to crash.
- `compare(model, data)` raises `TypeError` instead of hanging forever. `compare()` takes a *list* of
  models; a single fitted model passed positionally was iterated through the legacy `__getitem__`
  protocol, which never terminates.
- A dask client that mixle creates for itself no longer starts a diagnostics dashboard. The dashboard
  is a bokeh/tornado HTTP server: it bound a port the caller never requested, and recent bokeh refuses
  to stop it synchronously from inside a running event loop, so `optimize(..., backend='dask')` failed
  at cleanup inside a notebook kernel after its results were already computed. A client you create
  yourself is still discovered, reused untouched, and never closed by mixle.
- Release artifacts no longer ship the `mixle.tests` tree in the runtime wheel. Required Cython sources and
  quantitative-semantics data remain explicit package data, while the source distribution retains the
  changelog and generated release manifests.
- Stale pre-0.8 CPU/GPU benchmark results are archived and no longer presented as 0.8.0 performance evidence;
  new benchmark result documents carry version, release-line, and source-commit provenance.
- Registry alias reads and atomic-write/lineage regression tests close their files deterministically, removing
  resource warnings from the release persistence batch.
- Generated compatibility manifests now live under `manifests/`; tracked research scratch files under
  `experiments/group_attention` were removed and `/experiments/` is ignored.
- Block scheduling now prices density, responsibility, and parameter-update work together instead of
  treating density time as the whole block cost; learned controllers receive the same measured cost.
- Dirichlet-prior block and freeze/roll-up updates now use the exact MAP objective and carry the
  posterior weight prior; nested homogeneous mixtures preserve heterogeneous encoding depth.
- `task.regress`'s internal MLP student (`solve_regression`, `RegressionSolution.improve`, and any
  downstream distillation built on it, e.g. `mixle_pde.surrogate.distill_forward`) now builds its
  network at an explicit float32 instead of following `torch.get_default_dtype()`. A caller that had
  changed the process-global default dtype (mixle-pde's PDE code routinely does, for numerical
  precision) left the student's own weights at that ambient dtype while its inputs stayed explicitly
  float32, crashing with "mat1 and mat2 must have the same dtype, but got Float and Double".
- `learn_bayesian_network` (and therefore `optimize()`'s automatic structure-discovery path) raised
  `TypeError: '<' not supported between instances of 'NoneType' and 'str'` on any discrete field mixing
  a missing-value sentinel (`None`) with string/int/bool levels -- `sorted(set(...))` compared `None`
  against those types directly. Found while wiring real missingness (`None`, not a `"?"` placeholder
  string) into the F10.1 real-data validation work; fixed by sorting on `repr()`, the same guard `_GLMFactor.fit`
  already used a few lines away.
- Calibration, quantitative-semantics, and posterior-schema boundaries now reject empty, malformed,
  non-finite, asymmetric, or non-positive-semidefinite inputs instead of passing or clipping them.
- Reproducibility-sensitive modality reductions and corpus shingles now use portable digest-based
  hashes rather than process-randomized values.
- TreeHMM encoder equality handles array-backed state; real-option annotations resolve under runtime
  introspection; callable-arity probing no longer invokes user code twice or masks its exception;
  optimization failure uses explicit exceptions rather than removable assertions; and mixture search
  rejects detailed tuple scores before entering scalar optimizer loops.
- Empirical law discovery selects candidates on validation data and confirms the winner on a separate
  untouched holdout; invalid ranges, budgets, forms, and simulator outputs are rejected.
- Sequential-design acquisition budgets reject invalid values and explicitly count the initial fit;
  root lazy imports preserve nested dependency failures.
- Ordinary Pytest collection recognizes both supported test filename conventions.
- `mixle.utils` no longer lists `parallel_mpi`, `em`, `enumeration`, `estimation`, `mcmc`, `objectives`,
  or `priors` as importable submodules. Each was a stale name left over from earlier `pysp.utils`
  reorganizations (the real code lives at `mixle.utils.parallel.mpi`, `mixle.inference.*`, and
  `mixle.enumeration.algorithms`), so accessing any of them raised `ModuleNotFoundError`/`AttributeError`
  despite appearing in `mixle.utils.__all__` and `dir(mixle.utils)`. The public-API drift test now
  resolves every declared dynamic-package name via `getattr` instead of only diffing `__all__` as
  strings, so a stale export like this cannot pass silently again.
- `mixle.inference.structure`'s `dependency_gain`, `learn_structure`, and `learn_mixture_structure`
  (via `_init_matrix`) crashed with `TypeError: '<' not supported between instances of 'NoneType' and
  'str'` whenever a discrete parent/child column mixed a missing-value sentinel (`None`) with
  str/int/bool levels -- the same class of bug already fixed in `bayesian_network.py`'s
  `learn_bayesian_network`; all three sites now sort on `key=repr`.
- `mpmath` moves from a base runtime dependency to the new `highprec` extra (`pip install
  mixle[highprec]`); its one non-test consumer (`mixle.engines.highprec`) already treated it as an
  optional fallback behind `gmpy2`, so forcing it into every install left the module's `_BACKEND=None`
  degrade path practically unreachable and contradicted the base-install optional-dependency convention
  (worklist P2.2).
- `docs/development` and `docs/operations` no longer collide between their `.rst` and `.md` source
  twins (both are registered Sphinx source suffixes). Sphinx was silently building the `.md` file for
  each docname; for `operations` this meant the published site was missing the `mixle.ops` API
  reference entirely, since `docs/operations.rst` never won the build.

Findings from the 2026-07-13 full-tree code review (IDs reference its audit ledger; every fix ships
with a regression test that fails on the unfixed code):

- Fused numba kernels no longer compile with `fastmath=True` (`ninf`/`nnan` miscompiled -inf
  out-of-support scores into positive log-densities, reachable through `optimize()` auto-fusion), the
  fused E-step and nested emitters guard all-impossible rows, nested fusion declines
  min/max-statistic leaves, and the fused Pareto E-step tracks per-component support minima
  (D-1..D-5, D-7; #428).
- HMM `viterbi`/`seq_viterbi` perform real backpointer backtracking; the `taus` parameterization
  scores correctly in both scalar and vectorized paths; `seq_posterior` returns smoothing (not
  filtered) marginals; heterogeneous-emission models work through every read-out API;
  hierarchical-mixture EM is monotone on variable-length corpora; LDA handles empty trailing
  documents; structured/IO-HMM guard zero-mass observations and `fit(fast=True)` honors
  `final_states` (L-1..L-12; #435).
- `ops.quantize` brackets with the spatial quantile (was: half of every symmetric distribution
  silently discarded); `relations.ViterbiPath` delegates to the admissible k-best HMM search;
  split-conformal certification abstains honestly at small calibration sizes and certifies every
  class head; registry/relations/fault edge cases raise clear errors (C-1..C-11; #425).
- ppl: VI applies the non-centered transform (was: z-space values returned as posteriors);
  mixed-model EM converges on all parameters, with GLS standard errors; MAP point estimates drop the
  transform Jacobian (flat-prior MAP equals the MLE again); composite slot names no longer collide in
  summaries; half-normal log-density constant, IRLS coefficient-prior scaling, Kalman-EM
  initial-state timing, the NIG sigma summary, and `vi_fit(seed=)` (P-1..P-11; #431).
- inference: `optimize(schedule="auto")` can no longer accept a zero-support model (impossible rows
  score -inf, matching the mixture contract); keyed (tied) parameters survive the posterior-transform
  strategies and the heterogeneous executor; HMM conditioning answers past the evidence horizon with
  a true forward-algorithm joint; spatial-block folds clamp the max edge; Vuong pretests degenerate
  variance (I-1, I-3, I-6..I-9, I-11; #432).
- UQ/statistics: `brunner_munzel` one-sided p-values un-inverted; Wilcoxon `pratt` zero corrections;
  the particle filter carries SIS weights when not resampling; jackknife+/CV+ use the finite-sample
  order statistics with unbounded small-n endpoints; ESS uses Geyer's initial-monotone truncation per
  component; `nuts_numba` thinning returns the requested draw count; frailty-Cox hazard lookup,
  rank-normalization plotting position, Efron reported likelihood, canonical-link labels, m-out-of-n
  rescaling, and exact permutation p-values (U-1..U-10, I-4, I-5; #430).
- Neural leaves hold `eval()` during scoring and `train()` during fitting (dropout/batchnorm modules
  scored stochastically and mutated running statistics on mere `log_density` calls); multi-field
  accumulator fan-in no longer drops fields; module dtype follows the engine precision (G-1, G-2,
  G-5; #426).
- API consistency: `optimize(seed=)`; mixture/categorical constructor validation (negative
  categorical probabilities were accepted and returned negative densities); empty-data `ValueError`s
  and the `raise Exception` -> `ValueError` narrowing; ppl dialect aliases (`log_density`,
  `sample(size=)`, `GaussianObs`, HMM `components=`, pair-copula `log_density`); scalar
  `pseudo_count` broadcast; GP `fit` returns the model (S-1..S-4, S-6..S-10, S-13, S-14; #433).
- Completeness: structured/IO-HMM `TransitionOperator`s serialize (fitted models round-trip);
  `InputOutputHMM` gains sampler/viterbi/posterior-decode/state-posteriors; BetaBinomial honors the
  finite-support contract; LogSeries/Skellam/DirichletMultinomial enumerate; closed-form moments and
  entropies filled across the univariate catalog; `mixle.ppl` exports `waic`/`loo`;
  `ScheduledHMM.estimator()` restores the prototype convention (F-1, F-2, F-4, F-6, F-7, F-9, F-11,
  F-12; #434).
- Completeness follow-up: #434's "entropies filled across the univariate catalog" covered 5 of the
  11 families F-9 actually names. `SkewNormal`, `NegativeBinomial`, `Rician`, `Nakagami`, `Skellam`,
  and `LogSeries` now have `entropy()` too -- a closed form for Nakagami (via the Gamma-entropy
  identity under `X = sqrt(Y)`), a closed-form reduction plus one adaptively-quadrated term for
  SkewNormal, adaptive quadrature for Rician, and exact series summation (truncated at each
  distribution's own quantile, not a fixed-width heuristic) for NegativeBinomial/Skellam/LogSeries.
  Verified against independent numerical integration/Monte Carlo, not only scipy: scipy's own
  generic `entropy()` silently returns a wrong value ("sum did not converge") for NegativeBinomial
  and LogSeries at strongly over-dispersed parameters (F-9; #512).
- The MCMC parameter-posterior bridge (`sample_parameter_posterior`, reachable through
  `inference.posterior(..., over="params")`) hardcoded exactly 7 scalar families
  (`mcmc/parameter_bridge.py:287-290`) and raised `NotImplementedError` for every other
  distribution, despite the README reading as general ("a `prior=` is the only switch"). It now
  dispatches generically against each family's existing
  `mixle.stats.compute.declarations.DistributionDeclaration` -- the same per-parameter
  `constraint`/`differentiable` metadata `mixle.inference.gradient_fit` already uses for autograd
  fitting -- covering 33 families total (the 7 original plus 26 more, spanning continuous,
  discrete, and vector/multivariate families). Families with no declaration, a declaration
  describing a natural/scoring parameterization instead of the constructor's (e.g. von Mises), or
  an exotic constraint with no generic reparameterization yet (a covariance matrix, a coupled
  bound) still raise a clear `NotImplementedError` rather than silently guessing (F-5; #510).
- `tests/base_dist_test.py` now exercises 40 of its 41 base-distribution families end to end (was
  1 of 41 -- every `dists.append(...)` but `TreeHiddenMarkovModelDistribution`'s had been commented
  out since at latest the pysp->mixle rename); fixed the stale-argument constructor calls this
  uncovered plus a genuine `IntegerChowLiuTreeDistribution.__str__` bug (per-feature table strings
  were never joined and lost their shape via a blind `.flatten()`, so `eval(str(dist))` could not
  reconstruct a real instance), and the README's overclaim that this file exercises "each family"
  end to end (F-3; #516).

Findings from the 2026-08 adversarial statistical review, passes 2–23 (per-finding `STAT-RR…` IDs,
measured falsifiers, and reproductions live in `release-checklists/inference-stats-audit-2026-08-08.md`;
every fix ships with a regression test that fails on the unfixed code):

- Event studies (`mixle.inference.event_study`): the Poisson log-rate-ratio route now conditions on
  per-subject totals — `k_post | n ~ Binomial(n, p)` with `logit(p) = log(theta) + log(r)` — so the
  nuisance baseline rate cancels in the likelihood instead of being plugged in, and per-subject
  variances are conditional pmf moments that depend on totals alone, which keeps inverse-variance
  weights from correlating with sampling noise. Outcome-dependent exclusion is surfaced, not
  laundered: `poisson_lograte_effects` reports a selection receipt, and any zero-total drop demotes
  `hierarchical_event_study`'s label from ATT to a selected-sample (event-positive) association with
  `identified=False`. The intermediate "estimate the event-positive mean instead" repair was itself
  falsified — pooled debiasing left −0.092 bias and 0/80 CI coverage under heterogeneous effects — so
  the association label is what the estimator actually supports. Drop-free data gets a real ATT:
  equal-subject arithmetic arm means, one Welch–Student-t reference shared by the p-value and the CI,
  empirical variance floored at `mean(supplied variances)/n`, `df`/`ci_level`/population counts on
  the result, and the near-Normal arm-means condition stated together with its measured cost (31.0%
  skew-law rejection at n=5). `tipping_drift` reads the result's own CI edge instead of recomputing
  a different one.
- Conformal prediction: `weighted_conformal` takes per-query likelihood-ratio weights (an `(m,)`
  `test_weight`, or an explicit scalar) and computes per-query weighted quantiles — a single
  implicit weight was wrong whenever covariate shift varied across queries. `CrossModalModel.calibrate`
  splits its holdout into a scale half and a rank half (scaling and ranking the same residuals broke
  exchangeability), refuses fewer than 4 holdout points, and binds intervals to the exact predictor
  that produced the calibration scores; `fit` validates its arguments before mutating anything and
  invalidates conformal state before refitting, so a failed refit can no longer leave stale
  calibrated intervals behind.
- Calibration gates: promotion decisions replaced point estimates with one-sided 90% Clopper–Pearson
  bounds — measured-power lower bound ≥ 0.5, and tolerance rules additionally demand a
  null-rejection upper bound ≤ 0.5 sitting below the power lower bound, so a gate cannot be promoted
  in a regime where its replicate counts cannot distinguish calibration from miscalibration.
  `calibration_null_expectation` refuses `n_sim < 20` and selects the conservative order statistic
  `k = ceil(0.95 * (n_sim + 1))` with a strict `>` exceedance rule and an explicit tie caveat.
  Column-swap randomization checks are scoped to the two regimes where the swap null is exact
  (shared-draw construction under arbitrary row dependence; fully independent rows and entries);
  outside them — shared latent rows with fresh entries — the measured false-alarm rate was 83.9%,
  and the documentation now says so.
- Nonparametric tests (`mixle.inference.nonparametric`): `brunner_munzel` no longer fabricates a
  finite p-value under complete separation — it returns `pvalue=NaN` with the exact permutation
  bound labeled `p_exchangeability`; Wilcoxon signed-rank uses its exact null distribution through
  n=300; the runs test is exact through n=5000 via big-integer tail sums with a subnormal floor and
  a `log10_pvalue` field.
- Ensemble MCMC diagnostics (`mixle.ppl`, `mixle.inference.mcmc`): affine-invariant ensemble results
  carry walker provenance stamped by the sampler itself (surviving pickling through process pools),
  effective sample size sums per-walker Geyer estimates, and split R-hat is computed over ensembles
  — walkers interact, so the ensemble is the unit of replication. Half of each ensemble's walkers
  initialize from prior draws (scalar-shaped priors only), so a collapsed start cannot imitate
  convergence. `summarize()` gained an honest status ladder: recorded post-warmup NUTS divergences
  yield `divergent-transitions`; split R-hat above 1.01 or bulk/tail ESS below 100 yields
  `unconverged-by-diagnostics`; `ok` requires passing all three checks and is the only promotable
  status. Posterior summaries report `mcse` alongside each estimate.
- LLM answer calibration (`mixle.reason.llm`, `mixle.task`): calibration receipts bind the policy
  that produced them — `calibrate(..., policy_token=...)` records the sample size, generator, and
  policy identity; `answer()` refuses under a changed policy; and the stated validity condition is
  behavioral stability, not object identity. `fit_factuality` accepts only exact Boolean/0-1
  verdicts (scores silently coerced through truthiness corrupted the calibration target), and its
  discrimination number is labeled the resubstitution AUC it is. Calibrated generation declares its
  estimand: `sampling="constructed"` collapses duplicate prompts (uniform-over-distinct) while
  `sampling="iid-traffic"` counts rows as they arrive (traffic-weighted), and receipts carry the
  declaration with the effective count. Answered-slice bookkeeping is validated and transactional —
  marginal label-set coverage is not answered-slice risk (a 0.91 marginal figure coexisted with
  47.4% answered-slice error in the audited configuration; `calibrate_selective` is the
  answered-slice route).
- Entropy uncertainty (`mixle.inference.uncertainty`): the Miller–Madow entropy standard error comes
  from a 2048-replicate parametric bootstrap with a receipt (`entropy_se_receipt`), because the
  delta-method standard error degenerates exactly at equiprobable classes.

### Changed

- Exactness-preserving, parity-tested implementation changes: multivariate-Gaussian scoring
  precomputes the inverse Cholesky factor for single-gemm scoring; the torch objective fitters
  evaluate one forward per Adam iteration instead of two, with NaN-aware best-state tracking; and
  `seq_encode` chunking uses stride slices. These are implementation notes, not 0.8.0 performance
  claims; candidate-specific timing requires a retained benchmark receipt under the release policy
  (E-1, E-2/G-3, E-3, I-2; #436).
- The MPI backend (`MPIEncodedData`, `optimize(..., backend="mpi")`'s underlying handle) now folds
  per-rank sufficient statistics with an `O(log W)` `comm.reduce` binary tree instead of a
  gather-to-root loop, so no single rank folds more than `O(log W)` payloads — matching the technique
  already used by the Spark and local-heterogeneous executors. The standalone
  `mixle.inference.mpi_executor` transport (`mpi_fit`/`mpi_em_step`), a second, non-canonical MPI entry
  point kept only to preserve that tree-reduce technique, is removed now that the canonical backend
  uses it directly; verified equivalent to the removed transport under real `mpirun` before removal.
- Statistical-review signature changes, called out for scripts written against 0.7.0:
  `poisson_lograte_effects` returns `(effects, variances, selection_receipt)` — callers unpacking
  two values must take three; `weighted_conformal` requires a per-query `test_weight` (the previous
  implicit broadcast mis-stated coverage whenever shift varied across queries);
  `hierarchical_event_study` accepts `treated_selection=`/`control_selection=` receipts and labels
  any dropped-subject result an association (`identified=False`) rather than an ATT; and
  `mixle.ppl.summarize` `diagnostic_status` values now include `divergent-transitions`,
  `unconverged-by-diagnostics`, and `single-chain-mixing-unassessable` — code that string-matched
  the old vocabulary should use the documented ladder.

### Security

- `load_encoded` requires `trusted=True`. Its body is pickle, so loading it executes whatever the
  file contains, and the stored integrity digest is computed from and held inside that same file —
  it detects truncation and header tampering, not a file replaced wholesale. The decision now sits
  at the call site, where the provenance of the path is known.
- `Model.load`, `Registry.get`/`current`/`verify_chain`, and `Embedder.load` require `trust_code` to
  be exactly `True` or `False`. Each previously gated a code-executing decode with truthiness, so
  the string `"false"` — the form a flag takes in a config file, an environment variable, or a CLI
  argument — opened the gate it names the closing of.
- Substrate secret redaction now covers the whole surface the scanner reads: dictionary keys, set
  members, and opaque objects whose text carries a credential. Redaction is applied to a fixed point
  (a mask can re-trigger a broader rule), and `enforce_secret_policy` re-scans the sanitized item
  before returning it, so a future gap fails the write instead of leaking.
- Deployment names are contained under their declared artifact root. `Solution.deploy` joined a
  caller-supplied name onto the root unchecked, so `"../../escaped"` traversed out of it and an
  absolute name discarded it entirely; a symlink already inside the root is caught by resolving the
  result, which a check on the string alone cannot do.
- `Governance.approvers` and `.grants` are read-only views, and an approval can no longer redirect
  its own proposal. `approve(..., to=...)` substituted the target *and* the authorization was then
  checked against the substituted value, so an approver for one scope could publish into it an item
  proposed for another.

### Changed — behaviour, including reported values

Several repairs change numbers the library reports. Each is a correction; none is a tuning choice.

- **Acquisition rankings change.** `propose_local_penalization` carried an inline Expected
  Improvement that was wrong in two ways: the law itself (clamping the improvement term deletes its
  negative part) and a clamped sigma in place of the exact `sigma -> 0` limit. The error is
  `z`-dependent, so it reordered candidates. Max-value entropy search separately credited a
  deterministic candidate with `log 2` nats — the largest merit in the pool — and now returns zero.
- **Exchangeability verdicts change.** `exchangeability_check` compared each of two probes per field
  to `alpha` independently, so the aggregate error rate grew with the number of columns. Measured on
  genuinely exchangeable data with 20 columns, the old rule flagged a violation in 23 of 30 datasets;
  corrected, 0 of 30. The primary family is corrected together, both raw and adjusted p-values are
  reported per field, and `exchangeable` is documented as failure-to-reject rather than certification.
- **Expected-information-gain probes change.** The discrepancy-invention loop's simulator sampled an
  action-truncated law while its likelihood ignored the action, so four of five probe locations
  reported a *negative* information gain. The loop's default action moves from `3.3585` to `2.6969`
  and its reported gain at `probe_reweight_n=1` falls from `+0.2806` to `+0.0319`. The winning score
  is also re-estimated on an independent stream, removing a measured winner's-curse bias.
- **Multi-fidelity training budgets change.** LM fidelity rounded to whole epochs, so at
  `max_epochs=3` the budgets 0.05 through 0.4 all executed exactly one epoch, and at `max_epochs=1`
  every budget did. Fidelity is now denominated in training tokens; `budget=1.0` is unchanged.
  Recipe search additionally pins a shared seed — without one, two identical recipes at one budget
  returned 6.4616 and 7.6456 nats/token.
- **Endpoint graph laws are exact.** A declared `p=0` scored a present edge at `log(1e-12)` rather
  than `-inf`, so evidence the model calls impossible entered likelihood ratios and BIC comparisons
  as merely unlikely; a certain event scored `-1e-12` rather than `0`.
- **Scorecard identities change.** Question-set and scorer identities are content-based rather than
  `repr`-based. Identities persisted by an earlier build will not match, and a regression previously
  fabricated from address-bearing digests no longer is.
- State-space EM will not report convergence on an objective decrease, and records `monotone` and
  `max_objective_decrease` on its result. Across 3,200 fits at four tolerances the stopping iteration
  is unchanged.

### Changed — API

- Caller-supplied Boolean flags are exact at 101 public boundaries. `bool("false")` is `True`, so a
  flag arriving from configuration text previously enabled the behaviour it named the disabling of —
  graph directedness and self-loops, dataset shuffling, robust fitting, convergence requirements,
  normalization, approximation permission, low-memory execution, MCMC adaptation, and others.
- 35 durable records (`Receipt`, `Report`, `Verdict`, `Certificate`, `Provenance`, …) are frozen and
  detach their containers at construction. Eight records that are genuinely built incrementally stay
  mutable, deliberately, and are listed in the migration guide.
- `VerificationReceipt` carries `subject_hash`; `certify` will not raise a guarantee on a receipt
  that names no subject or a different one. `receipt_subject(model)` is exported so a caller
  supplying optimizer evidence can bind it, and `schedule` accepts `receipts=`.
- `CategoricalSampler.sample` requires an exact non-negative count and honours `batched`, which was
  previously declared and ignored. Backoff sequence encodings must carry exactly three elements.
- `System.answer(budget=...)` rejects a negative budget rather than returning a refused receipt,
  matching `improve`.


## [0.7.0] — 2026-07-09

Workstream: generic AI-capability platform pieces on top of the core estimation engine (task
decomposition, cross-modal reasoning, self-improvement loops, a system facade), plus a hardening
pass across the automatic-inference and design-of-experiments subsystems.

### Added

- New model families: `PINNRegression` (physics-informed neural network leaf), `HamiltonianNet`
  (conservation-law-preserving dynamics), `make_deep_set` (permutation-equivariant networks),
  monotonic MLPs and input-convex energy networks, `build_product_energy_net` (energy-based product
  of experts), `CopulaDistribution` (arbitrary marginals + a Sklar dependence core),
  `GatedMixtureDistribution` (input-dependent mixture-of-experts weights).
- Task-decomposition and agent-facing workstreams: plan models fit as Markov chains over agent
  traces, an outcome-trained decomposer, a minimal orchestrator loop, tabular Q-learning and
  maximum-entropy inverse RL, a local model registry, `ExecutionTrace` with bit-identical replay, the
  `Receipt` object (ledger + trace + calibration + provenance, offline re-verifiable), and the
  `System` facade (`answer`/`ingest`/`improve`).
- Cross-modal reasoning: `ModalityView`, per-edge conditional-transport premise checks, belief walks
  across chains of verified transports, cycle-consistency as a self-supervised abstention signal,
  task-sufficient projection, information-gain retrieval, and the workstream-F flagship harness.
- Self-improvement / knowledge-accumulation loops: the collapse monitor shared across amplification
  loops, the composition operator, `DesignModel`'s cross-round what-works prior, diagnosis-directed
  correction, a knowledge-accumulation flywheel measurement, degradation-policy handling for fault
  modes, and cost-aware routing threshold selection.
- `doe`: `VerifiableOracle` + the design-test-learn loop, noise-robust incumbent selection for
  Bayesian optimization, and a budgeted propose-verify-retrain loop over a discrete design space.

### Fixed

- `mixle.utils.automatic`: crashes on empty/degenerate input (`ZeroDivisionError` in the Poisson/
  Gaussian/log-normal estimator builders), every distribution detector silently dropping its own
  already-computed fit and the caller's `pseudo_count`, an `IndexError` on always-empty sequence
  fields, a modality-fingerprint diagnostic that could contradict the actual estimator built, and the
  model-suggestion logic ignoring its own held-out validation signal when it disagreed with the
  in-sample BIC pick.
- `mixle.data`: `Boolean.coerce` silently inverting string-typed values (`bool("False") == True`),
  `Schema.conform_record` silently truncating mismatched records via an unchecked `zip`, an
  inconsistent tuple-vs-list adjacency coercion in the graph data source, and `MaterializedSource`
  silently accepting non-reiterable one-shot iterators.
- `mixle.doe`: an unguarded Cholesky decomposition crashing on singular/near-singular input
  covariance, a Morris-screening `ZeroDivisionError` on a degenerate grid, a silent-`NaN` Gaussian
  Process surrogate when fit with zero observations, an infinite loop given a zero-cost fidelity in
  multi-fidelity optimization, several proposal functions crashing instead of validating
  `n_candidates`, TuRBO overshooting its evaluation budget on trust-region restart, silent
  batch-truncation/duplication under an oversized batch request, and `BayesianOptimizer.ask()`
  re-dispensing duplicate initial-design points in async/parallel ask-before-tell campaigns.
- A circular import that broke `mixle.inference` entirely; a mixture-correction term error in
  `explain()`'s decision-margin ledger; a stale `capacity.py` embedding-head rung mismatch; CI
  flakiness in EM's log-likelihood computation and a de-flaked mixture-of-trees test; Python
  3.10-specific abstention timing in the oracle-timeout path.
- Release-verification pass (found via a fresh, non-editable venv install of the built wheel --
  never caught by the dev environment, which has every optional extra installed): `import mixle`
  was completely broken (a missing-import `NameError` at class-definition time in
  `mixle.models.dpo_leaf` cascaded through `mixle.models.__init__`'s eager import chain into nearly
  every module), plus the same missing-import/stale-duplicate-method pattern recurring across 7
  sibling model files and `pinn.py`; 8 further modules importing torch-gated names unconditionally
  at module level (undermining their own already-correct optional-torch guards); a real
  `DPOAccumulator.value()` bug (weights returned as a list, not an array); a data-shape bug in
  `zero_shot_bootstrap`'s generic neural-density fallback (a 24-dim row was split into 24 scalar
  fields instead of one vector field); a stale test fixture double-wrapping `Registry.tier_stack`'s
  frontier callable; a layering violation (`mixle.experimental.long_context_eval` importing upward
  from `mixle.ppl`); and 2 more test files with the same unguarded-torch-import bug. Also: `numba`'s
  `tbb` dependency floor made `pip install mixle[numba]`/`mixle[all]` uninstallable on Apple Silicon
  (no arm64 wheels) -- now platform-gated; `ray` and `lightning` were used by real, documented
  optional backends with no corresponding `pip install mixle[...]` extra -- both added.

### Changed

- `pinn_leaf.py` renamed to `pinn.py`; the "leaf" suffix dropped from PINN naming throughout.
- Several `mixle/doe` tests re-marked `slow` (heavy Monte Carlo / neural-density fits) so the default
  fast test gate stays fast; a real duplicate-training bug (an estimator refit once per test instead
  of once per class) fixed alongside the re-marking.

## [0.6.2] — 2026-07-05

Workstream: the "frontier ecosystem" reasoning/knowledge stack (substrate, retrieval, the `Reasoner`
facade, `Harness` products, factuality receipts, governance/trust) built out across roughly twenty
parallel workstreams, plus a hardening pass on weighted accumulators, the model registry, MVN/HMM
numerics, and Torch DTensor sharding.

### Added

- Knowledge substrate and reasoning stack: typed/provenanced/scoped storage, multi-hop retrieval,
  `answer_from_substrate` with abstain-and-cite, the full `investigate()` action space
  (retrieve/compute/simulate/create/delegate), the `Reasoner` facade (`answer`/`ingest`/`improve`),
  and the `Harness` product with domain templates and a registry.
- Factuality receipts and a knowledge-graph/ontology stack: constrained decoding against an ontology,
  KG-RAG retrieval, estimation certificates and planner, calibration folded in as a post-condition,
  telemetry dashboards, learned pool/reasoner routing policies, governance/sharing controls, and a
  trust/audit trail.
- Cross-modal reasoning graph nodes, file connectors, exchangeability preconditions, and a
  context-packet/compression layer; four flagship example applications plus vision/edge-distillation
  demos; the "Scientist" laptop product.
- Neural-density families made directly constructible via kwargs (e.g. `VAE(dim=8, latent=2)`)
  instead of `build_*` factories; broadened distillation task support (response/multi-teacher/hint/
  attention/relational/sequence); pickle + `to_dict`/`to_json` serialization for `StreamingTransformer`
  and DPO leaves; optional percentile clipping in `quantize` to bound int4 outlier collapse.

### Fixed

- `mixle.models`: `DPOAccumulator` and `StreamingTransformerAccumulator` silently ignored per-sample/
  per-token weights -- `update`/`seq_update` dropped the weight and `value()`/`estimate()` computed an
  unweighted mean loss, so weighted EM, mixture responsibilities, streaming decay, or explicit sample
  weighting had no effect on DPO or streaming-transformer fits (bit-identical output regardless of
  weight). Weight now carried through the full accumulate/value/M-step path.
- `mixle.inference.production.registry`: `header()`/`metadata()` raised a bare `IndexError` and
  `get()` leaked a raw `FileNotFoundError` with the store path on an unregistered name/missing
  version -- unified behind a single `_resolve_version` guard that raises a consistent `KeyError`.
  Also fixed an unsanitized name/version/alias join onto the store root (a path-traversal vector if
  names are ever API-supplied).
- `mixle.inference.structure`: `_clone` cloned an estimator template via `eval(str(estimator))`;
  since most estimators use the default `<object at 0x...>` repr, the eval always raised
  `SyntaxError` and silently fell back to returning the same shared object rather than a copy --
  correct only by luck for stateless estimators. Replaced with `copy.deepcopy`.
- `mixle.task`: a saved `Solution` with `qhat=inf` reloaded as `None` and broke every subsequent call;
  `inf` is now persisted explicitly. `batch([])`/empty-input handling now returns `[]` uniformly.
- Neural leaves (`NeuralGaussian`, `softmax_leaf`, `mixture_density`, `energy`, `neural_density`):
  accumulators appended one ndarray per row and `np.stack`-ed the entire dataset every EM iteration
  (profiled as a major blowup at scale); rewritten to concatenate once at `value()`. Also fixed an
  array-truthiness bug (`if not xs` raised `ValueError` on an ndarray instead of detecting empty
  input), streamed `LM.nll` via chunking instead of one large `np.stack`, and made `make_mlp` raise
  on non-positive dims instead of silently building a degenerate constant net.
- `mixle.stats` MVN: `_robust_cho_factor` -- float32 (MPS/CUDA) MVN mixture EM crashed with "leading
  minor not positive definite" at higher dims from catastrophic cancellation; now symmetrizes and
  adds trace-scaled jitter only on failure (float64 path unchanged). Separately fixed an
  `(N,K,dim,dim)` memory blowup that OOM'd GPU MVN-mixture fits.
- `mixle.engines` (Torch/DTensor): component-sharding raised `ImportError` on torch 2.0-2.4 even
  though DTensor was reachable via a private module path pre-2.5; fixed with a public-then-private
  import fallback, and the sharded EM fit itself is now explicitly gated to torch >= 2.5 with an
  actionable error instead of crashing on an unsupported `logsumexp`/`isinf` sharding strategy.
- `mixle.inference.glm`: IRLS crashed on rank-deficient/collinear designs (e.g. correlated
  feature-vector parents in cross-modal graph fits); `_solve_psd` now falls back to minimum-norm
  `lstsq`/`pinv` at all three IRLS solve sites, bit-unchanged at full rank.
- `mixle.stats` HMM: the HMM distribution defaulted `use_numba=False` while the estimator defaulted
  it to `HAS_NUMBA`; since `optimize(prev_estimate=init)` encodes data through the distribution's
  encoder, the common "pass an init" HMM fit silently never used numba.
  Distribution default now matches the estimator.

### Changed

- `mixle.stats` MVN covariance accumulation switched from `np.einsum` to a BLAS `matmul`, with
  byte-exact parity. Historical timing is not a 0.8.0 claim.
- `mixle.stats` HMM distribution's `use_numba` default changed from `False` to `HAS_NUMBA`;
  behavior-preserving (bit-identical) but changes default performance characteristics, and an
  explicit `use_numba=False` is still respected.
- Neural-density model construction moved from `build_*` factory functions to direct constructible
  construction -- an API-shape change for consumers of the old factories.
- `mixle.utils.builder` removed as dead code; example/benchmark harnesses moved out of tracked
  `examples/` into gitignored `benchmarks/`.

## [0.6.1] — 2026-07-04

Workstream: the `mixle.task`/`solve()` lifecycle facade (rigid function to deployed, monitored
model), exact/approximate enumeration engines, neural-density adapters wired into the PPL, a
structured HMM/HSMM/Bayesian-network family, cross-modal fusion and LLM uncertainty quantification,
and a precision/distributed-compute engine push (LNS integer arithmetic, JAX/XLA jitted EM,
FSDP2/Spark/MPI transports).

### Added

- `mixle.task`/`solve()` lifecycle facade: `solve()` closing the loop from a rigid function to a
  deployed model with a reliability/OOD gate, `solve(synthesize=N)` generative dataset creation,
  `Solution` save/load/verification records across regression/multi-label/structured tasks,
  `Solution.health()` live-traffic conformal monitoring, `Cascade` serving with realized savings and
  self-improving harvest, cost-economics route recommendation, calibrated N-tier `Router`, generative
  and grammar-constrained plan decoding, distillation planners/tool-callers, edge distillation with
  int8/int4 quantization and device-budget search, and structured-record extraction tasks
  (`HashedRecord`, active labeling, `recommend_model`).
- Enumeration engines for exact/approximate ranking: `LatticeEnvelopeIndex`, `RescoredIndex`
  speculative enumeration, certified `branch_cap` pruning, `HMMPathIndex` quantized count-DP,
  `AREnvelopeIndex` for LLM deep enumeration, a persistent `SeekIndex`, a numpy/batched path,
  and quantized-inference certificates (`logit_error_bucket_slack`).
- Neural-density adapters wired into the PPL as first-class constructors (`NeuralDensity`,
  `NeuralConditionalDensity` wrapping VAE/MAF/MDN/autoregressive/flow torch models), `EnergyModel`
  (NCE + Langevin), `fisher_merge` closed-form Fisher-weighted parameter merge, closed-form
  variational GMM collapse/Runnalls KL mixture reduction, new conjugacy pairs (NIG, Gamma-rate,
  Categorical-Dirichlet, NegBinomial-Beta), and an `explain_fit`/`describe`/`how='laplace'`
  escalation ladder.
- Structured HMM/Bayesian-network family: `StructuredHMM` with low-rank/Kronecker/block-diagonal
  transitions and streaming/parallel Baum-Welch, `ExplicitDurationHMM`/HSMM with segment decoding,
  `InputOutputHMM`, scheduled (length/position-conditional) HMMs, and Bayesian networks with
  heterogeneous regression/GLM/linear-Gaussian edges, mixture-of-DAGs, and `counterfactual()`
  do/abduction-action-prediction.
- Cross-modal fusion and uncertainty: `ProductOfExpertsFusion`, `StructuredFusionClassifier`,
  `CrossModalModel` PoE-VAE with conformal intervals, `CrossModalStore` cross-modal RAG,
  `LLMUncertainty` semantic-entropy plus conformal abstain, claim-level UQ, `BeliefState`
  epistemic-aleatoric decomposition, `DiscreteAnswer.decide`, and the `mixle.reason` front door.
- Compute engines: LNS (logarithmic number system) integer arithmetic with an integer log-sum-exp
  kernel, packed binary/ternary/sub-byte precision kernels, an MPFR
  arbitrary-precision tail, a precision-spectrum planner with data-aware `optimize(precision=)`,
  torch/GPU scoring rolled out across roughly twenty distribution families, distributed EM transports
  (MPI, Spark, FSDP2/CUDA bf16 with DCP sharded checkpoints), and a JAX/XLA jitted EM path.
- `mixle.evolve` Phase 1: typed search space, bandit-population meta-search, and structure operators;
  declarative `LM`/streaming-transformer/DPO leaves with SFT loss-masking and CPT+EWC.

### Fixed

- `mixle` package `__all__`: `ExplicitDurationHMM` was listed but its import line had been dropped by
  a linter re-sort, so `from mixle import *` raised `AttributeError` and broke roughly 70 test
  collections.
- `mixle.inference.streaming`: a circular import through the `mixle.stats` package surface broke
  `mixle.inference` on import.
- `hmm_engine_forward_backward`: read a tensor's shape via `np.asarray` after it had already moved
  onto the torch/MPS engine, crashing GPU forward-backward with a device-conversion error; a related
  `max()` bug was also blocking tweedie torch/GPU scoring.
- `should_auto_fuse`: auto-fusion policy checked fusibility/workload but not numba availability, so a
  numba-free install crashed with `ModuleNotFoundError` mid-fit instead of falling back gracefully.
- PPL conjugate-bridge routing: an unsound route for a binary (logit) GLMM fit by PQL; `how='auto'`
  now warns instead of silently returning a MAP point estimate when the prior has no closed-form
  posterior.
- Constrained plan decoding: an earlier grammar constraint guaranteed output form but not content,
  making an undertrained model more confidently wrong (a silent correctness regression) -- fixed with
  a calibrated confidence floor.
- Enumeration: an earlier "NTT loses to Kronecker" benchmark conclusion was an implementation
  artifact rather than a real limitation; exact multi-prime NTT convolution now lands correctly.
- The README hero example referenced an unused/mislabeled estimator and didn't run; replaced with a
  correct, self-contained hierarchical-mixture topic model, and all runnable README blocks now
  execute standalone.
- Several flaky/order-dependent tests that were masking shared-state bugs: an unrestored
  `set_default_dtype(float64)` leaking across co-scheduled tests, neural-density adapter tests
  depending on global RNG state, and a wall-clock timing assertion invertible under load.

### Changed

- **Breaking**: `optimize(data)`/`fit(data)` now perform automatic dependency-structure discovery by
  default (previously opt-in); text fields also now join the dependency graph.
- Neural leaf classes renamed off the tree-position "...Leaf" suffix, with back-compat aliases kept
  (e.g. `NeuralDensityLeaf` -> `NeuralDensity`, `StreamingTransformerLeaf` -> `TransformerLMEstimator`).
- `mixle.program` (the closure-taking declarative optimization surface) demoted to
  `mixle.experimental.program` as not yet mature; no deletions.
- `benchmarks/` removed from the repo and gitignored; Sphinx `docs/` un-ignored and published to
  GitHub Pages instead of the ad hoc README-embedded examples.

## [0.6.0] — First mixle release

- PPL language core: deterministic-expression slots, `potential()` custom factors,
  `.each(by=)` indexed-flat hierarchical models, data-indexed latents (`theta[Field]`), non-Normal
  GLMMs, R-hat/ESS in `summary()`.
- Automatic-inference `fit` API: prototype/data coercion, `fit` forwards all `optimize` kwargs.
- Categorical/Dirichlet free-dimension inference.
- Streaming-estimator unification.
- Lower-bound version pins across the always-installed core and every optional-dependency extra, so
  users on too-old dependencies get a clear resolver error instead of obscure runtime breakage.

[Unreleased]: https://github.com/gmboquet/mixle/compare/v0.7.0...HEAD
[0.8.0]: https://github.com/gmboquet/mixle/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/gmboquet/mixle/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/gmboquet/mixle/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/gmboquet/mixle/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/gmboquet/mixle/releases/tag/v0.6.0
