# 02 — Leaf distributions (`pysp/stats/leaf/*.py`)

Scope: all 45 leaf modules. Every one realizes the **same core contract** — stated once below — and
adds zero or more capability FACETS on top. The body of this section groups the modules by facet; the
final coverage checklist lists all 45.

## The shared contract (stated once)

### `SequenceEncodableProbabilityDistribution` — ABC
- **Role:** a single probability distribution that can score one value, score a batch via an encoded
  sequence, sample, and hand back a matched estimator/accumulator/encoder quartet.
- **Formalized in:** `pysp/stats/compute/pdist.py:476` (subclass of `ProbabilityDistribution`,
  `pysp/stats/compute/pdist.py:64`).
- **Methods (the contract every leaf realizes):**
    log_density(self, x) -> float                       # scalar score of one realization (pdist.py:120)
    density(self, x) -> float                           # exp(log_density)
    seq_log_density(self, x) -> np.ndarray              # batch score of an encoded sequence (pdist.py:502)
    seq_encode / dist_to_encoder() -> DataSequenceEncoder  # build the batch encoder (pdist.py:525)
    sampler(self, seed) -> DistributionSampler          # pdist.py:125
    estimator(self, pseudo_count=None) -> ParameterEstimator  # pdist.py:130
    set_prior / get_prior                               # Bayesian hook (pdist.py:169/179)
    to_exponential_family(self, engine=None)            # default routes through compute_declaration (pdist.py:146)
    supported_engines / supports_engine                 # EngineResident facet (pdist.py:486/492)
    support_size() / support_is_finite() / enumerator() # Enumerable facet, default None/False/raise (pdist.py:206/218/198)
- **Companion ABCs each leaf supplies a quartet of** (all in `pdist.py`): `DistributionSampler`
  (530), `SequenceEncodableStatisticAccumulator` (692) + `StatisticAccumulatorFactory` (736),
  `ParameterEstimator` (743), `DataSequenceEncoder` (820). The estimator/accumulator/encoder triple
  is the fit path; `estimator(pseudo_count=...)` is the per-leaf entry point.
- **Implemented by:** all 45 modules (see checklist).

Below, only the **extra** surface beyond this contract is documented.

---

## Facet A — ExponentialFamily leaves (`compute_declaration` → `ExponentialFamilySpec`)

- **Facet:** a classmethod `compute_declaration(cls)` returns a `DistributionDeclaration` carrying an
  `ExponentialFamilySpec` (`pysp/stats/compute/declarations.py:43`). This drives the engine-resident
  exp-family scoring path (`to_exponential_family`). Two flags matter:
  - `fixed_base: bool = True` (declarations.py:53) — the base/support is the same for every instance,
    so the carrier can be cached.
  - `runtime_scoring: bool = True` (declarations.py:54) — the natural-parameter form is safe for the
    generic runtime scorer; `False` means the family keeps its own scoring (e.g. `-inf` η entries that
    would produce `0*-inf` under a generic dot product), and the engine falls back to the leaf's
    `seq_log_density`.

**29 EF leaves**, by flag setting:

| Default (`fixed_base=True, runtime_scoring=True`) | 24 modules |
|---|---|
| `bernoulli`, `beta`, `binomial`, `exponential`, `gamma`, `gaussian`, `geometric`, `gumbel`, `half_normal`, `inverse_gamma`, `inverse_gaussian`, `laplace`, `log_gaussian`, `logistic`, `logseries`, `poisson`, `rayleigh`, `student_t`, `uniform`, `von_mises`, `weibull`, `point_mass`* | |

(*`point_mass` carries a declaration but is a degenerate atom; see Facet F.)

Non-default flags:

| Module | `fixed_base` | `runtime_scoring` | Why |
|---|---|---|---|
| `categorical` | `False` | `False` | category set is per-instance `pmap`; η has `-inf` for unseen labels (would give `0*-inf`) |
| `integer_categorical` | `False` | `False` | same per-instance support / `-inf` η rationale |
| `integer_multinomial` | `False` | `False` | same |
| `negative_binomial` | `False` | (default `True`) | base depends on the per-instance dispersion `r` |
| `pareto` | `False` | (default `True`) | base/scale depends on the per-instance threshold |

**The 16 NON-EF leaves** (no `compute_declaration`): `beta_binomial`, `dirichlet_multinomial`,
`skellam`, `skew_normal`, `exgaussian`, `tweedie`, `generalized_extreme_value`, `generalized_pareto`,
`wrapped_cauchy`, plus the six point-process / temporal families and `chinese_restaurant_process`
(Facets E/F). These score through their own `log_density`/`seq_log_density` only.

---

## Facet B — Enumerable (and Rankable-by-index)

- **Facet:** overrides `enumerator() -> DistributionEnumerator` (`pdist.py:568`), `support_size()`,
  and `support_is_finite()`. A **finite** support ⟹ the enumerator is rankable-by-index (you can map
  an integer index to/from a value); a **countable-infinite** support is still iterable in descending
  density via `top_k`/`top_p` but reports no finite `support_size`.

**Discrete-finite (finite support, rankable-by-index):**
- `bernoulli` (size 2), `binomial` (size `n+1`), `categorical` (size `len(pmap)`),
  `integer_categorical` (size `num_vals`), `point_mass` (size 1), `integer_uniform_spike`,
  `integer_multinomial`, `categorical_multinomial` (multiset/multinomial enumerators —
  `MultisetProductEnumerator`, `MultinomialEnumerator`).

**Discrete-countable (infinite support; enumerator yields in density order, no finite size):**
- `geometric`, `negative_binomial`, `poisson`. (`logseries` is EF/count but ships no enumerator class.)

All other leaves inherit the base `support_is_finite()->False` / `enumerator()` raising
`EnumerationError` (`pdist.py:20`).

---

## Facet C — Continuous families (location-scale / positive-support / extreme-value)

All are continuous `log_density` leaves; sub-grouped by support geometry:

- **Location-scale (real line):** `gaussian`, `laplace`, `logistic`, `student_t`, `uniform`,
  `skew_normal`, `exgaussian` (exponentially-modified Gaussian). EF members:
  gaussian/laplace/logistic/student_t/uniform. Non-EF: skew_normal, exgaussian.
- **Positive-support:** `exponential`, `gamma`, `inverse_gamma`, `inverse_gaussian`, `log_gaussian`,
  `weibull`, `rayleigh`, `half_normal`, `pareto`, `tweedie`. All EF except `tweedie` (compound
  Poisson–Gamma; series scoring, no declaration). `pareto` is EF with `fixed_base=False`.
- **Extreme-value:** `gumbel` (EF), `generalized_extreme_value` (GEV, non-EF, shape ξ),
  `generalized_pareto` (GPD, non-EF, peaks-over-threshold tail). `weibull` is the EV-minimum companion.

---

## Facet D — Directional (circular) families

Circular support `[0, 2π)`; the normalizer is family-specific and lives only in the scalar path.

- `von_mises` — **EF** (default flags). Natural params `eta1=κcos μ, eta2=κsin μ`,
  `log_const = -log(2π I₀(κ))`. The Bessel normalizer `I₀` is computed stably via the
  exponentially-scaled `i0e`/`ive` (`scipy.special`); `_log_i0` and the mean-resultant Bessel ratio
  `A(κ)=I₁/I₀` are the directional-specific surface (`von_mises.py:46,51`).
- `wrapped_cauchy` — **non-EF**. Mean direction `mu`, mean-resultant length `rho∈[0,1)`;
  `log_density = log[(1-ρ²)/2π] - log(1+ρ²-2ρcos(θ-μ))`. Closed-form (first trigonometric moment)
  estimator: `rho e^{iμ}` is the resultant; sampling wraps a Cauchy of scale `γ=-log ρ`. No Bessel
  normalizer (rational normalizer instead).

---

## Facet E — Point-process / temporal families (event-time realizations)

**Variant contract:** a single "observation" is a *whole realization* (an event-time array or a
trajectory), NOT an iid scalar. They still subclass `SequenceEncodableProbabilityDistribution`, but:

- **Realization encoding:** `x` is a sorted event-time array on a fixed window `[0, T]`
  (`hawkes_process`, `inhomogeneous_poisson`), `(times, marks)` (`power_law_hawkes`), `(time, mark)`
  event lists (`multivariate_hawkes`), or `(n0, T, events)` with typed events
  (`birth_death`). Encoders validate ordering / window membership and reject events in zero-rate bins.
- **Intensity / compensator surface (extra methods):** `log_density` is `Σ log λ(tᵢ | history) -
  ∫₀ᵀ λ(t) dt` (the compensator). Public extras include `intensity(t, ...)`,
  `compensator`/expected-count over a window, and a `branching_ratio`/criticality accessor
  (`power_law_hawkes.intensity`/`expected_count`/`branching_ratio`).
- **Sampler:** Ogata thinning (`hawkes_process` exponential kernel; `power_law_hawkes` power-law
  kernel `(1+s/c)^{-p}`; `multivariate_hawkes` multivariate Ogata with a per-mark decaying upper
  bound); `inhomogeneous_poisson` does per-bin thinning; `birth_death` simulates the CTMC trajectory.
- **Fitting:** Veen–Schoenberg / Lewis–Mohler branching EM over the latent immigrant/offspring
  structure (`hawkes_process`, `multivariate_hawkes`); closed-form per-bin MLE
  (`inhomogeneous_poisson` rates = counts / (width·n)); closed-form rate MLEs from the trajectory
  replay (`birth_death`); direct MLE over `(mu,A,alpha,c,p)` keeping full event times
  (`power_law_hawkes` — its accumulator stores realizations, not closed-form sufficient statistics).

Modules: `hawkes_process`, `power_law_hawkes`, `multivariate_hawkes`, `inhomogeneous_poisson`,
`birth_death`. None are EF. `multivariate_hawkes`/`power_law_hawkes` are the **marked** variants.

---

## Facet F — Combinatorial atoms / degenerate

- `point_mass` — degenerate atom at `value`; `support_size()==1`, has an enumerator (Facet B),
  carries a trivial EF declaration. Estimator just re-asserts the atom.
- `chinese_restaurant_process` — combinatorial sequence-of-table-assignments distribution with
  concentration `alpha` over `n` customers; `log_density` over a partition/seating array. Non-EF;
  ships sampler + accumulator + `estimator(pseudo_count=...)`.

---

## Facet G — Conjugate / Bayesian hook (`set_prior` + `estimator(pseudo_count=...)`)

- **Facet:** the leaf recognizes a conjugate prior via `set_prior(prior)` which sets
  `has_conj_prior`/`conj_prior_params`; the matching `ParameterEstimator` then does a MAP/posterior
  update instead of a plain MLE, and `pseudo_count` inflates the leaf's own sufficient statistic as a
  pseudo-observation. (`pdist.py:130/169/179`.)

| Leaf | Conjugate prior class |
|---|---|
| `bernoulli`, `binomial`, `geometric` | `BetaDistribution` |
| `poisson`, `exponential` | `GammaDistribution` |
| `gaussian` | `NormalGammaDistribution` |
| `log_gaussian` | `NormalGammaDistribution` |
| `categorical` | `DictDirichletDistribution` |
| `integer_categorical` | `DirichletDistribution` / `SymmetricDirichletDistribution` |

All other leaves expose `set_prior`/`estimator(pseudo_count=...)` (the base contract) but have no
recognized conjugate family — `pseudo_count` still inflates sufficient statistics where the estimator
supports it. (`beta_binomial`/`dirichlet_multinomial` are themselves the compound/marginalized
conjugate forms, not conjugate-prior consumers.)

---

## Coverage checklist (all 45 modules)

```
bernoulli.py                  — core contract; EF(default); Enumerable-finite; Conjugate(Beta)
beta.py                       — core contract; EF(default); continuous (0,1)
beta_binomial.py              — core contract; non-EF count (overdispersed binomial / compound)
binomial.py                   — core contract; EF(default); Enumerable-finite; Conjugate(Beta)
birth_death.py                — core contract; Facet E point-process (CTMC trajectory, closed-form MLE)
categorical.py                — core contract; EF(fixed_base=False,runtime_scoring=False); Enumerable-finite; Conjugate(DictDirichlet)
categorical_multinomial.py    — core contract; EF? no-decl; Enumerable-finite (multiset/multinomial enumerators)
chinese_restaurant_process.py — core contract; Facet F combinatorial atom (CRP partition)
dirichlet_multinomial.py      — core contract; non-EF count (compound/marginalized multinomial)
exgaussian.py                 — core contract; non-EF continuous location-scale (ex-Gaussian)
exponential.py                — core contract; EF(default); positive-support; Conjugate(Gamma)
gamma.py                      — core contract; EF(default); positive-support
gaussian.py                   — core contract; EF(default); location-scale; Conjugate(NormalGamma)
generalized_extreme_value.py  — core contract; non-EF extreme-value (GEV, shape ξ)
generalized_pareto.py         — core contract; non-EF positive-tail (GPD, peaks-over-threshold)
geometric.py                  — core contract; EF(default); Enumerable-countable; Conjugate(Beta)
gumbel.py                     — core contract; EF(default); extreme-value (location-scale)
half_normal.py                — core contract; EF(default); positive-support
hawkes_process.py             — core contract; Facet E point-process (exp kernel, Ogata, branching EM)
inhomogeneous_poisson.py      — core contract; Facet E point-process (piecewise-constant, per-bin MLE)
integer_categorical.py        — core contract; EF(fixed_base=False,runtime_scoring=False); Enumerable-finite; Conjugate(Dirichlet)
integer_multinomial.py        — core contract; EF(fixed_base=False,runtime_scoring=False); Enumerable-finite
integer_uniform_spike.py      — core contract; non-EF; Enumerable-finite (uniform + spike)
inverse_gamma.py              — core contract; EF(default); positive-support
inverse_gaussian.py           — core contract; EF(default); positive-support
laplace.py                    — core contract; EF(default); location-scale
log_gaussian.py               — core contract; EF(default); positive-support; Conjugate(NormalGamma)
logistic.py                   — core contract; EF(default); location-scale
logseries.py                  — core contract; EF(default); count (log-series; no enumerator class)
multivariate_hawkes.py        — core contract; Facet E point-process (marked, multivariate Ogata, branching EM)
negative_binomial.py          — core contract; EF(fixed_base=False); Enumerable-countable
pareto.py                     — core contract; EF(fixed_base=False); positive-support (power-law tail)
point_mass.py                 — core contract; Facet F atom; EF(trivial); Enumerable-finite(size 1)
poisson.py                    — core contract; EF(default); Enumerable-countable; Conjugate(Gamma)
power_law_hawkes.py           — core contract; Facet E point-process (marked, power-law kernel, MLE)
rayleigh.py                   — core contract; EF(default); positive-support
skellam.py                    — core contract; non-EF count (difference of two Poissons; ℤ support)
skew_normal.py                — core contract; non-EF continuous location-scale (skewness α)
student_t.py                  — core contract; EF(default); location-scale (heavy-tailed)
tweedie.py                    — core contract; non-EF positive-support (compound Poisson–Gamma, series scoring)
uniform.py                    — core contract; EF(default); location-scale (bounded continuous)
von_mises.py                  — core contract; EF(default); Facet D directional (Bessel I₀ normalizer)
weibull.py                    — core contract; EF(default); positive-support / extreme-value-min
wrapped_cauchy.py             — core contract; non-EF; Facet D directional (rational normalizer)
```

45/45 modules accounted for (`__init__.py` excluded — re-export only).

---

## Notes / recommendations

- **No new ABC needed for the leaves** — they already share `SequenceEncodableProbabilityDistribution`
  cleanly. The facets (EF, Enumerable, Conjugate, point-process) are realized by *optional method
  overrides*, which is the library's existing convention.
- **Point-process variant should be formalized as a Protocol.** Facet E members override the same
  conceptual surface (`intensity`, compensator/`expected_count`, event-time realization encoding,
  Ogata sampler) but share no declared interface — a `TemporalPointProcess` Protocol (event-time
  realization + `intensity(t, history)` + windowed compensator) would document the contract that is
  currently implicit and inconsistently named (`expected_count` vs ad-hoc).
- **Enumerable / Rankable-by-index is already partly formal** via `enumerator()`/`support_size()`/
  `support_is_finite()`; worth promoting the finite-⟹-rankable invariant to a documented sub-protocol
  so quantization/enumeration code can dispatch on it.
- **Conjugate hook** (`set_prior`+`has_conj_prior`+MAP estimator) is implicit per-leaf; could be a
  `ConjugateUpdatable` Protocol so the fit layer can detect MAP-capable leaves uniformly.
```
