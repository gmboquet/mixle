# 03 — Multivariate & Combinator interfaces

Scope: `pysp/stats/multivariate/*.py` (10 leaf modules + `__init__`) and `pysp/stats/combinator/*.py`
(17 combinator modules + `__init__`). Every concrete class in both packages realizes the **core
contract** quintet defined in `pysp/stats/compute/pdist.py`; this section documents only the *extra*
interface surface each package adds on top of that contract.

The shared core contract (stated once, not repeated per leaf):

- `SequenceEncodableProbabilityDistribution` (`pdist.py:476`) — `log_density`, `seq_log_density`,
  `sampler`, `estimator`, `dist_to_encoder`, `supported_engines`/`supports_engine`.
- `DistributionSampler` (`pdist.py:530`) — `sample(size, *, batched)`.
- `SequenceEncodableStatisticAccumulator` (`pdist.py:692`) — `update`/`seq_update`,
  `initialize`/`seq_initialize`, `combine`, `value`/`from_value`, `key_merge`/`key_replace`,
  `acc_to_encoder`.
- `StatisticAccumulatorFactory` (`pdist.py:736`) — `make()`.
- `ParameterEstimator` (`pdist.py:743`) — `estimate(nobs, suff_stat)`, `accumulator_factory()`.
- `DataSequenceEncoder` (`pdist.py:820`) — `seq_encode`, `__eq__`.
- Optional facets: `DistributionEnumerator` (`pdist.py:568`), `to_exponential_family`
  (`pdist.py:146`), `get_prior`/`set_prior`, `compute_capabilities`/`compute_declaration`.

---

## Part A — Multivariate

All 10 multivariate leaves realize the core contract over a **vector / matrix / SPD / directional**
support. The interesting extra surface is two facets — `Conditionable`/`Marginalizable` (the
Gaussian-family conditional/marginal algebra) — plus the SPD-covariance, directional, and copula
sub-roles.

### `Conditionable`  — de-facto facet (recommend formalizing as a Protocol)
- **Role:** a joint distribution that can return the closed-form *conditional* over its unobserved
  coordinates given a subset of fixed coordinates.
- **Formalized in:** implicit — followed by convention by the elliptical location-scale families.
- **Methods:**
    condition(self, observed: dict[int, float]) -> Self
        # `observed` maps dim-index -> fixed value; returns the conditional over the remaining
        # dims in increasing index order. Raises if no dim is left unobserved or indices out of range.
        # Sampling the result is `given=`-style conditional sampling.
- **Implemented by:** `multivariate_gaussian.MultivariateGaussianDistribution`
  (`multivariate_gaussian.py:434`; Schur-complement update
  `mu_{u|o} = mu_u + S_uo S_oo^{-1}(x_o - mu_o)`, `S_{u|o} = S_uu - S_uo S_oo^{-1} S_ou`),
  `diagonal_gaussian.DiagonalGaussianDistribution` (`diagonal_gaussian.py:274`; diagonal => the
  conditional is just the kept-dim marginal, coords independent),
  `multivariate_student_t.MultivariateStudentTDistribution` (`multivariate_student_t.py:165`;
  conditional inflates scale by the observed Mahalanobis term and bumps df).
- **Facets:** condition; the result is itself the same Distribution type (closure under conditioning).
- **Notes:** the three implementers are exactly the elliptical families with a tractable
  partitioned-precision form. Wishart/inverse-Wishart/matrix-normal/copula/directional do **not**
  implement it. Should be a `Conditionable` Protocol.

### `Marginalizable`  — de-facto facet (recommend formalizing as a Protocol)
- **Role:** return the marginal distribution over a kept subset of coordinates.
- **Formalized in:** implicit — same three families as `Conditionable`.
- **Methods:**
    marginal(self, keep: Sequence[int]) -> Self
        # marginal over index set `keep` (order preserved). Gaussian marginals drop rows/cols:
        # N(mu, Sigma) -> N(mu[keep], Sigma[keep][:, keep]). Raises on empty / out-of-range keep.
- **Implemented by:** `MultivariateGaussianDistribution` (`multivariate_gaussian.py:465`),
  `DiagonalGaussianDistribution` (`diagonal_gaussian.py:291`),
  `MultivariateStudentTDistribution` (`multivariate_student_t.py:201`).
- **Facets:** marginalize; closed under the same family.
- **Notes:** pairs with `Conditionable` — together they make the Gaussian/Student-t families a
  closed **graphical-model leaf** usable inside latent-field / GP code.

### SPD / covariance-matrix family  — core-contract leaves over the SPD cone
- **Role:** distributions whose support is symmetric positive-definite matrices (or matrix-valued),
  serving as random-covariance models and conjugate priors for Gaussian (co)variance.
- **Implemented by:**
  - `wishart.WishartDistribution` (`wishart.py:32`) — `Wishart(df, scale)` on SPD matrices;
    conjugate prior for a Gaussian **precision**. Full accumulator/factory/estimator/encoder stack.
  - `inverse_wishart.InverseWishartDistribution` (`inverse_wishart.py:33`) — `X^{-1} ~ Wishart`
    => `X ~ InverseWishart(df, scale)`; conjugate prior for a Gaussian **covariance**.
  - `matrix_normal.MatrixNormalDistribution` (`matrix_normal.py:34`) — matrix-variate Gaussian
    `MN(M, U, V)` with row- and column-covariance Kronecker structure.
- **Facets:** score / sample / estimate; **conjugate** (Wishart / inverse-Wishart are the Gaussian
  conjugate priors — see the conjugate-family section). No condition/marginal.
- **Notes:** these are leaf distributions, but their conjugacy makes them the SPD half of the
  Normal–(Inverse-)Wishart conjugate pairs used by the Bayesian/HDP machinery.

### `covariance_shrinkage`  — estimator-only module (Ledoit–Wolf)
- **Role:** a regularized **covariance estimator**, not a distribution. Fits a shrinkage-blended
  covariance and returns an ordinary `MultivariateGaussianDistribution`.
- **Formalized in:** `covariance_shrinkage.py` — `LedoitWolfEstimator` (`:60`) +
  `LedoitWolfAccumulator` (`:84`, sums `s1..s4` moment stats) + factory (`:161`).
- **Methods:** realizes `ParameterEstimator` only —
    estimate(self, nobs, suff_stat) -> MultivariateGaussianDistribution   # mean + shrunk covar
        # blends the sample covariance toward a scaled-identity target by the analytic Ledoit-Wolf
        # delta; the shrinkage intensity is exposed on the result as `dist.shrinkage`.
- **Notes:** an estimator that targets a *different* distribution's type than its own module — the
  clean illustration that `ParameterEstimator.estimate` is decoupled from the producing module.
  No `*Distribution` class lives here.

### Directional family  — core-contract leaves on the sphere / projective space
- **Role:** distributions on directional supports (unit vectors / axes).
- **Implemented by:**
  - `von_mises_fisher.VonMisesFisherDistribution` (`von_mises_fisher.py:110`) — vMF on the unit
    `(d-1)`-sphere, mean direction `mu` and concentration `kappa`; full sampler/accumulator/
    estimator stack (estimator recovers `mu`, `kappa` from the resultant length).
  - `watson.WatsonDistribution` (`watson.py:54`) — Watson axial distribution on **projective**
    space (antipodal symmetry `x ~ -x`), mean axis + concentration.
- **Facets:** score / sample / estimate. No condition/marginal/conjugate.
- **Notes:** support-validation surface (inputs must be unit-norm) is the family-specific extra,
  analogous to the point-process event-time validation in other sections.

### `gaussian_copula`  — dependence-only leaf on the unit cube
- **Role:** a Gaussian copula on `(0,1)^d`: dependence carried by a correlation matrix `R`, with
  uniform margins. `log_density(u) = -0.5 logdet R - 0.5 z^T (R^{-1} - I) z`, `z = Phi^{-1}(u)`.
- **Formalized in:** `gaussian_copula.GaussianCopulaDistribution` (`gaussian_copula.py:36`);
  encoder stores the normal-score transform `z = Phi^{-1}(u)` so `seq_log_density` is a quadratic
  form. Estimator (`:175`) fits `R` by the inversion estimator from the encoded scores.
- **Facets:** score / sample / estimate. Implicitly composes with marginals at the application layer
  (copula models the dependence; margins are separate distributions) but it does **not** wrap child
  distributions itself — so it is a multivariate leaf, not a combinator.
- **Notes:** could be argued as a change-of-measure leaf, but in this tree it is a self-contained
  multivariate density on the cube.

---

## Part B — Combinator (the composition interface)

Combinators are the **composition layer**: each wraps one or more **child** distributions and
realizes the same core contract by *delegating to and composing the children*. There is no formal
`Combinator` ABC — it is a de-facto contract.

### `Combinator`  — de-facto composition contract (recommend a documented Protocol)
- **Role:** a distribution defined in terms of one or more child distributions; it composes the
  children's contract methods into its own.
- **Formalized in:** implicit — followed by convention across all 17 combinator modules. No
  `_base.py`, `SingleChildAccumulator`, or `MaskedBaseEncoder` exists in this tree (the task's
  reference to them is stale — the only `_base` token in the package is the literal `P_base` in a
  docstring). Each combinator re-implements the delegation inline. **This is exactly the place a
  shared `SingleChildCombinator` base would remove the most duplication.**
- **The composition contract (how a combinator composes its children):**
    __init__(self, <child(s)>, ...)        # store child dist(s): `self.dist` / `self.base` /
                                           #   `self.dists` (seq) / `self.dmap` (keyed)
    log_density(self, x)                   # combine child `log_density` (sum / mix / renorm / Jacobian)
    seq_log_density(self, enc)             # combine child `seq_log_density` over the encoded children
    dist_to_encoder() -> DataSequenceEncoder       # a combinator encoder that owns child encoder(s)
    estimator()/accumulator_factory()/make()       # build combinator accumulator that owns child accs
    enumerator() -> DistributionEnumerator         # present iff every child is enumerable (facet-preserving)
    compute_capabilities()                 # `intersect_engine_ready(children)` — engine-ready iff all children are
    compute_declaration()                  # declares `children=(...)` + `child_roles=(...)` for the compute layer
    get_prior()/set_prior()                # structural wrappers factor the joint prior over children
- **Facet preservation:** a combinator over enumerable children is enumerable
  (`enumerator()` builds a child-enumerator product/sum); engine-residency is the **intersection**
  of children (`intersect_engine_ready`, e.g. `composite.py:62`, `select.py`); priors factor over
  children (`composite.set_prior` distributes a per-component prior list, `composite.py:98`).
- **Notes:** the single biggest formalization opportunity in `pysp.stats`. A `SingleChildCombinator`
  mixin (covering truncated/censored/survival/hurdle/zero_inflated/transform/exponential_tilt/
  finite_stochastic_transform/optional/weighted — all of which carry exactly one `self.base`/
  `self.dist`) would collapse ~10 near-identical accumulator/encoder/prior-delegation bodies.

The 17 combinators split into four sub-roles.

### B.1 Structural product / composition  (`sequence`, `composite`, `record`, `conditional`, `select`, `ignored`, `null_dist`)
- **Role:** compose children by **position, key, or routing** — the log-density is a product (or a
  routed pick) of child log-densities; no renormalization or measure change.
- **Members & composition rule:**
  - `sequence.SequenceDistribution` (`sequence.py:55`) — i.i.d. variable-length vector: one child
    `dist` scored over each element times a `len_dist` over the length. Enumerable
    (`SequenceEnumerator`, `:555`).
  - `composite.CompositeDistribution` (`composite.py:56`) — product over a **heterogeneous tuple**
    `(D_0..D_{n-1})`; `log_density = sum_k D_k.log_density(x[k])`. Enumerable
    (`CompositeEnumerator` + `CompositeConditionalEnumerator`). Prior factors per component.
  - `record.RecordDistribution` (`record.py:84`) — like composite but keyed by **field name**
    (dict-valued observations). Enumerable.
  - `conditional.ConditionalDistribution` (`conditional.py:81`) — `p(x1 | x0)` via a `dmap`
    (`dict[key -> child]` or list indexed by `x0`) plus an optional `default_dist`; also realizes
    `ConditionalSampler.sample_given`. Enumerable.
  - `select.SelectDistribution` (`select.py:59`) — routes each `x` to `dists[choice_function(x)]`;
    density is the chosen child's density. Enumerable.
  - `ignored.IgnoredDistribution` (`ignored.py:30`) — wraps a child but returns constant
    log-density 0 (the variable is present but its likelihood is ignored); estimation no-ops.
  - `null_dist.NullDistribution` (`null_dist.py:32`) — degenerate placeholder (no real child;
    constant density), used as a default/absorbing slot. Enumerable trivially.
- **Facets:** all enumerable when children are; engine-ready = intersection.
- **Notes:** `ignored` / `null_dist` are the identity/absorbing elements of the composition algebra.

### B.2 Mixture-like latent weighting  (`weighted`, `optional`)
- **Role:** mix a child against an alternative or weight it — latent presence/weight rather than a
  pure product.
- **Members:**
  - `weighted.WeightedDistribution` (`weighted.py:37`) — observation is a `(x, w)` pair scored as
    `P_base(x)` carrying a per-observation weight; the weight scales the base in scoring/accumulation.
    Enumerable (delegates to base enumerator, `:197`).
  - `optional.OptionalDistribution` (`optional.py:40`) — value is present with prob `p` (scored by
    the child) or a missing sentinel with prob `1-p`; a 2-component latent mixture of "observed
    child" vs "missing". Enumerable (`OptionalEnumerator`, `:393`).
- **Facets:** latent (presence) weighting; enumerable.
- **Notes:** the lightweight latent layer — full mixtures live in `pysp/stats/mixture` (other section).

### B.3 Support surgery / renormalization  (`truncated`, `censored`, `survival`, `hurdle`, `zero_inflated`)
- **Role:** keep the child's *shape* but change its **support or atom structure**, renormalizing.
- **Members & rule:**
  - `truncated.TruncatedDistribution` (`truncated.py:50`) — restrict a `base` to an `allowed`
    (keep) or `forbidden` (drop) finite set and renormalize by `log Z` (computed in log-space with
    `logsumexp` / `log(1 - p_forbidden)` via `_log1mexp` for stability). Enumerable over the
    surviving atoms (`:173`).
  - `censored.CensoredDistribution` (`censored.py:69`) — observations beyond a bound contribute the
    tail mass `P(X >= c)` rather than a point density.
  - `survival.SurvivalDistribution` (`survival.py:56`) — reparametrize a base as a survival/hazard
    model (time-to-event), scoring via the survival function.
  - `hurdle.HurdleDistribution` (`hurdle.py:37`) — two-part: a Bernoulli "hurdle" at zero plus a
    **truncated-at-zero** base for the positives.
  - `zero_inflated.ZeroInflatedDistribution` (`zero_inflated.py:34`) — extra point mass `pi` at 0
    on top of a count base whose support includes 0; `log P(0) = logsumexp(log pi, log1mpi +
    base.log_density(0))`, `log P(x>0) = log1mpi + base.log_density(x)`.
- **Facets:** truncated is enumerable; the rest are score/sample/estimate. All single-child
  (`self.base`).
- **Notes:** these are the prime `SingleChildCombinator` candidates — each is `base` + a small
  renormalization constant computed once in `__init__`.

### B.4 Change of measure  (`transform`, `exponential_tilt`, `finite_stochastic_transform`)
- **Role:** push the child through a measure change — a deterministic transform (with Jacobian), an
  exponential tilt, or a stochastic (kernel) transform.
- **Members:**
  - `transform.TransformDistribution` (`transform.py:173`) — push a child through a fixed invertible
    **`Transform`** (sub-protocol below). `log_density(y) = base.log_density(T^{-1}(y)) +
    log|det J_{T^{-1}}(y)|` (the Jacobian term gated by `density_correction`). Encodes by
    inverse-transforming once, carrying `(child_enc, log_jac, valid)`. Has a `backend_seq_log_density`
    engine path. Enumerable (`TransformEnumerator`, `:348`).
  - `exponential_tilt.ExponentialTiltedDistribution` (`exponential_tilt.py:167`) — reweight a base
    by `exp(theta . T(x))` and renormalize by `Z(theta)`; `theta` scalar/vector/callable, default
    statistic `T(x)=log p_base(x)` (tempering => `p ~ p_base^{1+theta}`). Carries a `TiltResult`
    (`:47`) holding the normalizer. Enumerable (`:282`).
  - `finite_stochastic_transform.FiniteStochasticTransformDistribution`
    (`finite_stochastic_transform.py:55`) — push a finite-support child through a row-stochastic
    transition matrix (a measure change via a Markov kernel); `p(y) = sum_x K[x,y] p_base(x)`.
    Enumerable (`:133`).
- **Sub-protocol — `Transform`** (de-facto, in `transform.py`):
    forward(self, x) -> y
    inverse(self, y) -> x
    log_abs_det_inverse_jacobian(self, y) -> float
    invalid_inverse_value(self) -> float
  Implemented by `IdentityTransform` (`:33`), `AffineTransform` (`:57`), `ExpTransform` (`:88`),
  `LogTransform` (`:116`), `LogitTransform` (`:142`).
- **Facets:** all enumerable; transform/finite_stochastic are engine-bridgeable.
- **Notes:** `Transform` is a clean, already-near-formal Protocol — worth promoting to one and
  reusing wherever Jacobian corrections are needed.

---

## Coverage checklist

Multivariate (`pysp/stats/multivariate/`):
- `__init__.py` — package re-export surface only.
- `multivariate_gaussian.py` — core contract over R^d; **`Conditionable` + `Marginalizable`** (condition/marginal via Schur complement).
- `diagonal_gaussian.py` — core contract (independent coords); **Conditionable + Marginalizable** (degenerate, coords independent).
- `multivariate_student_t.py` — core contract; **Conditionable + Marginalizable** (df/scale-inflating conditional).
- `wishart.py` — core contract over SPD cone; **conjugate** prior for Gaussian precision.
- `inverse_wishart.py` — core contract over SPD cone; **conjugate** prior for Gaussian covariance.
- `matrix_normal.py` — core contract, matrix-variate Gaussian with Kronecker row/col covariance.
- `covariance_shrinkage.py` — **`ParameterEstimator` only** (Ledoit–Wolf), produces a `MultivariateGaussianDistribution`; no `*Distribution` class.
- `von_mises_fisher.py` — core contract on the unit sphere (directional); unit-norm support validation.
- `watson.py` — core contract on projective space (axial/antipodal directional).
- `gaussian_copula.py` — core contract on `(0,1)^d`; dependence-only (correlation matrix) leaf.

Combinator (`pysp/stats/combinator/`):
- `__init__.py` — package re-export surface only.
- `sequence.py` — `Combinator` (B.1 structural); i.i.d. element child × length child; enumerable.
- `composite.py` — `Combinator` (B.1 structural); heterogeneous tuple product; enumerable; prior factors per component.
- `record.py` — `Combinator` (B.1 structural); keyed/field product; enumerable.
- `conditional.py` — `Combinator` (B.1 structural) + `ConditionalSampler`; `dmap`-routed `p(x1|x0)`; enumerable.
- `select.py` — `Combinator` (B.1 structural); `choice_function`-routed pick; enumerable.
- `ignored.py` — `Combinator` (B.1 structural); identity element (constant 0 log-density).
- `null_dist.py` — `Combinator` (B.1 structural); absorbing/placeholder degenerate dist; enumerable.
- `weighted.py` — `Combinator` (B.2 latent weighting); per-observation `(x, w)` weighting of a base; enumerable.
- `optional.py` — `Combinator` (B.2 latent weighting); present-vs-missing 2-component latent mixture; enumerable.
- `truncated.py` — `Combinator` (B.3 support surgery); allowed/forbidden renormalization (log-space Z); enumerable.
- `censored.py` — `Combinator` (B.3 support surgery); tail-mass (interval) censoring.
- `survival.py` — `Combinator` (B.3 support surgery); survival/hazard reparametrization.
- `hurdle.py` — `Combinator` (B.3 support surgery); zero-hurdle + truncated-positive base.
- `zero_inflated.py` — `Combinator` (B.3 support surgery); extra point mass `pi` at 0 over a count base.
- `transform.py` — `Combinator` (B.4 change of measure) + **`Transform` sub-protocol**; change-of-variables Jacobian; enumerable; engine path.
- `exponential_tilt.py` — `Combinator` (B.4 change of measure); `exp(theta·T(x))` tilt + `Z(theta)`; enumerable.
- `finite_stochastic_transform.py` — `Combinator` (B.4 change of measure); row-stochastic Markov-kernel transform on finite support; enumerable.
