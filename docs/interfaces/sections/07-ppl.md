# 07 — Probabilistic-programming layer (`pysp/ppl/*.py`)

The PPL layer is a thin symbolic surface over `pysp.stats`: you write a model as an algebraic
expression of `RandomVariable`s with `free` holes, then `.fit(data, how=...)` *lowers* it to a
concrete `*Distribution`/`*Estimator` and routes to one of a dozen inference backends. There are
only a handful of *real* interfaces here — `RandomVariable` (the central object), the de-facto
`PosteriorResult` contract that every `.result` object follows, the **fitter** function contract
(the `how=` dispatch targets), the **Proxy** observation-model ABC, the **DynamicsOperator** ABC,
and the **forward-operator/adjoint** (`sparse_solve`) contract. Everything else is a concrete
implementation of one of these.

---

### `RandomVariable`  — de-facto contract (the central PPL object)
- **Role:** an immutable symbolic node in a model expression; carries an algebra (`_kind`) and lowers
  to a pysp distribution/estimator on demand.
- **Formalized in:** `pysp/ppl/core.py:798` (a `__slots__` class, not an ABC — should arguably be a
  Protocol so non-core families can present the same surface).
- **The `_kind` algebra** (tag in `__slots__` field `_kind`; private constructors build each):

      "sample"  RandomVariable._sample(family, args, name, keys, scope)   # a family applied to args (the workhorse)
      "bound"   RandomVariable._bound(dist, name, result)                 # wraps a concrete *Distribution (post-fit)
      "param"   _param_handle(dim, kind, support) via free(dim, ...)      # a structural vector/matrix parameter handle
      "apply"   RandomVariable._apply(base, transform)                    # a deterministic transform (exp/log/affine)
      "sum"     RandomVariable._sum(a, b)        a + b                     # convolution of independent RVs
      "prod"    RandomVariable._prod(a, b)       a * b                     # product of independent RVs
      "pow"     RandomVariable._pow(base, exp)   a ** p                    # power by a constant exponent
      "select"  RandomVariable._select(base, i)  a[i]                      # one component of a vector RV
      "given"   rv.given(event)                                           # conditioning on a region (rejection)
      "joint"   constrain(a < b, ...)                                     # several RVs under a relation

- **Construction surface:** `free` (the `_Free()` singleton at `core.py:75`; `free(dim, kind=..., support=...)`
  builds a `"param"` handle, `kind` ∈ vector/ordered/simplex/cholesky); `ordered` (`_Ordered()` singleton,
  `core.py:223`); a *concrete value or another RV placed in a constructor slot* (a prior); `Field`/`Group`/
  `_LinearPredictor` (`f * coef`, `f + g`) for regression slots. Family constructors (`Normal`, `Poisson`,
  `Mix`, `Seq`, `Markov`, `MVN`, …) live in `__init__.py` and return `"sample"` RVs.
- **Methods (operators):**

      __mul__/__add__/__sub__/__truediv__/__pow__/__neg__  -> apply/sum/prod RVs   # arithmetic algebra
      __getitem__(i) -> RV                                  # _select component
      exp(self)/log(self) -> RV                             # transform RVs
      __gt__/__ge__/__lt__/__le__/eq(other)/ne(other) -> Constraint   # build relations for constrain()
      given(self, constraint) -> RV                         # condition on a region
      each(self) -> RV                                      # broadcast a leaf family over a sequence

- **Methods (introspection — `@property`):**

      is_bound / has_free / name / scope -> bool|str
      dist        -> the lowered *Distribution (lower(self,'dist'))
      params      -> fitted parameter pytree
      result      -> the PosteriorResult object (or None for point fits)
      components  -> child RVs of a composite (mixture components, sequence body, ...)
      columns     -> per-variable list for a "joint" RV

- **Methods (sample / score / query):**

      sample(self, n=None, seed=None) -> ndarray            # per-_kind dispatch (convolve/reject/select/...)
      log_prob(self, x) -> float|ndarray                    # density; exact convolution or KDE for "sum"
      log_likelihood(self, data) -> float
      mean/var(self, samples=20000, seed=0)                 # Monte-Carlo; per-variable for "joint"
      prob(self, ...) / prob_of_event(self)                 # P(relation) / P(event) for joint/given
      predict(self, n=1, rng=None) -> ndarray               # posterior-predictive (integrates params if Bayesian)
      posterior(self, x)                                    # E-step latents (seq_posterior) OR param posterior draws
      aic/bic(self, data, k=None) ; waic/loo(self, data) -> dict ; pointwise_log_likelihood(self, data) -> (S,N)
      summary(self)                                         # delegates to result.summary() or returns .params

- **The `fit()` dispatch** (`core.py:1313`):

      fit(self, data, *, how="auto", max_its=100, delta=1e-8, backend="local",
          num_workers=None, engine=None, precision=None, print_iter=0, **kw) -> RandomVariable

  `how` ∈ {auto, em, map, mcmc, hmc, nuts, sample, ensemble, vi, vmp, conjugate, conjugate_mixture,
  hierarchical}. `auto` picks: `hierarchical` if grouped; `map` if inequality constraints; `conjugate`/
  `conjugate_mixture` if a registered conjugate pair; `map` if a structural/partial-free param; else
  `em`. Special pre-dispatch by family/slot shape routes to `regression.regression_fit` (a
  `_LinearPredictor` slot), `statespace.statespace_fit` (CompositeFamily `StateSpace`), and
  `pde.pde_fit` (CompositeFamily `PDEStateSpace`). The `em` path lowers to an estimator and calls
  `optimize(...)` (pysp's parallel/distributed EM). Every other `how` forwards to a **fitter** function
  (below) and returns a bound RV carrying a `.result`.
- **`lower(rv, target='dist'|'estimator')`** (`core.py:1578`): the single routing site symbolic→pysp.
  Caches per RV. Handles `bound` (return the dist), `apply` (wrap in `TransformDistribution`), and
  `sample` (delegate to the `Family`/`CompositeFamily` `dist_fn`/`est_fn`, recursing on child RVs).
- **Helpers:** `Family`/`CompositeFamily` (`core.py:585`/`662`) — the family registry entries
  (`make_dist`/`make_estimator`/`dist_fn`/`est_fn`/`seed_fn`); `Constraint` (`core.py:243`, `& | ~`,
  `.eval(env)`, `.contains(x)`, `.rv()`); `constrain(*constraints)` (`core.py:1517`) builds a `"joint"` RV.
- **Facets:** sampleable, scoreable, fittable; conditionable (`given`/`constrain`); transform-closed
  (apply/sum/prod/pow). Lowers to the full Distribution/Estimator contracts of §01–§06.
- **Notes:** immutable (`__setattr__` raises), pickled structurally via `__reduce__`. The structural
  vector/matrix param specs `_VectorSpec`/`_OrderedSpec`/`_SimplexSpec`/`_CholeskySpec`
  (`core.py:172–242`) are expanded into scalar inference slots by `inference._spec_slot_defs` and
  reassembled by `_spec_assemble`.

---

### `PosteriorResult`  — de-facto contract (the `.result` object)
- **Role:** the fitted-posterior / fit-summary object hung on `RandomVariable._result`; the surface
  `core.py` queries for prediction, posterior draws, model comparison, and summaries.
- **Formalized in:** *implicit — followed by convention*. **Recommend formalizing as a Protocol**
  (`pysp/ppl/_result.py` does **not** exist). The duck-typed surface `core.py` actually probes:

      summary(self) -> dict                          # used by RV.summary()           (always present in practice)
      samples(self, param=None) -> ndarray|dict      # used by RV.posterior(handle)    (Bayesian/conjugate/regression)
      mean(self, param=None)                         # posterior mean(s)
      predictive: Callable[[int, RandomState], ndarray]   # attr; used by RV.predict()  (set by fitter)
      pointwise_log_likelihood(self, data) -> (S,N)  # used by RV.waic()/loo()         (only Posterior/LocationScale)
      build: Callable[[dict], Distribution]          # attr; rebuild a dist from values (Posterior only)
      acceptance_rate: float | None                  # diagnostic; None for non-MCMC

- **Implemented by** (file:line, the concrete results):
  - `Posterior` (`inference.py:68`) — MCMC/HMC/NUTS/ensemble/VI; the most complete impl (all of the
    above + `rhat`/`ess`/`n_chains`). The reference for the protocol.
  - `ConjugatePosterior` (`inference.py:1078`) and `ConjugateMixturePosterior` (`inference.py:1322`)
    — exact closed-form; `samples(n,rng)`/`mean`/`summary`, no `pointwise_log_likelihood`/`build`.
  - `HierarchicalPosterior` (`inference.py:1430`) — per-group means/vars + hyper; `samples`/`summary`.
  - `_VIResult` (`inference.py:1607`) — a lightweight ELBO/mean/std metadata holder wrapped *inside* a
    `Posterior` after VI; not itself the `.result`.
  - `RegressionResult` (`regression.py:50`), `LMMResult` (`regression.py:218`),
    `LocationScaleResult` (`regression.py:352`) — GLM/LMM/heteroskedastic; coefficient `samples`,
    `predict(given)`, `summary`; `RegressionResult.to_exponential_family()`.
  - `GraphResult` (`vmp.py:193`) — VMP factor-graph node posteriors (`posterior(rv)`/`samples(rv)`).
  - `MixtureVMPResult` (`vmp.py:426`) — VB Gaussian-mixture (`weights`/`components`/`responsibilities`/
    `elbo_trace`/`summary`).
  - `StateSpaceResult` (`statespace.py:18`), `PDEStateSpaceResult` (`pde.py:25`) — Kalman/RTS/EM
    smoothed states + `summary`.
  - `FieldPosterior` (`field.py:519`) — Laplace/GN/VI field posterior; `mean/cov/sd/posterior(node)`,
    `field_posterior`, `sample(...)`, `summary` (the field analogue, queried directly not via `.result`).
- **Facets:** summarizable, posterior-sampleable, predictive, model-comparison-providing.
- **Notes:** `predictive` and `build` are *attributes set by the fitter*, not methods on the class —
  another argument for a documented Protocol so the optional-vs-required split is explicit.

---

### Fitter  — de-facto contract (the `how=` dispatch targets)
- **Role:** the inference backends; each takes the model RV + data and returns a bound RV carrying a
  `.result`. This is the table `RandomVariable.fit(how=...)` dispatches into.
- **Formalized in:** *implicit*. The shared signature:

      fitter(rv: RandomVariable, data, **kw) -> RandomVariable    # bound RV with ._result set

- **Members** (file:line → `how`):
  - `inference.map_fit` (1569) → `map`; L-BFGS (Torch analytic grad via `autograd.grad_target`) /
    Nelder-Mead fallback. No `.result` by default.
  - `inference.mcmc_fit` (909) → `mcmc` (adaptive RW-Metropolis); `hmc_fit` (950) → `hmc`;
    `nuts_fit` (1010) → `nuts`; `ensemble_fit` (854) → `ensemble` (Goodman–Weare); `sample_fit`
    (1067) → `sample` (auto ensemble≤12 params else nuts). All return `Posterior`. Common kwargs:
    `draws/burn/thin/chains/parallel/constraints/penalty/rng`.
  - `inference.vi_fit` (1622) → `vi` (ADVI; `family` meanfield/fullrank, Rényi `alpha`). → `Posterior`.
  - `inference.conjugate_fit` (1235) / `conjugate_mixture_fit` (1396) / `hierarchical_fit` (1549) →
    exact/EM Bayes. Spec detectors `conjugate_spec` (1214) / `conjugate_mixture_spec` (1360) gate `auto`.
  - `vmp.vmp_fit` (357) → `vmp` (nested-Normal factor graph → `GraphResult`); `vmp.mixture_vmp` (475)
    is a *standalone* `(data, K, ...)` function (not the `(rv, data)` shape) → `MixtureVMPResult`.
  - `regression.regression_fit` (545) — GLM (IRLS) / penalized (coord-descent) / LMM / quantile;
    pre-dispatched by a `_LinearPredictor` slot. → `RegressionResult`/`LMMResult`/`LocationScaleResult`.
  - `statespace.statespace_fit` (115) and `pde.pde_fit` (175) — Kalman/RTS + EM; pre-dispatched by
    CompositeFamily name.
- **Notes:** constraints/penalty are honored only by `map`/`mcmc`/`hmc`/`nuts`/`ensemble` (the others
  raise). `autograd.grad_target(rv, data)` (`autograd.py`) returns a `GradTarget` (`value_and_grad`,
  `advi`, `unpack`, `build`) used by `map_fit`/`vi_fit` when Torch is present. **Recommend formalizing
  the fitter signature as a Protocol** and a small registry instead of the inline `if how == ...` chain.

---

### `Proxy`  — ABC (the observation-model / likelihood contract for field inference)
- **Role:** one observation term in a joint latent-field model; maps a latent field + params to a
  log-likelihood (and optionally a Gaussian residual for Gauss-Newton).
- **Formalized in:** `pysp/ppl/field.py:357` (a base class with `NotImplementedError` stubs — a real
  ABC-by-convention).
- **Methods:**

      prefix: str ; field: str | None                       # namespace + which FieldSystem field
      params(self) -> list[_ParamSpec]                       # declare latent params (name, shape, support, init)
      loglik(self, field_t, params: dict, torch) -> tensor   # the negative loss (required)
      residual(self, field_t, params: dict, torch) -> tensor|None   # standardized resid for how='gauss_newton'

- **Implemented by:** `GaussianProxy`, `LogisticNicheProxy`, `PoissonProxy`, `CustomProxy` (all
  `field.py`); `_DifferentialProxy` (ODE/PDE forward, built by `inverse.Differential`); `_PenaltyProxy`
  (`priors.py:26`, the `TotalVariation`/`Potts` prior terms). Higher-level builders `Niche`, `Cox`,
  `GaussianField` return `(field, proxy)` pairs.
- **Facets:** differentiable-scoreable; composes into `joint([...])` (`field.py:1121`) → `FieldModel`
  → `fit_field(field, proxies, how='map'|'laplace'|'gauss_newton'|'vi')` → `FieldPosterior`.
- **Notes:** the `(field, proxy)` tuple is the lingua franca tying GP/GMRF fields (`FieldKernel`:
  `GP`/`RBF`/`AnisotropicRBF`/`GreatCircleRBF`/`GreatCircleMatern`/`RandomWalk`) to observations.

---

### `DynamicsOperator`  — ABC (method-of-lines spatial operator)
- **Role:** a discretized PDE spatial operator `G` plus its one-step time transition `A`, for the
  state-space PDE stack.
- **Formalized in:** `pysp/ppl/dynamics.py:89` (true `ABC`).
- **Methods:**

      __init__(self, n, length=1.0, bc="neumann", scheme="implicit"|"explicit"|"exact")
      @abstractmethod operator_matrix(self) -> ndarray       # the (n,n) spatial operator G (du/dt = G u)
      transition_matrix(self, dt) -> ndarray                 # A: implicit (I-dtG)^-1 / explicit I+dtG / expm(dtG)

- **Implemented by:** `DiffusionOperator`, `AdvectionOperator`, `AdvectionDiffusionOperator`
  (`dynamics.py:122/135/148`). Registry: `register_dynamics_operator(name, factory)`,
  `available_dynamics_operators()`, `make_operator(name, **kw)`.
- **Facets:** plug-in (registry); consumed by `pde.kalman_rts_em` / `pde_fit`.

---

### Forward-operator / adjoint  — de-facto contract (`sparse_solve` + grid assembly)
- **Role:** the differentiable PDE forward solve with an adjoint backward, enabling large-scale
  PDE-constrained Bayesian inversion (cost of the gradient is one extra factorization, independent of
  parameter count).
- **Formalized in:** *implicit* — `pysp/ppl/pde_solve.py`. The contract a forward operator exposes:

      sparse_solve(vals, rows, cols, n, b) -> u              # solve A u = b, A=sparse(rows,cols,vals); adjoint grads to vals,b
      _matvec(rows, cols, vals, n, x, torch) -> A x          # apply without solve (grads to vals,x)
      # grid assemblers return (rows, cols, vals, n): fixed integer pattern + differentiable values
      divergence_form(kappa, shape, *, spacing) -> (rows,cols,vals,n)        # -div(kappa grad u)
      helmholtz_operator(slowness2, shape, *, omega, spacing) -> (...)        # -lap u - omega^2 slowness2 u (complex)
      laplacian(shape, *, spacing) -> (...)
      _integrate_ops(rhs, y0, t_grid, torch, method)         # forward time integration
      _integrate_record(step, y0, n_steps, record, torch, checkpoint)   # checkpointed adjoint-state time loop

- **The `_Ops` namespace** (`ops.py`, `make_ops()`): the backend-agnostic façade handed to user
  `forward`/`rhs`/`observe`/`p` callbacks — math (`exp/log/sin/sqrt/clamp/sum/stack/matmul/solve`),
  the grid assemblers + `sparse_solve`/`matvec`, `integrate`/`integrate_record`, and
  `heaviside`/`level_set` for shape inference. Keeps physics decoupled from Torch.
- **Entry point:** `inverse.Differential(y, *, forward|rhs, y0, t_grid, method, observe, drivers, over,
  scale, family)` → `(field, _DifferentialProxy)` consumed by `joint([...]).fit(how=...)`.
- **Facets:** differentiable, adjoint-capable, complex-valued (Helmholtz/FWI). **Recommend a documented
  ForwardOperator Protocol** (`apply`/`adjoint`/grid-assembly) since the contract is currently spread
  across free functions.
- **Notes:** concrete physics solvers `NavierStokes2D` (`flow.py`), `WaveEquation2D` (`wave.py`),
  `solve_wave_pml` (`wave_pml.py`), `solve_poisson`/`CoupledPDESystem`/`solve_elasticity`
  (`multiphysics.py`), `fem_poisson` (`fem.py`), `shape_optimize`/`level_set_material` (`shape.py`)
  are implementations, not interfaces.

---

### Constraints & structural parameters  — supporting algebra
- **`Constraint`** (`core.py:243`): the relation object from RV comparisons; `& | ~` combine,
  `.eval(env)` evaluates over sampled columns, `.contains(x)` masks, `soft`/`residual` carry the
  penalty form. Helpers exported from `__init__.py`: `compare`, `eq`/`equal`/`ne`, `increasing`/
  `decreasing`/`monotone`, `convex`/`concave`/`lipschitz`, `ode_residual`. `constrain(...)` →
  `"joint"` RV; `.given(event)` → `"given"` RV (rejection).
- **Vector/matrix param specs** (`core.py:172–242`): `_VectorSpec(dim, support)`, `_OrderedSpec(dim)`,
  `_SimplexSpec(alpha, rows)`, `_CholeskySpec(dim)` — built via `free(dim, kind=...)`, expanded to
  scalar inference slots by `inference._spec_slot_defs` / reassembled by `_spec_assemble`.

---

### Coverage checklist (all 25 modules)

    core.py          — RandomVariable (the central de-facto contract) + _kind algebra, free/ordered, Constraint, constrain, lower(), fit() dispatch, Family/CompositeFamily registry, param specs.
    inference.py     — Fitter members map/mcmc/hmc/nuts/ensemble/sample/vi/conjugate/conjugate_mixture/hierarchical; PosteriorResult impls Posterior/Conjugate*/Hierarchical/_VIResult; conjugate_spec detectors.
    vmp.py           — Fitter vmp_fit + standalone mixture_vmp; PosteriorResult impls GraphResult, MixtureVMPResult (variational message passing / VBEM).
    regression.py    — Fitter regression_fit (GLM/IRLS, penalized, LMM, quantile); PosteriorResult impls RegressionResult/LMMResult/LocationScaleResult.
    field.py         — Proxy ABC + concrete proxies; FieldKernel/GP/RBF/... ; fit_field fitter; joint()/FieldModel; PosteriorResult impl FieldPosterior.
    inverse.py       — Differential() builder → (field, _DifferentialProxy); the ODE/PDE inverse-problem entry point onto the forward/adjoint contract.
    dynamics.py      — DynamicsOperator ABC + Diffusion/Advection/AdvectionDiffusion impls + operator registry (method-of-lines).
    pde_solve.py     — Forward-operator/adjoint contract: sparse_solve (adjoint grads), divergence_form/helmholtz_operator/laplacian assemblers, time integrators.
    ops.py           — _Ops backend-agnostic namespace (make_ops): math + grid assembly + sparse_solve + integrate + level_set; handed to user PDE callbacks.
    priors.py        — _PenaltyProxy (a Proxy impl) + TotalVariation/Potts prior-term builders → (field, proxy).
    pde.py           — PDEStateSpaceResult (PosteriorResult) + kalman_rts_em/pde_fit/fit_diffusivity/fit_pde_parameters/fit_reaction_diffusion (PDE state-space inference).
    statespace.py    — StateSpaceResult (PosteriorResult) + statespace_fit (univariate Kalman/RTS/EM); fit() pre-dispatch target.
    flow.py          — NavierStokes2D concrete forward solver (implementation of the physics layer).
    wave.py          — WaveEquation2D concrete forward solver (implementation).
    wave_pml.py      — solve_wave_pml concrete forward solver with PML absorbing boundary (implementation).
    fem.py           — fem_poisson/boundary_nodes concrete P1 finite-element solver (implementation).
    multiphysics.py  — solve_poisson/CoupledPDESystem/solve_elasticity concrete nD/coupled solvers (implementation).
    shape.py         — level_set_material/shape_optimize concrete level-set shape optimization (implementation).
    autograd.py      — GradTarget/MixtureGradTarget + grad_target(rv,data): Torch value_and_grad/advi/build helper for map_fit/vi_fit (infra, not a public contract).
    diagnostics.py   — waic/psis_loo/loo_stacking_weights/loo_stack model-comparison functions (consumed by RV.waic/loo).
    conformal.py     — conformal() + ConformalRegressor/Classifier/QuantileRegressor/Structure/LinkPredictor/KnowledgeGraph (split-conformal wrappers; interval/covers/predict_set surface).
    training_data.py — ModelingExample + generate_examples/families/build_model_from_code/fit_example (synthetic labels for an LLM-writes-pysp dataset; not a runtime interface).
    benchmark.py     — torch/pysp MLE+MCMC timing harness (bench_*); not an interface.
    benchmark_vs.py  — cross-library benchmark vs numpyro/emcee/pyro (task_*); not an interface.
    __init__.py      — the public PPL API surface (__all__): free/ordered/constrain/constraint helpers, family constructors, fit_field/Proxy/kernels, PDE/shape/Differential, conformal.
