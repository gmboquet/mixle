# 08 — Application + top-level layers

Scope: `pysp/models/*`, `pysp/doe/*`, `pysp/uq/*`, `pysp/infer/*`, `pysp/data/*`,
`pysp/relations.py`, `pysp/planner.py`, and the stats top-level modules
(`composition`, `hierarchical`, `temporal`, `errors_in_variables`, `max_stable`,
`block_gibbs`, `exp_family`, `sampling_api`, `_sampling`).

These are the **outer rings** of pysp: optimization-as-distribution (`relations`),
resource planning + the encoded-data backend registry (`planner`), and three
plugin registries (acquisitions / criteria, inference backends, encoded-data
backends). The application `models/` package is deliberately heterogeneous —
some wrappers honor the core Distribution/Estimator contract, most are
sklearn-style or plain inference objects (the "thin wrappers in app code" rule).

---

## A. Optimization-as-distribution (`relations.py`)

### `Relation` — **ABC** (real `abc.ABC`)
- **Role:** A constraint over a structured space whose members are *enumerated ranked by a residual*; the value is the whole ranked set, not the single optimum.
- **Formalized in:** `pysp/relations.py:181` (`class Relation(ABC)`).
- **Class attribute:** `sense: str = "min"` (`"min"` → members out in increasing cost; `"max"` → decreasing score).
- **Methods:**
    ```
    enumerator(self, k: int | None = None) -> Iterator[Solution]   # @abstractmethod; yield best-first, ≤k
    solve(self) -> Solution | None                                 # = next(enumerator(k=1), None)
    top(self, k: int) -> list[Solution]                            # = list(enumerator(k=k))
    sampler(self, seed=None, *, temperature=1.0, k=None,
            uniform=False, rng=None) -> RelationSampler            # Gibbs measure over members
    __iter__(self) -> Iterator[Solution]                           # = enumerator()
    ```
- **Item type:** `Solution(NamedTuple)` = `(value, objective)` (`relations.py:61`) — `value` is the member (assignment / string / state-seq / subset), `objective` its residual.
- **Implemented by:** `ShortestPath`, `Assignment` (Murty `k_best_assignments`), `SpanningTree` (Gabow `k_best_spanning_trees`), `EditDistance` (Dijkstra ball via `nearest_first`), `ViterbiPath` (k-best HMM paths), `BestSubsetRegression` (exhaustive AIC/BIC/RSS). Each delegates `enumerator` to whatever engine fits.
- **Facets:** **enumerable** (rank-from-index by residual), **samplable** (via `RelationSampler`). It is the *dual* of the Distribution sampler/estimator/enumerator triple: you specify the relation, ask for `enumerator()`, and it hands back a sampler — the same shape as a distribution.
- **Notes:** This is the canonical "optimization-as-distribution" surface — formal ABC, mirror of the core enumerator facet. The two shared low-level engines are module-level functions, not methods:
    ```
    best_first_paths(start, successors, is_goal=None, *, sense="min",
        heuristic=None, max_results=None, return_paths=True) -> Iterator[(path|state, total)]   # relations.py:71  (A* / k-best paths)
    nearest_first(start, neighbors, *, key=None, max_distance=None,
        max_results=None) -> Iterator[(state, distance)]                                        # relations.py:136 (Dijkstra outward ball)
    ```

### `RelationSampler` — de-facto sampler (mirrors `DistributionSampler`)
- **Role:** Draw members of a `Relation` under a Gibbs measure `exp(±objective / temperature)`.
- **Formalized in:** `pysp/relations.py:239` (constructed via `Relation.sampler`).
- **Methods:** `sample(self, size: int | None = None) -> Any` — one member value (`size=None`) or a list. Enumerates once, lazily, caches the categorical. `temperature→0` = point mass on optimum; `→inf`/`uniform=True` = uniform. Exact Gibbs only when finite and fully enumerated (`k=None`).
- **Facets:** **samplable**. **Notes:** the relation itself is a specification, not a random object, so it hands back a sampler that owns the RNG and the Gibbs measure — the same convention as every other pysp object.

---

## B. Resource planning + encoded-data registry (`planner.py`)

### `EncodedDataHandle` — **de-facto contract** (duck-typed orchestrator protocol)
- **Role:** The orchestrator interface `pysp.stats` consumes for distributed/local EM — local, multiprocessing, MPI, Spark, dask, torchrun, Lightning, Ray, and future handles implement it without shared inheritance.
- **Formalized in:** `pysp/planner.py:583` (base class documents the contract; `is_encoded_data_handle(obj)` is the structural check at `:618`).
- **Methods:**
    ```
    pysp_seq_log_density_sum(self, estimate) -> tuple[float, float]      # (num_obs, summed_log_density)
    pysp_seq_estimate(self, estimator, prev_estimate) -> Any            # one E-step fold + M-step
    pysp_seq_initialize(self, estimator, rng: RandomState, p: float) -> Any
    pysp_stream_accumulate(self, estimator, model) -> tuple[float, Any]  # folded suff-stats for streaming EM
    close(self) -> None ; __enter__/__exit__                            # context manager
    ```
- **Implemented by:** `LocalEncodedData` (`:822`, in-process; optional thread-pool over chunks), `SparkEncodedData` (`:1043`, RDD), `DaskEncodedData` (`:1137`, distributed futures); plus lazily-imported `MPEncodedData`, `MPIEncodedData`, `TorchRunEncodedData`, `LightningEncodedData`, `RayEncodedData` (in `pysp.utils.parallel.*`, registered here).
- **Facets:** **engine-resident** data movement + global sufficient-statistic folding. **Notes:** should be formalized as a `Protocol` — it is the cleanest implicit contract in this layer (four `pysp_seq_*` methods checked by `is_encoded_data_handle`).

### Encoded-data backend registry — **registry** ("register, don't branch")
- **Role:** Map a backend name → factory `factory(data, **params) -> EncodedDataHandle`; the extension point for new parallel/distributed frameworks.
- **Formalized in:** `pysp/planner.py` — `_ENCODED_DATA_BACKENDS: dict[str, Any]` (`:692`).
- **Surface:**
    ```
    register_encoded_data_backend(name: str, factory, aliases: tuple[str,...] = ()) -> None   # :695
    available_encoded_data_backends() -> list[str]                                              # :710
    encoded_data(data, estimator=None, model=None, encoder=None, placement=None,
        resources=None, engine=None, precision=None, num_chunks=None, sub_chunks=1,
        backend="local", num_workers=None, client=None, comm=None, root=0,
        root_only=False, parallel_chunks=False, chunk_workers=None) -> EncodedDataHandle        # :631 (dispatch)
    is_encoded_data_handle(obj) -> bool                                                          # :618
    ```
- **Built-ins registered at import:** `local`, `mp`/`multiprocessing`, `mpi`, `spark`, `dask`, `torchrun`, `lightning`/`pl`, `ray`.

### Planning data model — **plain dataclasses** (advisory placement)
- `DeviceSpec` (`:49`, frozen) — one placement target; `is_gpu`, `to_dict`/`from_dict`.
- `Resources` (`:89`) — collection of devices; classmethods `single_cpu`/`local`/`discover`/`from_specs`/`from_mpi`/`from_dask`/`from_spark`/`from_torchrun`; `fastest()`, JSON `to/from_json`, `save`/`load`.
- `CalibrationRecord` (`:358`, frozen) + `CalibrationCatalog` (`:408`) — append-only calibration store; `latest(...)`, `resources_for(...)`.
- `PlacementShard` (`:471`, frozen), `Placement` (`:493`, printable; `for_device`, `to_dict`), `ModelShard` (`:555`, frozen, component-axis).
- **Functions:** `plan(...) -> Placement` (`:1381`), `model_sharding_plan(model, resources, ...) -> tuple[ModelShard,...]` (`:1456`), `estimate_model_nbytes`, `estimate_estimator_stat_nbytes`, `calibrate_resources`.
- **Notes:** advisory only — orchestrators own actual data movement; the planner just estimates memory pressure and prints an editable placement.

---

## C. Design of Experiments (`doe/`)

Two plugin registries (acquisition + criterion) plus result dataclasses; all
results are **frozen dataclasses**, and the GP surrogate is a **duck contract**.

### Acquisition function — **de-facto contract** + **registry**
- **Role:** Merit score over candidate points from predictive `(mean, std)` and incumbent `best`.
- **Duck contract:** `fn(mean, std, best, *, maximize: bool, **params) -> np.ndarray` (scores maximized over candidates).
- **Formalized in:** `pysp/doe/bayesopt.py`. Built-ins: `expected_improvement` (`:30`, `xi`), `probability_of_improvement` (`:?`, `xi`), `upper_confidence_bound` (`kappa`).
- **Registry surface:** `register_acquisition(name, fn, aliases=())` (`:92`), `available_acquisitions() -> list[str]` (`:?`). Names: `ei`/`expected_improvement`, `pi`/`probability_of_improvement`, `ucb`/`lcb`/`cb`/`confidence_bound`/`upper_confidence_bound`.

### Optimal-design criterion — **de-facto contract** + **registry**
- **Role:** Scalar merit of an information matrix `M = FᵀF`.
- **Duck contract:** `fn(info: np.ndarray, *, ref: np.ndarray | None = None) -> float`.
- **Formalized in:** `pysp/doe/optimal.py`. Built-ins: `d_criterion` (logdet M), `a_criterion` (−tr M⁻¹), `i_criterion` (−mean predictive variance).
- **Registry surface:** `register_criterion(name, fn, aliases=())` (`:99`), `available_criteria()` (`:?`). Names: `d`/`d_optimal`/`det`, `a`/`a_optimal`/`trace`, `i`/`i_optimal`/`iv`.
- **Driver:** `optimal_design(bounds, n, *, candidates=None, model=None, criterion="D", n_candidates=256, n_restarts=5, max_iter=100, ref=None, seed=None) -> np.ndarray` (Fedorov exchange). Model matrix is a `ModelMatrix = Callable[[ndarray],(n,p)]`; `polynomial_features(degree=1, *, bias=True) -> ModelMatrix`.

### GP-surrogate — **duck contract**
- **Role:** Bayesian-optimization surrogate; reuses `pysp.models.gaussian_process.GaussianProcessRegressor`.
- **Expected surface (BO calls):** `fit(x, y, **fit_kwargs) -> ...` and `predict(x_obs, y_obs, x_pred, return_cov: bool) -> (mean, cov) | mean`.
- **Notes:** implicit — any object with this `fit`/`predict` pair can be passed via the `gp=` argument. Should be a `Protocol`.

### Ask-tell optimizer + functional drivers
- `BayesianOptimizer` (`optimizer.py:32`) — stateful ask-tell wrapper over the functional API.
    ```
    __init__(bounds, *, acq="ei", acq_kwargs=None, maximize=False, n_init=None,
             xi=0.0, n_candidates=512, fit_kwargs=None, seed=None)
    ask(self, q: int = 1) -> np.ndarray ; tell(self, x, y) -> BayesianOptimizer
    properties: x, y, n_observations, best -> BayesOptResult
    ```
- Functional BO (`bayesopt.py`): `propose_next(...)`, `propose_batch(...)` (kriging-believer), `minimize(objective, bounds, n_init=5, n_iter=15, ...) -> BayesOptResult`.
- Constrained (`constrained.py`): `propose_next_constrained`, `constrained_minimize(objective, constraints, bounds, ...) -> ConstrainedBayesOptResult`, `probability_of_feasibility(mean, std)`.
- Multi-objective (`multiobjective.py`): `multi_minimize(objectives, bounds, ..., rho=0.05) -> MultiObjectiveResult` (Tchebycheff), `pareto_mask(y)`.
- **Result dataclasses** (all frozen; *there is no shared `OptimizationResult` base* — each driver has its own):
  `BayesOptResult{best_x, best_y, x, y}`, `ConstrainedBayesOptResult{...,c, feasible}`, `MultiObjectiveResult{x, y, pareto_mask, pareto_x, pareto_y}`.
- **Space-filling designs** (`designs.py`, `Bounds = Sequence[(low,high)]`): `random_design`, `latin_hypercube(..., center=False)`, `maximin_latin_hypercube(..., trials=32)`, `sobol_design(..., scramble=True)`, `halton_design(..., scramble=True)`, `full_factorial(bounds, levels)`.

---

## D. Uncertainty quantification (`uq/`)

Purely **functional** — no Distribution/Estimator objects; one result class.

### Forward propagation — **standalone functions** (`propagate.py`)
- `propagate(func, mean, cov=None, *, n=10000, method="montecarlo", quantiles=(.05,.5,.95), seed=0) -> dict` — MC propagation.
- `unscented_transform(func, mean, cov, *, alpha=1e-3, beta=2.0, kappa=0.0) -> (mean_out, cov_out)` — sigma-point propagation.

### Global sensitivity — **standalone functions** (`sensitivity.py`)
- `sobol_indices(func, bounds, n=4096, *, seed=0, names=None) -> dict` — first- & total-order variance indices.
- `morris_screening(func, bounds, *, trajectories=20, levels=4, seed=0, names=None) -> dict` — elementary-effects screening.

### Calibration — **result class + driver** (`calibration.py`)
- `KOCalibration` (`:26`) — Kennedy-O'Hagan fitted model; `predict(x_new, *, with_discrepancy=True) -> ndarray`. **Standalone** (not the Distribution contract).
- `calibrate(simulator, x, y, theta0, *, discrepancy=True, discrepancy_lengthscale=None, seed=0, max_iter=300) -> KOCalibration`.

---

## E. Inference backends + diagnostics (`infer/`)

### `InferenceBackend` — **de-facto contract** (frozen dataclass) + **registry**
- **Role:** Pluggable MCMC engine description; abstracts engine selection from the public `nuts` facade.
- **Formalized in:** `pysp/infer/backends.py:40` (frozen dataclass).
- **Fields/methods:**
    ```
    name: str ; available: Callable[[], bool] ; target_kind: str  # "numpy_vg"|"njit_vg"|"torch_logp"|"jax_logp"
    nuts: Callable[..., NutsResult]
    ```
- **Registry surface:** `register_inference_backend(backend)` (`:53`), `get_inference_backend(name)` (`:58`), `available_backends() -> list[str]` (`:67`), `select_backend(backend="auto", target=None) -> str` (`:72`).
- **Built-ins:** `numpy` (always), `numba`, `torch`, `jax` (each self-registers when its engine imports).

### Public sampling facade + results (`infer/__init__.py`)
- `nuts(target, *, backend="auto", dim=None, init=None, num_samples=1000, warmup=1000, chains=1, mass=1.0, target_accept=0.8, max_tree_depth=10, thin=1, rng=None, parallel=None, **backend_kwargs) -> NutsResult`.
- `nuts_torch(logp, *, ...) -> NutsResult` (torch-native, GPU); `advi(target_batch, u0, s0, *, samples=1000, mc=16, steps=2000, lr=0.05, batch_size=None, family="meanfield", alpha=1.0, rng=None) -> AdviResult`.
- **Result dataclasses (frozen):** `NutsResult{samples, chains, rhat, ess, num_target_evals, step_size, extra}`, `AdviResult{samples, mean, scale, objective}`.

### Diagnostics — **standalone array utilities** (`diagnostics.py`)
- `rhat(chains) -> np.ndarray` (Gelman-Rubin, per-dim), `ess(samples, max_lag=None) -> np.ndarray` (effective sample size). Sampler-agnostic.

---

## F. Application models (`models/`)

The thin-wrapper layer. Three contract families coexist (per the orthogonal axes):
**(a)** full Distribution+Estimator, **(c)** sklearn-style `fit/predict`, **(d)**
plain inference/result objects. Most carry a `*FitResult`/`*Result` dataclass.

### Distribution/Estimator-contract models — **core contract** (rare in this layer)
- **`RandomForestConditional` / `RandomForestEstimator` / `RandomForestEncoder`** (`random_forest.py`, `_forest.py`) — the only model that fully honors the contract: `density`/`log_density`/`seq_log_density`, `sampler()`, `estimator()`, `dist_to_encoder()` on the conditional; `accumulator_factory()`, `estimate(nobs, suff_stat) -> RandomForestConditional` on the estimator; conditional `p(y|x)` (classification/regression). **Facets:** score, sample, estimate, engine-resident.
- **`TruncatedDirichletProcessMixtureModel`** (`dirichlet_process_mixture.py`) — Distribution-side surface (`log_density`, `density`, `component_log_density`, `responsibilities`, `effective_components`, `sample`) but is *fitted by a standalone function* `fit_truncated_dpm(...) -> TruncatedDirichletProcessMixtureFitResult` (variational CAVI), not a bound estimator. Aliases `TruncatedDPMModel`/`TruncatedDPMFitResult`.

### sklearn-style wrappers — **standalone `fit/predict`** (the dominant pattern here)
- **`GaussianProcessRegressor`** (`gaussian_process.py`) — `fit(x, y, max_its=500, lr=0.05, optimizer="adam", ..., return_result=False)`, `predict(x_train, y_train, x_new, return_cov=False)`, `log_marginal_likelihood`, `kernel`, `predict_monotone` (PAVA). Reused as the BO surrogate.
- **`SparseGaussianProcessRegressor`** (`sparse_gaussian_process.py`) — FITC inducing-point GP; `fit(x, y, *, optimize=True, seed=0, max_iter=100) -> self`, `predict(x_new, *, return_var=False)`.
- **`GaussianRegressionNeuralNetwork` / `CategoricalClassificationNeuralNetwork` / `PoissonRegressionNeuralNetwork`** (`neural.py`) — torch-module wrappers; `parameters()`, `log_likelihood(x, y)`, `fit(...)`, `predict`/`predict_proba`/`predict_rate`. Helper `make_mlp(...)`. Back-compat aliases `*NN`.
- **`TransEKnowledgeGraphModel`** (`knowledge_graph.py`) — embedding model; classmethod `random(...)`, `score_triples`/`distance_triples`, `margin_loss`, `negative_sample`, `fit_margin(...) -> KnowledgeGraphFitResult`.

### Plain inference / generative models — **(d)** (no contract, `fit_mle`/`sample`/inference)
- **`ErdosRenyiGraphModel` / `StochasticBlockGraphModel`** (`random_graph.py`) — `log_likelihood`, `sample`, `bic`; classmethod `fit_mle(...)`; standalone `hard_em_stochastic_block_model(...) -> HardEMResult`.
- **`PartiallyObservableMarkovDecisionProcessModel`** (`partially_observable_markov_decision_process.py`) — `belief_update`, `filter -> *FilterResult`, `sequence_log_likelihood`, `forward_backward`, `predict_observation`, `expected_reward`, `sample`; standalone `baum_welch_pomdp(...) -> *FitResult`. Aliases `POMDP*`.

### Result-only / function modules — **(d) dataclasses + standalone functions**
- **`grammar.py`** — no class of its own; operates on `HeterogeneousPCFGDistribution` (from `pysp.stats`). Dataclasses `GrammarLearningResult`, `PCFGParseNode`; functions `fit_induced_pcfg(...) -> GrammarLearningResult`, `pcfg_log_likelihood`, `viterbi_parse`, `grammar_rule_table`.
- **`dependence.py`** — causal-discovery utilities; dataclasses `ConditionalIndependenceResult`, `CausalSkeleton`, `PartiallyDirectedGraph`; functions `gaussian_partial_correlation`, `gaussian_conditional_independence`, `discrete_conditional_mutual_information`, `learn_pc_skeleton`, `orient_v_structures`.

**Cross-cutting note:** there is **no shared `FitResult`/`OptimizationResult` base** in `models/` — each model defines its own `*FitResult` dataclass (`model`, `history`, sometimes `responsibilities`/`validation_history`). Recommend a small common `FitResult` Protocol (`.model`, `.history`) if these are ever consumed generically.

---

## G. Data adapters (`data/`)

All **standalone adapters/encoders** — bridge external containers to the stats API.

- `dataframe.py` — `dataframe_records(df, fields=None, as_dict=False) -> list`, `seq_encode_dataframe(df, fields=None, encoder=None, estimator=None, model=None, num_chunks=1, chunk_size=None)` (routes through `seq_encode`). Pandas → pysp records/encoded data.
- `graph_data.py` — `GraphObservation` (frozen dataclass: `adjacency`, `block_assignments`); `GraphDataEncoder(DataSequenceEncoder)` — `seq_encode(x) -> tuple[GraphObservation,...]`, `nbytes`. **Realizes the `DataSequenceEncoder` contract** (the only data-package class that does).
- `rdd_sampler.py` — `take_sample(rdd, with_replacement, n, seed=None)`, `sample_seq_as_rdd(sc, dist, seq_len, count_per_split, num_splits, seed=None)`, `sample_rdd(sc, dist, count_per_split, num_splits, seed=None)`. Distributed-sampling helpers over Spark.

---

## H. Stats top-level modules

### Distribution/Estimator-contract families (full triple)
- **`composition.py`** — `AitchisonNormalDistribution` (logratio-normal on the simplex via `ilr`): `density`/`log_density`/`seq_log_density`, `mean_composition`, `sampler()`, `estimator()`, `dist_to_encoder()`; `AitchisonNormalEstimator`/`Sampler`/`Accumulator`/`DataEncoder`. Plus simplex helpers `closure`, `clr`/`clr_inv`, `ilr_basis`, `ilr`/`ilr_inv`. **Wraps the Gaussian** internally on ilr coordinates.
- **`hierarchical.py`** — `HierarchicalNormalDistribution` (two-level `y[g,i]~N(θ_g,σ²)`, `θ_g~N(μ,τ²)`): full contract plus `group_posterior(ybar, n) -> (mean,sd)`, `shrinkage(n)`. `HierarchicalNormalEstimator` does an EM fit in `estimate`.
- **`temporal.py`** — `PeriodicTimeDistribution` (von Mises on cycle phase): full contract; helpers `to_unix_seconds`, `cyclic_phase`, constant `PERIODS`.

### Stats modules that **break** the contract (the flagged leaks)
- **`temporal.py` — `SeasonalTimeSeries`** (`:220`): a **non-Distribution class carrying a stateful `fit()`** — `fit(times, values) -> self`, then `mean`, `conditional -> GaussianDistribution`, `log_density`, `sampler`, `decompose`. This is the **fit() leak** the spec flags: an sklearn-style mutate-and-return `fit` living in `pysp.stats` instead of the `estimator()`/`estimate()` split (no `dist_to_encoder`/accumulator). It is an *instance-method* fit (not a `@classmethod fit`), so it doesn't violate the harder "no `@classmethod fit`" rule, but it does break the domain-neutral Distribution convention and should be split into an estimator or moved to app code.
- **`max_stable.py` — `SmithMaxStable`** (`:19`): a **spatial process, intentionally not a Distribution** (no tractable full likelihood). Surface: `extremal_coefficient(h)`, `bivariate_cdf(z1,z2,h)`, `sampler(locations, seed) -> SmithMaxStableSampler` (Schlather). Fitted by standalone `fit_smith_maxstable(locations, fields)` (madogram/composite). Honest non-leaf exception.

### Standalone fit-utilities + result classes
- **`errors_in_variables.py`** — `deming_regression(x, y, variance_ratio=1.0) -> DemingFit`; `DemingFit` result class (`slope`, `intercept`, `variance_ratio`, `x_latent`, `conditional_mean(x_star)`). Standalone regression, not a Distribution.

### Block-inference dispatch — **de-facto block contract**
- **`block_gibbs.py`** — per-block update protocol. `ConjugateBlock(name, draw)` and `MetropolisBlock(name, log_conditional, scale=0.5)` both expose `name`, `kind`, `update(state: dict, rng) -> Any` (Metropolis adds `acceptance_rate`). `BlockGibbs(blocks, init).run(n_samples=2000, *, burn=500, seed=None) -> dict[str, ndarray]` dispatches per block by `kind`. **Implicit "inference block" contract** (`update(state, rng)`) — candidate for a `Protocol`.

### Exponential-family map surface — **capability facet + registry**
- **`exp_family.py`** — the canonical-form view `p(x)=h(x)·exp(⟨η,T(x)⟩−A(η))`. Registry entry points: `to_exponential_family(dist, engine=NUMPY_ENGINE) -> ExponentialFamilyForm | None`, `is_exponential_family(dist) -> bool`. The view objects (`ExponentialFamilyForm` + composite/iid/multinomial/conditional variants) expose `natural_parameters()`, `sufficient_statistics(x)`, `log_partition(eta=None)`, `log_base_measure(x)`, `log_density(x)`, `mean_parameters(...)` (∇A), `fisher_information(...)` (Cov T), `from_natural(eta) -> dist | None`.
  - **Spec source:** the per-distribution `ExponentialFamilySpec` lives at `pysp/stats/compute/declarations.py:43` (out of scope; a callable-bundle dataclass: `sufficient_statistics`, `natural_parameters`, `log_partition`, optional `base_measure`/`*_from_params`, flags `fixed_base`, `runtime_scoring`). `exp_family.py` *consumes* it; it does not define the ABC.
  - **Facet:** grants the **exp-family** capability to any declared distribution — the implicit ExponentialFamily facet from the core axes, surfaced as a value object here.

### Unified sampling entry point — **registry/dispatch**
- **`sampling_api.py`** — `sample(model, size=None, *, seed=None, rng=None, **kwargs) -> Any`: the single `pysp.stats.sample` facade. Dispatches `Relation`, `FieldPosterior`, `LatentPosterior`, and anything exposing `.sampler(seed).sample(size, **kwargs)`. Unifies the `DistributionSampler` facet across distributions, relations, and posteriors.
- **`_sampling.py`** — backing implementation: `scatter_component_draws(comp_state, comp_samplers, size) -> list` — vectorized scatter-dispatch for mixture-like models, bit-identical to a per-draw loop.

---

## Coverage checklist (every module in scope)

```
pysp/relations.py                        — Relation (real ABC) + RelationSampler; optimization-as-distribution; engines best_first_paths / nearest_first
pysp/planner.py                          — EncodedDataHandle contract + encoded-data backend registry (register_encoded_data_backend); Resources/DeviceSpec/Placement planning dataclasses
pysp/models/__init__.py                  — re-exports of the application models
pysp/models/gaussian_process.py          — GaussianProcessRegressor (sklearn-style fit/predict; BO surrogate duck contract)
pysp/models/sparse_gaussian_process.py   — SparseGaussianProcessRegressor (FITC; sklearn-style)
pysp/models/random_forest.py             — RandomForestEstimator / RandomForestConditional (full Distribution+Estimator contract)
pysp/models/_forest.py                   — NativeRandomForest backend for random_forest (contract internals)
pysp/models/neural.py                    — Gaussian/Categorical/Poisson NeuralNetwork wrappers (sklearn-style, torch); make_mlp
pysp/models/random_graph.py              — ErdosRenyi/StochasticBlock graph models (fit_mle/sample/bic) + HardEMResult
pysp/models/dirichlet_process_mixture.py — TruncatedDirichletProcessMixtureModel (Distribution-side) + fit_truncated_dpm / *FitResult
pysp/models/knowledge_graph.py           — TransEKnowledgeGraphModel (sklearn-style fit_margin) + KnowledgeGraphFitResult
pysp/models/grammar.py                   — fit_induced_pcfg / viterbi_parse over HeterogeneousPCFGDistribution; GrammarLearningResult/PCFGParseNode
pysp/models/partially_observable_markov_decision_process.py — POMDP model (filter/forward_backward/sample) + baum_welch_pomdp + *Filter/Fit results
pysp/models/dependence.py                — causal-discovery utils (CI tests, PC skeleton, v-structure orientation) + result dataclasses
pysp/doe/__init__.py                     — re-exports of the DoE surface
pysp/doe/bayesopt.py                     — acquisition contract + register_acquisition registry; propose_next/propose_batch/minimize; BayesOptResult
pysp/doe/optimal.py                      — optimality-criterion contract + register_criterion registry; optimal_design; polynomial_features
pysp/doe/optimizer.py                    — BayesianOptimizer (ask/tell stateful BO)
pysp/doe/designs.py                      — space-filling designs (random/LHS/maximin-LHS/Sobol/Halton/full-factorial)
pysp/doe/constrained.py                  — constrained BO (probability_of_feasibility, constrained_minimize) + ConstrainedBayesOptResult
pysp/doe/multiobjective.py               — multi-objective BO (multi_minimize, pareto_mask) + MultiObjectiveResult
pysp/uq/__init__.py                      — re-exports of the UQ surface
pysp/uq/propagate.py                     — propagate (Monte Carlo) + unscented_transform (standalone functions)
pysp/uq/sensitivity.py                   — sobol_indices + morris_screening (standalone functions)
pysp/uq/calibration.py                   — calibrate + KOCalibration (Kennedy-O'Hagan result class)
pysp/infer/__init__.py                   — nuts/nuts_torch/advi facade + NutsResult/AdviResult dataclasses; re-exports
pysp/infer/backends.py                   — InferenceBackend (frozen-dataclass contract) + register/get/available/select registry
pysp/infer/diagnostics.py                — rhat (Gelman-Rubin) + ess (effective sample size); standalone array utilities
pysp/data/__init__.py                    — re-exports of the data adapters
pysp/data/dataframe.py                   — dataframe_records / seq_encode_dataframe (pandas → pysp encoded data)
pysp/data/graph_data.py                  — GraphObservation + GraphDataEncoder (realizes DataSequenceEncoder)
pysp/data/rdd_sampler.py                 — take_sample / sample_seq_as_rdd / sample_rdd (Spark distributed sampling)
pysp/stats/composition.py                — AitchisonNormalDistribution (+Estimator/Sampler/Accumulator/Encoder); simplex helpers (Distribution contract)
pysp/stats/hierarchical.py               — HierarchicalNormalDistribution (+Estimator) partial pooling (Distribution contract; EM in estimate)
pysp/stats/temporal.py                   — PeriodicTimeDistribution (Distribution contract) + SeasonalTimeSeries (FIT() LEAK: instance-method fit in stats)
pysp/stats/errors_in_variables.py        — deming_regression + DemingFit (standalone errors-in-variables regression)
pysp/stats/max_stable.py                 — SmithMaxStable (+sampler) + fit_smith_maxstable (spatial process; honest non-Distribution)
pysp/stats/block_gibbs.py                — ConjugateBlock/MetropolisBlock (implicit block contract update(state,rng)) + BlockGibbs dispatcher
pysp/stats/exp_family.py                 — to_exponential_family / is_exponential_family + ExponentialFamilyForm view objects (exp-family facet; spec from declarations.py)
pysp/stats/sampling_api.py               — sample() unified dispatch (Relation/FieldPosterior/LatentPosterior/.sampler())
pysp/stats/_sampling.py                  — scatter_component_draws (vectorized mixture-component sampling backend)
```

---

## Recommendations — implicit contracts to formalize as Protocols/ABCs

1. **`EncodedDataHandle`** (`planner.py`) → `Protocol` — the four `pysp_seq_*` methods checked by `is_encoded_data_handle` are the cleanest implicit contract in the repo.
2. **Acquisition function** and **optimal-design criterion** (`doe/`) → `Protocol`s — both already have registries; the callable shapes (`fn(mean,std,best,*,maximize,**p)`, `fn(info,*,ref)`) are stable.
3. **GP surrogate** (`doe/`) → `Protocol` (`fit`, `predict(...,return_cov=bool)`) — currently duck-typed via the `gp=` argument.
4. **Inference block** (`block_gibbs.py`) → `Protocol` (`name`, `kind`, `update(state, rng)`).
5. **Common `FitResult`** (`models/`) → small `Protocol` (`.model`, `.history`) — every model rolls its own `*FitResult`; a shared shape would let tooling consume them generically. (`InferenceBackend` is already a clean dataclass-as-contract; no change needed.)

### Anti-pattern flagged
- **`SeasonalTimeSeries.fit()`** (`temporal.py`) is the one stats-layer **fit() leak**: an sklearn-style stateful `fit(times, values) -> self` on a non-Distribution class inside `pysp.stats`, bypassing the `estimator()`/`estimate(nobs, suff_stat)` split. Recommend either giving it a proper `ParameterEstimator` or moving it to app/notebook code, per the domain-neutral-core rule.
```
