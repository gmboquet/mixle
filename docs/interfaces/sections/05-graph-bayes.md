# 05 — Graph / structured / combinatorial (`pysp/stats/graph`) and Bayes (`pysp/stats/bayes`)

These two packages sit at the deep end of the composition axis (§ _SPEC.md axis 3): `graph/` realizes
the **core `SequenceEncodableProbabilityDistribution` contract over structured / combinatorial supports**
(permutations, trees, graphs, finite-state sequences, grammars), and `bayes/` realizes the
**conjugate / Bayesian-inference family** — priors *over parameters* plus the `ConjugatePosterior`
interface and the Bayesian-nonparametric (DP / PY / HDP) mixtures.

The shared leaf contracts (`SequenceEncodableProbabilityDistribution`, `DistributionSampler`,
`StatisticAccumulator`/`Factory`, `ParameterEstimator`, `DataSequenceEncoder`, `DistributionEnumerator`)
are defined in `pysp/stats/compute/pdist.py` and documented in the core-contract section; this section
states them once and focuses on the *extra* surface each combinatorial / Bayesian family adds.

---

## Part A — Graph / structured / combinatorial (`pysp/stats/graph`)

Every distribution class here subclasses `SequenceEncodableProbabilityDistribution` and ships the full
five-class bundle (`*Distribution`, `*Sampler`, `*Accumulator`, `*AccumulatorFactory`, `*Estimator`,
`*DataEncoder`). The structured support is what distinguishes these from leaves: `x` is a permutation /
ranking, an adjacency matrix / graph, a tree (as edges or a parent vector), a finite-state sequence, a
pair-of-count-sets, or a grammar.

### The combinatorial-support sub-interfaces

#### Finite-state / sequence
- `MarkovChainDistribution` — `markov_chain.py`. First-order chain over an arbitrary state alphabet;
  `x: list[T]`. Adds `expected_log_density(x) -> float`, `get_prior`/`set_prior`, gradient-fit hooks
  (`gradient_fit_state`, `_MarkovChainGradientFitState`), engine-resident `backend_*` kernels
  (`backend_stacked_params`/`_log_density`/`_sufficient_statistics_with_estimator`), and a
  `quantized_count_index(quantizer, max_fine_bucket)` for the count-budget seek index. **Enumerable**
  (`MarkovChainEnumerator`). Sampler adds `sample_paths(lengths)`, `sample_seq(size, v0, batched)`.
- `IntegerMarkovChainDistribution` — `integer_markov_chain.py`. The observed finite-state automaton:
  dense integer states, optional **lag** (higher-order context `tuple[int,...] -> int`). `x: Sequence[int]`.
  **Enumerable** (`IntegerMarkovChainEnumerator`); engine-resident `backend_*` kernels. Sampler adds
  `single_sample()`, `sample_given(x) -> int`.
- `MarkovTransformDistribution` — `markov_transform.py`. Transform model: a pair of count-sets is mapped
  to a third via a per-token transition matrix; `x` is a `(input, output)` pair. `backend_seq_log_density`.
  No enumerator (continuous-multiplicity support).
- `SparseMarkovAssociationDistribution` — `sparse_markov_transform.py`. Sparse hidden-association model
  over `x: tuple[list[(int,float)], list[(int,float)]]` (sparse count vectors); uses `scipy.sparse`
  (`lil_matrix`/`csr_matrix`) sufficient statistics. `backend_seq_log_density`; accumulator adds
  `initialize_rng(rng)`. No enumerator.

#### Ranking / permutation  — the *over-permutations* interface
All four score a permutation/ranking `x: Sequence[int]` (an ordering of `0..n-1`) and are **Enumerable**
(an enumerator over the `n!` support, streamed lazily in descending-probability order — the
**Rankable-by-index / k-best** facet).
- `MallowsDistribution` — `mallows.py`. Central permutation `sigma0` + dispersion `theta`; adds
  `kendall_distance(x) -> int`. `MallowsEnumerator`. numpy-only kernel (Kendall-tau is numpy-native).
- `PlackettLuceDistribution` — `plackett_luce.py`. Single worth vector over items; `PlackettLuceEnumerator`.
  Also ships a **partial-ranking** variant (`PlackettLucePartial{DataEncoder,Accumulator,AccumulatorFactory,Estimator}`)
  for top-k / censored orderings.
- `SpearmanRankingDistribution` — `spearman_rho.py`. Spearman-rho ranking law; `SpearmanRankingEnumerator`;
  engine-resident `backend_stacked_*` kernels + `backend_legacy_sufficient_statistics`/`backend_log_density_from_params`.
- `MatchingDistribution` — `matching.py`. Weighted bipartite perfect-matching / assignment law over `K_{n,n}`:
  `p(sigma) ∝ prod_i w[i,sigma(i)]`, normalizer = matrix **permanent** (Ryser, exponential in `n`,
  `max_nodes=12` cap). `MatchingEnumerator` streams in descending probability via **Murty's k-best**
  assignment (no `n!` materialization). Estimator = projected gradient ascent on log-weights to match edge marginals.

#### Tree / spanning
- `ChowLiuTreeDistribution` — `chow_liu_tree.py`. Generic tree-structured joint over fixed-length tuples
  `x: Sequence[Any]` (per-feature marginals + tree of pairwise edges). **Enumerable** (`ChowLiuTreeEnumerator`).
- `IntegerChowLiuTreeDistribution` — `integer_chow_liu_tree.py`. Dense-integer specialization
  (`x: Sequence[int]|np.ndarray`, `num_features × num_states`). **Enumerable** (`IntegerChowLiuTreeEnumerator`).
- `SpanningTreeDistribution` — `spanning_tree.py`. **Matrix-Tree** law over labeled spanning trees;
  `x` = `n-1` undirected `(i,j)` edges; `p(tree) ∝ prod edge-weights`, `Z` via the Matrix-Tree theorem.
  Distinct from ChowLiuTree (a *distribution shaped like* a tree) — this is a distribution *over trees*.
  **Enumerable**: `enumerator(max_edge_subsets=…)` → `SpanningTreeEnumerator`, lazy descending-probability
  k-best via **Gabow's** constrained-MST oracle (no edge-subset scan).

#### Random graphs
All score a binary graph `x` (square adjacency matrix, NetworkX-like graph, or mapping accepted by the
graph encoder).
- `ErdosRenyiGraphDistribution` — `erdos_renyi_graph.py`. Single edge probability. **Enumerable** (`ErdosRenyiGraphEnumerator`).
- `StochasticBlockGraphDistribution` — `stochastic_block_graph.py`. Block-membership + block-pair edge
  probabilities. **Enumerable** (`StochasticBlockGraphEnumerator`).
- `RandomDotProductGraphDistribution` — `random_dot_product_graph.py`. Latent-position model: each node
  carries `x_i ∈ R^d`, `P(edge_ij) = <x_i, x_j>`. Sampler only (no enumerator — continuous latent positions).
- `KnowledgeGraphDistribution` — `knowledge_graph.py`. Knowledge-graph embedding law over triples
  `x: Sequence[int]` (head, relation, tail). Sampler only on the main class, **but** ships two extra
  combinator-style roles: `KnowledgeGraphEnsemble(members)` (a bag of KG distributions) and
  `KnowledgeGraphPattern` — a query/pattern object with `log_density(binding)` and its own `enumerator()`
  over satisfying bindings (the **Enumerable** facet lives on the pattern, not the base distribution).

#### Grammar
- `GrammarDistribution` — `grammar.py`. The PCFG-estimation accumulator surface: a distribution over
  **graph grammars** (vertex-replacement / `VertexReplacementGrammar` + `GrammarRule`). Likelihood is
  computed rule-by-rule against the model grammar; estimation (`GrammarEstimator` /
  `GrammarEstimatorAccumulator`) re-derives a grammar from observed networks. Sampler only; no enumerator.

### `DistributionEnumerator` (combinatorial)  — de-facto contract
- **Role:** lazy iterator over a distribution's discrete support in **descending-probability** order,
  yielding `(value, log_prob)` pairs — the **Enumerable / Rankable-by-index / k-best** facet on combinatorial supports.
- **Formalized in:** `pysp/stats/compute/pdist.py:568` (`DistributionEnumerator`); obtained via
  `dist.enumerator(...)`.
- **Methods:**
    ```
    __init__(self, dist: SequenceEncodableProbabilityDistribution) -> None
    __iter__(self) / __next__(self) -> tuple[value, float]   # (support element, log p); StopIteration ends the support
    ```
- **Implemented by (graph):** `markov_chain`, `integer_markov_chain`, `mallows`, `plackett_luce`,
  `spearman_rho`, `matching`, `chow_liu_tree`, `integer_chow_liu_tree`, `spanning_tree`,
  `erdos_renyi_graph`, `stochastic_block_graph`, and `knowledge_graph` (via `KnowledgeGraphPattern`).
- **Facets:** Enumerable; the permutation/tree/assignment enumerators are genuinely **rankable** (k-best,
  no full-support materialization: Murty for matchings, Gabow for spanning trees, lazy product order for
  the rest). `EnumerationError` (`pdist.py:20`) is the documented escape hatch for the families that opt out.
- **Notes:** This is the cleanest candidate in this scope for promotion to a formal `Protocol` — the
  contract ("`enumerator()` returns a lazy descending-probability iterator of `(value, log_prob)`") is
  uniform across ~12 unrelated families and several use it as the public k-best ranking API.

---

## Part B — Bayes: the conjugate / Bayesian-inference family (`pysp/stats/bayes`)

### `ConjugatePosterior`  — de-facto contract (base class)
- **Role:** a closed-form conjugate posterior **over a likelihood's parameters**; consumes an exponential-family
  likelihood + data and exposes exact Bayesian inference (mean, exact parameter draws, evidence,
  point estimate, posterior predictive).
- **Formalized in:** `pysp/stats/bayes/conjugate.py:54` (base `ConjugatePosterior`); not an ABC — a base
  class with `NotImplementedError` stubs that each family fills in.
- **Methods:**
    ```
    mean(self) -> dict[str, Any]                                  # posterior mean of the parameters
    sample(self, n=1, rng=None) -> dict[str, np.ndarray]          # exact draws of parameter sets
    sampler(self, seed=None) -> ConjugatePosteriorSampler         # standard .sampler(seed).sample(size) adapter
    point_estimate(self)                                          # a fitted pysp dist at the posterior mean
    log_marginal_likelihood(self) -> float                       # evidence of the data under the prior
    posterior_predictive(self)                                   # pysp dist of a new draw
    summary(self) -> dict[str, Any]                              # {family, mean, hyper}
    hyper(self) -> dict[str, Any]                                # the posterior hyperparameters
    ```
  (`log_base` / `log_marginal_likelihood` are the absolute, base-measure-inclusive evidence terms.)
- **Implemented by:** the family-specific posteriors in `conjugate.py` — `BetaPosterior`,
  `GammaRatePosterior`, `DirichletPosterior`, `NormalInverseGammaPosterior`, `NormalInverseWishartPosterior`,
  `InverseGammaVariancePosterior`, `GammaParameterPosterior`, `DiagonalNIGPosterior`, `VonMisesMeanPosterior`,
  and `MixtureConjugatePosterior`.
- **Facets:** **Conjugate** (the defining facet); each carries exact `sample`/evidence so it is also a
  parameter-space `DistributionSampler`-like surface.
- **Notes:** the entry points are the two factories:
    ```
    conjugate_posterior(dist, data, prior=None, weights=None) -> ConjugatePosterior
    mixture_conjugate_posterior(dist, data, priors: list[dict], prior_weights=None, weights=None)
        -> MixtureConjugatePosterior
    ```
  `conjugate_posterior` dispatches on `type(dist)` through a registry (`_registry()`, `conjugate.py:920`)
  covering Bernoulli, Binomial, Geometric, Poisson, Exponential, Categorical, IntegerCategorical, Gaussian,
  MultivariateGaussian, Rayleigh, HalfNormal, LogGaussian, DiagonalGaussian, Gamma, InverseGamma,
  InverseGaussian, Pareto, NegativeBinomial, VonMises. For multi-parameter likelihoods whose conjugate is
  *conditional* (Gamma, InverseGamma, InverseGaussian, Pareto, NegativeBinomial, vonMises) the non-target
  parameter is taken as known from `dist`. Families with no closed-form conjugate raise via `_NO_CLOSED_FORM`
  (`conjugate.py:968`). `mixture_conjugate_posterior` realizes **Diaconis–Ylvisaker** (a mixture of conjugate
  priors is conjugate): components reweighted by `w'_m ∝ w_m · Z_m`; its `posterior_predictive()` returns a
  `MixtureDistribution` of the component predictives. **This is the natural candidate to formalize as a `Protocol`** (the method surface is uniform across all 10 posteriors).

### `ConjugatePosteriorSampler`
- **Role:** standard `.sample(size)` adapter over a `ConjugatePosterior` (each draw is a *parameter set*).
- **Formalized in:** `conjugate.py:102`. `sample(size=None)` → one parameter set (scalars) or a dict of
  length-`size` arrays; mirrors the distribution-sampling convention.

### Prior-distribution families — distributions *over parameters* (+ conjugate hooks)
Each of these subclasses `SequenceEncodableProbabilityDistribution` (so it is a first-class pysp
distribution over a parameter space) **and** carries the variational-Bayes hooks `cross_entropy(dist)` /
`entropy()` used as the ELBO global terms.
- `DirichletDistribution` — `dirichlet.py`. Over the simplex; `__init__(alpha, name, keys)`. Adds
  `cross_entropy(dist)`/`entropy()`, vectorized `seq_log_density`, `get_parameters() -> alpha` (lets it
  serve as the conjugate prior on a Categorical/Mixture weight simplex under the unified estimation
  protocol), and a `DirichletFisherView` (Fisher-information facet). `cross_entropy` accepts a
  `DirichletDistribution` or a `SymmetricDirichletDistribution` argument.
- `DictDirichletDistribution` — `dict_dirichlet.py`. Dirichlet over a `dict[Any,float]` simplex (sparse /
  keyed alphabet); `cross_entropy`/`entropy`.
- `SymmetricDirichletDistribution` — `symmetric_dirichlet.py`. Scalar-`alpha` Dirichlet (`__init__(alpha, dim, name)`);
  `entropy()` (broadcasts into Dirichlet `cross_entropy`).
- `NormalGammaDistribution` — `normal_gamma.py`. Over `(mu, tau)`; conjugate prior for the **univariate
  Gaussian**. `cross_entropy`/`entropy`, `seq_log_density`.
- `MultivariateNormalGammaDistribution` — `multivariate_normal_gamma.py`. Per-dimension Normal-Gamma
  (diagonal-precision MVN prior); `cross_entropy`/`entropy`.
- `NormalWishartDistribution` — `normal_wishart.py`. Over `(mu, Lambda)`; conjugate prior for the **full
  multivariate Gaussian**. Adds `expected_log_det() -> float`, `expected_precision() -> np.ndarray`
  (Wishart expectations for VB), `cross_entropy`/`entropy`.
- **Facets:** Conjugate-prior + a real (samplable, scorable) distribution over parameters; the
  `cross_entropy`/`entropy` pair is the implicit **variational-prior** facet (recommend naming /
  formalizing it — it appears on all six prior families and `DPM`).

### Bayesian-nonparametric mixtures — stick-breaking / CRP / EPPF surface
- `DirichletProcessMixtureDistribution` — `dirichlet_process_mixture.py`. DP mixture with a
  `Gamma(s1, 1/s2)` concentration hyper-prior. Adds `expected_log_density(x)` (the VB term) and a
  hyper-posterior accessor on `alpha`; estimator updates the Gamma hyper-posterior on `alpha` from the
  sufficient statistic. Stick-breaking truncation + a length estimator.
- `HierarchicalDirichletProcessMixtureDistribution` — `hierarchical_dirichlet_process_mixture.py`. HDP:
  group-level `Dirichlet(alpha·beta)` over a shared global `Dirichlet(gamma/K)` top stick; two
  concentrations (`alpha`, `gamma`), `expected_log_density`-style VB surface.
- `PitmanYorProcessDistribution` — `pitman_yor.py`. Two-parameter PY process **over set partitions**:
  `__init__(alpha, discount, ...)`, `x: List[int]` (cluster-label vector). Scores the **EPPF** directly
  (`log_density` = the exchangeable partition probability); `discount = 0` recovers the DP / CRP. Estimator
  fits `(alpha, discount)` by maximizing the aggregated EPPF log-likelihood (three integer-indexed histograms
  as the sufficient statistic; optional `estimate_discount`).
- **Facets:** these are the **LatentStructured / nonparametric** tail of the composition axis; PY exposes
  the **EPPF / partition** surface explicitly, DP/HDP the **stick-breaking + concentration-hyper-prior** surface.

---

## Coverage checklist

### `pysp/stats/graph` (17 files)
- `graph/__init__.py` — package aggregator / re-exports (no interface).
- `graph/markov_chain.py` — `SequenceEncodableProbabilityDistribution` (finite-state chain, arbitrary alphabet); **Enumerable**; engine-resident `backend_*`; gradient-fit + quantized-count-index hooks.
- `graph/integer_markov_chain.py` — observed finite-state automaton over integers with optional **lag**; **Enumerable**; engine-resident.
- `graph/markov_transform.py` — Markov transform over a pair-of-count-sets → third; `backend_seq_log_density`. (no enumerator)
- `graph/sparse_markov_transform.py` — sparse hidden-association Markov model (scipy.sparse suff-stats). (no enumerator)
- `graph/mallows.py` — Mallows distribution over permutations; **Enumerable**; `kendall_distance`.
- `graph/plackett_luce.py` — Plackett–Luce ranking law + partial-ranking variant; **Enumerable**.
- `graph/spearman_rho.py` — Spearman-rho ranking law; **Enumerable**; engine-resident `backend_stacked_*`.
- `graph/matching.py` — weighted bipartite matching / assignment (permanent, Ryser); **Enumerable** via Murty k-best.
- `graph/chow_liu_tree.py` — generic tree-structured joint over fixed-length tuples; **Enumerable**.
- `graph/integer_chow_liu_tree.py` — dense-integer Chow–Liu tree; **Enumerable**.
- `graph/spanning_tree.py` — Matrix-Tree spanning-tree law (distribution *over* trees); **Enumerable** via Gabow k-best.
- `graph/erdos_renyi_graph.py` — Erdős–Rényi random-graph distribution; **Enumerable**.
- `graph/stochastic_block_graph.py` — stochastic block-model graph distribution; **Enumerable**.
- `graph/random_dot_product_graph.py` — RDPG latent-position graph model; sampler only.
- `graph/knowledge_graph.py` — KG-embedding distribution over triples; `KnowledgeGraphEnsemble` (combinator) + `KnowledgeGraphPattern` (query w/ `enumerator()`).
- `graph/grammar.py` — PCFG / graph-grammar distribution + accumulator (`VertexReplacementGrammar`, `GrammarRule`); sampler only.

### `pysp/stats/bayes` (11 files)
- `bayes/__init__.py` — package aggregator / re-exports (no interface).
- `bayes/conjugate.py` — **`ConjugatePosterior`** base + 10 family posteriors + `ConjugatePosteriorSampler` + `MixtureConjugatePosterior`; factories `conjugate_posterior` / `mixture_conjugate_posterior` (registry dispatch on likelihood type; Diaconis–Ylvisaker mixture-of-conjugates).
- `bayes/dirichlet.py` — `DirichletDistribution`: distribution over the simplex + conjugate-prior hooks (`cross_entropy`/`entropy`/`get_parameters`) + `DirichletFisherView`.
- `bayes/dict_dirichlet.py` — `DictDirichletDistribution`: Dirichlet over a keyed/sparse simplex; `cross_entropy`/`entropy`.
- `bayes/symmetric_dirichlet.py` — `SymmetricDirichletDistribution`: scalar-`alpha` Dirichlet; `entropy`.
- `bayes/normal_gamma.py` — `NormalGammaDistribution`: conjugate prior for the univariate Gaussian (over `(mu,tau)`); `cross_entropy`/`entropy`.
- `bayes/multivariate_normal_gamma.py` — `MultivariateNormalGammaDistribution`: per-dim Normal-Gamma (diagonal-precision MVN prior); `cross_entropy`/`entropy`.
- `bayes/normal_wishart.py` — `NormalWishartDistribution`: conjugate prior for the full MVN (over `(mu,Lambda)`); `expected_log_det`/`expected_precision`/`cross_entropy`/`entropy`.
- `bayes/dirichlet_process_mixture.py` — `DirichletProcessMixtureDistribution`: DP mixture, stick-breaking + Gamma concentration hyper-prior; `expected_log_density`.
- `bayes/hierarchical_dirichlet_process_mixture.py` — `HierarchicalDirichletProcessMixtureDistribution`: HDP (group `Dirichlet(alpha·beta)` over global `Dirichlet(gamma/K)`).
- `bayes/pitman_yor.py` — `PitmanYorProcessDistribution`: two-parameter PY process over set partitions; scores the **EPPF**; DP/CRP at `discount=0`.

---

## Interfaces to formalize (recommendations)
1. **`DistributionEnumerator` as a `Protocol`** — already an ABC in `pdist.py`, but the combinatorial k-best
   variants (Murty / Gabow / lazy-product) form a uniform "lazy descending-probability `(value, log_prob)`
   iterator" contract used as a public ranking API across ~12 graph families; worth elevating + documenting the k-best guarantee.
2. **`ConjugatePosterior` as a `Protocol`** — its 7-method surface (`mean`/`sample`/`point_estimate`/
   `log_marginal_likelihood`/`posterior_predictive`/`hyper`/`summary`) is implemented identically by 10
   families through `NotImplementedError` stubs; a `Protocol` (or ABC) would make the registry/factory contract explicit.
3. **A `VariationalPrior` facet** — the `cross_entropy(dist) -> float` + `entropy() -> float` pair on the six
   prior families and the DP mixture is an implicit, name-it-able capability (the ELBO global terms).
