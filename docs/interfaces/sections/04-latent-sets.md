# 04 — Latent-state models & set/edit distributions

Scope: `pysp/stats/latent/*.py` and `pysp/stats/sets/*.py`, plus the latent-posterior spine
`pysp/stats/latent_posterior.py`.

Every distribution / sampler / accumulator / estimator / encoder in this area realizes the **core
contract ABCs** defined in `pysp/stats/compute/pdist.py`
(`SequenceEncodableProbabilityDistribution:476`, `DistributionSampler:530`,
`SequenceEncodableStatisticAccumulator:692`, `StatisticAccumulatorFactory:736`,
`ParameterEstimator:743`, `DataSequenceEncoder:820`, `DistributionEnumerator:568`). That contract is
stated once here and **not** repeated per leaf — the rest of this section catalogs the *extra* roles
these models add on top of it: the **`LatentPosterior` spine**, the **finite-latent / EM
responsibility** facet, the **sequential-latent (HMM) forward-backward / FFBS** facet, the
**grammar / branching** facet, and the **set / edit-distance** facet.

---

## A. The `LatentPosterior` spine — `q(z | x)` as a first-class object

### `LatentPosterior` — ABC
- **Role:** the posterior `q(z | x)` over a model's latent variables — *exact* for mixtures/HMMs,
  *mean-field* for LDA. Unifies the EM E-step output, latent sampling, the MAP configuration, and the
  ELBO entropy term into one object so models stop scattering loose `gamma`/`phi`/`xi` arrays.
- **Formalized in:** `pysp/stats/latent_posterior.py:45` (a real `abc.ABC`).
- **Methods:**
      marginals(self) -> Any        # per-latent marginal responsibilities — the EM M-step input
      sample(self, rng=None) -> Any # draw z ~ q(z|x)  (Gibbs / latent / posterior-predictive)
      mode(self) -> Any             # MAP latent configuration (argmax / Viterbi)
      entropy(self) -> Any          # H[q]  — the ELBO entropy term
- **Implemented by:** three concrete posteriors below; *produced by* `MixtureDistribution`,
  `HiddenMarkovModelDistribution`, `LDADistribution` via their `latent_posterior(x)` methods.
- **Facets:** latent-structured; sampling; entropy/ELBO.
- **Notes:** this is the keystone interface of the family. Mean-field realizations additionally expose
  `expected_complete_ll(dist)` / `update(dist)` / `elbo(dist)` (per the module docstring) for the
  variational E-step; exact posteriors don't need them. **Recommend formalizing the producer side as a
  Protocol** (`LatentStructured.latent_posterior(x) -> LatentPosterior`) — today only three of ~20
  latent models implement it, by convention.

### `CategoricalLatentPosterior` — concrete
- **Role:** independent categorical latents `q(z) = prod_i Cat(z_i; r_i)` — the exact posterior for a
  finite mixture's component labels (and an LDA document's per-token topic factor).
- **Formalized in:** `latent_posterior.py:65`.
- **Surface:** built from a row-stochastic `(N, K)` `responsibilities` matrix + optional `support`
  label map. `marginals()->(N,K)`, `sample(rng)->(N,)` support labels, `mode()->(N,)`,
  `entropy()->(N,)` per-observation.
- **Produced by:** `MixtureDistribution.latent_posterior` (`mixture.py:580`).

### `MarkovChainLatentPosterior` — concrete
- **Role:** chain-structured latents `q(z_1..z_T | x)` for an HMM — **exact via forward-backward**.
- **Formalized in:** `latent_posterior.py:105`.
- **Surface:** built from `log_pi (K,)`, `log_A (K,K)`, `log_b (T,K)`; runs a log-space forward filter
  on construction. `log_likelihood()->float` (forward normalizer); `marginals()->(T,K)` smoothing
  probs; `sample(rng)->(T,)` state path by **FFBS** (forward-filter backward-sample);
  `mode()->(T,)` **Viterbi** max-product path; `entropy()->float` exact scalar chain entropy via the
  FFBS factorization.
- **Produced by:** `HiddenMarkovModelDistribution.latent_posterior` (`hidden_markov.py:887`).

### `MeanFieldLDAPosterior` — concrete
- **Role:** mean-field variational posterior for one LDA document,
  `q(theta, z) = Dir(theta; gamma) prod_n Cat(z_n; phi_n)` (Blei–Ng–Jordan factorization as an object).
- **Formalized in:** `latent_posterior.py:186`.
- **Surface:** built from `gamma (K,)`, `phi (W,K)`, `counts (W,)`. `topic_proportions()->(K,)`
  (`E_q[theta]`); `marginals()->(W,K)`; `sample(rng)->(theta, z)` (continuous + discrete pair, draws
  per-*token* topics); `mode()->(W,)`; `entropy()->float` (`H[q(theta)] + sum_w count_w H[Cat(phi_w)]`).
  Heterogeneous latents (continuous theta + discrete z), hence the pair return.
- **Produced by:** `LDADistribution.latent_posterior` (`lda.py:414`).

---

## B. Finite-latent / exchangeable interface — the EM responsibility facet

### `FiniteMixture` (de-facto contract) — EM responsibility + component surface
- **Role:** a finite/exchangeable latent-variable model whose E-step produces per-observation
  **responsibilities** over a discrete latent (component / topic) and whose M-step delegates to the
  component sub-estimators.
- **Formalized in:** implicit — followed by convention across the mixture/topic family. **Recommend a
  `RespondsibilityModel` Protocol** capturing the method cluster below.
- **Methods (the de-facto surface; not all models expose every one):**
      component_log_density(self, x) -> np.ndarray       # (K,) per-component log p(x | z=k)
      seq_component_log_density(self, enc) -> np.ndarray # (N,K) vectorized
      posterior(self, x) -> np.ndarray                   # (K,) responsibilities  r_k = q(z=k|x)
      seq_posterior(self, enc) -> np.ndarray             # (N,K) vectorized E-step output
      expected_log_density(self, x) -> float             # variational E[log p] under a weight prior
      latent_posterior(self, x) -> LatentPosterior       # (mixture/LDA only) the q(z|x) object
      posterior_predictive(self, x, seed=None) -> list   # (mixture/LDA only) draw x' ~ p(x'|x)
- **Implemented by:**
    - `mixture.py` — `MixtureDistribution`: full surface; `posterior:368`, `seq_posterior:530`,
      `component_log_density:354`, `expected_log_density:270`, **`conditional(observed):327`** (returns
      a re-weighted `MixtureDistribution` given partially-observed dims), `latent_posterior:580`
      (→`CategoricalLatentPosterior`), `posterior_predictive:591`. Conjugate Dirichlet prior on weights.
    - `gaussian_mixture.py` — `GaussianMixtureDistribution`: full-covariance MVN components;
      `posterior:148`, `seq_posterior:245`, `component_log_density:136`.
    - `hierarchical_mixture.py` — `HierarchicalMixtureDistribution`: outer K-mixture of inner shared
      L-topic mixtures; `posterior:192` (outer), `seq_posterior:373`, `component_log_density:205`,
      `seq_component_log_density:245`. Estimator takes `num_mixtures` + a `len_dist`/`len_estimator`.
    - `heterogeneous_mixture.py` — `HeterogeneousMixtureDistribution`: components with **different
      types/encoders**; `posterior:185`, `seq_posterior:362`, `component_log_density:165`. Estimator
      supports `fixed_weights`.
    - `joint_mixture.py` — `JointMixtureDistribution`: mixture over paired `(x1, x2)` with a joint
      latent decomposition; estimator takes two component lists + a 3-tuple `pseudo_count`.
    - `semi_supervised_mixture.py` — `SemiSupervisedMixtureDistribution`: observation carries optional
      exogenous label evidence `(x0, [(k, weight)…])` that multiplicatively re-weights responsibilities;
      `posterior:188`, `seq_posterior:306`.
    - `spatial_mixture.py` — `SpatialMixture`: **non-standard surface** (not the full ABC stack). A
      grid-structured mixture with a Potts-MRF prior on labels, fit by mean-field variational EM:
      `fit(observations, *, max_iter=40, mf_iter=3, seed=0):73`, `responsibilities():109`,
      `labels():113` (MAP field), `entropy():117`, `component(j):122`. Constructor takes
      `shape`, `n_components`, an `emission` estimator, and Potts `beta`.
    - `probabilistic_pca.py` — `ProbabilisticPCADistribution`: continuous latent-factor Gaussian
      (closed-form ML); its "responsibility" analog is **`transform(x) -> E[z|x]`** (`:113`), the
      posterior latent-factor mean. Estimator takes `latent_dim`, `dim`.
    - `lda.py` — `LDADistribution`: Latent Dirichlet Allocation; `seq_log_density` returns the
      per-document variational ELBO (`:184`); `latent_posterior:414` (→`MeanFieldLDAPosterior`),
      `posterior_predictive:431`. Estimator: `topics`, `alpha_start`, `gamma_threshold`,
      `max_gamma_iter`.
    - `labeled_lda.py` — `LabeledLDADistribution`: LDA with per-document label sets selecting which
      topics are active; coupled per-label Dirichlet update; `seq_posterior:205`,
      `seq_component_log_density:179`, ELBO `seq_log_density:123`. Estimator takes `num_alpha`,
      optional `set_dist`/`len_dist`.
    - `integer_probabilistic_latent_semantic_indexing.py` — `IntegerProbabilisticLatentSemanticIndexingDistribution`:
      PLSI over integer-coded bags; document × state × word latent factorization; `log_density:180`
      marginalizes per-word topics. Estimator: `num_vals`, `num_states`, `num_docs`.
- **Facets:** latent-structured; responsibility/EM; (mixture, LDA) latent-posterior producer;
  (mixture) conditional/marginal; conjugate-Dirichlet weights.
- **Notes:** all realize the full core ABC stack (`*Distribution / *Sampler / *Accumulator /
  *AccumulatorFactory / *Estimator / *DataEncoder`) **except** `spatial_mixture.py`, which is a
  standalone `fit`-style class — worth flagging as an inconsistency with the rest of the library.

---

## C. Sequential-latent / HMM interface — forward-backward / Baum-Welch + FFBS

### `HiddenStateSequence` (de-facto contract) — "finite-state automaton with hidden state"
- **Role:** a latent *sequence* model: the latent is a coupled chain/tree of discrete states, the
  E-step is **forward-backward (Baum-Welch)**, decoding is **Viterbi**, sampling is **FFBS**, and the
  model carries optional **terminal-state** semantics (absorbing end states).
- **Formalized in:** implicit — followed by convention across the HMM family. **Recommend a
  `SequentialLatent` Protocol** for the forward-backward / `seq_posterior` / `viterbi` cluster.
- **Methods (de-facto; base `hidden_markov.py` is the reference surface):**
      log_density(self, x: list[T]) -> float                 # forward normalizer (terminal-aware)
      seq_log_density(self, enc) -> np.ndarray               # vectorized (numpy or numba kernel)
      seq_posterior(self, enc) -> list[np.ndarray] | None    # per-seq (T,K) Baum-Welch marginals
      viterbi(self, x) -> np.ndarray                         # MAP state path
      seq_viterbi(self, enc) -> ...                          # vectorized Viterbi
      latent_posterior(self, x) -> MarkovChainLatentPosterior# (base HMM only) the q(z|x) object
      posterior_predictive(self, x, seed=None) -> list       # (base HMM only) FFBS sample-predict
- **Implemented by:**
    - `hidden_markov.py` — `HiddenMarkovModelDistribution`: the reference. `viterbi:860`,
      `seq_posterior:825` (FFBS marginals via numba), `latent_posterior:887`
      (→`MarkovChainLatentPosterior`), `posterior_predictive:900`, `seq_viterbi:912`. Estimator carries
      `comp_ests`, `len_estimator`, `pseudo_count`, `terminal_states`, conjugate Dirichlet prior
      (`hmm_dirichlet_default_prior`). Module helpers `terminal_forward_loglik:156` /
      `terminal_forward_backward:174` implement the absorbing-end-state forward-backward.
    - `lookback_hidden_markov_model.py` — `LookbackHiddenMarkovModelDistribution`: **higher-order**
      (emissions condition on the previous `lag` observations); `viterbi_sequence:264`,
      `seq_posterior:391`. Estimator adds `lag` + `init_estimators` for the initial segment.
    - `tree_hidden_markov_model.py` — `TreeHiddenMarkovModelDistribution`: **branching** latent tree
      (upward-downward instead of a linear trellis); `viterbi:609`, `seq_viterbi:622`,
      `seq_posterior:504`; sampler `sample_tree`. Estimator: `terminal_level` depth bound, `use_numba`.
    - `quantized_hidden_markov_model.py` — `QuantizedHiddenMarkovModelDistribution`
      (**subclass of `HiddenMarkovModelDistribution`**): transition/emission/initial probabilities are
      `theta^exponent / Z` integer-count-DP forms; adds an `enumerator()`
      (`QuantizedHiddenMarkovModelEnumerator`, best-first over observations). Estimator does
      coordinate ascent on `theta` and the exponents (`init_mode` = `quantized`|`stationary`,
      `k_max`, `split_collapsed`).
    - `segmental_hidden_markov_model.py` — `SegmentalHiddenMarkovModelDistribution`: each state emits a
      variable-length **segment** (emissions are themselves distributions over sub-sequences); standard
      forward-backward `seq_log_density:219`; reuses `HiddenMarkovModelEnumerator`.
    - `semi_supervised_hidden_markov_model.py` — `SemiSupervisedHiddenMarkovModelDistribution`:
      uniform initial + per-position **soft state priors** folded multiplicatively into
      forward-backward; observation is `(emissions, prior_or_None)`; `log_density:169`,
      `seq_log_density:176`. No viterbi/enumerator.
    - `hidden_association.py` — `HiddenAssociationDistribution`: a **set/bag** sequential association
      — a `ConditionalDistribution` emits an output set given an input set, marginalizing hidden
      per-element states; `log_density:176`, `emission_mixture(s1):278` (returns a `MixtureDistribution`),
      `enumerator()`; sampler exposes `sample_given(x)` (the `ConditionalSampler` facet,
      `pdist.py:644`). Estimator: `cond_estimator`, `given_estimator`, `len_estimator`.
    - `integer_hidden_association.py` — `IntegerHiddenAssociationDistribution`: the integer-coded,
      numba-accelerated counterpart; `state_prob_mat (S×W)` + `cond_weights (W×S)` + `alpha` smoothing;
      `conditional_word_log_probs(s1):349`, `enumerator()`, `sample_given`. Estimator: `num_states`,
      `alpha`, `prev_estimator`, `use_numba`.
- **Facets:** latent-structured (chain/tree); responsibility/Baum-Welch; FFBS sampling; Viterbi/MAP;
  terminal-state; engine-resident (numba); (quantized/association) enumerable;
  (association) conditional-sampler.
- **Notes:** terminal states / `terminal_values` / `terminal_level` are a shared semantic across the
  family. Only base `hidden_markov.py` produces a `MarkovChainLatentPosterior` — the variants still
  return raw `(T,K)` arrays from `seq_posterior`.

### `_hidden_markov_numba_kernels.py` — shared engine-resident kernels (not a public class)
- **Role:** the numba-jit (`cache=True`) forward / forward-backward kernels shared by the HMM
  variants — the **EngineResident** implementation of the forward-backward facet.
- **Formalized in:** `pysp/stats/latent/_hidden_markov_numba_kernels.py` (module-level `@njit` funcs).
- **Kernels:**
      numba_seq_log_density(...)      # batched forward log-likelihood (parallel over sequences)
      numba_baum_welch(...)           # forward-backward E-step; init/transition/state count accumulators
      numba_baum_welch2(...)          # same, per-sequence accumulator arrays (parallel-safe)
      numba_baum_welch_alphas(...)    # forward-backward returning alphas (for state marginals), parallel
- **Imported by:** `hidden_markov.py`, `lookback_hidden_markov_model.py`
  (`numba_seq_log_density`, `numba_baum_welch2`, `numba_baum_welch_alphas`). Tree and integer-association
  models carry their own module-level kernels.
- **Notes:** made bit-identical to the numpy path; default-on when numba is available.

---

## D. Grammar / branching / nonparametric

### `heterogeneous_pcfg.py` — `HeterogeneousPCFGDistribution`
- **Role:** a Chomsky-normal-form **PCFG** (probabilistic context-free generalization of the HMM) with
  heterogeneous terminal-emission distributions; scores a `Sequence[Any]` of terminals.
- **Extra surface (inside-outside parsing):** `_inside(terminal_log_density):231`,
  `_inside_outside(...) -> (ll, root, binary, terminal):260`, `_log_density_from_nonterminal:336`,
  `to_fisher` (inside-outside Fisher view), `quantized_index`, `enumerator()`,
  `sampler` (recursive derivation with `max_depth`/`max_steps` budgets).
- **Estimators:** `HeterogeneousPCFGEstimator` (fixed rule structure) and
  `InducedHeterogeneousPCFGEstimator` (structure learner: `max_nonterminals`, `terminal_rule_mass`,
  `prune_threshold`, `min_rule_prob`, with an `initial_model(...)` builder).
- **Facets:** latent-structured (parse forest); enumerable; quantized-indexable; Fisher.

### `dirac_length.py` — `DiracLengthMixtureDistribution`
- **Role:** a 2-component mixture of a **Dirac point mass at `v`** (weight `1-p`) and a length
  distribution (weight `p`); scores an `int`. The "zero-inflation"/length-mixture leaf of the family.
- **Extra surface:** `component_log_density:151`, `posterior:167`, `seq_component_log_density:193`,
  `seq_local_elbo` — i.e. the finite-mixture responsibility facet at K=2 with a degenerate component.
  Enumerator deduplicates the Dirac point against the length support.
- **Estimator:** `DiracLengthMixtureEstimator(estimator, v, fixed_p, …)`.

### `indian_buffet_process.py` — `IndianBuffetProcessDistribution`
- **Role:** finite-truncated **Beta-Bernoulli Indian Buffet Process** — binary feature-allocation rows
  with a variational-Bayes posterior over per-feature inclusion probabilities. Scores a dense
  `[0,1]^K` vector or a sparse list/set of active indices (`data_format="auto"`).
- **Extra surface:** `expected_log_density` / `seq_expected_log_density` (VB E-step),
  `seq_local_elbo:294`, `enumerator()` (best-first product over independent Bernoulli features),
  `to_fisher`. Estimator: `num_features`, `alpha` concentration.
- **Facets:** nonparametric (truncated); exp-family/VB (expected-log-density); enumerable; Fisher.

---

## E. Sets / edit-distance interface

### `SetDistribution` (de-facto contract) — set-valued membership + Beta-conjugate priors
- **Role:** a distribution over **sets** (presence/absence of each support element, independent
  Bernoulli per element), with optional **required / forced membership** (elements with `p=1`) and a
  conjugate Beta prior facet (`set_prior` / `expected_log_density`).
- **Formalized in:** implicit. **Recommend a `SetValued` Protocol** (the `required` attribute +
  per-element membership log-probs).
- **Implemented by:**
    - `bernoulli_set.py` — `BernoulliSetDistribution`: arbitrary hashable support via a `pmap`;
      `required` = elements with `p=1`; conjugate Beta prior attached via `set_prior:145`
      (+ `posteriors`), with `expected_log_density:200` / `seq_expected_log_density:216` for the VB
      bound and `model_log_density` on the estimator (log p(model) under a shared Beta hyperprior).
    - `integer_bernoulli_set.py` — `IntegerBernoulliSetDistribution`: integer domain `[0, N-1]`, stored
      as `log_pvec`/`log_nvec`; `required` indices forced. Estimator takes `num_vals`.
- **Facets:** set-valued; conjugate (Beta); forced-membership; enumerable.

### `EditDistribution` (de-facto contract) — set-to-set edit / transition model
- **Role:** a distribution over a **pair of integer sets** `(prev, next)` parameterized by per-element
  **edit probabilities** (the insert / delete / keep / substitute transition rates), with an
  `init_dist` over the first set. The set-valued analog of a 1-step Markov transition.
- **Formalized in:** implicit (subset of the set facet, with a transition matrix).
- **Implemented by:**
    - `integer_bernoulli_edit.py` — `IntegerBernoulliEditDistribution`: `log_edit_pmat` is a
      `num_vals × {2 or 4}` matrix of per-element transition log-probs (2-col = inclusion given
      missing/present; 4-col = full miss/present × miss/present); `log_dvec` holds the 3 transitions
      relative to (miss|miss); `init_dist` scores the first set. `p(x1,x2) = P_init(x1)·prod_k p(edit_k)`.
    - `integer_step_bernoulli_edit.py` — `IntegerStepBernoulliEditDistribution`: a **step-function
      regularized** variant — after accumulating counts, the estimator fits a two-level step function
      (high prob for the top-k elements, low for the rest) per direction via binomial-likelihood
      maximization (`__effective_step_counts:728`, `__get_pqk:757`); enumerator subclasses the base
      edit enumerator.
- **Facets:** set-valued; edit/transition; (step) regularized parameterization; enumerable.

---

## Coverage checklist (all ~30 modules)

### `pysp/stats/latent/`
- `latent_posterior.py` — **`LatentPosterior` ABC spine** + `CategoricalLatentPosterior`,
  `MarkovChainLatentPosterior`, `MeanFieldLDAPosterior` (exact + mean-field `q(z|x)` objects).
- `__init__.py` — package re-exports (no interface).
- `mixture.py` — finite-mixture EM/responsibility contract; produces `CategoricalLatentPosterior`;
  `conditional` + `posterior_predictive` facets; conjugate-Dirichlet weights.
- `gaussian_mixture.py` — finite-mixture (full-covariance MVN components); responsibility surface.
- `hierarchical_mixture.py` — outer K × inner shared-L mixture; responsibility surface + `len_dist`.
- `heterogeneous_mixture.py` — finite mixture over heterogeneously-typed components; `fixed_weights`.
- `joint_mixture.py` — mixture over paired `(x1,x2)` with joint latent decomposition.
- `semi_supervised_mixture.py` — mixture with exogenous label evidence re-weighting responsibilities.
- `spatial_mixture.py` — Potts-MRF grid mixture, mean-field EM; **non-standard `fit`-style class**
  (`responsibilities/labels/entropy/component`), not the full ABC stack.
- `probabilistic_pca.py` — continuous latent-factor Gaussian; `transform(x)->E[z|x]` as the E-step.
- `lda.py` — LDA; produces `MeanFieldLDAPosterior`; ELBO `seq_log_density` + `posterior_predictive`.
- `labeled_lda.py` — supervised LDA (per-doc label sets, coupled per-label Dirichlet).
- `integer_probabilistic_latent_semantic_indexing.py` — PLSI (doc × state × word factorization).
- `hidden_markov.py` — **reference HMM**: forward-backward/Baum-Welch/Viterbi/FFBS, terminal states,
  produces `MarkovChainLatentPosterior` + `posterior_predictive`; conjugate Dirichlet prior.
- `lookback_hidden_markov_model.py` — higher-order HMM (emissions depend on previous `lag` obs).
- `tree_hidden_markov_model.py` — branching latent-tree HMM (upward-downward, `terminal_level`).
- `quantized_hidden_markov_model.py` — count-DP `theta^k/Z` HMM (subclass of base HMM); enumerable.
- `segmental_hidden_markov_model.py` — duration/segment HMM (states emit variable-length segments).
- `semi_supervised_hidden_markov_model.py` — HMM with per-position soft state priors.
- `hidden_association.py` — set/bag conditional-association sequential model; `sample_given`, enumerable.
- `integer_hidden_association.py` — integer numba-accelerated hidden-association; enumerable.
- `_hidden_markov_numba_kernels.py` — shared engine-resident numba forward/Baum-Welch kernels.
- `heterogeneous_pcfg.py` — CNF **PCFG**, inside-outside parsing; fixed + induced (structure-learning)
  estimators; enumerable, quantized-indexable, Fisher.
- `dirac_length.py` — Dirac-point + length-distribution 2-mixture over `int`; K=2 responsibility facet.
- `indian_buffet_process.py` — truncated Beta-Bernoulli **IBP** feature allocation; VB expected-log-
  density, enumerable, Fisher.

### `pysp/stats/sets/`
- `__init__.py` — package re-exports (no interface).
- `bernoulli_set.py` — **`SetDistribution`**: per-element Bernoulli over arbitrary support; `required`
  forced membership; conjugate Beta prior (`set_prior`, `expected_log_density`, `model_log_density`).
- `integer_bernoulli_set.py` — `SetDistribution` over integer domain `[0,N-1]` (`log_pvec`/`log_nvec`).
- `integer_bernoulli_edit.py` — **`EditDistribution`**: set-to-set `(prev,next)` per-element edit/
  transition model + `init_dist`.
- `integer_step_bernoulli_edit.py` — `EditDistribution` with step-function-regularized edit rates.
