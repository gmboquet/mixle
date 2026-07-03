# `mixle.models` — the applied-model layer

`mixle.stats` holds the **elementary** distributions — a Gaussian, a Poisson, a categorical — each a pointwise
density. `mixle.models` holds the models that are **more than one elementary density**: a neural network, a
Gaussian process, a random forest, a knowledge graph, a grammar, a decision process, a causal skeleton.

They are not a grab-bag — they share a purpose: **richer, domain-specialized model families exposed through the
same contract as the stats core**, so they *compose* with it. A neural leaf drops into a `CompositeDistribution`;
a Gaussian process becomes a mixture component; a grammar scores a structured record. Where a family is
supervised, sequential-decision, or causal (not a plain generative density), it keeps a small, task-appropriate
surface instead of forcing the five-piece contract.

Maturity varies (see the Project status table in the top-level README). Treat these as **specialist adapters
composable with the stable stats spine**, not the spine itself.

## Families

| Family | Modules | What it gives you |
| --- | --- | --- |
| **Neural & deep** | `neural`, `neural_leaf`, `neural_density`, `mixture_density`, `energy`, `softmax_leaf`, `transformer`, `language_model`, `streaming_transformer_leaf`, `embedding`, `dpo_leaf`, `continual`, `train_search` | Neural networks as mixle leaves — **adapters** that make a torch model a composable `Distribution`. Three axes: `NeuralGaussian` wraps a conditional net as a single-Gaussian `p(y\|x)`; `NeuralDensity` wraps *any* unconditional density module as `p(x)` (ready instances: `build_coupling_flow` and `build_maf`, exact continuous normalizing flows; `build_vae`, a latent-variable VAE; `build_autoregressive_categorical`, an exact density over discrete vectors; and `EnergyModel`, an energy-based `p(x) ∝ exp(-E(x))` trained by NCE and sampled by Langevin — *approximately* normalized, flagged like the VAE); `NeuralConditionalDensity` wraps *any* conditional density (`build_mdn`, a mixture density network — multimodal, heteroscedastic `p(y\|x)`; `build_conditional_flow`, an exact conditional normalizing flow capturing within-`y` structure; `build_conditional_autoregressive_categorical`, an exact `p(y\|x)` over discrete `y`). All fit jointly with classical families by the same EM M-step. Plus a causal-Transformer LM, a shared `CategoricalEmbedding`, and training utilities — DPO, continual-learning (EWC), and multi-fidelity DoE over the training recipe (`tune_training`). The rule: add neural models when they're new *distributions*, not new architectures. |
| **Non-parametric** | `gaussian_process`, `sparse_gaussian_process`, `random_forest` | Kernel and ensemble regressors as conditional `p(y \| x)` leaves — a GP (exact and sparse/inducing-point) and a random forest, usable as composite/mixture components. |
| **Relational / structured** | `knowledge_graph`, `random_graph`, `grammar` | Generative models over *structure*: a TransE knowledge-graph model, random-graph models (Erdős–Rényi / stochastic-block), and induced PCFG grammars whose `log_density` is the parse likelihood. |
| **Latent-variable** | `dirichlet_process_mixture` | Bayesian-nonparametric mixtures (truncated Dirichlet process) — a mixture whose number of clusters is inferred. |
| **Decision & control** | `partially_observable_markov_decision_process` | Sequential decision under partial observability: belief filtering and POMDP fitting/solving. |
| **Causal discovery** | `dependence` | Constraint-based causal structure learning — conditional-independence tests, a PC skeleton, and v-structure orientation into a partially-directed graph. |

## How this composes with the rest of mixle

- **`mixle.stats` / `mixle.inference`** — every model here is fit with the same `optimize`/`fit` machinery; the
  neural and GP leaves warm-start their gradient M-steps across EM iterations.
- **`mixle.inference.bayesian_network` / `.structure`** — the *automatic* structure learners (dependency
  forests, DAGs, mixtures of DAGs). `dependence` here is the constraint-based (PC) counterpart to those
  score-based learners.
- **`mixle.represent`** — the modality encoders and `CategoricalEmbedding` re-exported here are the
  representation-layer primitives; a neural leaf can consume their shared-space vectors.

## Quickstart

```python
from mixle.inference import optimize
from mixle.models import TransformerLMEstimator          # neural & deep
from mixle.stats import CompositeEstimator, GammaEstimator

# a neural next-token leaf beside a classical timing density, fit together in one call
est = CompositeEstimator((TransformerLMEstimator(vocab=500, d_model=128, n_layer=4, block=64), GammaEstimator()))
model = optimize(events, est, max_its=20)
```
