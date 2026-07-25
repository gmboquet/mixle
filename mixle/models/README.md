# `mixle.models` — applied model families

`mixle.stats` holds elementary distributions such as Gaussian, Poisson, and categorical families.
`mixle.models` holds applied model families that go beyond a single pointwise density: neural leaves,
Gaussian processes, random forests, knowledge graphs, grammars, decision processes, and causal discovery
helpers.

They share one purpose: richer, domain-specialized model families exposed through Mixle-compatible
contracts, so they can compose with the statistical core where appropriate. A neural leaf can drop into a
`CompositeDistribution`; a Gaussian process can act as a mixture component; a grammar can score a
structured record. Where a family is supervised, sequential-decision, or causal rather than a plain
generative density, it keeps a focused task-appropriate surface instead of forcing the full distribution
contract.

Maturity varies across this namespace. Treat these as specialist adapters that compose with the stable
statistics layer, and validate them on the target workflow before making production claims.

## Families

| Family | Modules | What it gives you |
| --- | --- | --- |
| **Neural & deep** | `neural`, `neural_leaf`, `neural_density`, `mixture_density`, `energy`, `softmax_leaf`, `transformer`, `language_model`, `streaming_transformer_leaf`, `embedding`, `dpo_leaf`, `continual`, `train_search`, `eval_harness` | Neural networks as mixle leaves — **adapters** that make a Torch density model a composable `Distribution`. `NeuralGaussian` wraps a Gaussian conditional; `NeuralDensity` wraps unconditional densities such as flows, VAEs, categorical autoregressive models, and energy models; `NeuralConditionalDensity` wraps true per-observation conditional densities such as MDNs, conditional flows, and conditional categorical autoregressive models. These fit jointly with classical families by EM. Batch-dependent objectives stay separate: `build_contrastive_projection` provides an explicitly weighted InfoNCE objective and retrieval ranker, not a probability leaf. The package also includes a causal Transformer LM, embeddings, DPO, continual learning, training search, and evaluation utilities. |
| **Neural & deep** | `neural`, `neural_leaf`, `neural_density`, `mixture_density`, `energy`, `softmax_leaf`, `transformer`, `language_model`, `streaming_transformer_leaf`, `embedding`, `dpo_leaf`, `continual`, `train_search` | Neural networks as Mixle leaves: adapters that make a torch model a composable `Distribution`. Three axes: `NeuralGaussian` wraps a conditional net as a single-Gaussian `p(y\|x)`; `NeuralDensity` wraps unconditional density modules as `p(x)`; `NeuralConditionalDensity` wraps conditional densities such as MDNs, conditional flows, and autoregressive categorical models. These can fit jointly with classical families through the same EM M-step pattern. The namespace also includes a causal-Transformer LM, shared `CategoricalEmbedding`, DPO, continual-learning utilities, and multi-fidelity DOE over training recipes (`tune_training`). |
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
from mixle.models import CategoricalEmbedding, TransformerLMEstimator
from mixle.stats import MixtureEstimator

# language-model experts that share one token embedding table
embedding = CategoricalEmbedding(num_categories=8000, dim=256, name="word")
experts = [
    TransformerLMEstimator(8000, d_model=256, n_layer=4, block=64, embedding=embedding)
    for _ in range(3)
]
model = optimize(token_windows, MixtureEstimator(experts), max_its=20)
```
