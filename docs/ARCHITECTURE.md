# pysparkplug — target architecture (reorganization proposal)

## The problem

The library grew object-first: everything is filed under the *thing* it belongs to
(`stats/leaf/gaussian.py`, `stats/latent/hidden_markov.py`). But the important structure is
**cross-cutting** — enumeration, sampling, inference, and the operations that transform one object
into another. Those concerns are currently smeared across `utils/`, `stats/compute/`, `stats/bayes/`,
`pdist.py`, and a pile of tiny `_`-prefixed files, so:

- you cannot *see* that relations and distributions are both enumerable — the contract lives in three
  files and nothing declares the shared concept;
- there is no model of **operations**: "if I quantize a distribution, what does it become?" has no
  answer because quantization isn't an operation, it's machinery buried in `utils/quantization`;
- the capability vocabulary is correct but scattered across ~8 modules.

## The organizing principle

> **Concerns are first-class modules that own a contract + its algorithms. Objects implement the
> contracts. Operations move objects between capability sets.**

Three kinds of thing, three kinds of module:

1. **Concerns** (`enumeration`, `sampling`, `inference`, `ops`, `compute`) — each owns *one* contract
   and the algorithms that consume it. Enumeration owns `Enumerable` + the k-best/unrank machinery;
   anything enumerable plugs in.
2. **Objects** (`dist`, `graph`, `process`, `relations`) — the families. They *implement* concern
   contracts; they don't define them.
3. **The kernel** (`contracts`, `capability`) — the universal object cast and the meta-layer that
   answers "what does this support?" by reading the concern contracts.

A `GaussianDistribution` *is a* Distribution (score/sample/estimate) but is **not** `Enumerable`.
A `Categorical` *is a* Distribution **and** `Enumerable` (from `pysp.enumeration`). A `Relation` is
`Enumerable` + `Sampler`. `pysp.ops.quantize(gaussian)` returns an object that *became* `Enumerable` +
`RankableByIndex`. The structure is the same everywhere and you can read it off the imports.

## Target layout

```
pysp/
  capability.py      meta-layer: supports / describe / catalog / require / what_supports
  contracts.py       the object cast: Distribution, Sampler, Estimator, Encoder, Accumulator, Factory
                     + light facets (Conditionable, ExponentialFamily, ConjugateUpdatable, SetValued)

  enumeration/       CONCERN — Enumerable / Enumerator / RankableByIndex; descending-probability
                     k-best; count-budget seek + unrank; the count semiring.
                     ← utils/enumeration, utils/model_enumeration, utils/quantization,
                       DistributionEnumerator (from pdist)
  sampling/          CONCERN — Sampler / ConditionalSampler; the unified sample(); LatentPosterior;
                     posterior_predictive.            ← stats/_sampling, sampling_api, latent_posterior
  inference/         CONCERN — Estimator / EMStrategy; MLE-EM, MAP, conjugate Bayes, MCMC, VI; the
                     fitter registry; Fisher geometry.
                     ← utils/{estimation,em,fit,objectives,fisher}, stats/bayes/conjugate, utils/mcmc
  ops/               OPERATIONS — quantize, truncate, condition, marginalize, mixture, transform,
                     tilt. Each documents its capability signature.   ← wraps combinators + new quantize

  dist/              OBJECTS (was stats) — leaf/ multivariate/ combinator/ latent/ priors/
  graph/             graph / ranking / set families            ← stats/graph + stats/sets
  process/           temporal / point-process families         ← the leaf hawkes/poisson/birth-death
  relations.py       Relation — implements enumeration + sampling

  engines/           numpy / torch / symbolic backends (ComputeEngine)
  compute/           declarations, kernels, exp-family stacked machinery   ← stats/compute

  ppl/ doe/ uq/ models/ data/    APPLICATIONS
  utils/             only genuinely generic helpers: special fns, vector, serialization, arithmetic, parallel
```

## How it answers the two questions

**"How do I know relations and distributions are both enumerable?"**
`pysp.enumeration` defines `Enumerable` once. `Categorical` and `Relation` both implement it; the
catalog and `pysp.what_supports(Enumerable, ...)` list them; the module's docstring is the one place
that says "these are the enumerable things." The concept has a home.

**"If I quantize a distribution, what does it become and how?"**
```python
from pysp import ops, describe
q = ops.quantize(GaussianDistribution(0, 1), bits=8)   # Distribution -> discrete on bins
describe(q)   # ... can: ... Enumerable · FiniteSupport · RankableByIndex ...
```
`ops.quantize` lives in the operations module with an explicit capability signature
(`any Distribution -> FiniteSupport ∧ Enumerable ∧ RankableByIndex`), implemented by binning into a
`Categorical`/`IntegerCategorical` over `pysp.enumeration`'s index. The operation is the answer.

## The capability-signature table for operations

| Operation | input requires | output gains |
|---|---|---|
| `quantize(dist, bits)` | Distribution (with cdf/quantile) | FiniteSupport · Enumerable · RankableByIndex |
| `truncate(dist, region)` | Distribution | (renormalized) Distribution; Enumerable if dist was |
| `condition(dist, observed)` | Conditionable | Distribution |
| `marginalize(dist, keep)` | Marginalizable | Distribution |
| `mixture(dists, w)` | Distributions | LatentStructured |
| `transform(dist, f)` | Distribution + invertible f | Distribution (Jacobian-corrected) |
| `tilt(dist, theta)` | ExponentialFamily | Distribution |

This table *is* the missing third axis. It belongs in `pysp.ops` and is rendered into
[`CAPABILITIES.md`](CAPABILITIES.md) next to the capability vocabulary.

## Migration strategy (how to do it without breaking the world)

The reorg is import churn across the tree; on an unmerged branch with a concurrent session it would be
a brutal merge. So:

1. **Re-export shims, not moves.** New concern modules `import` and re-export from the current
   locations first (`pysp/enumeration/__init__.py` re-exports `utils.enumeration` + the `Enumerable`
   capability). Zero behavior change, old paths keep working — the *structure* appears immediately.
2. **One concern at a time**, each its own PR: `enumeration` first (the cleanest, self-contained, and
   the exemplar), then `ops` (new, additive — adds `quantize`), then `sampling`, then `inference`.
3. **Flip the canonical home** per concern once the shim is in place: move the implementation into the
   concern module, leave a thin re-export at the old path (deprecated), update internal imports.
4. **Do it after `consistency-fixes` merges** (or accept the merge cost) so the reorg isn't racing the
   concurrent session.
5. The object families (`leaf/` etc.) move last and are mostly path renames (`stats` → `dist`), behind
   a `pysp.stats` re-export shim so external code is unaffected.

## Recommended first step

Build **`pysp/enumeration`** as the exemplar concern module: it re-exports today's
`utils/enumeration` + `utils/quantization` + the `Enumerable`/`RankableByIndex` capability + the
`DistributionEnumerator` contract under one roof, and add **`pysp/ops.py`** with a real `quantize` (+
the verb surface). Those two modules alone deliver the coherence — enumeration becomes a place, and
operations become an axis — and concretely answer both questions, with shims so nothing breaks.

---

## Status — namespaces live; self-contained concern machinery physically relocated

The structure above is **live**. Every namespace is reachable as `pysp.<ns>` (lazy `__getattr__`, so
`import pysp` stays cheap) or `from pysp.<ns> import …`, and **every existing import still works**.
All three concern packages now physically *own* their machinery (the files were `git mv`d in, with
thin re-export shims left at the old paths); only the object namespaces remain re-export shims.

| Module | Role | Physical state | Backed by |
|---|---|---|---|
| `pysp.enumeration` | concern | **package — machinery moved in** | `algorithms` (was utils.enumeration) + `quantization/` (was utils.quantization) + `model_enumeration` + `density_rank` (descending-probability rank/CDF, was utils.density_rank) + DistributionEnumerator (pdist) + Enumerable |
| `pysp.sampling` | concern | **package — machinery moved in** | `sampling_api` + `latent_posterior` + `_sampling` (were under stats) + pdist samplers + PosteriorPredictive |
| `pysp.inference` | concern | **package — machinery moved in** | `estimation` / `em` / `fit` / `objectives` / `fisher` / `priors` (were utils) + `mcmc/` (were utils.mcmc) + `target` / `backends` / `diagnostics` (the NUTS/ADVI facade, were `pysp.infer`) + bayes.conjugate (re-exported) |
| `pysp.ops` | operations | self-contained module | new (quantize) + the combinators, capability-gated |
| `pysp.contracts` | kernel | re-export shim | every ABC/Protocol in one import (capabilities eager, subsystem roles lazy) |
| `pysp.dist` / `pysp.process` / `pysp.models` | objects | re-export shims (by design) | aliases of stats / the point-process families / the generic-model package |

**Why the object namespaces stay as re-export shims (not a TODO — a decision).**
The distribution families (`pysp.dist`, `pysp.process`, `pysp.models`) keep their canonical home in
`pysp.stats` because **serialization type-ids are the fully-qualified module path** (`_type_id(cls) =
"{cls.__module__}.{cls.__name__}"`); physically moving a distribution would change its type-id and
break loading of previously-saved models.

**The inference machinery moved cleanly once the `stats` coupling was cut at its root.** `em`/
`estimation` used to import the *vectorized `seq_*` drivers* (`seq_encode` / `seq_estimate` /
`seq_initialize` / `seq_log_density_sum`) straight out of `pysp.stats.__init__`, where they were
defined inline — yet the stats leaves import this same machinery *during* `pysp.stats` import, so the
inference package could only re-enter a half-built `pysp.stats`. The real fix was to recognize those
`seq_*` functions for what they are — protocol dispatchers over the `pdist` contracts, not part of the
object surface — and move them to **`pysp.stats.compute.sequence`** (`pysp.stats` re-exports them, so
the public `pysp.stats.seq_estimate` API is unchanged). The inference machinery now imports only the
*compute layer* (`compute.{pdist,sequence}`), never the `pysp.stats` package surface, so
`pysp/inference/__init__.py` is a **plain eager package init** — no `__getattr__`.

**All sampling-based inference lives in `inference`.** Inference is one concern — *infer parameters
from data* — whose entry points differ only in what they **require of the input**: closed-form
conjugate Bayes needs a `ConjugateUpdatable` family; MLE/EM needs an `Estimator`; NUTS/ADVI need a
*sampleable / differentiable target*. That precondition is the whole reason MCMC/VI used to look like
a separate thing — it isn't, so the old `pysp.infer` facade and `pysp.utils.mcmc` samplers were folded
in (`pysp.inference.{target,backends,diagnostics,mcmc}`); `pysp.infer` is now a deprecated shim.
`conjugate` (bayes-family) keeps its home in `pysp.stats.bayes` and is re-exported. The only remaining
dividing line is serialization, which pins the *distribution objects* to `pysp.stats`.

---

## Revision — taxonomy review

Per review, two corrections to the namespace layout above:

- **A Markov chain is a distribution (over sequences/state-paths), not a graph.** The structured
  families are therefore *not* split by structure type. The object namespaces are **minimal**:
  `pysp.dist` is the umbrella over **every** distribution — graphs, rankings, sets, Markov chains,
  grammars all live here — and `pysp.process` holds the stochastic processes (Hawkes / Poisson /
  birth-death / CRP). **`pysp.graph` is dropped** (it was over-splitting; its members are reachable
  from `pysp.dist`).
- **`pysp.ops` is kept as-is.** It holds both the genuine unary transformations (`quantize`,
  `condition`, `marginalize`) and the thin combinator constructors (`truncate`, `mixture`,
  `transform`, `tilt`); the slight overlap with the combinator object layer is accepted.
