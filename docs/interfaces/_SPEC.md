# Interface-cataloging spec (read by every cataloging agent)

You are cataloging the **interfaces** (type contracts) of pysparkplug for a master design outline.
Working dir `/Users/grantboquet/codex/pysparkplug`. READ-ONLY except your one assigned section file.

## What an "interface" is here
The library has ~1700 concrete classes but only a few formal ABCs — most contracts are **implicit
/ de-facto** (followed by convention). Your job is to surface the *contracts*, not enumerate every
class. An interface is a reusable role/capability with a method surface, e.g.:
- the core contract ABCs (`SequenceEncodableProbabilityDistribution`, `DistributionSampler`,
  `DataSequenceEncoder`, `StatisticAccumulator`, `StatisticAccumulatorFactory`, `ParameterEstimator`,
  `DistributionEnumerator`),
- capability facets (Enumerable, Rankable-by-index, ExponentialFamily, Conditionable, Conjugate,
  EngineResident, LatentStructured) — some implicit, name them,
- package-specific roles (combinator composition, conjugate posterior, acquisition function, GP
  surrogate, inference backend, random variable, relation, compute engine, kernel, …).

## The three orthogonal axes (frame interfaces against these)
1. **Contract** — what the object IS (the distribution/estimator/sampler cast).
2. **Capability facet** — what it CAN DO (score / sample / estimate / enumerate / rank-from-index /
   engine-reside / exp-family / conjugate / condition / latent). Orthogonal — a model holds several.
3. **Composition depth** — leaf → combinator → latent-state (HMM/Markov/PCFG) → Bayesian/nonparametric.

## Output — write your section file
Write `docs/interfaces/sections/<NN-area>.md` (path in your task). For EACH interface in your scope:

```
### `InterfaceName`  — [ABC | Protocol | de-facto contract]
- **Role:** one line.
- **Formalized in:** `path.py` (or "implicit — followed by convention").
- **Methods:**
    method(self, arg: T, ...) -> Ret      # one-line contract / semantics
    ...
- **Implemented by:** modules / families (be specific; list them).
- **Facets:** which capability facets it grants (enumerable, exp-family, …).
- **Notes:** conjugacy, engine-residency, composition role, or "should be formalized as a Protocol".
```

Then end with a **coverage checklist**: every module in your assigned scope on its own line,
`path.py — <interface(s) it realizes / its role>`, so we can verify nothing is missed.

## Method
- Read enough of each module to get the real signatures (the `def`s on the public classes and the
  ABCs in `pysp/stats/compute/pdist.py`). Prefer accuracy over completeness of prose.
- Group the many concrete distributions under the contract they share — don't repeat the full
  Distribution interface per leaf; state it once and list which modules realize it (plus any
  *extra* surface a family adds, e.g. point processes' event-time validation, MVN's condition/marginal).
- Be precise about signatures (arg names + types + return). Cite `file:line` for the defining ABC.

## Part 2: return to me
A compact list: the interface names you cataloged, anything that is an implicit contract you
recommend formalizing as a Protocol/ABC, and the section file path. Keep it short — the detail is in
the file.
