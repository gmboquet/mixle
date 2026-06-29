"""Structured latent-variable models — LDA, IBP, and "models like that" — are first-class in mixle.

mixle ships a broad catalog of structured-latent models (all in ``mixle.stats``), each a full 5-part
family (Distribution / Sampler / Estimator / Accumulator / DataEncoder) fit by mean-field VI / EM:

  * admixtures / topic models -- ``LDADistribution`` (latent Dirichlet allocation), ``LabeledLDADistribution``,
    ``IntegerProbabilisticLatentSemanticIndexingDistribution`` (PLSI);
  * latent feature allocation -- ``IndianBuffetProcessDistribution`` (IBP, a nonparametric binary
    feature model);
  * continuous latent factors -- ``ProbabilisticPCADistribution``;
  * the GENERAL admixture base -- ``HierarchicalMixtureDistribution``: an outer mixture over per-group
    mixtures with *shared topics of any family*. LDA is its Categorical-topic special case; the same
    machinery does Gaussian / Poisson / heterogeneous topics, so "an LDA-like model over my own emission"
    needs no new class.

This example fits three of them on synthetic data and recovers the planted structure: classic LDA
(categorical topics), IBP (binary features), and -- the point -- a *non-categorical* admixture (Gaussian
topics) that no "topic model = words" framing covers.

To add another "model like that": compose ``HierarchicalMixtureDistribution`` with your topic family
(admixture over anything), reach for the built-in IBP/PLSI/PPCA, or register a bespoke family via
``register_family`` (the 5-part contract). Run: ``python examples/latent_variable_models_example.py``
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import mixle.stats as S
from mixle.inference import optimize


def demo_lda():
    """Latent Dirichlet Allocation: documents are bags of words; recover the topic word-distributions."""
    true = [{0: 0.60, 1: 0.25, 2: 0.10, 3: 0.05}, {0: 0.05, 1: 0.10, 2: 0.25, 3: 0.60}]
    gen = S.LDADistribution(
        [S.CategoricalDistribution(t) for t in true], alpha=[1.0, 1.0], len_dist=S.CategoricalDistribution({14: 1.0})
    )
    docs = [sorted(Counter(u).items()) for u in gen.sampler(seed=1).sample(400)]
    m = optimize(docs, S.LDAEstimator([S.CategoricalEstimator() for _ in range(2)]), max_its=60, out=None)
    topics = [[round(float(c.pmap.get(k, 0.0)), 2) for k in range(4)] for c in m.topics]
    print("LDA (categorical topics):")
    print(f"  true topics:      {[list(t.values()) for t in true]}")
    print(f"  recovered topics: {topics}")


def demo_ibp():
    """Indian Buffet Process: each object owns a sparse set of latent binary features (unbounded)."""
    gen = S.IndianBuffetProcessDistribution(3, alpha=2.0, data_format="dense")
    z = gen.sampler(seed=2).sample(300)
    est = S.IndianBuffetProcessEstimator(3, alpha=2.0, estimate_alpha=False, data_format="dense")
    fit = optimize(z, est, max_its=40, out=None)
    print("\nIBP (latent binary feature allocation):")
    print(f"  per-feature activation probs: {np.round(np.asarray(fit.feature_probs), 2).tolist()}")


def demo_gaussian_admixture():
    """The general admixture (HierarchicalMixture) with GAUSSIAN topics -- LDA beyond word-topics."""
    gen = S.HierarchicalMixtureDistribution(
        [S.GaussianDistribution(-4, 1), S.GaussianDistribution(4, 1)],  # two shared Gaussian "topics"
        [0.5, 0.5],
        [[0.85, 0.15], [0.15, 0.85]],  # two group-level mixing profiles
        len_dist=S.CategoricalDistribution({10: 1.0}),
    )
    seqs = gen.sampler(seed=3).sample(400)
    init = S.HierarchicalMixtureDistribution(
        [S.GaussianDistribution(-2, 1), S.GaussianDistribution(2, 1)],
        [0.5, 0.5],
        [[0.6, 0.4], [0.4, 0.6]],
        len_dist=S.CategoricalDistribution({10: 1.0}),
    )
    fit = optimize(seqs, init.estimator(), prev_estimate=init, max_its=60, out=None)
    means = sorted(round(t.mu, 2) for t in fit.topics)
    print("\nGaussian admixture (HierarchicalMixture, non-categorical topics):")
    print(f"  shared topic means: {means}   (true -4, +4)")


def main():
    print("# Structured latent-variable models in mixle (LDA, IBP, and 'models like that')\n")
    demo_lda()
    demo_ibp()
    demo_gaussian_admixture()
    print(
        "\nAll three are built-in 5-part families fit by the same EM/VI machinery. The admixture base "
        "(HierarchicalMixture) generalizes LDA to ANY topic family -- categorical for LDA, Gaussian here, "
        "Poisson / heterogeneous / structured leaves just as well. Add a new one by composing it, reaching "
        "for IBP/PLSI/PPCA, or registering a bespoke family."
    )


if __name__ == "__main__":
    main()
