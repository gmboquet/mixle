"""The epistemic loop: belief tracking, hypothesis portfolios, and EIG-driven action selection.

A library realization of the control loop OBSERVE -> UPDATE -> ABDUCE -> PREDICT -> DISCRIMINATE ->
ACT: maintain a weighted portfolio of typed hypotheses plus an explicit open-world mass (the "none
of the above" slot), reweight it against new observations through a pluggable likelihood strategy,
optionally propose new hypotheses when the evidence surprises every current one, pick the next
observation/action by expected information gain, and log every step to a replayable, JSON-serializable
decision journal.

This module is built entirely on *existing* mixle contracts rather than inventing new ones:
:mod:`mixle.inference.mcmc` for the SMC resampling precedent, :mod:`mixle.doe.active` /
:mod:`mixle.doe.oracle` for expected-information-gain estimation and verifiability tiers,
:mod:`mixle.evolve.ledger` for the append-only JSON-serializable journal shape, and
:mod:`mixle.data.exchangeability` for the permutation-test precedent behind the coherence checks.
Nothing here fits a specific scientific domain: every test and example uses synthetic toy models.

Scope, deliberately narrow (see ``notes/epistemic-loop-integration-workplan.md`` for the full design
and the two source specification documents it distills):

* **In scope:** :class:`~mixle.epistemic.portfolio.HypothesisPortfolio` (typed weighted hypotheses +
  open-world mass), :mod:`~mixle.epistemic.discrepancy` (KL/JS/Wasserstein/MMD between distributions
  or samples -- the "compare predicted vs. observed" hinge), :mod:`~mixle.epistemic.likelihood`
  (pluggable reweighting strategies at a declared verifiability tier),
  :func:`~mixle.epistemic.loop.step` (one loop iteration: update, optional abduction on surprise,
  optional EIG-based action selection), :class:`~mixle.epistemic.journal.EpistemicJournal`
  (append-only, replayable decision log), and :mod:`~mixle.epistemic.coherence` (exchangeability /
  martingale / evidence-conservation checks as plain testable functions).
* **Out of scope, explicitly:** modality encoders/decoders (use :mod:`mixle.represent` /
  :mod:`mixle.reason` at their current scope), any data corpus or training recipe, a simulator farm
  or MCP tool encapsulation (callers supply their own likelihood/action callables), RL, grammar-
  constrained token decoding, and any named scientific domain. Those remain future, separate work.
"""

from __future__ import annotations

from mixle.epistemic.coherence import (
    evidence_conservation_violation,
    exchangeability_violation,
    martingale_violation,
)
from mixle.epistemic.discrepancy import (
    DiscrepancyResult,
    discrepancy_report,
    js_divergence,
    kl_divergence,
    mmd,
    wasserstein_distance,
)
from mixle.epistemic.journal import DecisionRecord, EpistemicJournal
from mixle.epistemic.likelihood import CallableLikelihood, DiscrepancyLikelihood, LikelihoodStrategy
from mixle.epistemic.loop import EpistemicStep, step
from mixle.epistemic.portfolio import Hypothesis, HypothesisPortfolio

__all__ = [
    # discrepancy.py
    "DiscrepancyResult",
    "discrepancy_report",
    "kl_divergence",
    "js_divergence",
    "wasserstein_distance",
    "mmd",
    # portfolio.py
    "Hypothesis",
    "HypothesisPortfolio",
    # likelihood.py
    "LikelihoodStrategy",
    "DiscrepancyLikelihood",
    "CallableLikelihood",
    # loop.py
    "EpistemicStep",
    "step",
    # journal.py
    "DecisionRecord",
    "EpistemicJournal",
    # coherence.py
    "exchangeability_violation",
    "martingale_violation",
    "evidence_conservation_violation",
]
