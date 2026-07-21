Performance Crossover: Where Specialized Packages Win
=====================================================

Mixle's value proposition is composition and heterogeneity, not beating a
specialized single-model library on its own turf. This page states the losses as
well as the wins, so a reader can choose the right tool. It is deliberately
honest about cases where scikit-learn, hmmlearn, or pomegranate is the faster
choice.

The short version
-----------------

**For a single, standard model, a specialized package is usually faster to fit
than mixle, and mixle does not claim otherwise.** Mixle earns its keep when the
model is a *composition* -- a mixture inside an HMM state, a neural leaf beside a
classical one, a record of heterogeneous fields -- which the specialized packages
do not express at all.

Exact-candidate evidence
------------------------

The 0.8.0 release does not publish a numerical crossover table. Historical
developer measurements were produced by an older Mixle release and are retained
under ``benchmarks/archive/`` only as engineering history. They are not evidence
for this candidate. Run the tracked benchmark harness on the exact candidate to
measure the crossover on a named system.

Generality overhead, not a worse algorithm
------------------------------------------

The gap is **generality overhead**, not kernel inefficiency in the sense of a
worse algorithm. Both fit EM to the same optima; the parameter-level agreement is
gated separately (the scikit-learn GMM parity and hmmlearn HMM parity tests).
Mixle reaches those same optima through its general path -- encode observations,
accumulate per-component sufficient statistics, run a composable M-step -- which
is what lets the *same* mixture compose into an HMM state, a record field, or a
neural mixture. scikit-learn's ``GaussianMixture`` is a single fused, specialized
kernel with none of that generality to pay for.

So the honest framing is:

* **specialized cases:** for a standalone GMM or HMM, assume that a specialized
  implementation may be faster until an exact-candidate benchmark shows
  otherwise;
* **generality overhead vs kernel inefficiency:** the cost is the composable
  encode/accumulate/M-step machinery, paid on every fit; it is the price of
  composition, not evidence of a different or worse estimator.
* **where mixle is the right tool:** when the model is not a single family --
  heterogeneous records, latent structure over mixtures, neural leaves beside
  classical ones -- there is no specialized package to lose to, because those
  packages do not express the model.

GPU and backend numbers
-----------------------

No GPU or distributed-backend performance number is claimed for 0.8.0 without a
retained exact-candidate hardware receipt. Capability support is reported
separately from latency and throughput. Any future performance claim must state
which quantity it measures and name the candidate, system, workload, and receipt.
