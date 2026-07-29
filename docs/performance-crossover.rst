Performance Crossover: Where Specialized Packages Win
=====================================================

Mixle's value proposition is composition and heterogeneity. This page explains
how to measure the overhead relative to a specialized single-model library
without presupposing the result.

The short version
-----------------

For a single standard model, compare Mixle with a version-pinned specialized
implementation on the exact workload. Mixle also supports compositions such as
a mixture inside an HMM state, a neural leaf beside a classical one, and a record
of heterogeneous fields; compare only systems that express the same model.

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

Said plainly, because the paragraph above describes this without ever conceding it:
**for a standalone Gaussian mixture, scikit-learn wins.** Its ``GaussianMixture`` is
faster than mixle's general path, and mixle does not claim to overtake it on that
comparison. The same holds for hmmlearn on a standalone HMM. That is the expected
outcome of the design, not a defect to be explained away -- a fused single-purpose
kernel should beat a composable one at the single purpose it was fused for. What
mixle offers instead is that the *same* mixture drops into an HMM state, a record
field, or a neural mixture unchanged, which neither of those packages offers at all.

Naming the loss is the point. A comparison document that only lists methodological
caveats reads as an argument that the comparison cannot be made, and a reader is
right to discount it.

So the honest framing is:

* **specialized cases:** for a standalone GMM or HMM, make no performance
  conclusion until an exact-candidate benchmark measures both implementations;
* **generality overhead vs kernel inefficiency:** the cost is the composable
  encode/accumulate/M-step machinery, paid on every fit; it is the price of
  composition, not evidence of a different or worse estimator.
* **composition cases:** compare model semantics first, so a timing table does
  not silently compare different likelihoods.

GPU and backend numbers
-----------------------

No GPU or distributed-backend performance number is claimed for 0.8.0 without a
retained exact-candidate hardware receipt. Capability support is reported
separately from latency and throughput. Any future performance claim must state
which quantity it measures and name the candidate, system, workload, and receipt.
