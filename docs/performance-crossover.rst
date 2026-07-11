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

A measured example: plain Gaussian mixture
------------------------------------------

Fitting a plain one-dimensional 3-component Gaussian mixture (EM, 30 iterations,
matched single random initialization), mixle versus scikit-learn's
``GaussianMixture`` on one developer laptop, best of three runs after warmup:

============  =====================  =====================  =================
N (rows)      mixle fit time         scikit-learn fit time  faster
============  =====================  =====================  =================
200           ~2 ms                  ~0.5 ms                scikit-learn ~4x
2,000         ~4 ms                  ~1 ms                  scikit-learn ~4x
20,000        ~25 ms                 ~5 ms                  scikit-learn ~5x
200,000       ~174 ms                ~48 ms                 scikit-learn ~4x
============  =====================  =====================  =================

These numbers are illustrative -- one machine, one configuration, absolute
timings will differ -- but the *direction* is stable and is the point:
scikit-learn wins this comparison at every size, by roughly 3-5x. There is no N
at which mixle overtakes it for this model. Do not read this table as a promise
of specific milliseconds; read it as "expect a several-times gap on a
specialized package's home turf."

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

* **small-N and large-N alike:** for a standalone GMM, scikit-learn wins; for a
  standalone HMM, hmmlearn typically wins; a standalone single family often has a
  specialized package that is faster than the general path.
* **generality overhead vs kernel inefficiency:** the cost is the composable
  encode/accumulate/M-step machinery, paid on every fit; it is the price of
  composition, not evidence of a different or worse estimator.
* **where mixle is the right tool:** when the model is not a single family --
  heterogeneous records, latent structure over mixtures, neural leaves beside
  classical ones -- there is no specialized package to lose to, because those
  packages do not express the model.

GPU and backend numbers
-----------------------

Where GPU or distributed-backend numbers appear elsewhere in the documentation,
they are **throughput or capability demonstrations, not latency wins** over a
tuned CPU kernel at small N. A GPU engine helps large batched scoring and large
neural leaves; it does not make a small classical fit lower-latency than a
specialized CPU library. Any performance claim should state which of throughput,
latency, or capability it is demonstrating.
