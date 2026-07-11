What Mixle Is Not
=================

Mixle is a probabilistic-modeling library: it specifies, fits, and composes models of heterogeneous
data, with calibration and deferral built in. Knowing what it is *not* is the fastest way to decide
whether it fits your problem, and what to reach for alongside it. See :doc:`maturity` for how mature
each surface is, and :doc:`index` for the full scope.

Not a general-purpose supervised-ML / AutoML framework
------------------------------------------------------

If your task is a pure supervised prediction on fixed feature vectors — gradient-boosted trees on
tabular data, a fine-tuned classifier — a dedicated tool (scikit-learn, XGBoost/LightGBM, a trainer)
is usually the more direct choice. Mixle models the *generative* structure of heterogeneous data and
its uncertainty; it earns its place when you need a calibrated model, a latent/temporal structure, or
to compose classical, neural, and latent pieces in one fit — or when you want to distill an expensive
model into a small local one that knows when to defer (``mixle.task``).

Not a dedicated MCMC-first probabilistic programming language
-------------------------------------------------------------

Mixle does Bayesian inference (conjugate, variational, and NUTS) and ``mixle.ppl`` offers compact model
expressions, but that surface is still in active development. For large hierarchical Bayesian models
that lean on extensive MCMC diagnostics and a mature sampler ecosystem, a specialized PPL (Stan, PyMC,
NumPyro) is the deeper tool. Reach for ``mixle.ppl`` when the model composes naturally with the
stats/inference layer; check the generated model and route before depending on it.

Not a frontier pretraining or distributed-training platform
-----------------------------------------------------------

Mixle sits *above* the trainer. Its neural and language-model helpers, and the tensor/pipeline/context
parallelism plan, are the composition and orchestration layer that a real training run would sit on —
not a replacement for a dedicated large-scale trainer (Megatron, DeepSpeed, TorchTitan). Multi-node,
multi-GPU frontier pretraining is explicitly out of scope for this release; the parallelism knobs
validate a plan and, without an explicit opt-in, refuse to silently run a different one. Use mixle to
wrap, fine-tune, distill, and calibrate models; use a dedicated trainer to pretrain at frontier scale.

Not an MLOps, serving, or deployment platform
---------------------------------------------

``mixle.inference.production`` provides practical helpers — a local model registry, calibration,
cascades, a cost model — not a full deployment system: no autoscaling, no multi-tenant gateway, no
managed infrastructure. Serving, gateways, and cross-model routing at the service layer live in the
companion project ``mixle-mlops``. Keep deployment claims bounded to what the helpers actually do.

When mixle is the right tool
----------------------------

Reach for mixle when the problem is heterogeneous data that a single model should describe end to end;
when a classical distribution, a neural network, and a latent-variable model need to compose and fit in
one call; when you need calibrated uncertainty and principled deferral; or when you want to distill a
slow, expensive model into a cheap local one with a cost/quality receipt. For anything outside that,
compose mixle with the specialized tool rather than bending one to do the other's job.
