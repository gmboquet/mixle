A Torch Module Is A Distribution
=================================

A torch module is a distribution — the training code you didn't write. Any module exposing
``log_density(batch)`` fits with one call: no training loop, no batching/eval/convergence
boilerplate, no adapter classes. And because the fitted leaf *is* a distribution, it composes with
classical families and fits jointly by EM.

Quickstart
----------

.. code-block:: python

   import torch
   from mixle.inference import optimize
   from mixle.stats import GammaDistribution, MixtureDistribution

   class Flow(torch.nn.Module):        # your module: forward and objective, nothing else
       def log_density(self, x): ...   # (n, d) -> (n,)

   fitted = optimize(x, Flow())        # the loop, batching, eval, convergence — manufactured
   fitted.module                       # the raw torch module back — nothing is trapped

   # ...and it composes: a flow and a Gamma in ONE mixture, fit jointly by EM
   mix = MixtureDistribution([fitted, GammaDistribution(2.0, 1.0)], [0.5, 0.5])

A bare ``nn.Module`` that exposes ``log_density(x) -> (n,)`` coerces into a fitted leaf through
``optimize`` with no wrapper at all. Wrap it explicitly with :mod:`mixle.models`' ``GradLeaf`` when
you need to set knobs (``m_steps``, ``lr``, ``device``) or install hooks:

.. code-block:: python

   from mixle.models import GradLeaf

   leaf = GradLeaf(Flow(), m_steps=80, lr=1e-3, loss=my_loss, optimizer=my_optimizer)
   fitted = optimize(x, leaf, max_its=10, out=None)

The module owns forward and objective; mixle owns the loop. The only contract a module has to
satisfy is ``log_density(x) -> (n,)`` for scoring (also the default M-step objective) and, only if
you draw samples, ``sample(n) -> (n, d)``.

Control Never Leaves You
-------------------------

The whole point of wrapping a bare module instead of asking for a rewrite is that none of the usual
escape hatches disappear:

* **Freeze a backbone.** ``requires_grad_(False)`` on any submodule works as expected — the
  optimizer built by ``GradLeaf``/``GradEstimator`` only ever sees trainable parameters, so a
  projection head can train against a frozen encoder, or a LoRA-style adapter can train its
  low-rank delta over a base that never moves. A fully frozen module makes the M-step a no-op: a
  fixed distribution.
* **Hook the objective or the optimizer.** ``GradLeaf(module, loss=..., optimizer=...)`` overrides
  the default responsibility-weighted negative-log-likelihood M-step and the optimizer
  construction — custom objectives are a hook, not a subclass tree.
* **Drop back to raw torch at any time.** ``fitted.module`` is always the same torch module you
  handed in — nothing mixle does traps it. Everything above the module is still ordinary torch:
  parameters, ``state_dict()``, autograd.
* **Scale as a flag, not a rewrite.** ``build_causal_lm(..., gradient_checkpointing=True)`` trades
  recompute for activation memory on deep stacks or long blocks; the flag is a plain attribute so it
  can also be toggled on an existing model.

The receipts cover the manufactured loop and mixle's own leaves; frontier-scale multimodal stacks
remain torch/DeepSpeed territory — bring the trained module back as a leaf.

Receipts
--------

The claims above are pinned by tests, not prose:

* ``mixle/tests/torch_parity_test.py`` — the parity receipt. It trains the same module
  architecture on the same data through both a hand-written raw torch loop (tensor prep, optimizer,
  epoch loop, train/eval mode, ``no_grad`` eval) and through ``optimize(x, module)``, and asserts
  the two reach the same held-out log-likelihood. The manufactured training loop gives nothing away
  versus writing it by hand.
* ``mixle/tests/grad_control_test.py`` (``GradientCheckpointingTest``) — the identical-gradient
  receipt for gradient checkpointing. It builds a causal LM with and without
  ``gradient_checkpointing=True``, syncs their weights, runs a backward pass on the same batch
  through both, and asserts the loss and every parameter gradient match — the checkpointed
  recompute path is a memory/compute trade, not a model change.
* ``mixle/tests/grad_control_test.py`` (``AdapterThroughTheBridgeTest``) — the LoRA-style adapter
  receipt. It wraps a frozen base linear layer plus a trainable low-rank delta in a
  ``GradLeaf``, fits it with ``optimize``, and asserts the frozen base's ``state_dict()`` is
  bit-for-bit unchanged afterward while the low-rank delta moved and the fit's held-out likelihood
  genuinely improved — the deeper claim behind "peft just works": an adapter-wrapped module is
  still just a module.

See also :doc:`neural-llm` for the wider set of neural and LLM surfaces (``mixle.models``,
``mixle.task``, ``mixle.reason``) and where ``GradLeaf``/``NeuralDensity`` fit among them.
