Example Gallery
================

A fast index into the newer end-to-end examples: one entry per script, what
pattern it demonstrates, and the concrete **receipt** it prints and asserts --
a real measured number or invariant, not a restatement of the description. Each
example is paired with a smoke test under ``mixle/tests/`` that pins the same
receipt programmatically.

For the full, general example inventory (distribution families, HMMs,
enumeration, engines, and so on) see :doc:`examples`. This page covers the
newer applied adapter and multimodal-pretraining workflows, each landing as
its own example.

Adapter and Multimodal Pretraining Patterns
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Example
     - Pattern
     - Receipt
   * - ``examples/peft_lora_grad_leaf.py``
     - A real HuggingFace checkpoint (``hf-internal-testing/tiny-random-gpt2``),
       wrapped with actual ``peft.get_peft_model`` LoRA adapters, dropped into
       :class:`~mixle.models.GradLeaf` unchanged -- next-token log-likelihood
       summed over a sequence is the log density ``GradLeaf`` expects.
     - Base checkpoint weights are bitwise unchanged after fitting (drift
       ``0.0``); only the LoRA adapter parameters move; mean sequence
       log-density rises from ``-48.27`` to ``-46.18``.
   * - ``examples/multimodal_stage1_demo.py``
     - LLaVA-style stage-1 pretraining on synthetic volumes: a frozen 3-D
       ``Conv3d`` encoder and a frozen toy LM (embedding + GRU cell + head)
       bridged by a thin trainable projection, fit end to end through one
       ``GradLeaf``/``optimize`` call.
     - All 11 frozen backbone tensors are bitwise unchanged
       (``torch.equal`` before/after); the 4 projection tensors move; mean
       caption log-likelihood improves from ``-3.21`` to ``-0.60``.

Running the Examples
---------------------

.. code-block:: sh

   python examples/peft_lora_grad_leaf.py
   python examples/multimodal_stage1_demo.py

``peft_lora_grad_leaf.py`` additionally needs ``pip install "mixle[torch]" transformers peft``
(example-only dependencies, not a package extra, since ``GradLeaf`` has no
opinion on what module it is handed). ``multimodal_stage1_demo.py`` needs only
``mixle[torch]``.

Every receipt above is pinned by a paired smoke test:
``mixle/tests/peft_lora_grad_leaf_smoke_test.py`` and
``mixle/tests/multimodal_stage1_demo_smoke_test.py``.
