External Reproduction
=====================

This page is the protocol for an independent reviewer to reproduce mixle's claims from a clean environment
(worklist E14). It is deliberately runnable without insider knowledge: two commands, a captured environment,
and deterministic outputs to compare.

Why a receipt
-------------

A reproduction is only meaningful if two people can compare *the same thing*. mixle's claims are backed by
seeded, deterministic computations, so an independent run on the same version must produce identical numbers.
``scripts/reproduce.py`` captures the environment and those numbers as a JSON **receipt**; a difference in the
``checks`` block is a real, environment-dependent discrepancy worth investigating, not noise.

The protocol
------------

1. **Clean install.** In a fresh virtual environment, install the exact version under review::

     python -m venv repro-env
     repro-env/bin/pip install "mixle==<version>"        # or: pip install -e . from a clean clone

2. **Emit a receipt.** Run the reproduction script and save its output::

     repro-env/bin/python scripts/reproduce.py --out receipt.json

   The receipt records the environment (Python, platform, machine, mixle / numpy / scipy versions, git
   commit) and the deterministic claim checks -- a Gaussian fit recovering its parameters, scalar vs
   vectorized score agreement, a serialization round-trip, automatic family recovery, and a seeded sample.

3. **Compare.** The ``checks`` block must match a receipt produced from the same mixle version on any
   platform. The ``environment`` block documents *where* the receipt was produced, so a platform-specific
   difference (a BLAS build, a numpy version) is attributable rather than mysterious.

4. **Full suite (optional, stronger).** For a complete reproduction, run the correctness gate against the
   clean install::

     repro-env/bin/python -m pytest -m "not optional and not benchmark"

   and, with the optional backends installed, the optional and benchmark tiers. This is the same suite CI
   runs; the release checklist records the exact commands and the result counts for the released commit.

What a mismatch means
---------------------

* A **``checks`` difference on the same version** is a genuine reproducibility defect -- a nondeterminism, a
  platform-dependent numeric path, or an unpinned dependency drift. Treat it as a bug and file it with both
  receipts (the ``environment`` blocks localize the difference).
* A **``checks`` difference across versions** is expected when a release intentionally changes behavior; it
  must correspond to a changelog entry and a `release decision log <../release-checklists/0.8.0-decisions.md>`_
  entry.

The receipt's determinism is itself gated by ``mixle/tests/reproduce_receipt_test.py``, so the reproduction
path cannot silently rot.
