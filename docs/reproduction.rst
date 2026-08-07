External Reproduction
=====================

This page is the protocol for an independent reviewer to reproduce mixle's claims from a clean environment
(worklist E14). It is deliberately runnable without insider knowledge: two commands, a captured environment,
and deterministic outputs to compare.

Why a receipt
-------------

A reproduction is only meaningful if two people can compare *the same thing*. mixle's claims are backed by
seeded, deterministic computations, so an independent run on the same version must produce identical numbers.
The installed ``mixle-reproduce`` command captures the artifact, environment, expectations, measurements,
and per-check verdicts as a JSON **receipt**. It exits nonzero unless the wheel identity, installed-content
hashes, and every required check pass.

The protocol
------------

1. **Download and clean-install the exact artifact.** Keeping the wheel is necessary because an installed
   Python distribution does not retain the byte digest of the wheel archive from which it came::

     python -m venv repro-env
     repro-env/bin/pip download --no-deps --only-binary=:all: "mixle==<version>" --dest artifacts
     repro-env/bin/pip install artifacts/mixle-<version>-*.whl

2. **Emit a receipt.** Run the command shipped in that wheel and give it the retained artifact::

     repro-env/bin/mixle-reproduce --wheel artifacts/mixle-<version>-*.whl --out receipt.json

   The command verifies the wheel's SHA-256, name and version, its embedded source commit/tree, the hashes
   of installed files named by ``RECORD``, the deterministic claim checks, and -- decisively -- that the
   supplied wheel IS the distribution executing: the installed package's embedded build provenance must
   equal the wheel's, and every hashed entry in the wheel's ``RECORD`` must match the installed ``RECORD``.
   Version equality alone is not identity; without the binding, any other clean wheel of the same version
   could be presented as the receipt's subject while different bytes executed. A development checkout may
   instead use ``python scripts/reproduce.py --source-tree``; that explicitly binds the receipt to the
   repository containing the script and never consults the caller's ambient Git repository.

   **Scope.** The installed command runs the self-contained claim checks above and nothing else; its
   receipt says so in a ``scope`` field. The four reproduction-bundle entries
   (``gallery-univariate``, ``gallery-structured``, ``production-provenance``, ``scaling-backend``)
   are NOT part of it -- the wheel ships neither the bundle nor its inputs. Replaying them is a separate,
   source-bound step from a checkout of the same commit::

     python scripts/run_repro_entry.py --entry gallery-univariate

   run once per entry, attaching each validated receipt. A complete independent replay therefore
   consists of one installed-command receipt plus four entry receipts.

3. **Require success, then compare.** The process exit status and receipt-level ``passed`` must both indicate
   success. Each check records its observed value, required value, tolerance, and verdict. Compare the wheel
   SHA-256 with the digest attached to the GitHub Release; the environment block makes dependency/platform
   differences attributable.

4. **Full suite (optional, stronger).** For a complete reproduction, run the correctness gate against the
   clean install::

     repro-env/bin/python -m pytest -m "not optional and not benchmark"

   and, with the optional backends installed, the optional and benchmark tiers. This is the same suite CI
   runs; the release checklist records the exact commands and the result counts for the released commit.

What a mismatch means
---------------------

* A failed check or nonzero exit is a failed reproduction, not an ordinary receipt. Treat it as a bug and
  retain both receipts; their artifact and environment blocks localize the discrepancy.
* A **``checks`` difference across versions** is expected when a release intentionally changes behavior; it
  must correspond to a changelog entry and a `release decision log <../release-checklists/0.8.0-decisions.md>`_
  entry.

The receipt's determinism is itself gated by ``mixle/tests/reproduce_receipt_test.py``, so the reproduction
path cannot silently rot.
