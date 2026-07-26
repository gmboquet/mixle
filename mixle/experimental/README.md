# `mixle.experimental`

This directory is a holding area for unstable research prototypes. Nothing here is part of Mixle's stable API,
and a module name, docstring, type, or receipt object is not by itself evidence of correctness, scientific
validity, production scaling, or release readiness.

## Authoritative inventory

[`mixle/experimental/__init__.py`](./__init__.py) contains the machine-readable `EXPERIMENTAL_INVENTORY`.
Every top-level experimental module has:

- a narrow purpose statement;
- an explicit maturity label;
- executable focused acceptance-test paths when they exist;
- a durable status evidence ID only when the narrow claim was locally receipted at a specific revision; and
- limitations that prevent a focused result from being read as a broader guarantee.

Do not maintain a second hand-written module list here. The inventory-completeness test fails when a top-level
module is missing, a cited acceptance test does not exist, a maturity label is invalid, or a locally receipted
surface lacks a durable evidence ID.

## Maturity labels

- `prototype`: code and a focused executable specification exist, but no validated guarantee is asserted.
- `locally_receipted`: only the inventory's narrow purpose was exercised at the revision cited by its durable
  evidence record.
- `unvalidated`: known audit blockers remain; no behavioral guarantee should be inferred.
- `bookkeeping_only`: the module records evidence but does not execute or validate the experiment itself.

## Graduation

Graduation requires a separately approved stable-API proposal, relevant fixed-baseline and fixed-budget
acceptance evidence, documented assumptions and failure modes, compatibility and migration review, and all
normal release gates. `mixle.experimental.graduation` only stores supplied receipt metadata; it does not run an
evaluation or prove eligibility.

Tests under this directory are normally marked `experimental` so they can be selected independently. A test
path in the inventory is an executable specification, not proof that an arbitrary checkout currently passes.
