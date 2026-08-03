"""Exact scalar and sequence controls for typed-runtime public surfaces (MXR-080-1905).

The typed runtime's receipts were repaired one at a time, but the *controls* that feed them were
not: the arguments that decide how many rounds a node may starve, whether a partition is pinned,
which nodes are candidates, and what version a commit moves from. Each was annotated ``int`` or
``bool`` or ``tuple[str, ...]`` and checked, at most, for sign. That let three families of value
through, and all three were reproduced before this module was written:

* **fractional** -- ``SchedulerConfig(max_skip_rounds=2.5)`` and
  ``RuntimeVersions(model_version=0.0)`` constructed. A version that is a float cannot be recorded
  by :class:`~mixle.experimental.typed_runtime.transaction.CommitReceipt`, whose accepted-commit
  check is ``isinstance(before, int)`` -- so the coordinator advanced its version vector and *then*
  refused to write the receipt for it.
* **NumPy** -- ``RuntimeVersions(model_version=np.int64(0))`` constructed, but
  ``GainPerCostScheduler.schedule(model_version=np.int64(3))`` raised, because the receipt behind it
  tests ``isinstance(value, int)`` and ``np.int64`` is not an ``int``. Two halves of one runtime
  disagreed about whether the same value was a version.
* **string** -- ``GraphMemoryCache.put(partition, graph, pinned="no")`` pinned the partition, and
  ``GainPerCostScheduler.schedule(candidate_nodes="ab")`` scheduled the nodes ``a`` and ``b``
  because a string is a sequence of its characters.

The rule this module applies mirrors :mod:`mixle.utils.exact`: a NumPy scalar is a genuine value of
its kind and is canonicalized; everything else that merely *behaves* like one is refused, because
coercion silently changes what the control means. ``bool`` is refused where an integer is wanted --
``True`` is not a count -- and ``require_exact_bool`` from :mod:`mixle.utils.exact` is re-exported
so a caller of this module has one import for both.

What these helpers deliberately do NOT check: the *meaning* of the value. ``round_index`` being an
exact integer says nothing about whether it is the right round, and ``maximum_tokens`` being an
exact integer says nothing about whether the budget is achievable. Those remain the caller's and the
owning class's business -- this module only closes the gap between what a signature says a control
is and what it accepted.

PACKAGE-INTERNAL, and deliberately so: the module name is underscore-prefixed and it declares no
``__all__``, which keeps these helpers out of ``mixle.experimental.typed_runtime.__all__`` and
therefore out of ``manifests/api_manifest.json``. They are shared plumbing for the modules in this
package, not new public API -- a fix for a validation gap should not widen the frozen public
surface. Callers who want the same contract in their own code should use :mod:`mixle.utils.exact`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# Re-exported deliberately (the redundant alias marks it as such): the modules in this package want
# one import for "exact scalar control", and the Boolean half of that contract already exists and
# should not be forked.
from mixle.utils.exact import require_exact_bool as require_exact_bool


def require_exact_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as an ``int``, or raise ``TypeError``/``ValueError``.

    ``np.integer`` is accepted and canonicalized -- it is a genuine integer, and refusing it would
    reject the values the library's own array paths produce (this is the same allowance
    :func:`mixle.utils.exact.require_exact_bool` makes for ``np.bool_``). ``bool`` is refused even
    though it is an ``int`` subclass: ``True`` names a decision, not a count, and a Boolean arriving
    where a count belongs is a caller mistake worth surfacing rather than reading as ``1``.

    A float is refused even when integral (``2.0``). Accepting it would mean the runtime has two
    representations for one version or round, and the receipt classes that record them test
    ``isinstance(value, int)`` -- so ``2.0`` would be admitted here and refused there, after the
    state it describes had already moved (MXR-080-1905).
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an exact integer, got bool ({value!r}); a Boolean is a decision, not a count.")
    if isinstance(value, np.integer):
        value = int(value)
    elif not isinstance(value, int):
        raise TypeError(
            f"{name} must be an exact integer, got {type(value).__name__} ({value!r}). It is not "
            "coerced: a fractional or textual count would be admitted here and refused by the "
            "receipt that has to record it, after the state it describes had already moved."
        )
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def require_id_sequence(value: Any, name: str) -> tuple[str, ...]:
    """Return ``value`` as a tuple of non-blank identifier strings.

    A bare ``str`` (or ``bytes``) is refused rather than iterated: it is a sequence of its own
    characters, so ``candidate_nodes="ab"`` silently named two nodes nobody asked for
    (MXR-080-1905). Every other sequence is materialized, so a one-shot iterator cannot be consumed
    by a validation pass and arrive empty at the code that uses it.

    Membership is NOT checked here -- whether these ids exist in a graph or a plan is the caller's
    question, and each caller reports it differently.
    """
    if isinstance(value, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of ids, got {type(value).__name__} ({value!r}): a string "
            "would iterate as its characters and name ids nobody asked for. Pass a tuple/list, or "
            f"({value!r},) for a single id."
        )
    if not isinstance(value, Sequence) and not isinstance(value, (set, frozenset)):
        try:
            value = tuple(value)
        except TypeError as error:
            raise TypeError(f"{name} must be a sequence of ids, got {type(value).__name__}.") from error
    ids = tuple(value)
    blank = [item for item in ids if not isinstance(item, str) or not item.strip()]
    if blank:
        raise ValueError(f"{name} contains an id that names nothing: {blank!r}.")
    return ids
