"""Evidence-delivery contracts shared by the ``mixle.analysis`` risk surfaces (MXR-080-1900).

Two failure shapes recurred across the scientific result surfaces, and both are silent:

* **Short posterior delivery.** IC-1's ``Posterior.samples(n, rng)`` and
  ``Posterior.derived_quantity(fn, n, rng)`` both promise exactly ``n`` draws. Nothing in the
  protocol enforces that, and a real implementation can under-deliver for perfectly ordinary reasons
  -- a thinned chain, a rejection filter, a cached draw set. Where the consuming code then reduces
  the draws to a scalar (a mean risk, a decision value) or lets NumPy broadcast them, the shortfall
  leaves no trace: the answer comes back looking exactly like the one the caller asked for, computed
  on a fraction of the evidence. :func:`require_delivered_draws` is the receipt -- it converts that
  into an error naming the requested and delivered counts.

* **A Boolean in a numeric physical slot.** ``float(True) == 1.0``, so a flag that leaked into a
  dose, an exposure, an activity quantity or an emission factor is silently read as the physical
  quantity ``1``. It is the same class of defect ``mixle.utils.exact.require_exact_bool`` guards in
  the other direction (a non-Boolean read as a flag); :func:`reject_boolean_quantity` is the
  numeric-slot half. ``True`` is never a dose in mg/kg-day or a volume in litres, so there is no
  legitimate caller to refuse.

This module is private and declares no ``__all__``: the façade-drift guard (MXR-080-1593) reads a
submodule's ``__all__`` as a public-API declaration, and an internal contract helper is not one.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def require_delivered_draws(samples: Any, n: int, *, what: str) -> np.ndarray:
    """Return ``samples`` as a float array, or raise if it does not carry exactly ``n`` draws.

    ``what`` names the delivery being checked (e.g. ``"exposure posterior"``) and appears in the
    error, so a caller who hits this learns *which* posterior under-delivered rather than only that
    a shape was wrong.

    The check is on the LEADING axis only. A ``(n, d)`` draw matrix is the IC-1 shape and a ``(n,)``
    vector is its ``d == 1`` squeeze; both are legitimate, and which one a given carrier accepts is
    the carrier's own business, not this contract's. What is checked here is the one thing every
    carrier needs and none of them can recover after the fact: that the number of draws delivered is
    the number requested.

    Deliberately NOT checked here: finiteness and domain range. Those are per-quantity invariants
    (a probability's ``[0, 1]``, a dose's non-negativity) that the calling surfaces already enforce
    with their own, more specific errors; duplicating them here would report a count problem for a
    value problem.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 0:
        # A 0-d value IS one draw, so `n == 1` is satisfied and is promoted rather than refused --
        # the callers that adopted this check previously ran `np.atleast_1d` for exactly that case,
        # and turning it into an error would reject a delivery that is genuinely complete.
        if n == 1:
            return np.atleast_1d(arr)
        raise ValueError(f"{what} delivered a scalar, expected exactly n={n} draw(s) (MXR-080-1900)")
    delivered = int(arr.shape[0])
    if delivered != n:
        raise ValueError(
            f"{what} delivered {delivered} draw(s), expected exactly n={n} (MXR-080-1900). IC-1 promises "
            "exactly the requested number of draws; a short delivery (a thinned chain, a rejection "
            "filter, a cached draw set) would otherwise be averaged or broadcast away, and the result "
            "would look identical to one computed on the full requested evidence."
        )
    return arr


def reject_boolean_quantity(value: Any, name: str) -> None:
    """Raise ``TypeError`` if ``value`` is a Boolean (or a Boolean array) used as a physical quantity.

    Catches the ``bool``/``np.bool_`` scalar and the ``dtype=bool`` array -- the two forms a flag
    actually arrives in. A *mixed* Python list such as ``[True, 1.0]`` is NOT caught, because NumPy
    has already coerced it to a float array before this can see it; that is a real limit of this
    check and not something a leading-axis inspection can recover.

    Nothing else is judged here: an ``int`` dose of ``1`` is a legitimate quantity and stays one.
    Only the Boolean type is refused, and only because ``float(True) == 1.0`` makes it indetectable
    one line later.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a numeric physical quantity, got the Boolean {value!r} (MXR-080-1900). "
            "It is not coerced: float(True) is 1.0, so a flag that leaked into this slot would be "
            "read as the quantity 1 with nothing to distinguish it from a real measurement."
        )
    if isinstance(value, np.ndarray) and value.dtype == np.bool_:
        raise TypeError(
            f"{name} must be a numeric physical quantity, got a Boolean array (dtype=bool, shape "
            f"{value.shape}) (MXR-080-1900). Booleans are not coerced here: they would become an "
            "array of exact 0.0/1.0 quantities indistinguishable from real measurements."
        )
