"""Tying discovery via arrangement similarity -- the neural half of ConditionalJIT (H4).

The R1 copula note (see the roadmap doc, R1 -> G4, F6, I2, H4) observes that any weight tensor's flattened
value vector ``v`` decomposes as ``v = P . s``: a sorted **profile** ``s`` (the empirical quantile function --
the values in sorted order) composed with a **permutation** ``P`` (the arrangement -- which sorted rank each
original position maps to). That is exactly Sklar's theorem one level down: a joint distribution factors into
marginals (here: the profile / value distribution) and a dependence structure (here: the arrangement). This
module uses only the marginal half of that decomposition as a *discovery signal*: two tensors whose PROFILES
are close are, value-distribution-wise, "the same numbers, differently arranged" -- exactly the situation a
weight tie (two module attributes sharing one ``nn.Parameter``, the tensor analogue of this codebase's
``keys=`` mechanism for tying distribution parameters across combinator children, see
``mixle.stats.combinator``) can exploit for a real, measurable parameter reduction.

This module deliberately stays on the "discovery" side of the R1 note. Training a permutation (differentiable
OT / Sinkhorn / torchsort) is explicitly out of scope here -- profile comparison via sorted-vector L2 distance
is closed-form and needs no optimization loop. See :func:`profile_distance` for why sorted-vector L2 already
*is* the right 1-D optimal-transport quantity for this purpose.

Workflow:

1. :func:`tensor_profile` -- extract a fixed-length quantile-function profile from a weight tensor.
2. :func:`profile_distance` -- an L2 distance between two profiles (= 1-D Wasserstein-2 distance between the
   corresponding empirical distributions).
3. :func:`propose_ties` -- rank all pairs of named tensors by profile distance and return the most promising
   tying candidates.
4. :func:`apply_tie` -- build an isolated candidate with one shared ``nn.Parameter``, measure its output delta,
   and return it only when the caller's explicit parity budget accepts it.  The source model is never mutated.

Everything here lives in ``mixle/experimental/`` per F7: it graduates out once the mechanism has field mileage
and (once merged) should be registered against E0's graduation ledger -- a trivial follow-up, not done here
since ``mixle/experimental/graduation.py`` does not yet exist on this branch.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False


def tensor_profile(tensor: Any, n_quantiles: int = 256) -> Any:
    """Return the fixed-length quantile-function **profile** of a tensor's values.

    The profile is the empirical quantile function of the tensor's flattened values: sort the values, then
    resample that sorted curve onto ``n_quantiles`` evenly spaced quantile positions in ``[0, 1]`` by linear
    interpolation. Fixing ``n_quantiles`` makes profiles from differently shaped/sized tensors directly
    comparable (a ``(64, 64)`` weight and a ``(32, 128)`` weight both reduce to a length-``n_quantiles`` curve).

    This throws away the ARRANGEMENT entirely (sorting is exactly "forget the permutation P in v = P . s") and
    keeps only the value distribution -- by design, since arrangement similarity is not what tying discovery
    is asking about here: two tensors that hold "the same numbers" but scattered into different spatial slots
    are still excellent tying candidates once one of them is permuted (or once the model is simply insensitive
    to which of the two arrangements it uses, which the parity receipt in :func:`apply_tie` checks directly).

    Args:
        tensor: A torch tensor of any shape (or anything ``torch.as_tensor`` accepts).
        n_quantiles (int): Number of fixed quantile positions to resample the sorted values onto. Must be
            >= 2. Default 256 is small enough to be cheap and large enough that two profiles from
            differently-shaped tensors compare fairly.

    Returns:
        A 1-D torch tensor of length ``n_quantiles``, dtype float32, containing the interpolated quantile
        function.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("tensor_profile requires torch")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2, got %r" % (n_quantiles,))

    source = torch.as_tensor(tensor)
    if source.is_complex():
        raise TypeError("tensor_profile does not define an ordering for complex values")
    flat = source.detach().reshape(-1).to(torch.float32)
    n = flat.numel()
    if n == 0:
        raise ValueError("tensor_profile requires a non-empty tensor")
    if not torch.all(torch.isfinite(flat)):
        raise ValueError("tensor_profile requires finite tensor values")
    sorted_vals, _ = torch.sort(flat)
    if n == 1:
        return sorted_vals.expand(n_quantiles).clone()

    # Sample positions of the n sorted values along [0, 1], then linearly interpolate onto n_quantiles
    # evenly spaced query points -- torch.nn.functional.interpolate wants a batch/channel axis.
    source_positions = torch.linspace(0.0, 1.0, n, device=sorted_vals.device)
    query_positions = torch.linspace(0.0, 1.0, n_quantiles, device=sorted_vals.device)
    idx = torch.searchsorted(source_positions, query_positions, right=False).clamp(1, n - 1)
    lo, hi = idx - 1, idx
    lo_pos, hi_pos = source_positions[lo], source_positions[hi]
    span = (hi_pos - lo_pos).clamp_min(1e-12)
    frac = (query_positions - lo_pos) / span
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def profile_distance(profile_a: Any, profile_b: Any) -> float:
    """L2 distance between two equal-length quantile-function profiles.

    Sorted-vector-vs-sorted-vector L2 distance between two empirical quantile functions IS the (discretized)
    1-D Wasserstein-2 distance between the two underlying empirical distributions -- a standard, well-known
    fact worth stating explicitly rather than leaning on silently: for 1-D distributions, the optimal
    transport coupling between two empirical measures is exactly the sorted-to-sorted (rank-to-rank) matching,
    so ``W2(mu, nu) = ||sort(x) - sort(y)||_2`` up to the ``1/sqrt(n_quantiles)`` normalization used here. That
    is also the ``v = P . s`` decomposition's marginal half compared directly.  A fixed-length resampled
    profile is intentionally lossy: zero distance is a strong discovery signal, but is *not* an exact
    permutation certificate.  :func:`propose_ties` records that stricter fact separately by comparing the full
    sorted tensors.

    Args:
        profile_a: A 1-D tensor, as returned by :func:`tensor_profile`.
        profile_b: A 1-D tensor of the same length as ``profile_a``.

    Returns:
        float: The (root-mean-square-normalized) L2 distance between the two profiles, >= 0.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("profile_distance requires torch")
    a = torch.as_tensor(profile_a).to(torch.float32)
    b = torch.as_tensor(profile_b).to(torch.float32)
    if a.shape != b.shape:
        raise ValueError("profile_distance requires equal-length profiles, got %r vs %r" % (a.shape, b.shape))
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


@dataclass(frozen=True)
class TyingCandidate:
    """A single proposed weight tie between two named tensors.

    Attributes:
        name_a (str): Name of the first tensor (as passed into :func:`propose_ties`).
        name_b (str): Name of the second tensor.
        distance (float): Profile distance between the two tensors (lower = more similar = better candidate).
        shape_a (tuple[int, ...]): Shape of the first tensor.
        shape_b (tuple[int, ...]): Shape of the second tensor.
        exact_value_multiset (bool): Whether the full tensors contain exactly the same values with the same
            multiplicities.  Unlike a resampled profile comparison, this is an exact permutation certificate.
    """

    name_a: str
    name_b: str
    distance: float
    shape_a: tuple
    shape_b: tuple
    exact_value_multiset: bool


def _exact_value_multiset(tensor_a: Any, tensor_b: Any) -> bool:
    """Return whether two equal-shape real tensors contain the exact same value multiset."""
    original_a = torch.as_tensor(tensor_a).detach()
    original_b = torch.as_tensor(tensor_b).detach()
    if (
        original_a.shape != original_b.shape
        or original_a.dtype != original_b.dtype
        or original_a.device != original_b.device
    ):
        return False
    a = original_a.reshape(-1)
    b = original_b.reshape(-1)
    sorted_a = torch.sort(a).values
    sorted_b = torch.sort(b).values
    return bool(torch.equal(sorted_a, sorted_b))


def propose_ties(
    named_tensors: dict,
    n_quantiles: int = 256,
    max_distance: float | None = None,
    top_k: int | None = None,
) -> list:
    """Propose weight-tying candidates by pairwise profile similarity across a set of named tensors.

    Computes a :func:`tensor_profile` for every tensor, then ranks all pairs by :func:`profile_distance`
    (ascending -- most similar first). This is the discovery step of H4: the analogue of scanning distribution
    combinators for parameters that could share a ``keys=`` tag, applied to torch tensors instead.

    Args:
        named_tensors (dict[str, Any]): Mapping from a tensor name (e.g. ``"layer0.head2.q"``) to the tensor
            itself. Only shape/dtype/device-compatible pairs are proposed because a direct parameter tie cannot
            represent incompatible tensors.
        n_quantiles (int): Passed through to :func:`tensor_profile`.
        max_distance (float | None): If given, only pairs with ``distance <= max_distance`` are returned. Left
            unset (``None``) by default so callers can inspect the full ranked list and pick a threshold.
        top_k (int | None): If given, only the ``top_k`` closest pairs are returned (after ``max_distance``
            filtering, if any).

    Returns:
        list[TyingCandidate]: Candidates sorted by ascending distance (most similar / best tying candidate
            first).
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("propose_ties requires torch")
    if not isinstance(named_tensors, dict):
        raise TypeError("named_tensors must be a dict")
    if max_distance is not None and (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, (int, float))
        or not math.isfinite(max_distance)
        or max_distance < 0
    ):
        raise ValueError("max_distance must be finite and non-negative")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
        raise ValueError("top_k must be a non-negative integer")
    names = list(named_tensors.keys())
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("tensor names must be non-empty strings")
    profiles = {name: tensor_profile(named_tensors[name], n_quantiles=n_quantiles) for name in names}
    tensors = {name: torch.as_tensor(named_tensors[name]) for name in names}
    shapes = {name: tuple(tensors[name].shape) for name in names}

    candidates = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            if (
                shapes[name_a] != shapes[name_b]
                or tensors[name_a].dtype != tensors[name_b].dtype
                or tensors[name_a].device != tensors[name_b].device
            ):
                continue
            distance = profile_distance(profiles[name_a], profiles[name_b])
            if max_distance is not None and distance > max_distance:
                continue
            candidates.append(
                TyingCandidate(
                    name_a=name_a,
                    name_b=name_b,
                    distance=distance,
                    shape_a=shapes[name_a],
                    shape_b=shapes[name_b],
                    exact_value_multiset=_exact_value_multiset(
                        named_tensors[name_a],
                        named_tensors[name_b],
                    ),
                )
            )
    candidates.sort(key=lambda c: c.distance)
    if top_k is not None:
        candidates = candidates[:top_k]
    return candidates


@dataclass(frozen=True)
class ParityReceipt:
    """Measured output-parity receipt for a function-preserving-edit attempt.

    Attributes:
        max_abs_diff (float): Max absolute elementwise difference between pre- and post-edit outputs.
        relative_l2 (float): ``||after - before||_2 / ||before||_2``.  This is infinity when the baseline is
            identically zero but the candidate is not, and zero only when both are zero.
        params_before (int): Total parameter count before the edit.
        params_after (int): Total parameter count after the edit.
        max_relative_l2 (float): Caller-supplied relative-L2 acceptance limit.
        max_abs_tolerance (float | None): Optional caller-supplied maximum-absolute-error limit.
        accepted (bool): Whether the candidate satisfied every supplied limit and was returned.
    """

    max_abs_diff: float
    relative_l2: float
    params_before: int
    params_after: int
    max_relative_l2: float
    max_abs_tolerance: float | None
    accepted: bool

    @property
    def params_reduced(self) -> int:
        """Absolute parameter-count reduction (>= 0 for a real tie; can be 0 if no sharing occurred)."""
        return self.params_before - self.params_after

    @property
    def params_reduced_fraction(self) -> float:
        """Fractional parameter-count reduction, in ``[0, 1]``."""
        if self.params_before == 0:
            return 0.0
        return self.params_reduced / self.params_before


def _get_param(module: Any, dotted_name: str) -> Any:
    obj = module
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj, parts[-1]


def apply_tie(
    module: Any,
    name_a: str,
    name_b: str,
    inputs: Any,
    strategy: str = "average",
    *,
    max_relative_l2: float,
    max_abs_tolerance: float | None = None,
) -> tuple[Any, ParityReceipt]:
    """Transactionally evaluate a weight tie and return ``(selected_model, receipt)``.

    The edit is made only on a deep-copied candidate.  Both candidate attributes point to one
    ``nn.Parameter`` (an actual tie, not merely equal initialization).  The source model and all parameter
    identities referenced by its optimizer remain untouched.  The candidate is returned only when its measured
    error is within the explicit budget; otherwise the original ``module`` object is returned.  A caller that
    adopts an accepted candidate must construct or transactionally migrate an optimizer for that new model.
    ``strategy`` picks the shared value:

    - ``"average"`` (default): elementwise mean of the two original tensors. Requires identical shapes. This
      is the natural choice when the two tensors are similar-valued-but-not-identical (the realistic case
      tying discovery is meant to catch, per the H4 acceptance note) -- it minimizes the summed squared
      perturbation to both tensors simultaneously.
    - ``"keep_a"``: keep ``name_a``'s tensor verbatim and point ``name_b`` at it. Useful when one tensor is
      trusted more (e.g. it trained longer, or ``name_a`` is canonical by convention).

    Weight tying is NOT assumed to be output-preserving.  Profile similarity is only a discovery signal; this
    function gates adoption on a real forward comparison under the caller's error budget.

    Args:
        module: A ``torch.nn.Module``. Parity is measured in evaluation mode, then every module's prior local
            training/evaluation flag is restored.
        name_a (str): Dotted attribute path to the first ``nn.Parameter`` (e.g. ``"blocks.0.attn.qkv.weight"``).
        name_b (str): Dotted attribute path to the second ``nn.Parameter``. Must have the same shape as
            ``name_a``.
        inputs: Whatever ``module(inputs)`` accepts; used for the before/after forward pass.
        strategy (str): ``"average"`` or ``"keep_a"``; see above.
        max_relative_l2 (float): Required finite, non-negative relative-L2 acceptance limit.
        max_abs_tolerance (float | None): Optional finite, non-negative maximum-absolute-error limit.

    Returns:
        tuple[Any, ParityReceipt]: The isolated candidate when accepted, otherwise the original module, plus the
            measured decision receipt.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("apply_tie requires torch")
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if strategy not in ("average", "keep_a"):
        raise ValueError("strategy must be 'average' or 'keep_a', got %r" % (strategy,))
    for value, name in (
        (max_relative_l2, "max_relative_l2"),
        (max_abs_tolerance, "max_abs_tolerance"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative")

    owner_a, attr_a = _get_param(module, name_a)
    owner_b, attr_b = _get_param(module, name_b)
    param_a = getattr(owner_a, attr_a)
    param_b = getattr(owner_b, attr_b)
    if not isinstance(param_a, torch.nn.Parameter) or not isinstance(param_b, torch.nn.Parameter):
        raise TypeError("apply_tie requires both names to resolve to nn.Parameter attributes")
    if param_a is param_b:
        raise ValueError("apply_tie requires two parameters that are not already tied")
    if param_a.shape != param_b.shape:
        raise ValueError("apply_tie requires equal-shape tensors, got %r vs %r" % (param_a.shape, param_b.shape))
    if param_a.dtype != param_b.dtype or param_a.device != param_b.device:
        raise ValueError("apply_tie requires parameters with matching dtype and device")
    if param_a.requires_grad != param_b.requires_grad:
        raise ValueError("apply_tie cannot preserve different requires_grad settings in one shared parameter")

    params_before = sum(p.numel() for p in module.parameters())
    try:
        candidate = copy.deepcopy(module)
    except Exception as exc:
        raise TypeError("module must support deepcopy for transactional tie evaluation") from exc
    candidate_owner_a, candidate_attr_a = _get_param(candidate, name_a)
    candidate_owner_b, candidate_attr_b = _get_param(candidate, name_b)
    candidate_param_a = getattr(candidate_owner_a, candidate_attr_a)
    candidate_param_b = getattr(candidate_owner_b, candidate_attr_b)

    if strategy == "average":
        shared_value = (candidate_param_a.detach() + candidate_param_b.detach()) / 2.0
    else:  # "keep_a"
        shared_value = candidate_param_a.detach().clone()
    shared_param = torch.nn.Parameter(shared_value, requires_grad=candidate_param_a.requires_grad)
    setattr(candidate_owner_a, candidate_attr_a, shared_param)
    setattr(candidate_owner_b, candidate_attr_b, shared_param)

    training_states = [(submodule, bool(submodule.training)) for submodule in module.modules()]
    candidate_training_states = [(submodule, bool(submodule.training)) for submodule in candidate.modules()]
    module.eval()
    candidate.eval()
    try:
        with torch.no_grad():
            outputs_before = module(inputs)
            outputs_after = candidate(inputs)
    finally:
        for submodule, training in training_states:
            submodule.training = training
        for submodule, training in candidate_training_states:
            submodule.training = training

    if not isinstance(outputs_before, torch.Tensor) or not isinstance(outputs_after, torch.Tensor):
        raise TypeError("apply_tie parity evaluation requires tensor-valued model outputs")
    if outputs_before.shape != outputs_after.shape or outputs_before.numel() == 0:
        raise ValueError("apply_tie parity outputs must be non-empty tensors with matching shapes")
    if not torch.all(torch.isfinite(outputs_before)) or not torch.all(torch.isfinite(outputs_after)):
        raise ValueError("apply_tie parity outputs must be finite")

    diff = outputs_after - outputs_before
    max_abs_diff = float(diff.abs().max().item())
    before_norm = float(outputs_before.norm().item())
    diff_norm = float(diff.norm().item())
    relative_l2 = diff_norm / before_norm if before_norm > 0 else (0.0 if diff_norm == 0 else math.inf)

    params_after = sum(p.numel() for p in candidate.parameters())
    accepted = relative_l2 <= max_relative_l2 and (max_abs_tolerance is None or max_abs_diff <= max_abs_tolerance)

    receipt = ParityReceipt(
        max_abs_diff=max_abs_diff,
        relative_l2=relative_l2,
        params_before=params_before,
        params_after=params_after,
        max_relative_l2=float(max_relative_l2),
        max_abs_tolerance=float(max_abs_tolerance) if max_abs_tolerance is not None else None,
        accepted=accepted,
    )
    return (candidate if accepted else module), receipt
