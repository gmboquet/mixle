"""muP (Maximal Update Parametrization) -- width-independent lr/init transfer for :mod:`mixle.models.transformer`.

.. warning::

   **Experimental frontier-training prototype.** The parametrization math is exact and tested at small
   width, but this is not a validated production scaling recipe. Verify the coordinate check at your target
   width before relying on transfer; treat it as a research prototype, not a supported training API.

**The idea.** In the standard parametrization, the learning rate and init scale that work best for a
transformer depend on its width (``d_model``): as a model gets wider, activations/gradients grow (or
shrink) with width unless the parametrization compensates, so a wider model needs a *different* tuned
lr than a narrower one -- hyperparameter search has to be repeated at every scale. muP (Yang et al.,
"Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", 2022) chooses
init-variance and lr scaling rules per layer *role* (input/embedding, hidden, output/readout) such
that, in the infinite-width limit, the optimal lr stops moving with width. In practice this means: tune
lr (and init scale) once, cheaply, on a small model, then *transfer* those hyperparameters to a much
larger model at (almost) zero extra tuning cost -- the mechanism a capacity ladder needs so its tuning
bill doesn't scale with every rung.

**Why implemented from scratch instead of depending on the `mup` package.** Microsoft's reference
``mup`` library is not installed in this environment (checked: ``import mup`` fails) and adding it as a
dependency is unnecessary weight for what it buys here: the abc-parametrization rules for a transformer
are a short, precisely published table (a handful of closed-form multipliers), not a large or fiddly
algorithm, and mixle already treats torch as an optional soft dependency (see the ``_HAS_TORCH`` guard
pattern used across ``mixle.models``) -- pulling in a required third-party package for ~10 lines of
well-specified math would work against that. This module implements the rules directly against
:class:`mixle.models.transformer.CausalLM`'s actual module structure.

**The rules (muP for Adam; Tensor Programs V, Table 8 / the ``mup`` library's Adam column).** Let
``width_mult = target_width / base_width`` (the ratio of the layer's fan-in-scaling width to the base
width it was tuned at). Every parameter is classified into one of three roles:

* **input** -- embeddings (token, position) and LayerNorm affine params. Fan-in does not scale with
  ``d_model`` (it's the fixed vocab size, or LayerNorm has no fan-in at all), so muP leaves both init
  variance and lr **unscaled** (``Theta(1)`` in ``width_mult``).
* **hidden** -- every ``Linear`` inside a transformer block (attention qkv/proj, MLP in/out). Fan-in
  scales linearly with width. Init variance stays at the standard ``1/fan_in`` (std multiplier
  ``width_mult**-0.5``); the lr is scaled down as ``width_mult**-1`` -- *this* is the headline muP
  rule and the reason a wide model doesn't blow up (or stall) with the narrow model's lr.
* **output** -- the unembedding/readout. Fan-in scales with width, fan-out (vocab) is fixed. Init
  variance gets an *extra* ``1/width_mult`` beyond the hidden rule (std multiplier ``width_mult**-1``),
  lr scales as ``width_mult**-1`` like hidden, and the forward pass gets an explicit
  ``width_mult**-1`` multiplier on the logits (the "c" of the abc-parametrization) so the *output scale*
  is also width-independent at init.

**Weight tying.** ``CausalLM`` ties ``head.weight = tok.weight`` (see ``mixle/models/transformer.py``)
-- the same ``nn.Parameter`` object plays both the embedding-lookup and the unembedding-projection role.
Since a single tensor can't have two different init/lr rules at once, this module follows the standard
treatment for muP with tied embeddings (the same one the ``mup`` library's ``MuReadout`` implements):
the shared parameter is classified under the **input** rule (fixed variance/lr -- it *is* the embedding
table), and the muP **output** role is instead realized as a multiplicative rescale applied to the
*logits* at readout time (:func:`output_forward_multiplier`), independent of the (shared) weight's own
init/lr. This reproduces the correct output-scale behavior without touching weight tying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

Role = Literal["input", "hidden", "output"]
_ROLES: tuple[Role, ...] = ("input", "hidden", "output")


def _check_width_mult(width_mult: float) -> float:
    width_mult = float(width_mult)
    if width_mult <= 0:
        raise ValueError(f"width_mult must be positive, got {width_mult}.")
    return width_mult


def init_std_multiplier(role: Role, width_mult: float) -> float:
    """Return the muP multiplier on init std for a parameter of ``role`` at the given ``width_mult``.

    ``width_mult = target_width / base_width``. Multiply the BASE width's tuned/measured init std by
    this factor to get the init std to use at the target width:

    * ``"input"``  -> ``1``                  (fan-in fixed, e.g. vocab size -- no rescale)
    * ``"hidden"`` -> ``width_mult ** -0.5``  (variance ``1/fan_in``, standard, unchanged form)
    * ``"output"`` -> ``width_mult ** -1``    (variance ``1/fan_in**2`` -- an extra ``1/width_mult``)
    """
    width_mult = _check_width_mult(width_mult)
    if role == "input":
        return 1.0
    if role == "hidden":
        return width_mult**-0.5
    if role == "output":
        return width_mult**-1.0
    raise ValueError(f"unknown muP role {role!r}; expected one of {_ROLES}.")


def lr_multiplier(role: Role, width_mult: float) -> float:
    """Return the muP multiplier on learning rate for a parameter of ``role`` at the given ``width_mult``.

    ``width_mult = target_width / base_width``. Multiply the BASE width's tuned lr by this factor to
    get the transferred lr to use at the target width:

    * ``"input"``  -> ``1``                (constant lr -- the muP "don't touch it" role)
    * ``"hidden"`` -> ``width_mult ** -1``  (the headline muP rule: lr shrinks as the model widens)
    * ``"output"`` -> ``width_mult ** -1``  (same shrink as hidden, for Adam)
    """
    width_mult = _check_width_mult(width_mult)
    if role == "input":
        return 1.0
    if role in ("hidden", "output"):
        return width_mult**-1.0
    raise ValueError(f"unknown muP role {role!r}; expected one of {_ROLES}.")


def output_forward_multiplier(width_mult: float) -> float:
    """Return the muP readout multiplier (the "c" of abc-parametrization) applied to output-role logits.

    Multiply the raw ``head`` output by this factor so the readout's output scale stays width-independent
    at init, even though (due to weight tying, see the module docstring) ``head.weight`` itself is
    parametrized under the ``"input"`` rule rather than a separate ``"output"`` init/lr rule.
    """
    return _check_width_mult(width_mult) ** -1.0


def transfer_lr(base_lr: float, base_width: int, target_width: int, *, role: Role = "hidden") -> float:
    """Rescale ``base_lr`` (tuned at ``base_width``) to the muP-predicted optimum at ``target_width``.

    This is the deliverable that "collapses the ladder's tuning bill": tune ``role="hidden"`` lr once,
    cheaply, at a small ``base_width``, then call this to predict the optimal lr at any larger
    ``target_width`` with (ideally) no further search. Defaults to ``role="hidden"`` -- the dominant
    parameter group (attention + MLP) and the one muP's headline lr-transfer guarantee is about.
    """
    if base_width <= 0 or target_width <= 0:
        raise ValueError("base_width and target_width must be positive.")
    width_mult = target_width / base_width
    return float(base_lr) * lr_multiplier(role, width_mult)


def transfer_init_std(base_std: float, base_width: int, target_width: int, *, role: Role = "hidden") -> float:
    """Rescale ``base_std`` (tuned/measured at ``base_width``) to the muP init std at ``target_width``."""
    if base_width <= 0 or target_width <= 0:
        raise ValueError("base_width and target_width must be positive.")
    width_mult = target_width / base_width
    return float(base_std) * init_std_multiplier(role, width_mult)


def _is_layernorm_param(name: str) -> bool:
    parts = name.split(".")
    return "ln1" in parts or "ln2" in parts or (len(parts) >= 2 and parts[-2] == "ln")


def classify_causal_lm_params(model) -> dict[str, Role]:
    """Map every named parameter of a :class:`mixle.models.transformer.CausalLM` to its muP role.

    * ``tok.weight`` / ``pos.weight`` (embeddings) -> ``"input"`` -- fan-in is the fixed vocab / block
      length, not ``d_model``.
    * LayerNorm affine params (``blocks.*.ln1``/``ln2``, top-level ``ln``) -> ``"input"`` -- no
      fan-in at all, muP leaves them at their standard ``Theta(1)`` scale/shift regardless of width.
    * ``head.weight`` -> not a separate entry: it is the *same* ``nn.Parameter`` as ``tok.weight``
      (weight tying), so ``model.named_parameters()`` already reports it once, under ``"tok.weight"``.
      See the module docstring for how the muP output role is instead applied at readout time.
    * everything else (the attention qkv/proj and MLP ``Linear`` weights/biases inside each block) ->
      ``"hidden"`` -- fan-in scales linearly with ``d_model``.
    """
    roles: dict[str, Role] = {}
    for name, _ in model.named_parameters():
        top = name.split(".")[0]
        if top in ("tok", "pos"):
            roles[name] = "input"
        elif top == "ln" or _is_layernorm_param(name):
            roles[name] = "input"
        else:
            roles[name] = "hidden"
    return roles


def enable_mup_attention(model, enabled: bool = True) -> None:
    """Turn on (or off) muP attention-logit scaling on every block of a :class:`CausalLM`.

    Standard attention scales ``QK^T`` by ``1/sqrt(head_dim)`` (the usual "softmax temperature"
    choice). muP (Tensor Programs V, Table 3 -- the attention-logit row of the abc-parametrization)
    instead requires ``1/head_dim``: because muP's hidden-role init/lr rules make the *correlation*
    structure of ``q``/``k`` grow with width (not just their per-coordinate variance, which standard
    ``1/sqrt(head_dim)`` scaling was already designed to control), the standard scale under-divides
    at wide models and lets attention-logit scale drift with width -- exactly the kind of hidden
    per-layer-role mismatch that breaks the zero-shot lr-transfer guarantee. See
    ``mixle.models.transformer.CausalAttention.mup_attention``, which this flips.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("enable_mup_attention requires torch.")
    for block in model.blocks:
        block.attn.mup_attention = bool(enabled)


def apply_mup_init(model, *, base_width: int, base_std: float = 0.02) -> None:
    """Re-initialize ``model`` in place per the muP init rules, relative to ``base_width``.

    ``base_std`` is the hidden-role init std tuned/measured at ``base_width`` (the mixle default,
    ``0.02``, matches common transformer practice and is a reasonable base-width value on its own).
    ``model.d_model`` is read as the target width, so ``width_mult = model.d_model / base_width``.
    LayerNorm weight/bias keep their identity init (``1`` / ``0``, unaffected by width, matching the
    ``"input"`` role's no-rescale treatment); every other bias is zero-initialized (muP does not
    rescale bias init); every other weight matrix is drawn ``Normal(0, base_std * init_std_multiplier(role, width_mult))``.
    Also turns on muP attention-logit scaling (:func:`enable_mup_attention`) on every block -- the
    ``1/head_dim`` QK scaling is as much a part of "the model is parametrized under muP" as the
    init/lr rules above, and previously being left at the standard ``1/sqrt(head_dim)`` scale was an
    unintentional gap between what this module documented and what it actually configured.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("apply_mup_init requires torch.")
    width_mult = float(model.d_model) / float(base_width)
    roles = classify_causal_lm_params(model)
    named = dict(model.named_parameters())
    for name, role in roles.items():
        p = named[name]
        if _is_layernorm_param(name) or (name == "ln.weight" or name == "ln.bias"):
            if name.endswith(".weight"):
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)
            continue
        if p.dim() < 2:
            nn.init.zeros_(p)
            continue
        std = base_std * init_std_multiplier(role, width_mult)
        nn.init.normal_(p, mean=0.0, std=std)
    enable_mup_attention(model, enabled=True)


@dataclass(frozen=True)
class MuPParamGroup:
    """One torch-optimizer param group under muP, tagged with the role it was scaled for."""

    role: Role
    lr: float
    n_params: int


def mup_param_groups(model, *, base_width: int, lr: float) -> list[dict]:
    """Build torch optimizer param groups implementing muP's per-role lr scaling for ``model``.

    ``lr`` is the BASE (hidden-role) learning rate -- the one hyperparameter tuned once, cheaply, at
    ``base_width``. ``model.d_model`` is read as the target width. Returns a list of
    ``{"params": [...], "lr": ..., "mup_role": ...}`` dicts suitable for ``torch.optim.Adam(groups)``;
    the ``"input"`` group's lr is unscaled, the ``"hidden"``/``"output"`` groups get
    ``lr * lr_multiplier(role, width_mult)`` -- i.e. passing the same tuned ``lr`` at any target width
    reproduces exactly what :func:`transfer_lr` predicts for that role. ``transfer_lr`` is the formula;
    this is the mechanism that applies it to a live model + optimizer.
    """
    width_mult = float(model.d_model) / float(base_width)
    roles = classify_causal_lm_params(model)
    named = dict(model.named_parameters())
    by_role: dict[Role, list] = {"input": [], "hidden": [], "output": []}
    for name, role in roles.items():
        by_role[role].append(named[name])
    return [
        {"params": params, "lr": float(lr) * lr_multiplier(role, width_mult), "mup_role": role}
        for role, params in by_role.items()
        if params
    ]


__all__ = [
    "MuPParamGroup",
    "Role",
    "apply_mup_init",
    "enable_mup_attention",
    "classify_causal_lm_params",
    "init_std_multiplier",
    "lr_multiplier",
    "mup_param_groups",
    "output_forward_multiplier",
    "transfer_init_std",
    "transfer_lr",
]
