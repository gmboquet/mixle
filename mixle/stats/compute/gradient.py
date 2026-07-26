"""Gradient-fitting utilities shared by distribution-layer Torch hooks.

The helpers normalize prior specifications, collect child-prior structures, and
provide small state objects used by generic differentiable M-step routines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class GradientFitError(NotImplementedError):
    """Raised when generic gradient fitting cannot handle a model family."""

    pass


def prior_zero(torch, engine, ref=None):
    """Return a scalar zero on the active Torch/device context."""
    if ref is not None:
        return torch.as_tensor(0.0, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(0.0, dtype=engine.dtype, device=engine.device)


def prior_family(prior):
    """Return the normalized prior family name, if ``prior`` is mapping-like."""
    return prior.get("family") if isinstance(prior, Mapping) else None


def _unexpected_keys(values: Mapping[Any, Any], allowed: set[Any], label: str) -> None:
    extras = [key for key in values if key not in allowed]
    if extras:
        raise ValueError("%s contains unknown keys: %s." % (label, ", ".join(map(repr, extras))))


def _prior_alignment(priors: Any) -> tuple[Any, str, bool]:
    """Return payload, alignment mode, and whether an aligned wrapper was removed."""
    if not isinstance(priors, Mapping):
        return priors, "exact", False
    family = prior_family(priors)
    if family == "aligned":
        _unexpected_keys(priors, {"family", "prior", "alignment_receipt"}, "aligned prior")
        if "prior" not in priors:
            raise ValueError("aligned prior requires a 'prior' payload.")
        payload = priors["prior"]
        wrapped = True
    else:
        payload = priors
        wrapped = False
    receipt = priors.get("alignment_receipt")
    if receipt is None:
        if wrapped:
            raise ValueError("aligned prior requires an alignment_receipt.")
        return payload, "exact", wrapped
    if not isinstance(receipt, Mapping):
        raise TypeError("alignment_receipt must be a mapping.")
    _unexpected_keys(receipt, {"mode", "rationale"}, "alignment_receipt")
    mode = receipt.get("mode")
    rationale = receipt.get("rationale")
    if mode not in ("partial", "broadcast"):
        raise ValueError("alignment_receipt.mode must be 'partial' or 'broadcast'.")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("non-exact prior alignment requires a non-empty rationale.")
    if wrapped and mode != "broadcast":
        raise ValueError("an aligned single-prior wrapper requires broadcast mode.")
    return payload, mode, wrapped


def _without_alignment(priors: Mapping[Any, Any]) -> dict[Any, Any]:
    return {key: value for key, value in priors.items() if key != "alignment_receipt"}


def prior_sequence(values, n: int, *, mode: str = "exact", label: str = "prior sequence"):
    """Align a prior sequence exactly, or under an acknowledged partial/broadcast mode."""
    seq = list(values)
    if mode == "exact":
        if len(seq) != n:
            raise ValueError("%s has %d entries; exact alignment requires %d." % (label, len(seq), n))
        return seq
    if mode == "partial":
        if len(seq) > n:
            raise ValueError("%s has %d entries; structure has only %d." % (label, len(seq), n))
        return seq + [None] * (n - len(seq))
    if mode == "broadcast":
        if len(seq) != 1:
            raise ValueError("%s broadcast requires exactly one entry, got %d." % (label, len(seq)))
        return seq * n
    raise ValueError("unknown prior alignment mode %r." % mode)


def _aligned_mapping(values: Any, keys: Sequence[Any], mode: str, label: str) -> dict[Any, Any]:
    expected = tuple(keys)
    if not isinstance(values, Mapping):
        if mode == "broadcast":
            return {key: values for key in expected}
        if isinstance(values, (list, tuple)):
            aligned = prior_sequence(values, len(expected), mode=mode, label=label)
            return dict(zip(expected, aligned))
        if len(expected) == 1 and mode == "exact":
            return {expected[0]: values}
        raise TypeError("%s must be a key mapping or aligned sequence." % label)
    _unexpected_keys(values, set(expected), label)
    present = set(values)
    missing = [key for key in expected if key not in present]
    if mode == "exact" and missing:
        raise ValueError("%s is missing keys required by exact alignment: %s." % (label, ", ".join(map(repr, missing))))
    if mode == "broadcast":
        if len(values) != 1:
            raise ValueError("%s broadcast requires exactly one keyed entry." % label)
        prior = next(iter(values.values()))
        return {key: prior for key in expected}
    return {key: values.get(key) for key in expected}


def composite_child_priors(priors, n: int):
    """Return child priors for a composite-style product distribution."""
    if priors is None:
        return [None] * n
    priors, mode, wrapped = _prior_alignment(priors)
    if wrapped:
        return [priors] * n
    family = prior_family(priors)
    if family == "composite":
        _unexpected_keys(priors, {"family", "children", "alignment_receipt"}, "composite prior")
        return prior_sequence(priors.get("children", ()), n, mode=mode, label="composite child priors")
    if isinstance(priors, (list, tuple)):
        return prior_sequence(priors, n, label="composite child priors")
    if isinstance(priors, Mapping) and "children" in priors:
        _unexpected_keys(priors, {"children", "alignment_receipt"}, "composite prior")
        return prior_sequence(priors.get("children", ()), n, mode=mode, label="composite child priors")
    if n == 1 and priors is not None:
        return [priors]
    raise ValueError("a multi-child composite prior requires exact children or an explicit alignment receipt.")


def conditional_priors(priors, keys):
    """Return per-condition, default, and given priors for conditional models."""
    keys = tuple(keys)
    if priors is None:
        return {key: None for key in keys}, None, None
    priors, mode, wrapped = _prior_alignment(priors)
    if wrapped:
        return {key: priors for key in keys}, priors, priors
    family = prior_family(priors)
    if family == "conditional":
        _unexpected_keys(
            priors,
            {"family", "conditions", "dmap", "default", "given", "alignment_receipt"},
            "conditional prior",
        )
        if "conditions" in priors and "dmap" in priors:
            raise ValueError("conditional prior cannot specify both 'conditions' and 'dmap'.")
        condition_priors = priors.get("conditions", priors.get("dmap", {}))
        condition_priors = _aligned_mapping(condition_priors, keys, mode, "conditional condition priors")
        return condition_priors, priors.get("default"), priors.get("given")
    if isinstance(priors, Mapping) and any(k in priors for k in ("conditions", "dmap", "default", "given")):
        _unexpected_keys(
            priors,
            {"conditions", "dmap", "default", "given", "alignment_receipt"},
            "conditional prior",
        )
        if "conditions" in priors and "dmap" in priors:
            raise ValueError("conditional prior cannot specify both 'conditions' and 'dmap'.")
        condition_priors = priors.get("conditions", priors.get("dmap", {}))
        condition_priors = _aligned_mapping(condition_priors, keys, mode, "conditional condition priors")
        return condition_priors, priors.get("default"), priors.get("given")
    if isinstance(priors, Mapping) and prior_family(priors) is None and any(key in priors for key in keys):
        direct = _without_alignment(priors)
        return _aligned_mapping(direct, keys, mode, "conditional condition priors"), None, None
    if isinstance(priors, (list, tuple)):
        return _aligned_mapping(priors, keys, mode, "conditional condition priors"), None, None
    if len(keys) == 1:
        return {keys[0]: priors}, None, None
    raise ValueError("a multi-condition prior requires exact keys or an explicit alignment receipt.")


def record_child_priors(priors, fields, n: int):
    """Return field-aligned child priors for a named record model."""
    fields = tuple(fields[:n])
    if len(fields) != n:
        raise ValueError("record metadata has %d fields; expected %d children." % (len(fields), n))
    if priors is None:
        return [None] * n
    priors, mode, wrapped = _prior_alignment(priors)
    if wrapped:
        return [priors] * n
    family = prior_family(priors)
    if family == "record":
        _unexpected_keys(priors, {"family", "fields", "alignment_receipt"}, "record prior")
        field_priors = priors.get("fields", {})
        aligned = _aligned_mapping(field_priors, fields, mode, "record field priors")
        return [aligned[field] for field in fields]
    if isinstance(priors, Mapping) and "fields" in priors:
        _unexpected_keys(priors, {"fields", "alignment_receipt"}, "record prior")
        field_priors = priors.get("fields", {})
        aligned = _aligned_mapping(field_priors, fields, mode, "record field priors")
        return [aligned[field] for field in fields]
    if isinstance(priors, Mapping) and family is None and any(field in priors for field in fields):
        aligned = _aligned_mapping(_without_alignment(priors), fields, mode, "record field priors")
        return [aligned[field] for field in fields]
    if isinstance(priors, (list, tuple)):
        return prior_sequence(priors, n, mode=mode, label="record field priors")
    if n == 1 and priors is not None:
        return [priors]
    raise ValueError("a multi-field record prior requires exact fields or an explicit alignment receipt.")


def transform_prior(priors):
    """Return the base-child prior for a transform wrapper."""
    if priors is None:
        return None
    priors, _, wrapped = _prior_alignment(priors)
    if wrapped:
        return priors
    family = prior_family(priors)
    if family == "transform":
        _unexpected_keys(priors, {"family", "base", "child", "alignment_receipt"}, "transform prior")
        if "base" in priors and "child" in priors:
            raise ValueError("transform prior cannot specify both 'base' and 'child'.")
        return priors.get("base", priors.get("child"))
    if isinstance(priors, Mapping) and ("base" in priors or "child" in priors):
        _unexpected_keys(priors, {"base", "child", "alignment_receipt"}, "transform prior")
        if "base" in priors and "child" in priors:
            raise ValueError("transform prior cannot specify both 'base' and 'child'.")
        return priors.get("base", priors.get("child"))
    return priors


def select_child_priors(priors, n: int):
    """Return choice-child priors for a select/routed model."""
    if priors is None:
        return [None] * n
    priors, mode, wrapped = _prior_alignment(priors)
    if wrapped:
        return [priors] * n
    family = prior_family(priors)
    if family == "select":
        _unexpected_keys(priors, {"family", "children", "alignment_receipt"}, "select prior")
        return prior_sequence(priors.get("children", ()), n, mode=mode, label="select child priors")
    if isinstance(priors, Mapping) and "children" in priors:
        _unexpected_keys(priors, {"children", "alignment_receipt"}, "select prior")
        return prior_sequence(priors.get("children", ()), n, mode=mode, label="select child priors")
    if isinstance(priors, (list, tuple)):
        return prior_sequence(priors, n, label="select child priors")
    if n == 1 and priors is not None:
        return [priors]
    raise ValueError("a multi-choice select prior requires exact children or an explicit alignment receipt.")


def mixture_priors(priors, n: int):
    """Return component priors plus an optional weight prior for mixtures."""
    if priors is None:
        return [None] * n, None
    priors, mode, wrapped = _prior_alignment(priors)
    if wrapped:
        return [priors] * n, None
    family = prior_family(priors)
    if family == "mixture":
        _unexpected_keys(priors, {"family", "components", "weights", "alignment_receipt"}, "mixture prior")
        components = prior_sequence(
            priors.get("components", ()), n, mode=mode, label="mixture component priors"
        )
        return components, priors.get("weights")
    if isinstance(priors, Mapping) and ("components" in priors or "weights" in priors):
        _unexpected_keys(priors, {"components", "weights", "alignment_receipt"}, "mixture prior")
        components = prior_sequence(
            priors.get("components", ()), n, mode=mode, label="mixture component priors"
        )
        return components, priors.get("weights")
    if isinstance(priors, (list, tuple)):
        return prior_sequence(priors, n, label="mixture component priors"), None
    if family == "dirichlet":
        return [None] * n, priors
    if n == 1:
        return [priors], None
    raise ValueError("broadcasting one prior across mixture components requires an explicit broadcast receipt.")


def sequence_priors(priors, has_length: bool = True):
    """Return element and length priors for an iid sequence model."""
    if priors is None:
        return None, None
    priors, mode, wrapped = _prior_alignment(priors)
    if wrapped:
        return priors, priors if has_length else None
    family = prior_family(priors)
    if family == "sequence":
        _unexpected_keys(priors, {"family", "element", "length", "alignment_receipt"}, "sequence prior")
        if mode == "exact" and "element" not in priors:
            raise ValueError("sequence prior exact alignment requires an 'element' entry.")
        if has_length and mode == "exact" and "length" not in priors:
            raise ValueError("sequence prior exact alignment requires a 'length' entry.")
        if not has_length and priors.get("length") is not None:
            raise ValueError("sequence has no length model but a length prior was supplied.")
        return priors.get("element"), priors.get("length")
    if isinstance(priors, Mapping) and ("element" in priors or "length" in priors):
        _unexpected_keys(priors, {"element", "length", "alignment_receipt"}, "sequence prior")
        if mode == "exact" and "element" not in priors:
            raise ValueError("sequence prior exact alignment requires an 'element' entry.")
        if has_length and mode == "exact" and "length" not in priors:
            raise ValueError("sequence prior exact alignment requires a 'length' entry.")
        if not has_length and priors.get("length") is not None:
            raise ValueError("sequence has no length model but a length prior was supplied.")
        return priors.get("element"), priors.get("length")
    if isinstance(priors, (list, tuple)):
        seq = prior_sequence(priors, 2 if has_length else 1, mode=mode, label="sequence priors")
        return seq[0], seq[1] if has_length else None
    if has_length:
        raise ValueError("a sequence with a length model requires exact priors or an explicit alignment receipt.")
    return priors, None


def markov_chain_priors(priors, row_keys):
    """Return initial, transition-row, and length priors for Markov chains."""
    row_keys = tuple(row_keys)
    if priors is None:
        return None, {key: None for key in row_keys}, None
    priors, mode, wrapped = _prior_alignment(priors)
    if wrapped:
        return priors, {key: priors for key in row_keys}, priors
    family = prior_family(priors)
    if family == "markov_chain":
        _unexpected_keys(
            priors,
            {"family", "initial", "init", "transitions", "transition_map", "length", "alignment_receipt"},
            "Markov-chain prior",
        )
        if "initial" in priors and "init" in priors:
            raise ValueError("Markov-chain prior cannot specify both 'initial' and 'init'.")
        if "transitions" in priors and "transition_map" in priors:
            raise ValueError("Markov-chain prior cannot specify both 'transitions' and 'transition_map'.")
        rows = priors.get("transitions", priors.get("transition_map", {}))
        rows = _aligned_mapping(rows, row_keys, mode, "Markov transition-row priors")
        return priors.get("initial", priors.get("init")), rows, priors.get("length")
    if isinstance(priors, Mapping) and any(
        k in priors for k in ("initial", "init", "transitions", "transition_map", "length")
    ):
        _unexpected_keys(
            priors,
            {"initial", "init", "transitions", "transition_map", "length", "alignment_receipt"},
            "Markov-chain prior",
        )
        if "initial" in priors and "init" in priors:
            raise ValueError("Markov-chain prior cannot specify both 'initial' and 'init'.")
        if "transitions" in priors and "transition_map" in priors:
            raise ValueError("Markov-chain prior cannot specify both 'transitions' and 'transition_map'.")
        rows = priors.get("transitions", priors.get("transition_map", {}))
        rows = _aligned_mapping(rows, row_keys, mode, "Markov transition-row priors")
        return priors.get("initial", priors.get("init")), rows, priors.get("length")
    if family == "dirichlet":
        raise ValueError("broadcasting a Dirichlet prior across a Markov chain requires an explicit broadcast receipt.")
    if isinstance(priors, (list, tuple)):
        seq = prior_sequence(priors, 3, mode=mode, label="Markov-chain priors")
        transition_priors = seq[1]
        if isinstance(transition_priors, Mapping):
            transition_priors = _aligned_mapping(
                transition_priors, row_keys, mode, "Markov transition-row priors"
            )
        elif isinstance(transition_priors, (list, tuple)):
            transition_priors = _aligned_mapping(
                transition_priors, row_keys, mode, "Markov transition-row priors"
            )
        else:
            if mode != "broadcast":
                raise ValueError("one transition prior requires an explicit broadcast receipt.")
            transition_priors = {key: transition_priors for key in row_keys}
        return seq[0], transition_priors, seq[2]
    raise ValueError("Markov-chain prior requires exact structure or an explicit alignment receipt.")


def dirichlet_alpha_tensor(alpha, labels, logits, engine, torch):
    """Broadcast Dirichlet concentration values to a logits tensor."""
    if alpha is None:
        alpha = 1.0
    if isinstance(alpha, Mapping):
        if labels is None:
            raise ValueError("Dirichlet alpha mappings require categorical labels.")
        _unexpected_keys(alpha, set(labels), "Dirichlet alpha")
        missing = [label for label in labels if label not in alpha]
        if missing:
            raise ValueError(
                "Dirichlet alpha is missing labels required by exact alignment: %s."
                % ", ".join(map(repr, missing))
            )
        alpha = [alpha[label] for label in labels]
    alpha_t = engine.asarray(alpha)
    if alpha_t.ndim == 0:
        return alpha_t + torch.zeros_like(logits)
    return alpha_t


def normal_gamma_log_prior(mu, sigma2, priors, torch):
    """Return a Normal-Gamma log prior over a Gaussian-style mean/variance."""
    if prior_family(priors) != "normalgamma":
        return None
    tau = 1.0 / sigma2
    alpha = float(priors.get("alpha", priors.get("a", 1.0)))
    beta = float(priors.get("beta", priors.get("b", 0.0)))
    lp = (alpha - 1.0) * torch.log(tau) - beta * tau
    kappa = float(priors.get("kappa", 0.0))
    if kappa > 0.0:
        lp = lp + 0.5 * torch.log(tau) - 0.5 * kappa * tau * (mu - float(priors.get("mu0", 0.0))) ** 2
    return lp


class CategoricalGradientFitState:
    """Autograd state for finite categorical simplex maps."""

    def __init__(self, template: Any, labels: Sequence[Any], logits: Any) -> None:
        self.template = template
        self.labels = tuple(labels)
        self.logits = logits

    def shadow(self, torch, shadow_child):
        """Build a temporary distribution object backed by live Torch logits."""
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        probs = torch.softmax(self.logits, dim=0)
        shadow._backend_labels = self.labels
        shadow._backend_log_probs = torch.log(probs)
        shadow._backend_log_default = getattr(self.template, "log_default_value", -np.inf)
        return shadow

    def score(self, enc, engine, torch, score_child):
        """Score encoded categorical observations through the shadow object."""
        from mixle.stats.compute.backend import backend_seq_log_density

        return backend_seq_log_density(self.shadow(torch, None), enc, engine)

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted categorical distribution from optimized logits."""
        probs = torch.softmax(self.logits, dim=0).detach().cpu().numpy()
        pmap = {label: float(prob) for label, prob in zip(self.labels, probs)}
        return type(self.template)(
            pmap, default_value=getattr(self.template, "default_value", 0.0), name=getattr(self.template, "name", None)
        )

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return the Dirichlet or weak fallback prior contribution."""
        if prior_strength == 0.0 and priors is None:
            return prior_zero(torch, engine, self.logits)
        alpha = None
        if prior_family(priors) == "dirichlet":
            alpha = priors.get("alpha")
        if alpha is None:
            alpha = 1.0 + float(prior_strength) / max(1, self.logits.numel())
        alpha_t = dirichlet_alpha_tensor(alpha, self.labels, self.logits, engine, torch)
        return torch.sum((alpha_t - 1.0) * torch.log_softmax(self.logits, dim=0))


class OptionalGradientFitState:
    """Autograd state for optional/missing-value wrappers."""

    def __init__(self, template: Any, child: Any, logit_p: Any) -> None:
        self.template = template
        self.child = child
        self.logit_p = logit_p

    def shadow(self, torch, shadow_child):
        """Build a temporary optional wrapper backed by live child/raw p state."""
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        shadow.dist = shadow_child(self.child, torch)
        shadow.has_p = self.logit_p is not None
        if self.logit_p is not None:
            shadow.p = torch.sigmoid(self.logit_p)
            shadow.log_p = torch.log(shadow.p)
            shadow.log_pn = torch.log1p(-shadow.p)
        return shadow

    def score(self, enc, engine, torch, score_child):
        """Score encoded optional observations, including missing values."""
        sz, z_idx, nz_idx, enc_data = enc
        rv = engine.zeros(sz)
        if self.logit_p is not None:
            p = torch.sigmoid(self.logit_p)
            if len(z_idx):
                rv[engine.asarray(z_idx)] = torch.log(p)
            if len(nz_idx):
                rv[engine.asarray(nz_idx)] = score_child(self.child, enc_data, engine, torch) + torch.log1p(-p)
        else:
            if len(nz_idx):
                rv[engine.asarray(nz_idx)] = score_child(self.child, enc_data, engine, torch)
        return rv

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted optional distribution from optimized state."""
        p = None if self.logit_p is None else float(torch.sigmoid(self.logit_p).detach().cpu().item())
        return type(self.template)(
            build_child(self.child, torch),
            p=p,
            missing_value=getattr(self.template, "missing_value", None),
            name=getattr(self.template, "name", None),
        )

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return observed-child plus optional missingness prior contribution."""
        family = prior_family(priors)
        if family == "optional":
            child_prior = priors.get("observed")
            missing_prior = priors.get("missing")
        elif family == "beta":
            child_prior = None
            missing_prior = priors
        else:
            child_prior = priors
            missing_prior = None
        rv = prior_child(self.child, child_prior, prior_strength, torch, engine, initial_leaves_by_id)
        if self.logit_p is not None:
            p = torch.sigmoid(self.logit_p)
            if prior_family(missing_prior) == "beta":
                alpha = float(missing_prior.get("alpha", 1.0))
                beta = float(missing_prior.get("beta", 1.0))
                rv = rv + (alpha - 1.0) * torch.log(p) + (beta - 1.0) * torch.log1p(-p)
            elif prior_strength != 0.0:
                alpha = 1.0 + float(prior_strength) / 2.0
                rv = rv + (alpha - 1.0) * (torch.log(p) + torch.log1p(-p))
        return rv


class CompositeGradientFitState:
    """Autograd state for product distributions over tuple-like observations."""

    def __init__(self, template: Any, children: Sequence[Any]) -> None:
        self.template = template
        self.children = list(children)

    def shadow(self, torch, shadow_child):
        """Build a temporary composite with live child shadows."""
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        shadow.dists = tuple(shadow_child(child, torch) for child in self.children)
        shadow.count = len(shadow.dists)
        return shadow

    def score(self, enc, engine, torch, score_child):
        """Return summed child log densities for encoded tuple fields."""
        rv = score_child(self.children[0], enc[0], engine, torch)
        for i in range(1, len(self.children)):
            rv = rv + score_child(self.children[i], enc[i], engine, torch)
        return rv

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted composite from fitted children."""
        return type(self.template)(tuple(build_child(child, torch) for child in self.children))

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return the sum of field-aligned child prior contributions."""
        priors_by_child = composite_child_priors(priors, len(self.children))
        rv = prior_zero(torch, engine)
        for child, child_prior in zip(self.children, priors_by_child):
            rv = rv + prior_child(child, child_prior, prior_strength, torch, engine, initial_leaves_by_id)
        return rv


class ConditionalGradientFitState:
    """Autograd state for conditional keyed children."""

    def __init__(self, template: Any, dmap: Mapping[Any, Any], default_child: Any, given_child: Any) -> None:
        self.template = template
        self.dmap = dict(dmap)
        self.default_child = default_child
        self.given_child = given_child

    def shadow(self, torch, shadow_child):
        """Build a temporary conditional model with live child shadows."""
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        shadow.dmap = {key: shadow_child(child, torch) for key, child in self.dmap.items()}
        if self.default_child is not None:
            shadow.default_dist = shadow_child(self.default_child, torch)
            shadow.has_default = True
        if self.given_child is not None:
            shadow.given_dist = shadow_child(self.given_child, torch)
            shadow.has_given = True
        return shadow

    def score(self, enc, engine, torch, score_child):
        """Score grouped conditional encodings by condition key."""
        sz, cond_vals, eobs_vals, idx_vals, given_enc = enc
        rv = engine.zeros(sz)
        for i in range(len(cond_vals)):
            key = cond_vals[i]
            idx = engine.asarray(idx_vals[i])
            if key in self.dmap:
                scores = score_child(self.dmap[key], eobs_vals[i], engine, torch)
            elif self.default_child is not None:
                scores = score_child(self.default_child, eobs_vals[i], engine, torch)
            else:
                scores = engine.zeros(len(idx_vals[i])) + float("-inf")
            rv = engine.index_add(rv, idx, scores)
        if self.given_child is not None and given_enc is not None:
            rv = rv + score_child(self.given_child, given_enc, engine, torch)
        return rv

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted conditional distribution from children."""
        fitted_map = {key: build_child(child, torch) for key, child in self.dmap.items()}
        default_dist = (
            getattr(self.template, "default_dist", None)
            if self.default_child is None
            else build_child(self.default_child, torch)
        )
        given_dist = (
            getattr(self.template, "given_dist", None)
            if self.given_child is None
            else build_child(self.given_child, torch)
        )
        return type(self.template)(
            fitted_map,
            default_dist=default_dist,
            given_dist=given_dist,
            name=getattr(self.template, "name", None),
            keys=getattr(self.template, "keys", None),
        )

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return keyed/default/given prior contributions for all active children."""
        condition_priors, default_prior, given_prior = conditional_priors(priors, tuple(self.dmap.keys()))
        rv = prior_zero(torch, engine)
        for key, child in self.dmap.items():
            rv = rv + prior_child(child, condition_priors.get(key), prior_strength, torch, engine, initial_leaves_by_id)
        if self.default_child is not None:
            rv = rv + prior_child(
                self.default_child, default_prior, prior_strength, torch, engine, initial_leaves_by_id
            )
        if self.given_child is not None:
            rv = rv + prior_child(self.given_child, given_prior, prior_strength, torch, engine, initial_leaves_by_id)
        return rv


class RecordGradientFitState(CompositeGradientFitState):
    """Autograd state for named record distributions."""

    def score(self, enc, engine, torch, score_child):
        """Score named-record encodings using product-distribution logic."""
        if not self.children:
            n = int(enc[0]) if isinstance(enc, tuple) and len(enc) == 1 else 0
            return engine.zeros(n)
        return super().score(enc, engine, torch, score_child)

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted record distribution preserving field sources."""
        fields = tuple(zip(getattr(self.template, "fields", ()), getattr(self.template, "sources", ())))
        return type(self.template)(fields, tuple(build_child(child, torch) for child in self.children))

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return field-aligned child prior contributions."""
        priors_by_child = record_child_priors(priors, getattr(self.template, "fields", ()), len(self.children))
        rv = prior_zero(torch, engine)
        for child, child_prior in zip(self.children, priors_by_child):
            rv = rv + prior_child(child, child_prior, prior_strength, torch, engine, initial_leaves_by_id)
        return rv


class SelectGradientFitState(CompositeGradientFitState):
    """Autograd state for choice-routed child distributions."""

    def score(self, enc, engine, torch, score_child):
        """Score each routed subset with the selected child model."""
        xi, idx, enc_tuple = enc
        rv = engine.zeros(sum(len(u) for u in xi))
        for i in range(len(idx)):
            child_scores = score_child(self.children[idx[i]], enc_tuple[i], engine, torch)
            rv = engine.index_add(rv, engine.asarray(xi[i]), child_scores)
        return rv

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted select distribution preserving the router."""
        return type(self.template)(
            tuple(build_child(child, torch) for child in self.children), self.template.choice_function
        )

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return child prior contributions aligned to select choices."""
        priors_by_child = select_child_priors(priors, len(self.children))
        rv = prior_zero(torch, engine)
        for child, child_prior in zip(self.children, priors_by_child):
            rv = rv + prior_child(child, child_prior, prior_strength, torch, engine, initial_leaves_by_id)
        return rv


class TransformGradientFitState:
    """Autograd state for fixed transforms with differentiable children."""

    def __init__(self, template: Any, child: Any) -> None:
        self.template = template
        self.child = child

    def shadow(self, torch, shadow_child):
        """Build a temporary transform wrapper with a live base child."""
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        shadow.dist = shadow_child(self.child, torch)
        return shadow

    def score(self, enc, engine, torch, score_child):
        """Score transformed encodings with optional Jacobian correction."""
        child_enc, log_jac, valid = enc
        rv = score_child(self.child, child_enc, engine, torch)
        if self.template.density_correction:
            rv = rv + engine.asarray(log_jac)
        invalid = engine.zeros(rv.shape) + float("-inf")
        return engine.where(engine.asarray(valid), rv, invalid)

    def build(self, torch, build_child, detach_value):
        """Reconstruct a transform wrapper around the fitted base child."""
        return type(self.template)(
            build_child(self.child, torch),
            transform=getattr(self.template, "transform", None),
            density_correction=getattr(self.template, "density_correction", None),
            name=getattr(self.template, "name", None),
            keys=getattr(self.template, "keys", None),
        )

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return the base-child prior contribution for a transform wrapper."""
        return prior_child(self.child, transform_prior(priors), prior_strength, torch, engine, initial_leaves_by_id)


class SequenceGradientFitState:
    """Autograd state for iid sequence distributions."""

    def __init__(self, template: Any, child: Any, len_child: Any) -> None:
        self.template = template
        self.child = child
        self.len_child = len_child

    def shadow(self, torch, shadow_child):
        """Build a temporary sequence model with live element/length children."""
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        shadow.dist = shadow_child(self.child, torch)
        if self.len_child is not None:
            shadow.len_dist = shadow_child(self.len_child, torch)
            shadow.null_len_dist = False
        return shadow

    def score(self, enc, engine, torch, score_child):
        """Score flattened sequence elements plus optional length model."""
        idx, icnt, inz, enc_seq, enc_nseq = enc
        rv = engine.zeros(len(icnt))
        if len(idx) > 0:
            elem_ll = score_child(self.child, enc_seq, engine, torch)
            eidx = engine.asarray(idx)
            if self.template.len_normalized:
                elem_ll = elem_ll * engine.asarray(icnt)[eidx]
            rv = engine.index_add(rv, eidx, elem_ll)
        if self.len_child is not None and enc_nseq is not None:
            rv = rv + score_child(self.len_child, enc_nseq, engine, torch)
        return rv

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted sequence distribution from child fits."""
        length_dist = (
            getattr(self.template, "len_dist", None) if self.len_child is None else build_child(self.len_child, torch)
        )
        return type(self.template)(
            build_child(self.child, torch),
            len_dist=length_dist,
            len_normalized=getattr(self.template, "len_normalized", False),
            name=getattr(self.template, "name", None),
        )

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return element and length child prior contributions."""
        child_prior, len_prior = sequence_priors(priors, has_length=self.len_child is not None)
        rv = prior_child(self.child, child_prior, prior_strength, torch, engine, initial_leaves_by_id)
        if self.len_child is not None:
            rv = rv + prior_child(self.len_child, len_prior, prior_strength, torch, engine, initial_leaves_by_id)
        return rv


class MixtureGradientFitState:
    """Autograd state for mixture components and simplex weights."""

    def __init__(self, template: Any, components: Sequence[Any], w_logits: Any) -> None:
        self.template = template
        self.components = list(components)
        self.w_logits = w_logits

    def shadow(self, torch, shadow_child):
        """Build a temporary mixture backed by live component/weight tensors."""
        shadow = object.__new__(type(self.template))
        shadow.__dict__.update(getattr(self.template, "__dict__", {}))
        shadow.components = [shadow_child(child, torch) for child in self.components]
        shadow.num_components = len(shadow.components)
        shadow.w = torch.softmax(self.w_logits, dim=0)
        shadow.log_w = torch.log_softmax(self.w_logits, dim=0)
        shadow.zw = [False] * shadow.num_components
        return shadow

    def score(self, enc, engine, torch, score_child):
        """Return mixture log densities via log-sum-exp over components."""
        scores = [score_child(child, enc, engine, torch) for child in self.components]
        comp = engine.stack(scores, axis=1)
        return engine.logsumexp(comp + torch.log_softmax(self.w_logits, dim=0)[None, :], axis=1)

    def build(self, torch, build_child, detach_value):
        """Reconstruct a fitted mixture from optimized components and weights."""
        components = [build_child(child, torch) for child in self.components]
        weights = detach_value(torch.softmax(self.w_logits, dim=0))
        return type(self.template)(components, list(weights / weights.sum()), name=getattr(self.template, "name", None))

    def log_prior(self, priors, prior_strength: float, torch, engine, initial_leaves_by_id, prior_child):
        """Return component prior contributions plus mixture-weight prior."""
        component_priors, weight_prior = mixture_priors(priors, len(self.components))
        rv = prior_zero(torch, engine)
        for child, child_prior in zip(self.components, component_priors):
            rv = rv + prior_child(child, child_prior, prior_strength, torch, engine, initial_leaves_by_id)
        if prior_family(weight_prior) == "dirichlet":
            alpha = dirichlet_alpha_tensor(weight_prior.get("alpha"), None, self.w_logits, engine, torch)
            rv = rv + torch.sum((alpha - 1.0) * torch.log_softmax(self.w_logits, dim=0))
        elif prior_strength != 0.0:
            alpha = 1.0 + float(prior_strength) / max(1, len(self.components))
            rv = rv + torch.sum((alpha - 1.0) * torch.log_softmax(self.w_logits, dim=0))
        return rv
