"""E5 (part 1): the selective-scan (S6 / Mamba) module -- a third ``ContextMechanism`` (see
``mixle/experimental/context_spine.py``) alongside E1's ``SlidingWindowSpine``, targeting long, smooth,
low-curvature dependencies that don't compress into a fixed local window. See ``notes/designs/E5.md`` for
the full design: why input-DEPENDENT (selective) ``Delta, A, B, C`` -- not S4's fixed, input-independent
recurrence -- is the property this mechanism exists for, why ``mamba-ssm`` is not a realistic dependency on
this machine (no CUDA toolkit), and the exact S4D-real initialization this module uses.

``_scan_layer`` is the ONE S6 recurrence implementation (a literal sequential Python loop over ``T``, v1
per the design note's explicit scope decision -- a chunked/parallel scan is documented future work, not
attempted here); both :meth:`SelectiveScan.step` and ``mixle.experimental.ssm_hybrid.HybridBlock``'s SSM
branch call it, so there is exactly one scan, not two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from mixle.experimental.graduation import REGISTRY, ExperimentalMechanism

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

__all__ = ["SelectiveScanState", "SelectiveScan"]


@dataclass
class SelectiveScanState:
    """Per-layer recurrent state ``h`` (``(batch, d_inner, d_state)``, ``None`` until the first ``step``)
    plus the running absolute position counter -- the SSM analogue of ``SlidingWindowState``'s KV cache,
    except the state is already fixed-size (no window/cache-length bookkeeping needed)."""

    h: list[Any] = field(default_factory=list)
    batch_size: int = 0
    pos: int = 0


if _HAS_TORCH:
    # dt_proj bias init: choose the bias so softplus(bias) is log-uniform in [_DT_MIN, _DT_MAX], then invert
    # softplus -- exactly Mamba's `Mamba.__init__` dt-bias init (verified directly against the mamba-ssm
    # 2.3.2.post1 sdist source, mamba_ssm/modules/mamba_simple.py, not from memory -- see notes/designs/E5.md
    # Risks, which explicitly flagged this init as unverified pending implementation). Starting Delta small
    # (rather than at an arbitrary scale) is what lets the scan begin near a slow, controllable decay instead
    # of either freezing (Delta ~ 0, no update) or blowing through history (Delta large) at step 0.
    _DT_MIN = 0.001
    _DT_MAX = 0.1
    _DT_INIT_FLOOR = 1e-4

    def _positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive exact integer, got {value}")
        return int(value)

    def _require_finite(value: torch.Tensor, name: str) -> None:
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must contain only finite values")

    def _dt_bias_init(d_inner: int) -> torch.Tensor:
        dt = torch.exp(torch.rand(d_inner) * (math.log(_DT_MAX) - math.log(_DT_MIN)) + math.log(_DT_MIN)).clamp(
            min=_DT_INIT_FLOOR
        )
        return dt + torch.log(-torch.expm1(-dt))  # inverse softplus: softplus(inv_dt) == dt

    def _s4d_real_a_log_init(d_inner: int, d_state: int) -> torch.Tensor:
        """S4D-real init: ``A[d, n] = n`` for ``n = 1..d_state``, IDENTICAL across every ``d_inner`` channel;
        ``A_log = log(A)``. Verified directly against mamba-ssm 2.3.2.post1's ``mamba_simple.py`` (the
        ``# S4D real initialization`` block: ``A = repeat(torch.arange(1, d_state+1), "n -> d n", d=d_inner)``),
        not asserted from training-data recall -- notes/designs/E5.md's Risks section explicitly flagged this
        as the one unverified number the Selective Copying parity receipt depends on."""
        a = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
        return torch.log(a)

    def _scan_layer(
        u: torch.Tensor,
        A_log: torch.Tensor,
        W_delta: nn.Linear,
        W_B: nn.Linear,
        W_C: nn.Linear,
        D: torch.Tensor,
        h_prev: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The S6 recurrence for one layer, one chunk (notes/designs/E5.md, "The scan itself"):

            Delta_t = softplus(W_delta u_t)                  A = -exp(A_log)   (always negative)
            A_bar   = exp(Delta_t (x) A)                      B_t = W_B u_t      C_t = W_C u_t
            h_t     = A_bar * h_{t-1} + (Delta_t (x) B_t) * u_t     y_t = (h_t * C_t).sum(-1) + D * u_t

        ``u``: ``(batch, T, d_inner)``, already projected into the mixer's inner dimension (the "x_t" of the
        design note's math). A literal sequential Python loop over ``T`` -- v1, not the parallel/log-depth
        scan a fused kernel would use (see module docstring). Returns ``(h_last, y)``,
        ``y: (batch, T, d_inner)``. SHARED by :meth:`SelectiveScan.step` and
        ``mixle.experimental.ssm_hybrid.HybridBlock``'s SSM branch -- one scan implementation, not two.
        """
        if not torch.is_tensor(u) or not u.is_floating_point() or u.ndim != 3:
            raise TypeError("u must be a real floating-point tensor with shape (batch, time, d_inner)")
        b, t, d_inner = u.shape
        if b == 0 or t == 0 or d_inner == 0:
            raise ValueError("u must have non-empty batch, time, and d_inner dimensions")
        if not torch.is_tensor(A_log) or A_log.ndim != 2 or A_log.shape[0] != d_inner:
            raise ValueError("A_log must have shape (d_inner, d_state)")
        d_state = A_log.shape[1]
        if d_state == 0:
            raise ValueError("A_log must have a non-empty d_state dimension")
        if A_log.device != u.device:
            raise ValueError("u and A_log must be on the same device")
        if not torch.is_tensor(D) or D.shape != (d_inner,) or D.device != u.device:
            raise ValueError("D must have shape (d_inner,) on u's device")
        expected_linears = {
            "W_delta": (W_delta, d_inner, d_inner),
            "W_B": (W_B, d_inner, d_state),
            "W_C": (W_C, d_inner, d_state),
        }
        for name, (layer, in_features, out_features) in expected_linears.items():
            if (
                not isinstance(layer, nn.Linear)
                or layer.in_features != in_features
                or layer.out_features != out_features
            ):
                raise ValueError(f"{name} must map {in_features} inputs to {out_features} outputs")
            if layer.weight.device != u.device:
                raise ValueError(f"{name} and u must be on the same device")
        if h_prev is not None:
            if (
                not torch.is_tensor(h_prev)
                or h_prev.shape != (b, d_inner, d_state)
                or h_prev.device != u.device
                or h_prev.dtype != u.dtype
            ):
                raise ValueError(f"h_prev must have shape {(b, d_inner, d_state)} on u's device and dtype")
            _require_finite(h_prev, "h_prev")
        _require_finite(u, "u")
        _require_finite(A_log, "A_log")
        _require_finite(D, "D")
        delta = F.softplus(W_delta(u))  # (b, t, d_inner), > 0
        B = W_B(u)  # (b, t, d_state)
        C = W_C(u)  # (b, t, d_state)
        A = -torch.exp(A_log)  # (d_inner, d_state), always negative -- see module docstring
        _require_finite(delta, "Delta")
        _require_finite(B, "B")
        _require_finite(C, "C")
        _require_finite(A, "A")

        h = h_prev if h_prev is not None else u.new_zeros(b, d_inner, A.shape[-1])
        ys: list[torch.Tensor] = []
        for step_t in range(t):
            delta_t = delta[:, step_t]  # (b, d_inner)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (b, d_inner, d_state)
            dB = delta_t.unsqueeze(-1) * B[:, step_t].unsqueeze(1)  # (b, d_inner, d_state)
            h = A_bar * h + dB * u[:, step_t].unsqueeze(-1)  # (b, d_inner, d_state)
            y_t = (h * C[:, step_t].unsqueeze(1)).sum(-1) + D * u[:, step_t]  # (b, d_inner)
            _require_finite(h, "recurrent state")
            _require_finite(y_t, "scan output")
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (b, t, d_inner)
        return h, y

    class SelectiveScan(nn.Module):
        """E5 baseline: the S6/Mamba selective scan as a ``ContextMechanism``.

        Block shape mirrors ``SlidingWindowSpine``'s pre-norm-residual convention (``ln1 -> mixer ->
        residual -> ln2 -> mlp -> residual``, weight-tied head) so the two mechanisms differ only in what
        the "mixer" is (notes/designs/E5.md). ``d_inner = expand * d_model`` (Mamba's convention); the
        recurrent state ``h`` is carried across ``step`` calls exactly like ``SlidingWindowState``'s KV
        cache, and ``detach`` does ``h.detach()`` per layer -- same TBPTT contract, no window/cache-length
        bookkeeping needed since the state is already fixed-size.
        """

        def __init__(
            self, vocab: int, *, d_model: int = 32, d_state: int = 16, n_layer: int = 2, expand: int = 2
        ) -> None:
            super().__init__()
            self.vocab = _positive_integer(vocab, "vocab")
            self.d_model = _positive_integer(d_model, "d_model")
            self.d_state = _positive_integer(d_state, "d_state")
            self.n_layer = _positive_integer(n_layer, "n_layer")
            self.expand = _positive_integer(expand, "expand")
            self.d_inner = self.expand * self.d_model

            self.tok = nn.Embedding(self.vocab, self.d_model)
            self.ln1 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layer)])
            self.in_proj = nn.ModuleList([nn.Linear(self.d_model, self.d_inner) for _ in range(self.n_layer)])
            self.W_delta = nn.ModuleList([nn.Linear(self.d_inner, self.d_inner) for _ in range(self.n_layer)])
            self.W_B = nn.ModuleList([nn.Linear(self.d_inner, self.d_state) for _ in range(self.n_layer)])
            self.W_C = nn.ModuleList([nn.Linear(self.d_inner, self.d_state) for _ in range(self.n_layer)])
            self.out_proj = nn.ModuleList([nn.Linear(self.d_inner, self.d_model) for _ in range(self.n_layer)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layer)])
            self.mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.d_model, 4 * self.d_model),
                        nn.GELU(),
                        nn.Linear(4 * self.d_model, self.d_model),
                    )
                    for _ in range(self.n_layer)
                ]
            )
            self.ln_f = nn.LayerNorm(self.d_model)
            self.head = nn.Linear(self.d_model, self.vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying, matching SlidingWindowSpine's convention

            # S4D-real init (see _s4d_real_a_log_init) -- one A_log per (layer, d_inner, d_state).
            self.A_log = nn.Parameter(
                torch.stack([_s4d_real_a_log_init(self.d_inner, self.d_state) for _ in range(self.n_layer)])
            )
            self.A_log._no_weight_decay = True
            self.D = nn.Parameter(torch.ones(self.n_layer, self.d_inner))
            self.D._no_weight_decay = True

            with torch.no_grad():
                for layer in range(self.n_layer):
                    self.W_delta[layer].bias.copy_(_dt_bias_init(self.d_inner))
                    self.W_delta[layer].bias._no_reinit = True

        def init_state(self, batch_size: int, *, device: str = "cpu") -> SelectiveScanState:
            batch_size = _positive_integer(batch_size, "batch_size")
            h = [
                torch.zeros(
                    batch_size,
                    self.d_inner,
                    self.d_state,
                    device=device,
                    dtype=self.A_log.dtype,
                )
                for _ in range(self.n_layer)
            ]
            return SelectiveScanState(h=h, batch_size=batch_size, pos=0)

        def detach(self, state: SelectiveScanState) -> SelectiveScanState:
            return SelectiveScanState(
                h=[hi.detach() if hi is not None else None for hi in state.h],
                batch_size=state.batch_size,
                pos=state.pos,
            )

        def step(self, state: SelectiveScanState, chunk: tuple[Any, Any]) -> tuple[SelectiveScanState, Any]:
            if not isinstance(state, SelectiveScanState):
                raise TypeError("state must be a SelectiveScanState")
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise TypeError("chunk must be an (input_tokens, target_tokens) tuple")
            x, y = chunk
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.dtype != torch.long or y.dtype != torch.long:
                raise TypeError("input and target tokens must be torch.long tensors")
            if x.ndim != 2 or y.shape != x.shape or x.shape[0] == 0 or x.shape[1] == 0:
                raise ValueError("input and target tokens must have equal non-empty (batch, time) shape")
            b, t = x.shape
            if state.batch_size != b:
                raise ValueError(f"state batch_size={state.batch_size} does not match chunk batch_size={b}")
            if isinstance(state.pos, bool) or not isinstance(state.pos, Integral) or state.pos < 0:
                raise ValueError("state.pos must be a non-negative exact integer")
            if len(state.h) != self.n_layer:
                raise ValueError(f"state.h must contain exactly {self.n_layer} layer entries")
            if y.device != x.device or x.device != self.tok.weight.device:
                raise ValueError("state, model, input, and target tensors must be on the same device")
            if bool(((x < 0) | (x >= self.vocab) | (y < 0) | (y >= self.vocab)).any().item()):
                raise ValueError(f"input and target token IDs must lie in [0, {self.vocab})")
            for layer, state_h in enumerate(state.h):
                if (
                    not torch.is_tensor(state_h)
                    or state_h.shape != (b, self.d_inner, self.d_state)
                    or state_h.device != x.device
                    or state_h.dtype != self.A_log.dtype
                ):
                    raise ValueError(
                        f"state.h[{layer}] must have shape {(b, self.d_inner, self.d_state)} "
                        "on the model device and dtype"
                    )
                _require_finite(state_h, f"state.h[{layer}]")
            h = self.tok(x)
            new_h: list[Any] = []
            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                u = self.in_proj[layer](hn)
                h_last, y_out = _scan_layer(
                    u,
                    self.A_log[layer],
                    self.W_delta[layer],
                    self.W_B[layer],
                    self.W_C[layer],
                    self.D[layer],
                    state.h[layer],
                )
                h = h + self.out_proj[layer](y_out)
                h = h + self.mlp[layer](self.ln2[layer](h))
                new_h.append(h_last)

            logits = self.head(self.ln_f(h))  # (b, t, vocab)
            _require_finite(logits, "logits")
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            new_state = SelectiveScanState(h=new_h, batch_size=state.batch_size, pos=state.pos + t)
            return new_state, loss

        def log_density(self, x: Any, y: Any) -> Any:
            """Return conditional sequence log likelihoods ``sum_t log p(y_t | x_<=t)``.

            ``x`` and ``y`` are equal-shape, non-empty ``(n_sequences, time)`` long tensors. Each row
            is an independent event with a freshly initialized recurrent state. The return has shape
            ``(n_sequences,)`` and is additive over token log probabilities; it is deliberately not a
            length-normalized training score.
            """
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.dtype != torch.long or y.dtype != torch.long:
                raise TypeError("x and y must be torch.long tensors")
            if x.ndim != 2 or y.shape != x.shape or x.shape[0] == 0 or x.shape[1] == 0:
                raise ValueError("x and y must have equal non-empty (n_sequences, time) shape")
            out = []
            for i in range(x.shape[0]):
                state = self.init_state(1, device=str(x.device))
                _, mean_nll = self.step(state, (x[i : i + 1], y[i : i + 1]))
                out.append(-mean_nll * x.shape[1])
            return torch.stack(out)

    REGISTRY.register(ExperimentalMechanism(name="selective_scan"))
