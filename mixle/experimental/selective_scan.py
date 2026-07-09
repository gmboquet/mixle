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
        b, t, d_inner = u.shape
        delta = F.softplus(W_delta(u))  # (b, t, d_inner), > 0
        B = W_B(u)  # (b, t, d_state)
        C = W_C(u)  # (b, t, d_state)
        A = -torch.exp(A_log)  # (d_inner, d_state), always negative -- see module docstring

        h = h_prev if h_prev is not None else u.new_zeros(b, d_inner, A.shape[-1])
        ys: list[torch.Tensor] = []
        for step_t in range(t):
            delta_t = delta[:, step_t]  # (b, d_inner)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (b, d_inner, d_state)
            dB = delta_t.unsqueeze(-1) * B[:, step_t].unsqueeze(1)  # (b, d_inner, d_state)
            h = A_bar * h + dB * u[:, step_t].unsqueeze(-1)  # (b, d_inner, d_state)
            y_t = (h * C[:, step_t].unsqueeze(1)).sum(-1) + D * u[:, step_t]  # (b, d_inner)
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
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.d_state = int(d_state)
            self.n_layer = int(n_layer)
            self.expand = int(expand)
            self.d_inner = self.expand * self.d_model

            self.tok = nn.Embedding(vocab, d_model)
            self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
            self.in_proj = nn.ModuleList([nn.Linear(d_model, self.d_inner) for _ in range(n_layer)])
            self.W_delta = nn.ModuleList([nn.Linear(self.d_inner, self.d_inner) for _ in range(n_layer)])
            self.W_B = nn.ModuleList([nn.Linear(self.d_inner, d_state) for _ in range(n_layer)])
            self.W_C = nn.ModuleList([nn.Linear(self.d_inner, d_state) for _ in range(n_layer)])
            self.out_proj = nn.ModuleList([nn.Linear(self.d_inner, d_model) for _ in range(n_layer)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
            self.mlp = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
                    for _ in range(n_layer)
                ]
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying, matching SlidingWindowSpine's convention

            # S4D-real init (see _s4d_real_a_log_init) -- one A_log per (layer, d_inner, d_state).
            self.A_log = nn.Parameter(
                torch.stack([_s4d_real_a_log_init(self.d_inner, d_state) for _ in range(n_layer)])
            )
            self.A_log._no_weight_decay = True
            self.D = nn.Parameter(torch.ones(n_layer, self.d_inner))
            self.D._no_weight_decay = True

            with torch.no_grad():
                for layer in range(n_layer):
                    self.W_delta[layer].bias.copy_(_dt_bias_init(self.d_inner))
                    self.W_delta[layer].bias._no_reinit = True

        def init_state(self, batch_size: int, *, device: str = "cpu") -> SelectiveScanState:
            del batch_size  # state grows lazily from None on first step, like SlidingWindowState's cache
            return SelectiveScanState(h=[None] * self.n_layer, pos=0)

        def detach(self, state: SelectiveScanState) -> SelectiveScanState:
            return SelectiveScanState(
                h=[hi.detach() if hi is not None else None for hi in state.h],
                pos=state.pos,
            )

        def step(self, state: SelectiveScanState, chunk: tuple[Any, Any]) -> tuple[SelectiveScanState, Any]:
            x, y = chunk
            b, t = x.shape
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
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            new_state = SelectiveScanState(h=new_h, pos=state.pos + t)
            return new_state, loss

        def log_density(self, x: Any, y: Any) -> Any:
            """``x, y``: ``(n, T)`` long tensors. Returns ``-mean_per_position_nll`` for each of the ``n``
            sequences, each scored independently (state re-initialized per row) -- one non-streaming
            forward per row, computed by calling ``init_state`` + ``step`` once per row exactly as a
            length-``T``, single-chunk stream would, not a separately-written scoring path
            (notes/designs/E5.md, "GradLeaf citizenship")."""
            out = []
            for i in range(x.shape[0]):
                state = self.init_state(1, device=str(x.device))
                _, mean_nll = self.step(state, (x[i : i + 1], y[i : i + 1]))
                out.append(-mean_nll)
            return torch.stack(out)

    REGISTRY.register(ExperimentalMechanism(name="selective_scan"))
