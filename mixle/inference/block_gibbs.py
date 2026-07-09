"""Block-coordinate Gibbs with per-block inference dispatch -- a different update method per parameter.

Real models are heterogeneous: some parameters have a conjugate full conditional (sample it in closed
form, exactly, no tuning), others do not (fall back to Metropolis), others are best marginalized or
optimized. A single global ``how=`` wastes the structure. BlockGibbs cycles the blocks and lets each one
declare its own conditional update -- a closed-form draw where the conditional is conjugate, a
Metropolis step where it is not -- so the low-cost exact updates run exactly and only the hard blocks pay
for sampling. The composition-expressiveness piece: mixed inference across one model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["BlockGibbs", "ConjugateBlock", "MetropolisBlock"]


class ConjugateBlock:
    """A block whose full conditional is conjugate: ``draw(state, rng)`` returns an exact closed-form sample."""

    def __init__(self, name: str, draw: Callable[[dict, np.random.RandomState], Any]):
        self.name = name
        self._draw = draw
        self.kind = "conjugate"

    def update(self, state: dict, rng: np.random.RandomState) -> Any:
        """Draw an exact full-conditional sample for this block."""
        return self._draw(state, rng)


class MetropolisBlock:
    """A non-conjugate block updated by a random-walk Metropolis step on its log full-conditional.

    ``log_conditional(value, state)`` returns the unnormalized log density of this block's value given the
    rest of the state; ``scale`` sets the proposal width (adapted lightly toward a ~0.4 acceptance rate).
    """

    def __init__(self, name: str, log_conditional: Callable[[Any, dict], float], scale: float = 0.5):
        self.name = name
        self._logp = log_conditional
        self.scale = float(scale)
        self.kind = "metropolis"
        self._acc = 0
        self._tot = 0

    def update(self, state: dict, rng: np.random.RandomState) -> Any:
        """Run one random-walk Metropolis update for this block."""
        cur = state[self.name]
        prop = cur + self.scale * rng.standard_normal(np.shape(cur))
        log_alpha = self._logp(prop, state) - self._logp(cur, state)
        self._tot += 1
        accept = np.log(rng.uniform()) < log_alpha
        if accept:
            self._acc += 1
        if self._tot % 50 == 0:  # light proposal adaptation during burn-in
            rate = self._acc / self._tot
            self.scale *= np.exp((rate - 0.4) * 0.5)
        return prop if accept else cur

    @property
    def acceptance_rate(self) -> float:
        """Return the realized Metropolis acceptance rate."""
        return self._acc / max(self._tot, 1)


class BlockGibbs:
    """Block-coordinate sampler that dispatches each block's own conditional update each sweep."""

    def __init__(self, blocks: list, init: dict):
        self.blocks = blocks
        self.init = dict(init)

    def run(self, n_samples: int = 2000, *, burn: int = 500, seed: int | None = None) -> dict[str, np.ndarray]:
        """Run the chain; returns ``{block_name: array of post-burn-in samples}``."""
        rng = np.random.RandomState(seed)
        state = dict(self.init)
        chains: dict[str, list] = {b.name: [] for b in self.blocks}
        for it in range(burn + n_samples):
            for b in self.blocks:
                state[b.name] = b.update(state, rng)
            if it >= burn:
                for b in self.blocks:
                    chains[b.name].append(state[b.name])
        return {k: np.array(v) for k, v in chains.items()}
