"""Block-coordinate Gibbs with per-block inference dispatch -- a different update method per parameter.

Real models are heterogeneous: some parameters have a conjugate full conditional (sample it in closed
form, exactly, no tuning), others do not (fall back to Metropolis), others are best marginalized or
optimized. A single global ``how=`` wastes the structure. BlockGibbs cycles the blocks and lets each one
declare its own conditional update -- a closed-form draw where the conditional is conjugate, a
Metropolis step where it is not -- so the low-cost exact updates run exactly and only the hard blocks pay
for sampling. The composition-expressiveness piece: mixed inference across one model.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["BlockGibbs", "ConjugateBlock", "MetropolisBlock"]

#: Types that cannot be mutated in place, so a reference to one is already a safe snapshot.
_IMMUTABLE = (bool, int, float, complex, str, bytes, type(None), np.number, np.bool_)


def _snapshot(value: Any) -> Any:
    """An independent copy of a block value, so retained samples cannot be rewritten later.

    A conditional update is free to build its result in place -- reusing one preallocated buffer per
    block is the obvious way to write a fast conjugate draw, and an in-place torch/NumPy update is
    idiomatic. Storing the returned object directly makes every retained sample an alias of that one
    buffer, so the "chain" ends up holding N copies of the final state and the caller's own
    initialisation is mutated along with it. Snapshotting at the boundary keeps that legitimate
    implementation style working instead of forbidding it.
    """
    if isinstance(value, _IMMUTABLE):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise TypeError(
            f"BlockGibbs cannot snapshot a block value of type {type(value).__name__!r}: {exc}. "
            "Block values must be copyable, otherwise retained samples would alias mutable state and "
            "the chain could be rewritten in place after the fact."
        ) from exc


def _finite_positive_scale(value: Any) -> float:
    """A random-walk proposal scale must be a finite, strictly positive width."""
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"MetropolisBlock scale must be a real number, got {value!r}") from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"MetropolisBlock scale must be finite and positive, got {scale!r}")
    return scale


def _exact_int(value: Any, label: str, *, minimum: int) -> int:
    """``value`` as an exact integer at least ``minimum`` (``bool`` and floats rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an exact integer, got {value!r}")
    count = int(value)
    if count < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {count!r}")
    return count


class ConjugateBlock:
    """A block whose full conditional is conjugate: ``draw(state, rng)`` returns an exact closed-form sample."""

    def __init__(self, name: str, draw: Callable[[dict, np.random.RandomState], Any]):
        self.name = name
        self._draw = draw
        self.kind = "conjugate"

    def reset(self) -> None:
        """Clear per-run state. A conjugate draw carries none -- present for a uniform block protocol."""

    def update(self, state: dict, rng: np.random.RandomState, in_burn_in: bool = True) -> Any:
        """Draw an exact full-conditional sample for this block.

        ``in_burn_in`` is accepted (and ignored) only to keep a uniform call signature with
        :class:`MetropolisBlock` for :class:`BlockGibbs`'s dispatch loop -- a conjugate draw needs
        no proposal tuning.
        """
        return self._draw(state, rng)


class MetropolisBlock:
    """A non-conjugate block updated by a random-walk Metropolis step on its log full-conditional.

    ``log_conditional(value, state)`` returns the unnormalized log density of this block's value given the
    rest of the state; ``scale`` sets the proposal width (adapted lightly toward a ~0.4 acceptance rate).
    """

    def __init__(self, name: str, log_conditional: Callable[[Any, dict], float], scale: float = 0.5):
        self.name = name
        self._logp = log_conditional
        # A zero proposal scale proposes the current point every sweep: every "proposal" is accepted,
        # so the block reports acceptance 1.0 -- the healthiest-looking number it can produce -- while
        # returning a constant chain. A NaN or infinite scale is the mirror image: every proposal is
        # rejected (NaN comparisons are false, infinite steps land at -inf density), so the chain
        # freezes at its initialisation and reports acceptance 0.0. Neither is a hard-to-tune
        # sampler; both are an unusable one, and both are invisible in the returned chain.
        self.scale = self._initial_scale = _finite_positive_scale(scale)
        self.kind = "metropolis"
        self._acc = 0
        self._tot = 0

    def reset(self) -> None:
        """Restore the proposal scale and acceptance counters this block was constructed with.

        Adaptation state is per-run: a fresh :meth:`BlockGibbs.run` must not inherit the tuning, nor
        the cumulative acceptance rate, that a previous run happened to leave behind.
        """
        self.scale = self._initial_scale
        self._acc = 0
        self._tot = 0

    def update(self, state: dict, rng: np.random.RandomState, in_burn_in: bool = True) -> Any:
        """Run one random-walk Metropolis update for this block.

        ``in_burn_in`` gates the scale adaptation below: adapting the proposal after burn-in ends
        would keep changing the transition kernel during the "stationary" phase the retained
        samples are drawn from, breaking the chain's own validity requirement that an adaptive
        proposal converges before samples are collected -- contrary to this class's own docstring
        promise ("adapted... toward a ~0.4 acceptance rate" during burn-in specifically).
        """
        cur = state[self.name]
        prop = cur + self.scale * rng.standard_normal(np.shape(cur))
        # A NaN anywhere in the Metropolis ratio makes `log(u) < log_alpha` false, so a broken target
        # was indistinguishable from a chain that legitimately never moved: the run completed with
        # every draw equal to the initial state and an acceptance rate of 0.0, which reads as a badly
        # tuned proposal rather than a target that cannot be evaluated. -inf on the *proposal* is
        # different and stays a plain rejection -- that is how a target says "zero probability here".
        cur_lp = float(self._logp(cur, state))
        if not np.isfinite(cur_lp):
            raise ValueError(
                f"block {self.name!r}: the log conditional at the current state is {cur_lp!r}; "
                "the chain cannot occupy a state the target does not admit"
            )
        prop_lp = float(self._logp(prop, state))
        if np.isnan(prop_lp):
            raise ValueError(
                f"block {self.name!r}: the log conditional returned NaN for a proposal; "
                "use -inf to reject a state outside the support"
            )
        log_alpha = prop_lp - cur_lp
        self._tot += 1
        accept = np.log(rng.uniform()) < log_alpha
        if accept:
            self._acc += 1
        if in_burn_in and self._tot % 50 == 0:  # light proposal adaptation during burn-in only
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
        self.blocks = list(blocks)
        if not self.blocks:
            raise ValueError("BlockGibbs requires at least one block")
        names = [b.name for b in self.blocks]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            # Two blocks sharing a name write the same state key and append to the same chain, so one
            # block's draws silently overwrite the other's and the returned chain interleaves two
            # different conditionals under one label.
            raise ValueError(f"BlockGibbs block names must be unique, got duplicates: {duplicated}")
        missing = [n for n in names if n not in init]
        if missing:
            raise ValueError(f"init has no starting value for block(s) {missing}; known keys: {sorted(init)}")
        # Snapshot at construction: the caller's dict AND its values stay theirs. Without this an
        # in-place conditional update mutates the very initialisation the caller passed in, so a second
        # run -- or the caller's own later inspection of it -- silently starts somewhere else.
        self.init = {k: _snapshot(v) for k, v in dict(init).items()}

    def run(
        self, n_samples: int = 2000, *, burn: int = 500, seed: int | None = None, resume: bool = False
    ) -> dict[str, np.ndarray]:
        """Run the chain; returns ``{block_name: array of post-burn-in samples}``.

        Args:
            n_samples: number of post-burn-in draws to retain (exact positive integer).
            burn: number of burn-in sweeps to discard, during which proposals adapt (exact >= 0).
            seed: RNG seed. With ``resume=False`` the same seed reproduces the same chain exactly.
            resume: ``False`` (default) starts a fresh run -- the state returns to ``init`` and every
                block's per-run adaptation state is reset, so the run is a pure function of
                ``(blocks, init, n_samples, burn, seed)``. ``True`` continues from whatever tuning the
                previous run left behind, for deliberately staged runs; the chain then depends on the
                run history and is not reproducible from the seed alone.

        Raises:
            TypeError: if ``n_samples`` or ``burn`` is not an exact integer.
            ValueError: if ``n_samples`` is not positive or ``burn`` is negative.
        """
        n_samples = _exact_int(n_samples, "n_samples", minimum=1)
        burn = _exact_int(burn, "burn", minimum=0)
        if not resume:
            # `run` already resets the RNG and the state; leaving each block's adapted proposal scale
            # and acceptance counters untouched made a re-run with the same seed produce a different
            # chain, and let the previous run's post-burn-in samples feed this run's burn-in
            # adaptation rate. Reset makes a default run reproducible from its arguments alone.
            for b in self.blocks:
                reset = getattr(b, "reset", None)
                if callable(reset):
                    reset()
        rng = np.random.RandomState(seed)
        state = {k: _snapshot(v) for k, v in self.init.items()}
        chains: dict[str, list] = {b.name: [] for b in self.blocks}
        for it in range(burn + n_samples):
            in_burn_in = it < burn
            for b in self.blocks:
                state[b.name] = b.update(state, rng, in_burn_in)
            if it >= burn:
                for b in self.blocks:
                    chains[b.name].append(_snapshot(state[b.name]))
        return {k: np.array(v) for k, v in chains.items()}
