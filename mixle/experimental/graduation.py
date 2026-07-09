"""Graduation bookkeeping for mechanisms living in :mod:`mixle.experimental`.

Policy (see ``mixle/experimental/README.md``): a mechanism graduates out of ``experimental/`` into the
stable package when it (a) beats the E1 baseline on the E7 long-context referee suite at matched FLOPs,
and (b) has misfit/truncation receipts -- measured error-characterization artifacts, not just "it works".

Neither the E1 baseline nor the E7 referee suite exists yet at the time this scaffold was written; this
module only encodes the *shape* of the bookkeeping those later items will populate and check against.
Nothing here enforces the rule today -- :meth:`ExperimentalMechanism.is_eligible` is pure bookkeeping over
whatever receipts a mechanism happens to have attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentalMechanism:
    """One entry in the experimental-mechanism graduation ledger.

    - ``name``: short identifier, e.g. ``"chunked_recurrent_spine"`` (E1) or ``"sketch_state_attention"`` (E3).
    - ``graduated``: whether this mechanism has already been promoted into the stable package.
    - ``baseline_receipt``: the matched-FLOPs comparison against the E1 baseline on the E7 suite, once both
      exist -- e.g. ``{"metric": "bpb", "mechanism": 1.02, "baseline": 1.05, "flops": 3.1e20}``. ``None``
      until measured.
    - ``misfit_receipt``: honest error-characterization artifacts for the mechanism's state structure --
      e.g. sketch collision rate, tree truncation error, moment-closure residual. ``None`` until measured.
    """

    name: str
    graduated: bool = False
    baseline_receipt: dict | None = None
    misfit_receipt: dict | None = None

    def is_eligible(self) -> bool:
        """Whether both receipts required by the graduation rule are present.

        This does not check ``graduated`` -- it answers "could this graduate", not "has it".
        """
        return self.baseline_receipt is not None and self.misfit_receipt is not None


@dataclass
class _GraduationRegistry:
    """In-memory ledger of :class:`ExperimentalMechanism` entries, keyed by name."""

    _mechanisms: dict[str, ExperimentalMechanism] = field(default_factory=dict)

    def register(self, mechanism: ExperimentalMechanism) -> ExperimentalMechanism:
        """Add (or replace) a mechanism entry and return it."""
        self._mechanisms[mechanism.name] = mechanism
        return mechanism

    def get(self, name: str) -> ExperimentalMechanism:
        """Look up a registered mechanism by name."""
        return self._mechanisms[name]

    def __iter__(self):
        return iter(self._mechanisms.values())

    def __len__(self) -> int:
        return len(self._mechanisms)


REGISTRY = _GraduationRegistry()
"""The process-wide graduation ledger. Track-E items register their ``ExperimentalMechanism`` here."""
