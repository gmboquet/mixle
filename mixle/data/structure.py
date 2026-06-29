"""Sample structure -- the exchangeability tag carried by every :class:`~mixle.data.core.DataSource`.

``seq_encode(data, num_chunks=C)`` partitions a dataset by *striding* -- chunk ``i`` is ``data[i::C]`` --
which silently reorders observations. That is correct only when the records are exchangeable. This module
makes the intended joint structure explicit so partitioning is *justified* rather than assumed, and so a
model can be checked against the data it is handed:

* ``IID``                   -- independent & identically distributed records.
* ``EXCHANGEABLE``          -- the joint law is permutation-invariant (de Finetti): order is irrelevant
                              but latent coupling is allowed (mixtures, Dirichlet-process models, ...).
* ``PARTIALLY_EXCHANGEABLE`` (``by``) -- exchangeable *within* groups keyed by ``by`` (hierarchical /
                              grouped / panel data): groups must stay intact on a partition.
* ``SEQUENTIAL``            -- each record is a whole ordered sequence (HMM / Markov / Hawkes / AR); the
                              records are mutually exchangeable, so they may be strided, but a record is
                              never split internally (the encoder owns the within-record order).

The first three (and ``SEQUENTIAL``, whose records are atomic) may stride records freely; only
``PARTIALLY_EXCHANGEABLE`` constrains partitioning -- groups are distributed whole. The default for an
un-annotated dataset is ``EXCHANGEABLE``, which is exactly today's striding behavior, so nothing changes
until a user opts in by tagging a source.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SampleStructure:
    """The joint structure of a dataset's records (an exchangeability class)."""

    kind: str  # "iid" | "exchangeable" | "partially_exchangeable" | "sequential"
    by: str | Callable[[Any], Any] | None = None  # grouping key for partial exchangeability

    @property
    def strides_records(self) -> bool:
        """True if records may be strided/shuffled across partitions (everything but grouped data)."""
        return self.kind != "partially_exchangeable"

    def group_key(self, record: Any) -> Any:
        """Return the group key of ``record`` for partial exchangeability (else ``None``)."""
        if self.by is None:
            return None
        if callable(self.by):
            return self.by(record)
        if isinstance(record, dict):
            return record[self.by]
        return getattr(record, self.by)

    def __str__(self) -> str:
        return self.kind if self.by is None else "%s(by=%r)" % (self.kind, self.by)


IID = SampleStructure("iid")
EXCHANGEABLE = SampleStructure("exchangeable")
SEQUENTIAL = SampleStructure("sequential")


def partially_exchangeable(by: str | Callable[[Any], Any]) -> SampleStructure:
    """Return a ``PARTIALLY_EXCHANGEABLE`` structure grouped by field name or key function ``by``."""
    return SampleStructure("partially_exchangeable", by)


# --- model <-> structure compatibility (the capability check) -------------------------------------
#
# A model declares which sample structures it can consume. Sequential models (HMM/Markov/Hawkes/AR)
# read order off each record; exchangeable latent models (mixtures, Dirichlet-process) are invariant to
# order; grouped models (HDP, labeled-LDA, hierarchical) consume partial exchangeability; plain leaves
# are IID. The default for a bare list is EXCHANGEABLE, so the check is opt-in and never fires on
# existing call sites -- it only catches a mismatch once a user explicitly tags a source.

_SEQUENTIAL_HINTS = (
    "markov",
    "hmm",
    "hawkes",
    "pcfg",
    "grammar",
    "renewal",
    "sequence",
    "segmental",
    "lookback",
    "inhomogeneous",
    "birth_death",
    "temporal",
    "autoreg",
)
_GROUPED_HINTS = ("hdp", "hierarchical", "labeled_lda", "labeledlda", "ldadistribution", "lda")
_EXCHANGEABLE_HINTS = ("mixture", "dirichletprocess", "dirichlet_process", "pitman", "buffet", "latent")


def supported_structures(model: Any) -> frozenset[str]:
    """Return the ``SampleStructure`` kinds a model/estimator can consume (by capability + name)."""
    cls = type(model).__name__.lower()
    if any(h in cls for h in _SEQUENTIAL_HINTS):
        return frozenset({"sequential"})
    if any(h in cls for h in _GROUPED_HINTS):
        return frozenset({"partially_exchangeable", "exchangeable", "iid"})
    if any(h in cls for h in _EXCHANGEABLE_HINTS):
        return frozenset({"exchangeable", "iid"})
    return frozenset({"iid", "exchangeable"})  # a plain leaf is i.i.d./exchangeable


def check_model_structure(model: Any, structure: SampleStructure, *, strict: bool = False) -> None:
    """Warn (or, if ``strict``, raise) when a model cannot consume a source's sample structure.

    Catches the silent footgun: a source tagged ``SEQUENTIAL`` handed to an i.i.d. leaf ("did you mean an
    HMM?"), or grouped data handed to a model that ignores groups. A no-op when compatible.
    """
    if structure.kind in supported_structures(model):
        return
    msg = (
        "data is %s, but %s consumes %s -- the structure assumption does not match (e.g. an i.i.d. "
        "model on an ordered series silently strides away the order)."
        % (structure, type(model).__name__, "/".join(sorted(supported_structures(model))))
    )
    if strict:
        raise ValueError(msg)
    import warnings

    warnings.warn(msg, stacklevel=3)
