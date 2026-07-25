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
from dataclasses import dataclass, field
from typing import Any

_KINDS = frozenset({"iid", "exchangeable", "partially_exchangeable", "sequential"})


@dataclass(frozen=True)
class GroupingPolicy:
    """A callable grouping rule with durable identity and an explicit semantics version."""

    policy_id: str
    version: str
    function: Callable[[Any], Any] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be a non-empty stable identifier")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty grouping-policy version")
        if not callable(self.function):
            raise TypeError("function must be callable")

    def __call__(self, record: Any) -> Any:
        return self.function(record)

    def __str__(self) -> str:
        return f"{self.policy_id}@{self.version}"


@dataclass(frozen=True)
class SampleStructure:
    """The joint structure of a dataset's records (an exchangeability class)."""

    kind: str  # "iid" | "exchangeable" | "partially_exchangeable" | "sequential"
    by: str | GroupingPolicy | None = None  # grouping key for partial exchangeability

    def __post_init__(self) -> None:
        """Validate ``kind``/``by`` at the declaration boundary, before either is used to drive a
        partition policy. An unknown ``kind`` used to be silently treated as strideable (every kind
        other than exactly ``"partially_exchangeable"`` strides), and a ``partially_exchangeable``
        instance with no usable ``by`` used to silently group every record into a single bucket via
        :meth:`group_key` -- an invalid scientific assumption quietly became an operational partition
        policy instead of failing here."""
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {self.kind!r}")
        if self.kind == "partially_exchangeable" and not isinstance(self.by, (str, GroupingPolicy)):
            raise ValueError(
                "partially_exchangeable requires a string field name or a versioned GroupingPolicy for "
                f"`by`, got {self.by!r} -- an unusable key would silently group every record together."
            )

    @property
    def strides_records(self) -> bool:
        """True if records may be strided/shuffled across partitions (everything but grouped data)."""
        return self.kind != "partially_exchangeable"

    def group_key(self, record: Any) -> Any:
        """Return the group key of ``record`` for partial exchangeability (else ``None``)."""
        if self.by is None:
            return None
        if isinstance(self.by, GroupingPolicy):
            return self.by(record)
        if isinstance(record, dict):
            return record[self.by]
        return getattr(record, self.by)

    def __str__(self) -> str:
        if self.by is None:
            return self.kind
        identity = self.by if isinstance(self.by, str) else str(self.by)
        return f"{self.kind}(by={identity!r})"


IID = SampleStructure("iid")
EXCHANGEABLE = SampleStructure("exchangeable")
SEQUENTIAL = SampleStructure("sequential")


def grouping_policy(policy_id: str, version: str, function: Callable[[Any], Any]) -> GroupingPolicy:
    """Bind a callable grouping rule to a stable, versioned provenance identity."""
    return GroupingPolicy(policy_id, version, function)


def partially_exchangeable(by: str | GroupingPolicy) -> SampleStructure:
    """Return a ``PARTIALLY_EXCHANGEABLE`` structure grouped by field name or key function ``by``."""
    return SampleStructure("partially_exchangeable", by)


# --- model <-> structure compatibility (the capability check) -------------------------------------
#
# A model declares which sample structures it can consume. Sequential models (HMM/Markov/Hawkes/AR)
# read order off each record; exchangeable latent models (mixtures, Dirichlet-process) are invariant to
# order; grouped models (HDP, labeled-LDA, hierarchical) consume partial exchangeability; plain leaves
# are IID. The default for a bare list is EXCHANGEABLE, so the check is opt-in and never fires on
# existing call sites -- it only catches a mismatch once a user explicitly tags a source.

_EXPLICIT_MODEL_STRUCTURES: dict[str, frozenset[str]] = {
    # ordered records / event histories
    "mixle.stats.combinator.sequence.SequenceEstimator": frozenset({"sequential"}),
    "mixle.stats.latent.hidden_markov.HiddenMarkovEstimator": frozenset({"sequential"}),
    "mixle.stats.latent.lookback_hidden_markov_model.LookbackHiddenMarkovModelEstimator": frozenset({"sequential"}),
    "mixle.stats.latent.quantized_hidden_markov_model.QuantizedHiddenMarkovEstimator": frozenset({"sequential"}),
    "mixle.stats.latent.scheduled_hidden_markov_model.ScheduledHMMEstimator": frozenset({"sequential"}),
    "mixle.stats.latent.segmental_hidden_markov_model.SegmentalHiddenMarkovEstimator": frozenset({"sequential"}),
    "mixle.stats.latent.semi_supervised_hidden_markov_model.SemiSupervisedHiddenMarkovEstimator": frozenset(
        {"sequential"}
    ),
    "mixle.stats.latent.tree_hidden_markov_model.TreeHiddenMarkovEstimator": frozenset({"sequential"}),
    "mixle.stats.sequences.integer_markov_chain.IntegerMarkovChainEstimator": frozenset({"sequential"}),
    "mixle.stats.sequences.markov_chain.MarkovChainEstimator": frozenset({"sequential"}),
    "mixle.stats.sequences.markov_transform.MarkovTransformEstimator": frozenset({"sequential"}),
    "mixle.stats.sequences.sparse_markov_transform.SparseMarkovAssociationEstimator": frozenset({"sequential"}),
    "mixle.stats.processes.ctmc.ContinuousTimeMarkovChainEstimator": frozenset({"sequential"}),
    "mixle.stats.processes.hawkes_process.HawkesProcessEstimator": frozenset({"sequential"}),
    "mixle.stats.processes.multivariate_hawkes.MultivariateHawkesProcessEstimator": frozenset({"sequential"}),
    "mixle.stats.processes.power_law_hawkes.PowerLawHawkesEstimator": frozenset({"sequential"}),
    "mixle.stats.processes.renewal_process.RenewalProcessEstimator": frozenset({"sequential"}),
    "mixle.stats.processes.birth_death.BirthDeathSamplingEstimator": frozenset({"sequential"}),
    # grouped/document hierarchies
    "mixle.stats.bayes.hierarchical_dirichlet_process_mixture.HierarchicalDirichletProcessMixtureEstimator": frozenset(
        {"partially_exchangeable", "exchangeable", "iid"}
    ),
    "mixle.stats.latent.hierarchical.HierarchicalNormalEstimator": frozenset(
        {"partially_exchangeable", "exchangeable", "iid"}
    ),
    "mixle.stats.latent.hierarchical_mixture.HierarchicalMixtureEstimator": frozenset(
        {"partially_exchangeable", "exchangeable", "iid"}
    ),
    "mixle.stats.latent.lda.LDAEstimator": frozenset({"partially_exchangeable", "exchangeable", "iid"}),
    "mixle.stats.latent.labeled_lda.LabeledLDAEstimator": frozenset(
        {"partially_exchangeable", "exchangeable", "iid"}
    ),
}


def supported_structures(model: Any) -> frozenset[str]:
    """Return an explicit structure contract; never infer scientific semantics from a class name."""
    cls = model if isinstance(model, type) else type(model)
    key = f"{cls.__module__}.{cls.__qualname__}"
    declared = _EXPLICIT_MODEL_STRUCTURES.get(key)
    if declared is None:
        declared = getattr(model, "supported_sample_structures", None)
        if callable(declared):
            declared = declared()
    if declared is None:
        raise TypeError(
            f"{key} does not declare supported_sample_structures; structure compatibility cannot be inferred safely"
        )
    result = frozenset(declared)
    if not result or not result <= _KINDS:
        raise ValueError(f"{key} has an invalid supported_sample_structures declaration: {sorted(result)!r}")
    return result


def check_model_structure(model: Any, structure: SampleStructure, *, strict: bool = True) -> None:
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
