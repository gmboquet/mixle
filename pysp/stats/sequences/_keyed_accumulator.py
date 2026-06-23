"""Shared key-merge plumbing for the (init, trans) two-key Markov-transform accumulators.

Both ``MarkovTransformAccumulator`` (dense, 3-set) and ``SparseMarkovAssociationAccumulator``
(sparse, 2-set) hold two independently keyed sufficient statistics -- an ``init_count`` vector and a
``trans_count`` (sparse) matrix -- plus a delegated size/length accumulator. The single-key default
on ``StatisticAccumulator`` (keyed on ``self.keys``) does not fit this two-key shape, so the pooling
logic below is shared here rather than re-implemented in each module.

The E-step itself (responsibility scatter, ``_track_ll`` log-density byproduct) is *not* shared: the
dense and sparse paths differ in row-weight construction (outer product vs broadcast), init-term
smoothing, normalization denominator, sparse-matrix type, and scatter mechanism, and the sparse model
carries an extra low-memory bincount encoding with no dense analogue. Only the genuinely identical
key plumbing is collapsed.
"""

from typing import Any


class InitTransKeyedAccumulator:
    """Mixin providing ``key_merge``/``key_replace`` for an (init_key, trans_key) accumulator.

    Expects the host accumulator to define ``init_key``, ``trans_key``, ``init_count``,
    ``trans_count``, and ``size_accumulator``. ``size_accumulator`` may be ``None`` (the dense model)
    or a real accumulator/``NullAccumulator`` (the sparse model); the ``is not None`` guard handles
    both, since ``NullAccumulator.key_merge``/``key_replace`` are no-ops.
    """

    def key_merge(self, stats_dict: dict[str, Any]) -> None:
        """Merge the keyed ``init_count`` and ``trans_count`` into ``stats_dict``, then the size acc."""
        if self.init_key is not None:
            if self.init_key in stats_dict:
                stats_dict[self.init_key] += self.init_count
            else:
                stats_dict[self.init_key] = self.init_count

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                stats_dict[self.trans_key] += self.trans_count
            else:
                stats_dict[self.trans_key] = self.trans_count

        if self.size_accumulator is not None:
            self.size_accumulator.key_merge(stats_dict)

    def key_replace(self, stats_dict: dict[str, Any]) -> None:
        """Replace ``init_count`` and ``trans_count`` from ``stats_dict``, then the size acc."""
        if self.init_key is not None:
            if self.init_key in stats_dict:
                self.init_count = stats_dict[self.init_key]

        if self.trans_key is not None:
            if self.trans_key in stats_dict:
                self.trans_count = stats_dict[self.trans_key]

        if self.size_accumulator is not None:
            self.size_accumulator.key_replace(stats_dict)
