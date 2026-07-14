"""Every keys-bearing family obeys the pooling protocol -- swept over the full public catalog.

The key_merge/key_replace protocol is hand-implemented per family, and that duplication produced
FIVE distinct failure shapes before this sweep existed: pull-without-write-back (12 families kept
only the first site's statistics -- 8 scalar families plus the 4 attention accumulators), dead
keys (CTMC never inserted, so tying was a silent no-op), a reversed key_replace that pushed one
site's statistics OVER the pool (Optional), a key_replace that crashed subscripting the pooled
accumulator object (DiagonalGaussian), and combine/from_value adopting live array references so
pooling mutated the DONOR accumulator (vMF, MVG, DG). Hand-picked per-family tests can't keep up
with ~100 implementations, so this drives the protocol property over every distribution in the
public catalog (the same one sampler_seed_test validates against ``stats.__all__``):

    two accumulators sharing a key, fed DIFFERENT data, must both end holding the pool of ALL the
    data -- exactly what an explicit ``combine`` of the two sites gives -- in either merge order.

Families whose accumulators use a different keying convention (tuple/slot keys on combinators) or
that cannot round-trip sample->encode->accumulate are skipped WITH a reason, a minimum-swept floor
keeps silent skips from hollowing the test out, and the families from the 2026-07-14 pooling fixes
are required by name. A family whose ``key_merge`` mentions ``self.keys`` but never registers the
key FAILS as dead-keys (the CTMC shape) rather than skipping.
"""

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

# the catalog lives in a sibling test module; mixle/tests is not a package, so load it by path
_SEED_TEST = Path(__file__).with_name("sampler_seed_test.py")
_spec = importlib.util.spec_from_file_location("_sampler_seed_catalog", _SEED_TEST)
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_sampler_seed_catalog", _mod)
_spec.loader.exec_module(_mod)
_CATALOG = dict(_mod._stats_public_distribution_catalog())

# the nine families fixed in the 2026-07-14 pooling sweep: they must be SWEPT, never skipped
_REQUIRED = (
    "GaussianDistribution",
    "ExponentialDistribution",
    "GammaDistribution",
    "GeometricDistribution",
    "SkellamDistribution",
    "HurdleDistribution",
    "ZeroInflatedDistribution",
    "InhomogeneousPoissonProcessDistribution",
    "ContinuousTimeMarkovChainDistribution",
)
_MIN_SWEPT = 60  # coverage floor: silent skips must never hollow the sweep out


def _flat(v):
    out = []

    def walk(u):
        if isinstance(u, (tuple, list)):
            for piece in u:
                walk(piece)
        elif isinstance(u, dict):
            for k in sorted(u, key=repr):
                walk(u[k])
        elif isinstance(u, np.ndarray):
            out.extend(np.asarray(u, dtype=np.float64).ravel().tolist())
        elif isinstance(u, (int, float, np.integer, np.floating)):
            out.append(float(u))
        # non-numeric leaves (strings, None, objects) carry no poolable mass: ignored

    walk(v)
    return np.asarray(out)


def _key_tree(root, depth):
    """Key the top accumulator (depth=0) or also its DIRECT children (depth=1); return the keys.

    Hybrid accumulators (Hurdle, ZeroInflated) manage their own key AND hold a child accumulator
    whose statistics their key does not cover -- those need depth 1 to pool fully. But keying a
    child TOGETHER with a parent whose key already covers the whole value (Composite) double-pools
    through object aliasing, so depth 0 must be tried first and depth 1 only when the shallow
    config leaves child statistics site-local. Deeper structures are out of scope on purpose:
    their middle layers are site-local by design. Traversal order is deterministic (sorted
    attribute names), so two accumulators from the same factory pool position by position.
    """
    assigned = []
    counter = iter(range(10_000))

    def key_one(obj):
        if hasattr(obj, "key_merge") and hasattr(obj, "keys") and obj.keys is None:
            key = f"k{next(counter)}"
            obj.keys = key
            assigned.append(key)
            return True
        return False

    if key_one(root) and depth >= 1:
        for _name, value in sorted(vars(root).items()):
            candidates = value if isinstance(value, (list, tuple)) else [value]
            for child in candidates:
                if hasattr(child, "key_merge"):
                    key_one(child)
    return assigned


def _accumulate(dist, fac, batches, depth):
    acc = fac.make()
    assigned = _key_tree(acc, depth)
    enc = dist.dist_to_encoder()
    for data in batches:
        acc.seq_update(enc.seq_encode(data), np.ones(len(data), dtype=np.float64), dist)
    return acc, assigned


def _run_protocol(first, second):
    stats_dict: dict = {}
    first.key_merge(stats_dict)
    second.key_merge(stats_dict)
    first.key_replace(stats_dict)
    second.key_replace(stats_dict)
    return stats_dict


class KeyedProtocolSweepTest(unittest.TestCase):
    def test_every_scalar_keyed_family_pools_across_sites(self):
        swept, skipped = [], {}
        for name, dist in sorted(_CATALOG.items()):
            with self.subTest(family=name):
                outcome = self._check_family(name, dist)
                if outcome is None:
                    swept.append(name)
                else:
                    skipped[name] = outcome
        missing = [n for n in _REQUIRED if n not in swept]
        self.assertFalse(missing, f"required (previously broken) families were not swept: {missing}")
        self.assertGreaterEqual(
            len(swept),
            _MIN_SWEPT,
            f"only {len(swept)} families swept -- the skip list has hollowed the sweep out: {skipped}",
        )

    def _check_family(self, name, dist):
        """Run the pooling property for one catalog entry; return a skip reason or None (=swept).

        The oracle is COMBINE-based: after the protocol, both tied sites must hold exactly what an
        explicit ``combine`` of the two sites' pre-protocol statistics gives. (An
        accumulate-everything-at-one-site oracle is wrong for Monte-Carlo accumulators, whose
        statistics depend on the rng path, and combine-vs-seq_update parity is its own protocol,
        not the keying one.) Config fallback: the top-only keying regime is tried first; families
        whose own key does not cover their child accumulator's statistics (Hurdle, ZeroInflated)
        retry with direct children keyed too.
        """
        try:
            est = dist.estimator()
            fac = est.accumulator_factory()
        except Exception as e:  # noqa: BLE001 -- estimator-less catalog entries skip with a reason
            return f"no estimator/accumulator ({type(e).__name__})"
        try:
            data_a = list(dist.sampler(seed=101).sample(size=5))
            data_b = list(dist.sampler(seed=202).sample(size=3))
        except Exception as e:  # noqa: BLE001
            return f"sampler unsupported ({type(e).__name__})"

        last_err = None
        for depth in (0, 1):
            try:
                outcome = self._check_config(name, dist, fac, data_a, data_b, depth)
            except AssertionError as e:
                last_err = e
                continue
            return outcome
        raise last_err

    def _check_config(self, name, dist, fac, data_a, data_b, depth):
        try:
            acc_a, assigned = _accumulate(dist, fac, [data_a], depth)
            acc_b, _ = _accumulate(dist, fac, [data_b], depth)
        except Exception as e:  # noqa: BLE001 -- families that can't round-trip sample->accumulate
            return f"sample/encode/accumulate round-trip unsupported ({type(e).__name__})"
        if not assigned:
            return "no scalar keys attribute on the accumulator (tuple/slot keying convention)"

        # the combine-based pool oracle, snapshotted BEFORE the protocol mutates either site
        ref = fac.make()
        ref.combine(acc_a.value())
        ref.combine(acc_b.value())
        expected = _flat(ref.value())

        stats_dict = _run_protocol(acc_a, acc_b)
        if not any(k in stats_dict for k in assigned):
            # never registered: EITHER a different keying convention (fine) or the CTMC dead-keys
            # bug shape. If this key_merge reads self.keys, it participates by design -- FAIL loud.
            try:
                src = inspect.getsource(type(acc_a).key_merge)
            except (OSError, TypeError):
                src = ""
            self.assertNotIn(
                "self.keys",
                src,
                f"{name}: key_merge reads self.keys but never registered any assigned key -- dead "
                "keys (both protocol passes are silent no-ops, tied sites never pool)",
            )
            return "key_merge does not use the scalar keys attribute"

        va, vb = _flat(acc_a.value()), _flat(acc_b.value())
        if expected.size == 0 and va.size == 0:
            return "stat-less accumulator (nothing numeric to pool)"
        np.testing.assert_allclose(
            va, vb, rtol=1e-9, atol=1e-12, err_msg=f"{name} (depth={depth}): tied sites diverge after replace"
        )
        np.testing.assert_allclose(
            np.sort(va),
            np.sort(expected),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{name} (depth={depth}): pooled statistics != combine of both sites",
        )
        # order invariance: merging (b, a) must give the same pool (sorted: list-carrying stats
        # may concatenate in merge order without changing the multiset)
        acc_a2, _ = _accumulate(dist, fac, [data_a], depth)
        acc_b2, _ = _accumulate(dist, fac, [data_b], depth)
        _run_protocol(acc_b2, acc_a2)
        np.testing.assert_allclose(
            np.sort(_flat(acc_a2.value())),
            np.sort(va),
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{name} (depth={depth}): pooling is merge-order dependent",
        )
        return None


if __name__ == "__main__":
    unittest.main()
