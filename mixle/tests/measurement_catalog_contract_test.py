"""Measured cost must not round toward "free", and a catalog is evidence (MXR-080-1878).

Three defects in one surface. ``estimate`` built a ``CostEstimate`` with ``int(median(...))``, which
TRUNCATES -- and a median over an even sample count is a midpoint, so two measurements of 0 and 1
bytes estimated 0 bytes of communication. A scheduler divides gain by that number.
``EffectiveContextMeasurement`` enforced only the outer of two nested count relations, so a receipt
could attend to more tokens than it ever materialized. And the catalog's record list was public and
mutable, so a caller could append a fabricated measurement without passing the one type check.
"""

import unittest

from mixle.experimental.typed_runtime.contracts import UpdateKind
from mixle.experimental.typed_runtime.measurement import (
    EffectiveContextMeasurement,
    MeasurementCatalog,
    WorkMeasurement,
)

KEY = ("n", UpdateKind.EXACT_CLOSED_FORM, "b")


def _measurement(**overrides) -> WorkMeasurement:
    fields = dict(
        node_type="n",
        update_kind=UpdateKind.EXACT_CLOSED_FORM,
        backend="b",
        wall_time_seconds=1.0,
    )
    fields.update(overrides)
    return WorkMeasurement(**fields)


class CostRoundingTest(unittest.TestCase):
    def test_a_fractional_median_does_not_truncate_to_free(self):
        catalog = MeasurementCatalog()
        catalog.record(_measurement(communication_bytes=0))
        catalog.record(_measurement(communication_bytes=1))
        # median is 0.5; int() made it 0, i.e. communication is free.
        self.assertEqual(catalog.estimate(*KEY).communication_bytes, 1)

    def test_an_exact_median_is_unchanged(self):
        catalog = MeasurementCatalog()
        for value in (4, 4, 4):
            catalog.record(_measurement(communication_bytes=value))
        self.assertEqual(catalog.estimate(*KEY).communication_bytes, 4)

    def test_rounding_is_upward_so_an_estimate_never_understates_cost(self):
        catalog = MeasurementCatalog()
        for value in (2, 3):
            catalog.record(_measurement(communication_bytes=value))
        self.assertEqual(catalog.estimate(*KEY).communication_bytes, 3)

    def test_an_empty_catalog_still_reports_no_evidence(self):
        self.assertIsNone(MeasurementCatalog().estimate(*KEY))


class CatalogOwnershipTest(unittest.TestCase):
    def test_records_are_a_detached_read_only_view(self):
        catalog = MeasurementCatalog()
        catalog.record(_measurement())
        records = catalog.records
        self.assertIsInstance(records, tuple)
        with self.assertRaises(AttributeError):
            records.append(_measurement())  # type: ignore[attr-defined]

    def test_the_view_does_not_track_later_records(self):
        catalog = MeasurementCatalog()
        catalog.record(_measurement())
        snapshot = catalog.records
        catalog.record(_measurement())
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(catalog.records), 2)

    def test_construction_still_accepts_and_checks_initial_records(self):
        self.assertEqual(len(MeasurementCatalog([_measurement()]).records), 1)
        with self.assertRaisesRegex(TypeError, "must be WorkMeasurement values"):
            MeasurementCatalog(["not a measurement"])

    def test_record_still_type_checks(self):
        with self.assertRaisesRegex(TypeError, "accept WorkMeasurement values"):
            MeasurementCatalog().record("not a measurement")


class NestedCountTest(unittest.TestCase):
    """Attention is over materialized context, so it is a subset of it."""

    def test_attending_to_more_than_was_materialized_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            EffectiveContextMeasurement(materialized_tokens=10, attended_tokens=50)

    def test_the_legitimate_nesting_still_constructs(self):
        row = EffectiveContextMeasurement(source_horizon_tokens=100, materialized_tokens=10, attended_tokens=10)
        self.assertAlmostEqual(row.active_to_source_ratio, 0.1)

    def test_the_outer_relation_is_still_enforced(self):
        with self.assertRaisesRegex(ValueError, "source horizon"):
            EffectiveContextMeasurement(source_horizon_tokens=5, materialized_tokens=10)


if __name__ == "__main__":
    unittest.main()
