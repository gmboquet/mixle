"""A snapshot must not execute caller code to obtain something still mutable (MXR-080-1872).

``mixle.doe.oracle`` and ``mixle.inference.scenario`` both convert a payload to immutable form, and
both used to deep-copy any value they could not convert, warning when the copy failed or came back as
the original. That fallback failed three ways at once: it ran the caller's ``__deepcopy__``, it kept
the alias whenever that hook returned ``self``, and even on success it left the receipt holding a
mutable object. A warning names a problem; it does not fix one.
"""

import unittest
from dataclasses import FrozenInstanceError

from mixle.doe.oracle import _deeply_frozen
from mixle.inference.scenario import _frozen_point_mass
from mixle.utils.immutable import OpaqueSnapshot, is_immutable_atom

FREEZERS = (("scenario point mass", _frozen_point_mass), ("late oracle receipt", _deeply_frozen))


class Sneaky:
    """A payload whose copy hook runs caller code and defeats the copy."""

    executed = 0

    def __init__(self, payload):
        self.payload = payload

    def __deepcopy__(self, memo):
        type(self).executed += 1
        return self


class Uncopyable:
    def __init__(self):
        self.payload = [1]

    def __deepcopy__(self, memo):
        raise RuntimeError("this object refuses to be copied")


class CustomObjectSnapshotTest(unittest.TestCase):
    def setUp(self):
        Sneaky.executed = 0

    def test_the_copy_hook_is_never_executed(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                Sneaky.executed = 0
                freeze(Sneaky([1]))
                self.assertEqual(Sneaky.executed, 0)

    def test_the_snapshot_is_not_the_caller_object(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                victim = Sneaky([1])
                self.assertIsNot(freeze(victim), victim)

    def test_the_snapshot_stops_tracking_the_caller(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                victim = Sneaky([1])
                snapshot = freeze(victim)
                victim.payload.append(2)
                self.assertEqual(snapshot.state["payload"], (1,))

    def test_the_snapshot_is_immutable(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                snapshot = freeze(Sneaky([1]))
                with self.assertRaises(TypeError):  # mappingproxy rejects assignment
                    snapshot.state["payload"] = "replaced"
                with self.assertRaises(FrozenInstanceError):
                    snapshot.type_name = "replaced"

    def test_an_object_that_refuses_to_be_copied_is_still_recorded(self):
        # Previously this warned and returned the caller's object by reference.
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                snapshot = freeze(Uncopyable())
                self.assertIsInstance(snapshot, OpaqueSnapshot)
                self.assertEqual(snapshot.type_name, "Uncopyable")

    def test_a_reference_cycle_terminates(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                victim = Sneaky(None)
                victim.payload = victim
                snapshot = freeze(victim)
                self.assertIsNone(snapshot.state["payload"].state)


class ImmutableAtomTest(unittest.TestCase):
    """Scalars must pass through: the container branches above the fallback do not name them, and
    ``copy.deepcopy`` used to return them unchanged."""

    def test_scalars_survive_the_fallback_unchanged(self):
        for label, freeze in FREEZERS:
            for value in (1, 2.5, True, None, "s", b"b", complex(1, 2)):
                with self.subTest(freezer=label, value=repr(value)):
                    self.assertEqual(freeze(value), value)

    def test_numpy_scalars_are_atoms(self):
        import numpy as np

        self.assertTrue(is_immutable_atom(np.int64(3)))
        self.assertTrue(is_immutable_atom(np.float32(1.5)))

    def test_a_mutable_container_is_not_an_atom(self):
        for value in ([1], {"a": 1}, {1, 2}, bytearray(b"x")):
            with self.subTest(value=repr(value)):
                self.assertFalse(is_immutable_atom(value))


if __name__ == "__main__":
    unittest.main()
