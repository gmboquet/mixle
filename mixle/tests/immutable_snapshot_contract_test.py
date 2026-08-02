"""A snapshot must not execute caller code to obtain something still mutable (MXR-080-1872).

``mixle.doe.oracle`` and ``mixle.inference.scenario`` both convert a payload to immutable form, and
both used to deep-copy any value they could not convert, warning when the copy failed or came back as
the original. That fallback failed three ways at once: it ran the caller's ``__deepcopy__``, it kept
the alias whenever that hook returned ``self``, and even on success it left the receipt holding a
mutable object. A warning names a problem; it does not fix one.
"""

import datetime
import enum
import numbers
import unittest
import uuid
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction

import numpy as np

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
        self.assertTrue(is_immutable_atom(np.int64(3)))
        self.assertTrue(is_immutable_atom(np.float32(1.5)))

    def test_a_mutable_container_is_not_an_atom(self):
        for value in ([1], {"a": 1}, {1, 2}, bytearray(b"x")):
            with self.subTest(value=repr(value)):
                self.assertFalse(is_immutable_atom(value))


class MutableNumberSubclass(numbers.Number):
    """A mutable object wearing an immutable value's abstract base class."""

    def __init__(self):
        self.payload = [1]


class Colour(enum.Enum):
    RED = "red"


class Tier(enum.IntEnum):
    LOW = 1


class LoadedEnum(enum.Enum):
    """An enum whose members carry caller state of their own."""

    ONE = 1

    def __init__(self, _value):
        self.notes = []


class ClosedAtomSetTest(unittest.TestCase):
    """``is_immutable_atom`` must fail safe in BOTH directions (MXR-080-1880).

    The first version asked ``isinstance(value, (Number, Enum, type, datetime.date, ...))``, which is
    an open question wearing a closed question's clothes, and it was wrong on each side: ``Number`` is
    an abstract base any class may subclass or register against, so a mutable custom object answered
    "immutable" and stayed aliased; while ``UUID`` -- an ordinary immutable domain value -- matched
    nothing and was destroyed into an ``OpaqueSnapshot``, changing what a fully observed rollout draw
    contains.
    """

    def test_a_mutable_number_subclass_is_not_an_atom(self):
        self.assertFalse(is_immutable_atom(MutableNumberSubclass()))

    def test_a_mutable_number_subclass_is_snapshotted_not_aliased(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                victim = MutableNumberSubclass()
                snapshot = freeze(victim)
                victim.payload.append(2)
                self.assertIsInstance(snapshot, OpaqueSnapshot)
                self.assertIsNot(snapshot, victim)
                self.assertEqual(snapshot.state["payload"], (1,))

    def test_canonical_immutable_values_pass_through_unchanged(self):
        values = (
            uuid.UUID(int=1),
            Decimal("1.5"),
            Fraction(1, 3),
            np.int64(3),
            np.float32(1.5),
            datetime.date(2026, 1, 1),
            datetime.datetime(2026, 1, 1),
            datetime.timedelta(days=1),
            frozenset({1}),
            complex(1, 2),
            5,
            2.5,
            "x",
            b"y",
            True,
            None,
        )
        for label, freeze in FREEZERS:
            for value in values:
                with self.subTest(freezer=label, value=repr(value)):
                    self.assertTrue(is_immutable_atom(value))
                    self.assertEqual(freeze(value), value)

    def test_a_uuid_survives_a_rollout_draw_as_itself(self):
        # It became an OpaqueSnapshot, which changes what a fully observed record contains.
        identifier = uuid.UUID(int=7)
        self.assertEqual(_frozen_point_mass(identifier), identifier)

    def test_an_ordinary_enum_member_is_an_atom(self):
        for member in (Colour.RED, Tier.LOW):
            with self.subTest(member=repr(member)):
                self.assertTrue(is_immutable_atom(member))

    def test_an_enum_whose_members_carry_mutable_state_is_not_an_atom(self):
        # Enum membership alone is not enough: a subclass may hang a list off each member.
        self.assertFalse(is_immutable_atom(LoadedEnum.ONE))


if __name__ == "__main__":
    unittest.main()
