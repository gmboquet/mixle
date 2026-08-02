"""Freezing a payload must reach what its pointers point at (MXR-080-1850 / MXR-080-1851).

Both helpers already recursed through lists, dicts and sets. Both still leaked through a numpy
*object* array: ``copy()`` duplicates the pointers, and ``setflags(write=False)`` stops an element
being rebound but not the object behind it being mutated. That is the one container whose copy is
shallow in a way the others' are not, and it is the container each finding actually named.
"""

import unittest

import numpy as np

from mixle.doe.oracle import _deeply_frozen
from mixle.inference.scenario import _frozen_point_mass

FREEZERS = (("scenario point mass", _frozen_point_mass), ("late oracle receipt", _deeply_frozen))


def _object_array(value):
    holder = np.empty(1, dtype=object)
    holder[0] = value
    return holder


class ObjectArrayFreezeTest(unittest.TestCase):
    def test_a_list_inside_an_object_array_is_not_aliased(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                original = [1]
                frozen = freeze(_object_array(original))
                original.append(2)
                self.assertEqual(frozen[0], (1,))

    def test_a_nested_object_array_is_reached(self):
        # The auditor's shape: a dict holding a list, inside an array, inside an array.
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                payload = {"answer": [1]}
                frozen = freeze(_object_array(_object_array(payload)))
                payload["answer"].append(2)
                self.assertEqual(frozen[0][0]["answer"], (1,))

    def test_the_frozen_container_still_refuses_rebinding(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                frozen = freeze(_object_array([1]))
                self.assertFalse(frozen.flags.writeable)
                with self.assertRaises(ValueError):
                    frozen[0] = "replaced"

    def test_a_zero_dimensional_object_array_is_handled(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                holder = np.empty((), dtype=object)
                original = [1]
                holder[()] = original
                frozen = freeze(holder)
                original.append(2)
                self.assertEqual(frozen[()], (1,))


class NumericArrayTest(unittest.TestCase):
    """Numeric arrays are the common case and must not be disturbed by the object-array path."""

    def test_dtype_and_values_survive(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                frozen = freeze(np.array([1.5, 2.5]))
                self.assertEqual(frozen.dtype, np.float64)
                np.testing.assert_array_equal(frozen, [1.5, 2.5])
                self.assertFalse(frozen.flags.writeable)

    def test_a_copy_is_taken_so_later_writes_do_not_show_through(self):
        for label, freeze in FREEZERS:
            with self.subTest(freezer=label):
                source = np.array([1.0])
                frozen = freeze(source)
                source[0] = 99.0
                self.assertEqual(frozen[0], 1.0)


if __name__ == "__main__":
    unittest.main()
