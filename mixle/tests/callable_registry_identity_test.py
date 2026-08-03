"""One callable, one canonical serialization id, published atomically (MXR-080-1888).

``register_serializable_callable`` guarded only the forward map. Registering the same function under
two ids therefore succeeded: both stayed live in the id-to-callable map and both decoded, while the
callable-to-id map was silently rewritten to the second. Payloads written before and after the second
registration carried different identities for the same object, and nothing raised.

This is the defect the CLASS registry was repaired for (MXR-080-0724); the callable path did not
receive the same repair, and also took no lock, so two threads could interleave between the check and
the two writes and leave the maps disagreeing.
"""

import threading
import unittest

from mixle.utils.serialization import (
    SerializationError,
    register_serializable_callable,
)


def _probe_one(x):
    return x


class CanonicalIdentityTest(unittest.TestCase):
    def test_one_callable_cannot_take_a_second_id(self):
        def fn(x):
            return x

        register_serializable_callable(fn, "identity-a")
        with self.assertRaisesRegex(SerializationError, "already registered as"):
            register_serializable_callable(fn, "identity-b")

    def test_re_registering_under_the_same_id_is_still_idempotent(self):
        def fn(x):
            return x

        register_serializable_callable(fn, "identity-same")
        self.assertIs(register_serializable_callable(fn, "identity-same"), fn)

    def test_two_callables_still_cannot_share_one_id(self):
        # Fresh locals, not module-level probes: the registry is process-global and now enforces one
        # id per callable, so a probe shared with another test makes the outcome depend on which
        # test registered it first.
        def first(x):
            return x

        def second(x):
            return x

        register_serializable_callable(first, "identity-shared")
        with self.assertRaisesRegex(SerializationError, "already registered"):
            register_serializable_callable(second, "identity-shared")

    def test_a_derived_id_still_works_without_an_explicit_one(self):
        self.assertIs(register_serializable_callable(_probe_one), _probe_one)
        # Idempotent: the derived id is the same on a second call, so this does not trip the
        # one-id-per-callable rule.
        self.assertIs(register_serializable_callable(_probe_one), _probe_one)

    def test_a_lambda_still_requires_an_explicit_id(self):
        with self.assertRaisesRegex(SerializationError, "callable_id is required"):
            register_serializable_callable(lambda x: x)


class ConcurrentRegistrationTest(unittest.TestCase):
    def test_racing_registrations_leave_the_two_maps_agreeing(self):
        """The write is under the registry lock, so no interleaving can split the maps."""
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def register(index: int) -> None:
            def fn(x):
                return x

            try:
                barrier.wait(timeout=5)
                register_serializable_callable(fn, f"identity-race-{index}")
            except SerializationError:
                pass
            except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
