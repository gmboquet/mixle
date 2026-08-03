"""Registry (mixle.system.registry): a dir-backed catalog of registered task models, queried by capability/fingerprint.

Card REG-a (workstream J2): register writes a real task-artifact directory + a JSON index entry; find_for and
tier_stack read the index back, including in a fresh Registry instance pointed at the same dir.

Also regression-tests two review findings fixed in this file:

1. Path traversal: an unvalidated ``entry_id`` (``"../escaped"``, an absolute path, a value containing
   separators, ...) joined onto the registry ``dir`` could write a task artifact OUTSIDE the registry
   root. Fixed with ``_safe_entry_id`` (single-path-component validation) plus a resolved-path
   containment check.
2. Concurrent overwrite: two ``Registry`` instances (or two threads/processes sharing one) opened on the
   same ``dir`` each cache ``self._entries`` once at construction and never otherwise refresh it, so one
   instance's ``register`` could silently overwrite another's already-persisted index row -- even with
   zero entry_id collisions. Fixed with an ``fcntl.flock``-serialized read-modify-write (re-reading the
   on-disk index fresh under the lock) plus an independent claim-file conflict guard.
"""

import dataclasses
import json
import multiprocessing as mp
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle.stats.univariate.discrete.categorical import CategoricalDistribution  # noqa: E402
from mixle.system import Registry  # noqa: E402
from mixle.task.calibrate import ESCALATE  # noqa: E402
from mixle.task.distill import distill_for_routing  # noqa: E402
from mixle.task.model import StructuredClassifierIO, TaskModel  # noqa: E402
from mixle.task.router import Router  # noqa: E402


def _spam_teacher(texts):
    words = {"free", "winner", "prize"}
    return ["spam" if any(w in t.split() for w in words) else "ham" for t in texts]


def _billing_teacher(texts):
    words = {"invoice", "overdue", "payment"}
    return ["billing" if any(w in t.split() for w in words) else "other" for t in texts]


def _toy_model(teacher, vocab, seed):
    texts = [f"{a} {b} filler {c}" for a in vocab for b in vocab for c in ["x", "y", "z"]]
    return distill_for_routing(teacher, texts, dim=64, hidden=[16], epochs=40, seed=seed, calibration_frac=0.3)


def _json_task_model():
    """A minimal (torch-free payload) TaskModel -- cheap to construct, for tests exercising registry
    id/index mechanics rather than model quality (mirrors core_review_fixes_test.py's C-6/C-7 helper)."""
    return TaskModel(
        model=CategoricalDistribution({"x": 0.5, "y": 0.5}),
        adapter=StructuredClassifierIO(field_keys=None, label_index=0, labels=["x", "y"]),
        payload="json",
    )


class _DelayedWriteRegistry(Registry):
    """``Registry`` whose ``_write_index`` pauses briefly before persisting.

    Widens the window between "re-read the index under the lock" and "write it back" inside
    ``register()``, so the concurrency tests below deterministically exercise overlapping critical
    sections (real threads actually interleaved while each holds/awaits the lock) instead of depending
    on scheduler timing luck to hit it in a single trial -- mirrors
    ``registry_versioning_test.py``'s ``_DelayedVersionsRegistry``. Module-level (not a local class) so a
    spawned worker process inherits it cleanly too.
    """

    def _write_index(self) -> None:
        time.sleep(0.05)
        super()._write_index()


def _mp_register_worker(root: str, i: int, barrier, queue) -> None:
    """Module-level (picklable) worker process target for the cross-process race test."""
    reg = _DelayedWriteRegistry(root)
    barrier.wait()
    entry = reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id=f"writer_{i}")
    queue.put(entry.entry_id)


class RegistryTest(unittest.TestCase):
    def test_find_for_matches_capability_not_the_other(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            spam_model = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=0)
            billing_model = _toy_model(_billing_teacher, ["invoice", "overdue", "payment", "meeting", "lunch"], seed=1)
            reg.register(spam_model, capabilities=["spam_filter"], fingerprint=[0.0, 0.0, 0.0, 0.0, 0.0], cost=0.01)
            reg.register(
                billing_model, capabilities=["billing_router"], fingerprint=[9.0, 9.0, 9.0, 9.0, 9.0], cost=0.02
            )

            spam_matches = reg.find_for("spam_filter")
            self.assertEqual(len(spam_matches), 1)
            self.assertEqual(spam_matches[0].capabilities, ["spam_filter"])

            billing_matches = reg.find_for("billing_router")
            self.assertEqual(len(billing_matches), 1)
            self.assertEqual(billing_matches[0].capabilities, ["billing_router"])

            near_spam = reg.find_for([0.1, 0.0, 0.0, 0.0, 0.0])
            self.assertEqual(len(near_spam), 1)
            self.assertEqual(near_spam[0].capabilities, ["spam_filter"])

    def test_tier_stack_cheapest_first_frontier_last(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            cheap = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=2)
            pricier = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=3)
            reg.register(pricier, capabilities=["spam_filter"], cost=0.05)
            reg.register(cheap, capabilities=["spam_filter"], cost=0.01)

            def frontier(texts):
                # Router calls the frontier as a BATCHED callable (texts -> [label]) and does its own
                # single-item wrap/unwrap (see Router.__call__) -- matching Cascade._teacher_label's
                # convention of "teacher is the raw batched function". A frontier that already unwraps
                # to one text in/out double-wraps and breaks (a list-of-one-string gets .split()'d).
                return _spam_teacher(texts)

            stack = reg.tier_stack("spam_filter", frontier=frontier, costs=[0.01, 0.05, 1.0])

            self.assertEqual(len(stack), 3)
            self.assertEqual([c for _, _, c in stack], [0.01, 0.05, 1.0])
            self.assertEqual(stack[-1][0], "frontier")
            self.assertIs(stack[-1][1], frontier)
            for name, model, _cost in stack[:-1]:
                self.assertTrue(hasattr(model, "decide"))

            # the exact shape Router's constructor wants: cheapest calibrated tiers, frontier fallback last
            router = Router(tiers=stack)
            decisions = [router(t) for t in ["free prize now", "team meeting today"]]
            for d_ in decisions:
                self.assertTrue(d_ is ESCALATE or isinstance(d_, str))

    def test_round_trips_through_a_fresh_registry_instance(self):
        with tempfile.TemporaryDirectory() as d:
            first = Registry(d)
            model = _toy_model(_spam_teacher, ["free", "winner", "prize", "meeting", "lunch"], seed=4)
            entry = first.register(model, capabilities=["spam_filter"], fingerprint=[1.0, 2.0, 3.0], cost=0.02)

            reopened = Registry(d)
            self.assertEqual(len(reopened.find_for("spam_filter")), 1)
            reloaded = reopened.load(entry.entry_id)
            self.assertEqual(reloaded.decide("free prize now"), model.decide("free prize now"))

    # ----------------------------------------------------------------------------------------------
    # Finding (a): entry_id path traversal
    # ----------------------------------------------------------------------------------------------

    def test_register_rejects_entry_ids_that_escape_the_registry_root(self):
        """entry_id='../escaped' (or any other traversal / absolute / separator-bearing id) must be
        rejected before any filesystem write, not silently write a task artifact outside the registry
        root directory."""
        unsafe_ids = [
            "../escaped",
            "../../etc/cron.d/evil",
            "..",
            ".",
            "a/../../b",
            "sub/dir",
            "/etc/passwd",
            "\x00nullbyte",
            "",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            registry_root = os.path.join(tmp, "registry")
            reg = Registry(registry_root)
            before = set(os.listdir(tmp))
            for unsafe_id in unsafe_ids:
                with self.assertRaises(ValueError, msg=f"entry_id={unsafe_id!r} should have been rejected"):
                    reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id=unsafe_id)
            # nothing was written anywhere in tmp beyond the registry directory itself, and nothing
            # landed in the index either
            self.assertEqual(before, set(os.listdir(tmp)))
            self.assertEqual(reg.find_for("cap"), [])

    def test_register_rejects_a_dangling_symlink_that_resolves_outside_the_root(self):
        """A single-component entry_id (passes the format check) can still escape via a pre-placed
        symlink inside the registry directory -- e.g. from a restored or tampered registry. The
        resolved-path containment check must catch that independently of the format check: a dangling
        symlink target isn't caught by the existing os.path.exists()-based duplicate check (which
        follows symlinks and reports False for one that dangles), so this exercises the containment
        check on its own."""
        with tempfile.TemporaryDirectory() as tmp:
            registry_root = os.path.join(tmp, "registry")
            os.makedirs(registry_root, exist_ok=True)
            dangling_target = os.path.join(tmp, "sibling", "nonexistent")
            symlink_path = os.path.join(registry_root, "evil_link")
            os.symlink(dangling_target, symlink_path)
            self.assertFalse(os.path.exists(symlink_path))  # dangling: exists() follows and reports False

            reg = Registry(registry_root)
            with self.assertRaises(ValueError, msg="a dangling symlink escape should have been rejected"):
                reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="evil_link")
            self.assertFalse(os.path.exists(os.path.join(tmp, "sibling")))

    def test_register_accepts_a_safe_entry_id_after_rejecting_unsafe_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_root = os.path.join(tmp, "registry")
            reg = Registry(registry_root)
            with self.assertRaises(ValueError):
                reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="../escaped")

            entry = reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="safe_id")
            self.assertEqual(entry.entry_id, "safe_id")
            real_root = os.path.realpath(registry_root)
            real_entry = os.path.realpath(entry.path)
            self.assertTrue(real_entry == real_root or real_entry.startswith(real_root + os.sep))
            self.assertEqual([e.entry_id for e in reg.find_for("cap")], ["safe_id"])

    # ----------------------------------------------------------------------------------------------
    # Finding (b): concurrent Registry instances silently overwriting each other's index entries
    # ----------------------------------------------------------------------------------------------

    def test_concurrent_register_with_distinct_ids_does_not_drop_index_entries(self):
        """Isolates the pure index-level race described by finding (b): every writer uses an explicit,
        pairwise-distinct entry_id (no artifact-path or id collision is even possible), yet -- before
        the fix -- index rows still went missing, because register() never re-read the on-disk index
        before its own full-overwrite _write_index() call. Each of N SEPARATE Registry instances (not
        threads sharing one instance) races register() against the same dir, matching finding (b)'s
        "two ordinary Registry instances" framing."""
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "root")
            n_writers = 6
            # pre-construct every instance BEFORE any register() call, so all instances share the same
            # stale initial view of the (empty) index -- the exact scenario finding (b) describes.
            instances = [_DelayedWriteRegistry(root) for _ in range(n_writers)]
            barrier = threading.Barrier(n_writers)

            def writer(i):
                barrier.wait()  # line every writer up so they all enter register() together
                return (
                    instances[i]
                    .register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id=f"writer_{i}")
                    .entry_id
                )

            with ThreadPoolExecutor(max_workers=n_writers) as pool:
                results = [f.result(timeout=10) for f in [pool.submit(writer, i) for i in range(n_writers)]]

            self.assertEqual(sorted(results), [f"writer_{i}" for i in range(n_writers)])

            reopened = Registry(root)
            indexed_ids = {e.entry_id for e in reopened.find_for("cap")}
            self.assertEqual(
                indexed_ids,
                {f"writer_{i}" for i in range(n_writers)},
                "one or more concurrently-registered index rows were silently dropped",
            )
            for i in range(n_writers):
                reopened.load(f"writer_{i}")  # every artifact is present AND indexed, not orphaned

    def test_concurrent_register_with_auto_ids_does_not_collide(self):
        """Same race, but with auto-generated ids (the common, no-entry_id-argument call pattern):
        confirms id allocation itself is also race-free under the lock, in addition to the index write."""
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "root")
            n_writers = 6
            instances = [_DelayedWriteRegistry(root) for _ in range(n_writers)]
            barrier = threading.Barrier(n_writers)

            def writer(i):
                barrier.wait()
                return instances[i].register(_json_task_model(), capabilities=["cap"], cost=0.01).entry_id

            with ThreadPoolExecutor(max_workers=n_writers) as pool:
                results = [f.result(timeout=10) for f in [pool.submit(writer, i) for i in range(n_writers)]]

            self.assertEqual(len(set(results)), n_writers, f"expected {n_writers} distinct auto ids, got {results}")

            reopened = Registry(root)
            indexed_ids = {e.entry_id for e in reopened.find_for("cap")}
            self.assertEqual(indexed_ids, set(results))
            for entry_id in results:
                reopened.load(entry_id)

    def test_register_raises_a_conflict_error_when_the_claim_marker_is_already_staked(self):
        """Belt-and-suspenders guard, tested independently of the lock: pre-stake a claim marker for an
        entry_id that is in neither the index nor on disk as an artifact yet (as if another writer's
        register() were mid-flight -- past the claim step but not yet done saving), and confirm
        register() raises a clear, typed conflict error instead of proceeding to share or overwrite that
        writer's in-progress artifact."""
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "root")
            reg = Registry(root)
            os.makedirs(root, exist_ok=True)
            claim_path = os.path.join(root, ".e1.claim")
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)

            with self.assertRaises(RuntimeError):
                reg.register(_json_task_model(), capabilities=["cap"], cost=0.01, entry_id="e1")

            self.assertFalse(os.path.exists(os.path.join(root, "e1")))
            self.assertEqual(reg.find_for("cap"), [])

    @pytest.mark.slow
    def test_concurrent_register_across_processes_does_not_drop_index_entries(self):
        """Same race as the threading tests above, but with real separate OS processes -- the realistic
        "two ordinary Registry instances" scenario for a filesystem-backed local registry (e.g. two
        independent capture/accumulation workflows registering into the same dir at once).
        ``fcntl.flock`` is a cross-process lock; a fix that only serialized threads (e.g. a bare
        ``threading.Lock``) would pass the threading tests above but not this one.

        Uses the ``spawn`` start method (the platform default, and the only one guaranteed safe with
        threads already running in this process). Each spawned child pays the full cost of importing
        mixle's scientific stack cold in a fresh interpreter (tens of seconds), which is why this one
        test is marked ``slow`` rather than living in the default fast gate (mirrors
        ``registry_versioning_test.py``'s analogous cross-process test for the sibling registry).
        """
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "root")
            os.makedirs(root, exist_ok=True)
            n_writers = 4
            barrier = ctx.Barrier(n_writers)
            queue = ctx.Queue()
            procs = [ctx.Process(target=_mp_register_worker, args=(root, i, barrier, queue)) for i in range(n_writers)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=180)
                self.assertEqual(p.exitcode, 0, f"writer process pid={p.pid} failed or hung (exitcode={p.exitcode})")

            results = [queue.get(timeout=5) for _ in range(n_writers)]
            self.assertEqual(sorted(results), [f"writer_{i}" for i in range(n_writers)])

            reg = Registry(root)
            indexed_ids = {e.entry_id for e in reg.find_for("cap")}
            self.assertEqual(indexed_ids, {f"writer_{i}" for i in range(n_writers)})
            for i in range(n_writers):
                reg.load(f"writer_{i}")


def _write_index(root, records):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(records, f)


def _record(entry_id, root, **kw):
    base = {
        "entry_id": entry_id,
        "path": os.path.join(root, entry_id),
        "kind": "task",
        "capabilities": ["cap"],
        "fingerprint": None,
        "profile": {},
        "cost": 0.0,
    }
    base.update(kw)
    return base


class UntrustedIndexTest(unittest.TestCase):
    """MXR-080-1695: path containment and kind validation applied only while registering. Index
    deserialization trusted both fields, so a record naming an absolute path outside the registry and
    ``kind="invented"`` loaded straight through load()'s fallthrough TaskModel branch."""

    def test_an_unknown_kind_is_rejected_rather_than_treated_as_a_task_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            _write_index(root, [_record("e0", root, kind="invented")])
            with self.assertRaises(ValueError):
                Registry(root)

    def test_a_serialized_path_cannot_redirect_a_load_outside_the_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            outside = os.path.join(tmp, "outside", "artifact")
            _write_index(root, [_record("e0", root, path=outside)])
            reg = Registry(root)
            self.assertEqual(reg._get("e0").path, os.path.join(root, "e0"))

    def test_an_unsafe_entry_id_in_the_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            _write_index(root, [_record("../escaped", root)])
            with self.assertRaises(ValueError):
                Registry(root)


class ImmutableRecordsTest(unittest.TestCase):
    """MXR-080-1696: find_for() returned the very objects held in Registry._entries, so mutating a
    lookup result rewrote later routing and loads with no mutation API, validation or persistence."""

    def test_a_lookup_result_cannot_rebind_the_registrys_artifact_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            _write_index(root, [_record("a", root)])
            reg = Registry(root)
            found = reg.find_for("cap")[0]
            with self.assertRaises(dataclasses.FrozenInstanceError):
                found.path = "/tmp/redirected"
            self.assertEqual(reg._get("a").path, os.path.join(root, "a"))

    def test_clearing_a_lookup_results_capabilities_does_not_unregister_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            _write_index(root, [_record("a", root)])
            reg = Registry(root)
            reg.find_for("cap")[0].capabilities.clear()
            self.assertEqual([e.entry_id for e in reg.find_for("cap")], ["a"])

    def test_nested_profile_data_is_defensively_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            _write_index(root, [_record("a", root, profile={"capture": {"n": 1}})])
            reg = Registry(root)
            reg.find_for("cap")[0].profile["capture"]["n"] = 99
            self.assertEqual(reg._get("a").profile, {"capture": {"n": 1}})


class FingerprintSelectionTest(unittest.TestCase):
    """MXR-080-1697: registration and deserialization accepted arbitrary fingerprint values and
    dimensions -- a NaN entry could be selected as "nearest" over an exact match, and one length-3
    record made every two-dimensional query raise a broadcasting ValueError."""

    def test_a_non_finite_fingerprint_is_rejected_at_the_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            nan = float("nan")
            _write_index(root, [_record("bad", root, fingerprint=[nan, nan])])
            with self.assertRaises(ValueError):
                Registry(root)

    def test_an_incompatible_dimension_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            _write_index(
                root,
                [
                    _record("wrong_dim", root, fingerprint=[1.0, 2.0, 3.0]),
                    _record("healthy", root, fingerprint=[0.0, 0.0]),
                ],
            )
            reg = Registry(root)
            self.assertEqual([e.entry_id for e in reg.find_for([0.0, 0.0])], ["healthy"])

    def test_a_non_finite_query_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "reg")
            _write_index(root, [_record("healthy", root, fingerprint=[0.0, 0.0])])
            reg = Registry(root)
            with self.assertRaises(ValueError):
                reg.find_for([float("nan"), 0.0])


class RegistryControlFileTest(unittest.TestCase):
    def test_the_lock_file_is_never_followed_through_a_symlink(self):
        # MXR-080-1683: register() opened the fixed .registry.lock path with mode "w" before applying
        # flock, following any existing symlink and truncating its target. A lock symlink aimed at an
        # external file was followed during an otherwise successful registration and emptied it;
        # entry-path containment does not protect this control file.
        with tempfile.TemporaryDirectory() as tmp:
            outside = os.path.join(tmp, "DO_NOT_TOUCH.txt")
            with open(outside, "w") as f:
                f.write("DO NOT TOUCH")
            reg_dir = os.path.join(tmp, "registry")
            os.makedirs(reg_dir)
            os.symlink(outside, os.path.join(reg_dir, ".registry.lock"))

            reg = Registry(reg_dir)
            with self.assertRaises(RuntimeError):
                reg.register(_json_task_model(), capabilities=["c"])

            with open(outside) as f:
                self.assertEqual(f.read(), "DO NOT TOUCH")  # untouched, not truncated

    def test_a_failed_index_write_leaves_nothing_claimed_behind(self):
        # MXR-080-1684: the artifact was published and its exclusive claim marker retained before the
        # index was written, and only artifact-save failures dropped the claim. Forcing _write_index()
        # to fail left both the artifact and .<id>.claim on disk with no index.json: the registration
        # raised, retrying that id was rejected as already existing, and no read could discover it.
        with tempfile.TemporaryDirectory() as tmp:
            reg = Registry(tmp)
            original_write_index = reg._write_index
            state = {"fail": True}

            def failing_write_index():
                if state["fail"]:
                    raise OSError("disk full")
                return original_write_index()

            reg._write_index = failing_write_index
            with self.assertRaises(OSError):
                reg.register(_json_task_model(), capabilities=["c"], entry_id="e")

            self.assertFalse(os.path.exists(os.path.join(tmp, "e")))
            self.assertFalse(os.path.exists(os.path.join(tmp, ".e.claim")))
            self.assertEqual(reg._entries, [])

            # the rolled-back id is genuinely free again: a retry succeeds and is discoverable
            state["fail"] = False
            entry = reg.register(_json_task_model(), capabilities=["c"], entry_id="e")
            self.assertEqual(entry.entry_id, "e")
            self.assertEqual([e.entry_id for e in Registry(tmp)._entries], ["e"])


class RegisterValidatesBeforeItPersistsTest(unittest.TestCase):
    """MXR-080-1902 (High): ``RegistryEntry.__post_init__`` is the real validator for
    ``capabilities``/``profile``/``cost``, and ``register`` used to construct the row only AFTER it
    had claimed the id and written the artifact. A rejected field therefore raised with ``<id>/`` and
    ``.<id>.claim`` left on disk and no ``index.json`` naming them -- the id was permanently
    poisoned: no read could discover the entry, and every retry of that id (in this process or a
    fresh one) was rejected as "registry already has an entry". The ``fingerprint`` pre-check already
    in place proves the intent; this generalizes it to the rest of the record."""

    def _assert_nothing_persisted(self, root, entry_id):
        self.assertFalse(os.path.exists(os.path.join(root, entry_id)), "an artifact survived a rejected register")
        self.assertFalse(
            os.path.exists(os.path.join(root, f".{entry_id}.claim")), "a claim marker survived a rejected register"
        )
        self.assertFalse(os.path.exists(os.path.join(root, "index.json")))

    def test_a_rejected_entry_field_leaves_no_artifact_claim_or_index_behind(self):
        cases = {
            "non-string capability": (dict(capabilities=["ok", 123]), "capabilities must be strings"),
            "non-finite cost": (dict(capabilities=["ok"], cost=float("nan")), "cost must be finite"),
            "non-finite fingerprint": (
                dict(capabilities=["ok"], fingerprint=[1.0, float("nan")]),
                "non-finite fingerprint",
            ),
        }
        for label, (kwargs, message) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                reg = Registry(tmp)
                with self.assertRaisesRegex(ValueError, message):
                    reg.register(_json_task_model(), entry_id="e0", **kwargs)
                self._assert_nothing_persisted(tmp, "e0")
                self.assertEqual(reg._entries, [])
                # the id is genuinely free again: the retry the caller would naturally make succeeds
                entry = reg.register(_json_task_model(), capabilities=["ok"], entry_id="e0")
                self.assertEqual(entry.entry_id, "e0")
                self.assertEqual([e.entry_id for e in Registry(tmp)._entries], ["e0"])

    def test_a_rejected_auto_id_register_does_not_consume_the_id(self):
        # The auto-id path leaked the same way, just less visibly: the orphaned artifact made the
        # next auto-generated id scan past it, so a rejected call silently burned entry_0000.
        with tempfile.TemporaryDirectory() as tmp:
            reg = Registry(tmp)
            with self.assertRaises(ValueError):
                reg.register(_json_task_model(), capabilities=["ok", None])
            self._assert_nothing_persisted(tmp, "entry_0000")
            self.assertEqual(reg.register(_json_task_model(), capabilities=["ok"]).entry_id, "entry_0000")

    def test_a_valid_register_is_unaffected(self):
        # Negative control: reordering validation ahead of the write must not change the happy path.
        with tempfile.TemporaryDirectory() as tmp:
            reg = Registry(tmp)
            entry = reg.register(
                _json_task_model(),
                capabilities=["cap"],
                fingerprint=[1.0, 2.0],
                profile={"k": "v"},
                cost=0.5,
                entry_id="good",
            )
            self.assertEqual(entry.entry_id, "good")
            self.assertEqual(entry.fingerprint, [1.0, 2.0])
            self.assertTrue(os.path.exists(os.path.join(tmp, "good")))
            self.assertEqual([e.entry_id for e in Registry(tmp)._entries], ["good"])


if __name__ == "__main__":
    unittest.main()
