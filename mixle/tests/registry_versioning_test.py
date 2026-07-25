"""Regression tests for four fixed defects in mixle.inference.production.registry.Registry.

1. ``register`` numbered the next version from ``len(versions())``, not the highest existing version
   number -- so a registry missing a version (deleted out of band, or a future ``delete()``) reused
   that number and the next ``register`` call silently overwrote a DIFFERENT, later version.
2. ``promote`` wrote the alias file directly (``open(..., "w")``), not atomically -- a crash or a
   concurrent reader mid-write could observe a truncated/partial alias file instead of the old or
   new value.
3. ``register``'s version-number allocation was a plain read-modify-write: two concurrent writers
   (threads or processes) could both read the same existing versions, both compute the same "next"
   number, and both write it -- one write silently clobbered the other, with no error to either
   caller. Fixed with an ``fcntl.flock`` serializing allocation+write per model name, plus an
   ``O_CREAT | O_EXCL`` write as an independent conflict-detection guard.
4. ``register``'s version-file write, even after (3)'s fix, still wrote the payload directly into the
   ``O_CREAT | O_EXCL``-opened file at the final ``<version>.json`` path: a SINGLE writer failing
   partway through -- serialization raising mid ``json.dump``, the disk filling up, the process
   crashing -- left that exact path holding truncated/invalid JSON, permanently: a later ``register``
   sees the (corrupt) file already exists and silently moves on to the next version number, orphaning
   the corrupt one, and the ``O_CREAT | O_EXCL`` guard then refuses forever to let anything overwrite
   it. Fixed by writing to a private, fsynced temp file first and publishing it with ``os.link``
   (which -- like the ``O_CREAT | O_EXCL`` it replaces -- atomically creates the destination only if
   absent, unlike ``os.replace``), so a failed write leaves nothing at the final path at all.
"""

import json
import multiprocessing as mp
import os
import stat
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import mixle.inference.production.registry as registry_module
import mixle.stats as st
from mixle.inference.production.registry import Registry


class _DelayedVersionsRegistry(Registry):
    """``Registry`` whose ``versions()`` pauses briefly before returning.

    Widens the window between "read existing versions" and "write the new version file" inside
    ``register()``, so the concurrency tests below deterministically exercise the race (real
    threads / real processes actually interleaved inside the critical section) instead of
    depending on scheduler timing luck to hit it in a single trial. Module-level (not a local
    class) so a forked worker process inherits it cleanly.
    """

    def versions(self, name: str) -> list[str]:
        out = super().versions(name)
        time.sleep(0.05)
        return out


def _mp_register_worker(root: str, i: int, barrier, queue) -> None:
    """Module-level (picklable) worker process target for the cross-process race test."""
    reg = _DelayedVersionsRegistry(root)
    model = st.GaussianDistribution(float(i), 1.0)
    barrier.wait()
    ver = reg.register(model, "m")
    queue.put((i, ver, float(i)))


def test_register_after_a_missing_version_does_not_reuse_its_number():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        v1 = reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        v2 = reg.register(st.GaussianDistribution(1.0, 1.0), "m")
        v3 = reg.register(st.GaussianDistribution(2.0, 1.0), "m")
        assert (v1, v2, v3) == ("v1", "v2", "v3")

        # simulate v2 having been removed (deleted out of band; the class has no delete() of its own)
        os.remove(os.path.join(reg._dir("m"), "v2.json"))
        assert reg.versions("m") == ["v1", "v3"]

        v4 = reg.register(st.GaussianDistribution(3.0, 1.0), "m")
        assert v4 == "v4", "the next version must come from the highest existing number, not the count"

        # v3 must still be exactly what it was -- never overwritten
        model_v3, _ = reg.get("m", "v3")
        assert model_v3.mu == 2.0
        model_v4, _ = reg.get("m", "v4")
        assert model_v4.mu == 3.0


def test_promote_writes_the_alias_atomically():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        reg.register(st.GaussianDistribution(1.0, 1.0), "m")
        reg.promote("m", "v2")

        alias_path = os.path.join(reg._dir("m"), "production.alias")
        assert os.path.exists(alias_path)
        with open(alias_path) as alias_file:
            assert alias_file.read() == "v2"
        # no leftover temp file from the atomic-write dance
        assert [f for f in os.listdir(reg._dir("m")) if f.endswith(".tmp")] == []

        model, _ = reg.current("m")
        assert model.mu == 1.0


def test_promote_fsyncs_both_the_alias_file_and_registry_directory(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        calls: list[str] = []
        real_fsync = os.fsync

        def recording_fsync(fd):
            calls.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
            return real_fsync(fd)

        monkeypatch.setattr(registry_module.os, "fsync", recording_fsync)
        reg.promote("m", "v1")

        assert "file" in calls
        assert "directory" in calls


def test_concurrent_promotions_use_private_temp_files_and_never_collide(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        reg.register(st.GaussianDistribution(1.0, 1.0), "m")
        barrier = threading.Barrier(2)
        real_replace = os.replace

        def synchronized_replace(source, target):
            barrier.wait(timeout=5)
            return real_replace(source, target)

        monkeypatch.setattr(registry_module.os, "replace", synchronized_replace)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(reg.promote, "m", version) for version in ("v1", "v2")]
            for future in futures:
                future.result(timeout=5)

        alias_path = os.path.join(reg._dir("m"), "production.alias")
        with open(alias_path) as f:
            assert f.read() in {"v1", "v2"}
        assert [name for name in os.listdir(reg._dir("m")) if name.endswith(".tmp")] == []


def test_promote_rejects_an_unknown_version_without_touching_the_alias():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        reg.promote("m", "v1")
        with pytest.raises(KeyError):
            reg.promote("m", "v99")
        # the failed promote must not have clobbered the existing alias
        alias_path = os.path.join(reg._dir("m"), "production.alias")
        with open(alias_path) as alias_file:
            assert alias_file.read() == "v1"


def test_concurrent_register_with_threads_does_not_silently_clobber_a_version():
    """Real OS threads race ``register()`` for the SAME model name at (as close to) the same
    instant. Before the fix this reliably lost writes: two threads would both read the registry
    as empty, both compute ``next_n == 1``, and both write ``v1.json`` via plain ``open(..., "w")``
    -- one silently clobbered the other and neither caller saw an error."""
    with tempfile.TemporaryDirectory() as tmp:
        reg = _DelayedVersionsRegistry(os.path.join(tmp, "root"))
        n_writers = 6
        barrier = threading.Barrier(n_writers)

        def writer(i: int) -> tuple[str, float]:
            model = st.GaussianDistribution(float(i), 1.0)
            barrier.wait()  # line every writer up so they all enter register() together
            return reg.register(model, "m"), float(i)

        with ThreadPoolExecutor(max_workers=n_writers) as pool:
            futures = [pool.submit(writer, i) for i in range(n_writers)]
            results = [f.result(timeout=10) for f in futures]

        versions_seen = [ver for ver, _ in results]
        assert len(set(versions_seen)) == n_writers, (
            f"expected {n_writers} distinct version ids from {n_writers} concurrent successful "
            f"registrations, got {versions_seen} -- a version number was reused"
        )
        for ver, expected_mu in results:
            model, _ = reg.get("m", ver)
            assert model.mu == expected_mu, (
                f"writer that registered mu={expected_mu} as {ver!r} was silently clobbered by "
                f"another concurrent writer: that slot now holds mu={model.mu}"
            )


@pytest.mark.slow
def test_concurrent_register_across_processes_does_not_silently_clobber_a_version():
    """Same race as above, but with real separate OS processes -- the realistic "concurrent
    writer" scenario for a filesystem-backed production registry (e.g. two training jobs
    registering checkpoints for the same model name at once). ``fcntl.flock`` is a cross-process
    lock; a fix that only serialized threads (e.g. a bare ``threading.Lock``) would pass the
    threading test above but not this one.

    Uses the ``spawn`` start method (the platform default, and the only one guaranteed safe with
    threads already running in this process -- ``fork`` here throws "process is multi-threaded,
    use of fork() may lead to deadlocks in the child"). Each spawned child pays the full cost of
    importing this scientific stack cold in a fresh interpreter (tens of seconds), which is why
    this one test is marked ``slow`` rather than living in the default fast gate.
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
            assert p.exitcode == 0, f"writer process pid={p.pid} failed or hung (exitcode={p.exitcode})"

        results = [queue.get(timeout=5) for _ in range(n_writers)]
        versions_seen = [ver for _, ver, _ in results]
        assert len(set(versions_seen)) == n_writers, (
            f"expected {n_writers} distinct version ids from {n_writers} concurrent successful "
            f"registrations, got {versions_seen} -- a version number was reused"
        )

        reg = Registry(root)
        for _, ver, expected_mu in results:
            model, _ = reg.get("m", ver)
            assert model.mu == expected_mu, (
                f"writer that registered mu={expected_mu} as {ver!r} was silently clobbered by "
                f"another concurrent writer: that slot now holds mu={model.mu}"
            )


def test_register_raises_instead_of_silently_overwriting_an_existing_version_file(monkeypatch):
    """Belt-and-suspenders guard, tested independently of the lock.

    Under normal operation ``versions()`` always sees a pre-existing file before ``register()``
    picks a number, so the lock alone is enough -- there is no black-box way to make a *correct*
    read collide with what's on disk. The guard exists for the case where the read is somehow
    stale anyway (a filesystem where ``flock`` does not actually exclude, or any other bug that
    lets two writers into the critical section together): simulate that by making ``versions()``
    under-report what's on disk, and check the final ``O_CREAT | O_EXCL`` write still turns the
    resulting collision into a raised, visible error instead of a silent ``open(..., "w")``
    overwrite of the pre-existing ``v1.json``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        d = reg._dir("m")
        sentinel = {"version": "v1", "model": "sentinel-do-not-clobber"}
        with open(os.path.join(d, "v1.json"), "w") as f:
            json.dump(sentinel, f)

        # simulate a stale read of the existing versions (as if flock had not actually excluded a
        # concurrent writer) so register() computes next_n == 1 despite v1.json already existing.
        monkeypatch.setattr(Registry, "versions", lambda self, name: [])

        with pytest.raises(RuntimeError, match="registry conflict"):
            reg.register(st.GaussianDistribution(0.0, 1.0), "m")

        # the pre-existing version file must be left completely untouched
        with open(os.path.join(d, "v1.json")) as f:
            assert json.load(f) == sentinel


def test_register_write_failure_leaves_no_corrupt_version_file_and_stays_retriable(monkeypatch):
    """A single writer failing partway through the version-file write -- serialization raising mid
    ``json.dump``, disk full, a crash -- must never leave a truncated/corrupt ``<version>.json`` on
    disk. Before the fix, ``register`` wrote the payload directly into the ``O_CREAT | O_EXCL``-opened
    file at the FINAL path: a failure after some bytes were already flushed left that exact path
    holding invalid JSON, permanently -- a later ``register`` call sees the (corrupt) file already
    exists and silently allocates the next number instead, orphaning the corrupt one forever, and the
    ``O_CREAT | O_EXCL`` guard then refuses any future write to that same path, so it can never
    self-heal. This simulates that failure by making ``json.dump`` write a real partial chunk to the
    file handle -- so, pre-fix, actual truncated bytes land on disk -- and then raise, mirroring a
    disk-full ``OSError`` mid-write.
    """
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        d = reg._dir("m")

        def _crash_mid_dump(obj, fp, *_a, **_kw):
            fp.write('{"version": "v1", "registered_at": "trunc')  # real bytes hit the file handle
            fp.flush()
            raise OSError("simulated: No space left on device")

        monkeypatch.setattr(json, "dump", _crash_mid_dump)
        with pytest.raises(OSError, match="No space left on device"):
            reg.register(st.GaussianDistribution(0.0, 1.0), "m")
        monkeypatch.undo()  # restore the real json.dump before the clean retry below

        # the failed write must leave NOTHING at the final path -- not a truncated file
        v1_path = os.path.join(d, "v1.json")
        assert not os.path.exists(v1_path), "a failed write left a corrupt version file behind"
        # ...and no temp-file debris either
        leftovers = [f for f in os.listdir(d) if f != ".register.lock"]
        assert leftovers == [], f"a failed write leaked temp files: {leftovers}"

        # the version number must stay cleanly retriable, not permanently stranded behind a corrupt file
        ver = reg.register(st.GaussianDistribution(5.0, 1.0), "m")
        assert ver == "v1", "a failed write must not permanently consume its version number"
        model, _ = reg.get("m", "v1")
        assert model.mu == 5.0


def test_register_never_temporarily_removes_the_live_models_header(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        model = st.GaussianDistribution(0.0, 1.0)
        header = {"run": "immutable-snapshot"}
        model.header = header
        observed_live_headers = []
        real_serialize = registry_module.to_serializable

        def observing_serialize(subject):
            observed_live_headers.append(getattr(model, "header", None))
            assert not hasattr(subject, "header")
            return real_serialize(subject)

        monkeypatch.setattr(registry_module, "to_serializable", observing_serialize)
        reg.register(model, "m")

        assert observed_live_headers == [header]
        assert model.header is header


def test_version_envelope_digest_detects_header_or_metadata_tampering():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Registry(os.path.join(tmp, "root"))
        reg.register(
            st.GaussianDistribution(0.0, 1.0),
            "m",
            header={"run": "original"},
            metadata={"stage": "fit"},
        )
        path = os.path.join(reg._dir("m"), "v1.json")
        with open(path) as f:
            payload = json.load(f)
        payload["metadata"]["stage"] = "tampered"
        with open(path, "w") as f:
            json.dump(payload, f)

        with pytest.raises(ValueError, match="integrity failure"):
            reg.metadata("m", "v1")
