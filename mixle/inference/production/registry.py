"""A versioned model registry: store fitted models + their provenance, list versions, promote/swap.

A filesystem-backed store so a production system can register every fitted model (with its
:class:`~mixle.inference.production.provenance.Header`), list and load any version, and promote a chosen version
to an alias (e.g. ``"production"``) -- the swap point a serving layer reads from. Models serialize through
``mixle.utils.serialization`` (the safe registry-keyed JSON); headers are plain JSON dicts.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Callable
from copy import copy, deepcopy
from datetime import UTC, datetime
from typing import Any

from mixle.utils.exact import require_explicit_true
from mixle.utils.serialization import (
    SerializationError,
    ensure_pysp_serialization_registry,
    from_serializable,
    to_serializable,
)

# Distinguishes "caller passed expected_tip=None" (meaning: the name must be empty) from
# "caller did not ask for a tip check at all".
_UNSPECIFIED = object()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fsync_directory(path: str) -> None:
    """Durably commit a link/rename/unlink performed inside ``path``."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _record_digest(payload: dict[str, Any]) -> str:
    """Content digest for a version envelope, excluding its own digest field.

    The digest authenticates the *persisted* envelope, so it has to be computed over the form that
    will be read back, not the in-memory one. ``_canonical`` deliberately tags lists and tuples
    differently (MXR-080-1601: otherwise ``[[1, 2]]`` and ``[(1, 2)]`` collide), but JSON has only
    arrays -- so a payload holding a tuple anywhere (``header["schema"]`` holds
    ``("value", "Real")`` pairs for every fitted model) digested one way on write and the other way
    on read, and every such version file failed its own integrity check on load. Normalizing through
    JSON first makes writer and reader agree by construction rather than by coincidence.
    """
    from mixle.data.hashing import _canonical

    subject = dict(payload)
    subject.pop("record_digest", None)
    return hashlib.sha256(_canonical(json.loads(json.dumps(subject)))).hexdigest()


def _transition_digest(metadata: dict[str, Any]) -> str:
    """Bind one checkpoint transition to its exact persisted predecessor."""
    from mixle.data.hashing import _canonical

    fields = {
        "lineage_schema": metadata.get("lineage_schema"),
        "run_id": metadata.get("run_id"),
        "checkpoint_iter": metadata.get("checkpoint_iter"),
        "model_hash": metadata.get("model_hash"),
        "parent_hash": metadata.get("parent_hash"),
        "parent_version": metadata.get("parent_version"),
        "parent_record_digest": metadata.get("parent_record_digest"),
        "parent_transition_digest": metadata.get("parent_transition_digest"),
    }
    return hashlib.sha256(_canonical(fields)).hexdigest()


def _safe_segment(seg: str, kind: str = "name") -> str:
    """Reject a model name / version / alias that is not a single path component under the registry root.

    The registry is a filesystem store that may be fed names/aliases from an API (e.g. via
    ``Service.from_registry``); joining a raw ``../escape`` (or an absolute path, or one with separators)
    onto the root would read or write outside it. Constrain each segment to a plain basename."""
    if not isinstance(seg, str) or not seg:
        raise ValueError(f"registry {kind} must be a non-empty string, got {seg!r}")
    if (
        seg in (os.curdir, os.pardir)
        or os.sep in seg
        or (os.altsep and os.altsep in seg)
        or "\x00" in seg
        or os.path.isabs(seg)
        or os.path.basename(seg) != seg
    ):
        raise ValueError(f"unsafe registry {kind} {seg!r}: must be a single path component (no separators or '..')")
    return seg


class Registry:
    """A directory of named models, each with numbered versions and movable aliases."""

    def __init__(self, root: str) -> None:
        ensure_pysp_serialization_registry()
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _model_dir(self, name: str, *, create: bool) -> str:
        """Resolve the on-disk directory for model ``name``, refusing any path that escapes the store root.

        ``_safe_segment`` blocks traversal *inside* the name string, but a symlink pre-placed in the root
        (from an untrusted or restored registry, or a tar extraction) named like a model would still let a
        read or write follow it outside the root. Reject an entry that is a symlink, or whose real path is
        not contained in the root's real path, before any ``open`` / ``makedirs`` follows it.
        """
        d = os.path.join(self.root, _safe_segment(name))
        if os.path.lexists(d):
            root_real = os.path.realpath(self.root)
            real = os.path.realpath(d)
            if os.path.islink(d) or (real != root_real and not real.startswith(root_real + os.sep)):
                raise ValueError(f"unsafe registry name {name!r}: entry resolves outside the store root")
        if create:
            os.makedirs(d, exist_ok=True)
        return d

    def _dir(self, name: str) -> str:
        return self._model_dir(name, create=True)

    def names(self) -> list[str]:
        """Registered model names."""
        return sorted(n for n in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, n)))

    def versions(self, name: str) -> list[str]:
        """Version ids for ``name`` in registration order (``v1``, ``v2``, ...)."""
        d = self._model_dir(name, create=False)
        if not os.path.isdir(d):
            return []
        vs = [f[:-5] for f in os.listdir(d) if f.endswith(".json")]
        return sorted(vs, key=lambda v: int(v[1:]) if v[1:].isdigit() else 0)

    def register(
        self,
        model: Any,
        name: str,
        *,
        header: Any = None,
        metadata: dict | None = None,
        expected_tip: Any = _UNSPECIFIED,
    ) -> str:
        """Store ``model`` under ``name`` as a new version; return its version id.

        ``expected_tip`` makes the write conditional: the current latest version must equal it (or
        the name must be empty when it is ``None``) at the moment of allocation, checked under the
        same lock that allocates the version. A caller extending a lineage passes the tip it
        adopted, and a concurrent writer that got there first makes this call refuse instead of
        persisting a sibling successor.

        ``header`` defaults to ``model.header`` if present. The model is serialized with the safe mixle
        registry; the header (a :class:`Header` or dict) and ``metadata`` are stored alongside.

        Allocating the next version number is a read-modify-write over ``versions(name)``: two
        concurrent callers (two threads, or two processes) can both read the same existing versions,
        both compute the same "next" number, and both write it -- one write silently clobbers the
        other, with no error to either caller. To prevent that, the number allocation and the write
        are serialized per model name with an ``fcntl.flock`` on a lock file in the model's directory
        (held for both the read and the write, so a second writer waits and is correctly given the
        NEXT free number rather than racing for the same one). The write itself additionally uses
        ``O_CREAT | O_EXCL`` as an independent, belt-and-suspenders guard: if a file for that version
        exists anyway (e.g. flock is a no-op on the underlying filesystem), it raises instead of
        silently overwriting the earlier registration.

        The version file is also published atomically: the payload is written to a private temp file
        in the model's directory and fsynced *before* it is ever linked to the final ``<version>.json``
        path (via ``os.link``, after which the temp name is dropped). A failure anywhere in between --
        serialization raising partway through ``json.dump``, the disk filling up, the process crashing
        -- therefore never leaves a truncated or corrupt version file on disk: the file only ever exists
        fully-formed or not at all. This mirrors the temp-file-plus-atomic-publish idiom used by
        :func:`mixle.task.artifact._atomic_json_dump` and ``mixle.system.registry``'s ``_write_index``,
        except the publish step uses ``os.link`` rather than ``os.replace``: this write must still refuse
        to clobber an existing (or concurrently-claimed) version file, and ``os.replace`` would silently
        overwrite one -- reopening the bug the ``O_CREAT | O_EXCL`` guard above fixes. Unlike writing
        directly into the ``O_CREAT | O_EXCL``-opened file, a failed attempt here also leaves nothing
        behind at ``path``, so the same version number stays cleanly retriable instead of being
        permanently stuck behind a corrupt file that neither parses nor can be overwritten.
        """
        d = self._dir(name)
        attached = getattr(model, "header", None)
        if header is None:
            header = attached
        hdr_source = header.to_dict() if hasattr(header, "to_dict") else header
        hdr = deepcopy(hdr_source)
        metadata_snapshot = deepcopy(metadata or {})
        # The header is stored separately. Serialize a shallow snapshot without it rather than
        # deleting and restoring ``model.header`` on the caller's live object: the latter lets a
        # concurrent scorer or second registration observe a transiently headerless model.
        had_attr = hasattr(model, "__dict__") and "header" in vars(model)
        subject = copy(model) if had_attr else model
        if had_attr:
            vars(subject).pop("header", None)
        model_ser = to_serializable(subject)

        lock_path = os.path.join(d, ".register.lock")
        # O_NOFOLLOW: _model_dir screens the model directory for symlink escape, but not the files
        # inside it. A preplaced .register.lock symlink was followed, and opening it for writing
        # truncated whatever it pointed at, outside the store. O_TRUNC is deliberately absent too --
        # a lock file's contents are never read, so there is nothing here that needs truncating.
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "r+") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # the NEXT version number, not the current COUNT: a deleted version (v2 removed from
                # v1,v2,v3) must not free up its number for reuse, or the next register() overwrites
                # the surviving v3.
                existing = self.versions(name)
                # Checked INSIDE the lock, so "the tip is still what I adopted" and "allocate the
                # next version" are one atomic step. Two checkpointers that adopted the same tip
                # each passed a tip check taken outside this lock and then both registered,
                # persisting sibling successors of one parent; two created on an EMPTY name each
                # wrote an independent root. Both returned success and left the chain unverifiable
                # (SYS3-06). Under the lock the second writer sees the moved tip and refuses.
                if expected_tip is not _UNSPECIFIED:
                    current_tip = existing[-1] if existing else None
                    if current_tip != expected_tip:
                        raise RuntimeError(
                            f"registry conflict: {name!r} tip is {current_tip!r}, not the expected "
                            f"{expected_tip!r}; another writer appended first. Refusing to write a "
                            f"sibling of the same predecessor."
                        )
                next_n = max((int(v[1:]) for v in existing if v[1:].isdigit()), default=0) + 1
                ver = f"v{next_n}"
                payload = {
                    "version": ver,
                    "registered_at": _now(),
                    "model": model_ser,
                    "header": hdr,
                    "metadata": metadata_snapshot,
                }
                payload["record_digest"] = _record_digest(payload)
                path = os.path.join(d, ver + ".json")
                # Write fully to a private temp file (fsynced) before it ever touches `path`, so a
                # failure mid-write -- serialization raising partway through json.dump, disk full, a
                # crash -- leaves `path` completely untouched instead of a truncated/corrupt version
                # file. os.link (not os.replace) publishes it: link atomically creates `path` only if
                # it does not already exist, raising FileExistsError otherwise, which preserves the
                # O_CREAT | O_EXCL conflict-detection this write has always needed (os.replace would
                # silently overwrite an existing or concurrently-claimed version file instead).
                fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{ver}.", suffix=".json.tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(payload, f)
                        f.flush()
                        os.fsync(f.fileno())
                    try:
                        os.link(tmp, path)
                    except FileExistsError:
                        raise RuntimeError(
                            f"registry conflict: {name!r} version {ver!r} already exists -- another writer "
                            "claimed it concurrently; retry register()"
                        ) from None
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    _fsync_directory(d)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        return ver

    def checkpointer(
        self, name: str, *, every: int = 1, resume: bool = True, trust_code: bool = False
    ) -> Callable[[Any], None]:
        """Return an ``optimize(on_step=...)`` callback that snapshots the model under ``name`` every
        ``every`` iterations (recording the iteration + log-density in the version metadata).

        Each checkpoint records its model ``model_hash`` and the previous checkpoint's ``parent_hash``,
        so the saved snapshots form a verifiable chain (see :meth:`verify_chain`). Resume an interrupted
        run from the latest checkpoint::

            reg = Registry("ckpts")
            optimize(data, est, on_step=reg.checkpointer("run", every=5))
            model, _ = reg.get("run")              # latest checkpoint
            optimize(data, est, prev_estimate=model)   # continue training

        A checkpointer created for a name that already holds a VERIFIABLE chain resumes it: it
        adopts that chain's run identity, links its first new checkpoint to the existing tip, and
        continues the iteration count, so appending recovered work leaves ``verify_chain`` true.
        Pass ``resume=False`` to deliberately root a new lineage instead. ``trust_code`` is only
        used to re-load models while checking whether the existing chain is adoptable, and carries
        the same meaning as in :meth:`get`.
        """
        from mixle.data.hashing import model_hash

        parent: str | None = None
        parent_version: str | None = None
        parent_record_digest: str | None = None
        parent_transition_digest: str | None = None
        run_id = secrets.token_hex(16)
        iteration_offset = 0
        # What register() must see as the tip at write time. Three intents, kept distinct:
        #   resume=True and a chain was adopted  -> the adopted tip (a moved tip is a fork; refuse)
        #   resume=True and the name was empty   -> None (we are the root; a second root collides)
        #   resume=False                         -> no condition: the caller ASKED for a fresh
        #                                           unlinked lineage after whatever exists, so
        #                                           the write is unconditional by design.
        # Passing None for resume=False conflated the last two and made every deliberate
        # resume=False on a non-empty name refuse as a "conflict".
        expected_tip: Any = _UNSPECIFIED
        if resume:
            expected_tip = None

        # Resume: adopt the existing lineage rather than rooting a second one under the same name.
        # A fresh run_id with a null parent made the COMBINED chain unverifiable the moment a
        # recovered run appended to it -- verify_chain requires one run_id, strictly increasing
        # iterations, and exact parent linkage, and a restart broke all three (SYS-04). Adoption is
        # conditional on the existing versions actually verifying, so a chain is only ever extended
        # from an authenticated tip; the parent identity taken here is the same quadruple
        # verify_chain will re-derive.
        #
        # If the existing versions do NOT verify -- a truncated checkpoint, or plain register() calls
        # under the same name that carry no lineage metadata -- resume REFUSES rather than rooting a
        # new run on top of them. The earlier text here argued the fallthrough was harmless because
        # "it did not verify before this call either"; that is only true for lineage-less
        # registrations, and false for a chain that was valid until one file was damaged, which the
        # fallthrough silently made unrecoverable (SYS3-03). Starting over is a decision the caller
        # states with resume=False or a new name, never something inferred from damage.
        if resume:
            existing = self.versions(name)
            if existing:
                # NOT wrapped in a broad except. verify_chain() already reports semantic corruption
                # as False; anything it RAISES is a different state -- verification could not be
                # performed at all. The commonest case is its deliberate trust refusal for a
                # NeuralLeaf-family checkpoint when trust_code was not supplied, and swallowing
                # that turned "I am not allowed to check this" into "a new root is safe here":
                # the callback then appended an unlinked root to a valid neural chain and made it
                # permanently unverifiable (CP2-01). An exception propagates, so the caller either
                # passes trust_code=True to adopt or resume=False to root a new lineage on purpose.
                if not self.verify_chain(name, trust_code=trust_code):
                    # verify_chain returned False: the persisted lineage under this name does not
                    # verify -- a truncated or corrupted checkpoint, or versions that never carried
                    # lineage at all. Either way there is no authenticated tip to extend, and the
                    # first repair's answer here was to fall through and quietly root a NEW run id
                    # with a null parent on top of it. That is the SAME conflation CP2-01 fixed one
                    # branch over ("cannot verify" treated as "safe to start over"), and it turned a
                    # chain that had been valid until one file was damaged into one that verifies
                    # nowhere, with no error raised at any point (SYS3-03). Refuse. A caller who
                    # actually wants a fresh lineage says so with resume=False or a new name.
                    raise ValueError(
                        f"checkpoint lineage for {name!r} does not verify, so there is no "
                        f"authenticated tip to resume from. Refusing to append a new root to it: "
                        f"repair or remove the damaged versions, use a different name, or pass "
                        f"resume=False to deliberately start an unlinked lineage."
                    )
                tip = existing[-1]
                tip_metadata = self.metadata(name, tip)
                run_id = tip_metadata["run_id"]
                parent = tip_metadata["model_hash"]
                parent_version = tip
                parent_record_digest = self.record_digest(name, tip)
                parent_transition_digest = tip_metadata["transition_digest"]
                iteration_offset = int(tip_metadata["checkpoint_iter"])
                expected_tip = tip

        def _save(step: Any) -> None:
            nonlocal parent, parent_record_digest, parent_transition_digest, parent_version, expected_tip
            if every <= 1 or step.iter % every == 0:
                # The predecessor was snapshotted when this callback was CONSTRUCTED. If another
                # checkpointer adopted the same tip and appended first, writing against the cached
                # predecessor forks the lineage. The first fix re-read the tip HERE and refused on a
                # mismatch -- but that check and the register() below were two separate operations,
                # so two writers could both pass the check and then both register (SYS3-06). The
                # tip condition now travels INTO register() as expected_tip and is evaluated under
                # the same lock that allocates the version number: exactly one of two racing
                # writers appends, the other refuses. `expected_tip=None` on an empty name means
                # "I am the root", so two roots created concurrently on one name also collide.
                h = model_hash(step.model)
                metadata = {
                    "lineage_schema": "mixle-checkpoint-lineage-v1",
                    "run_id": run_id,
                    # a resumed optimize() restarts step.iter at 1; the chain requires strictly
                    # increasing iterations across the whole lineage, so continue from the tip.
                    "checkpoint_iter": iteration_offset + step.iter,
                    "log_density": step.log_density,
                    "model_hash": h,
                    "parent_hash": parent,
                    "parent_version": parent_version,
                    "parent_record_digest": parent_record_digest,
                    "parent_transition_digest": parent_transition_digest,
                }
                metadata["transition_digest"] = _transition_digest(metadata)
                version = self.register(
                    step.model,
                    name,
                    metadata=metadata,
                    expected_tip=expected_tip,
                )
                parent = h
                parent_version = version
                # this callback's next write must expect the version IT just produced; a
                # resume=False callback stays unconditional (it never pinned a tip to begin with).
                if expected_tip is not _UNSPECIFIED:
                    expected_tip = version
                parent_record_digest = self.record_digest(name, version)
                parent_transition_digest = metadata["transition_digest"]

        return _save

    def _resolve_version(self, name: str, version: str) -> str:
        """Resolve ``"latest"`` to the highest version and raise a clear KeyError for an unknown name or
        version -- rather than a bare IndexError on an unregistered name or a raw FileNotFoundError (which
        leaks the store path) on a missing version. Mirrors the guard get() already had."""
        _safe_segment(name)
        vs = self.versions(name)
        if not vs:
            raise KeyError(f"no versions registered for model {name!r}")
        if version == "latest":
            return vs[-1]
        if version not in vs:  # returned value is therefore always a known-safe version id
            raise KeyError(f"{name!r} has no version {version!r}")
        return version

    def get(self, name: str, version: str = "latest", *, trust_code: bool = False) -> tuple[Any, dict | None]:
        """Load ``(model, header)`` for a version (``"latest"`` = highest-numbered).

        A registered model containing a NeuralLeaf-family component embeds its weights as a pickle
        blob (see :mod:`mixle.models._neural_serial`); deserializing that executes code, so ``get``
        requires ``trust_code=True`` for such an entry -- trust the registry root, not just the JSON
        extension. A pure-statistical entry loads either way.
        """
        # `if trust_code:` is truthiness, and what it opens is a trusted-deserialization scope:
        # trust_code="false" -- the string, straight out of a config file or CLI argument -- entered
        # it (MXR-080-1881). The flag must be the True singleton, matching load_encoded's contract.
        if trust_code is not False:
            require_explicit_true(
                trust_code,
                "Registry.get trust_code",
                because="It opens a trusted-deserialization scope, and a registered model may embed "
                "a pickle blob whose decode executes code.",
            )
        version = self._resolve_version(name, version)
        payload = self._load_payload(name, version)
        if trust_code:
            from mixle.utils.serialization import trusted_deserialization

            with trusted_deserialization():
                return from_serializable(payload["model"]), payload.get("header")
        return from_serializable(payload["model"]), payload.get("header")

    def header(self, name: str, version: str = "latest") -> dict | None:
        """Just the provenance header of a version (no model deserialization)."""
        version = self._resolve_version(name, version)
        return self._load_payload(name, version).get("header")

    def metadata(self, name: str, version: str = "latest") -> dict:
        """Just the ``metadata`` of a version (no model deserialization) -- e.g. a checkpoint's iteration."""
        version = self._resolve_version(name, version)
        return self._load_payload(name, version).get("metadata") or {}

    def _load_payload(self, name: str, version: str) -> dict[str, Any]:
        """Read one immutable envelope and verify its content digest when present."""
        # O_NOFOLLOW for the same reason as the registration lock: a symlinked version file would
        # otherwise be followed and read JSON from outside the store as though it were a record.
        version_path = os.path.join(self._model_dir(name, create=False), version + ".json")
        with os.fdopen(os.open(version_path, os.O_RDONLY | os.O_NOFOLLOW)) as f:
            payload = json.load(f)
        stored = payload.get("record_digest")
        if stored is not None and stored != _record_digest(payload):
            raise ValueError(f"registry integrity failure for {name!r} version {version!r}")
        return payload

    def record_digest(self, name: str, version: str = "latest") -> str:
        """Verified digest of the exact persisted version envelope."""
        version = self._resolve_version(name, version)
        payload = self._load_payload(name, version)
        stored = payload.get("record_digest")
        if not isinstance(stored, str):
            raise ValueError(f"{name!r} version {version!r} has no authenticated record digest")
        return stored

    def promote(self, name: str, version: str, alias: str = "production") -> None:
        """Point ``alias`` (e.g. ``"production"``) at ``version`` -- the atomic model swap.

        Written via a temp file + ``os.replace`` in the same directory (same filesystem, so the
        rename is atomic): a concurrent reader of :meth:`current` either sees the old alias target or
        the new one, never a truncated/partial write, and a crash mid-write leaves the old alias
        file untouched rather than corrupted.
        """
        if version not in self.versions(name):
            raise KeyError(f"{name!r} has no version {version!r}")
        d = self._dir(name)
        target = os.path.join(d, _safe_segment(alias, "alias") + ".alias")
        fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{_safe_segment(alias, 'alias')}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(version)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
            _fsync_directory(d)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    def current(self, name: str, alias: str | None = None, *, trust_code: bool = False) -> tuple[Any, dict | None]:
        """Load the model an ``alias`` points at.

        Two deliberately different behaviours, split on whether the caller NAMED an alias:

        * ``current(name)`` -- no alias requested. Resolves the default ``"production"`` alias if it
          exists and falls back to ``latest`` if it does not. This is the bootstrap path: a registry
          with registrations but no promotion yet still serves something.
        * ``current(name, "production")`` -- an alias was explicitly requested. It must exist;
          an absent alias raises :class:`KeyError`. A caller that names an alias is asserting "serve
          the version promoted to this alias", and quietly substituting ``latest`` answers a
          different question with an UNPROMOTED model -- which also silently defeats a rollback,
          since rolling an alias back to a known-good version has no effect on a caller whose alias
          name is misspelled or was never created (SYS-05).

        See :meth:`get` -- ``trust_code`` is required in the same way and for the same reason.
        """
        requested = alias is not None
        p = os.path.join(
            self._model_dir(name, create=False), _safe_segment(alias if requested else "production", "alias") + ".alias"
        )
        # the version READ FROM the alias file is still resolved against the known version list by get(),
        # so a tampered alias file cannot traverse either. O_NOFOLLOW for the same reason as the
        # registration lock and _load_payload: a symlinked alias would otherwise be followed and its
        # target read as the pointer, so a planted link could redirect "production" at whatever file
        # it names. Opening with O_NOFOLLOW instead of testing os.path.exists first also closes the
        # TOCTOU window between the check and the open.
        try:
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            if requested:
                raise KeyError(
                    f"model {name!r} has no alias {alias!r}; promote a version to it "
                    f"(Registry.promote) or call current({name!r}) without an alias to accept the "
                    f"latest registered version"
                ) from None
            version = "latest"
        else:
            with os.fdopen(fd, encoding="utf-8") as f:
                version = f.read().strip()
        return self.get(name, version, trust_code=trust_code)

    def verify_chain(self, name: str, *, trust_code: bool = False) -> bool:
        """Verify the persisted checkpoint lineage for ``name`` (see :meth:`checkpointer`).

        Requires every version to carry a complete authenticated lineage record. Checks exact parent
        versions, record digests, model hashes, transition digests, run identity, and iteration order,
        and re-hashes each loaded model. Missing or partial lineage is unverified and returns False.

        See :meth:`get` -- ``trust_code`` is required in the same way and for the same reason: a chain
        that contains a NeuralLeaf-family checkpoint must be loaded to be re-hashed, so verifying it is
        subject to the same code-execution trust gate as loading it directly.
        """
        from mixle.data.hashing import model_hash

        versions = self.versions(name)
        if not versions:
            return False
        required = {
            "lineage_schema",
            "run_id",
            "checkpoint_iter",
            "model_hash",
            "parent_hash",
            "parent_version",
            "parent_record_digest",
            "parent_transition_digest",
            "transition_digest",
        }
        previous_hash: str | None = None
        previous_version: str | None = None
        previous_record_digest: str | None = None
        previous_transition_digest: str | None = None
        previous_iter: int | None = None
        run_id: str | None = None
        try:
            for ver in versions:
                metadata = self.metadata(name, ver)
                if not required.issubset(metadata):
                    return False
                if metadata["lineage_schema"] != "mixle-checkpoint-lineage-v1":
                    return False
                if run_id is None:
                    run_id = metadata["run_id"]
                if not isinstance(run_id, str) or metadata["run_id"] != run_id:
                    return False
                checkpoint_iter = metadata["checkpoint_iter"]
                if (
                    isinstance(checkpoint_iter, bool)
                    or not isinstance(checkpoint_iter, int)
                    or (previous_iter is not None and checkpoint_iter <= previous_iter)
                ):
                    return False
                if metadata["parent_hash"] != previous_hash:
                    return False
                if metadata["parent_version"] != previous_version:
                    return False
                if metadata["parent_record_digest"] != previous_record_digest:
                    return False
                if metadata["parent_transition_digest"] != previous_transition_digest:
                    return False
                if metadata["transition_digest"] != _transition_digest(metadata):
                    return False
                stored = metadata["model_hash"]
                if not isinstance(stored, str):
                    return False
                model, _ = self.get(name, ver, trust_code=trust_code)
                if model_hash(model) != stored:
                    return False
                previous_hash = stored
                previous_version = ver
                previous_record_digest = self.record_digest(name, ver)
                previous_transition_digest = metadata["transition_digest"]
                previous_iter = checkpoint_iter
        except SerializationError:
            raise
        except (KeyError, TypeError, ValueError):
            return False
        return True
