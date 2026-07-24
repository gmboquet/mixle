"""Optionally compile the FMA double-double kernels (:mod:`mixle.engines._dd_kernels`).

mixle imports and runs fine WITHOUT any compiled extension (``mixle.engines.extended`` falls back to the
pure-numpy double-double path). Calling :func:`compile_dd_kernels` builds the optional accelerator;
afterwards ``dd_dot`` automatically uses it (~3x via hardware FMA). Requires Cython + a C compiler.
Keeping the build on demand avoids making a C compiler a hard installation dependency.

Every ``compile_*`` builder here writes into an isolated, user-writable cache directory
(:func:`_kernel_cache_dir`) rather than the installed ``mixle`` package -- some installs (a wheel
unpacked into a shared/system ``site-packages`` tree, various read-only-filesystem container
deployments) cannot be written to at all, and a build that requires writing there fails outright even
though the documented pure-Python fallback would have been perfectly usable instead. Each builder then
best-effort copies its result next to the ``.pyx`` source too, so a normal writable install keeps
finding it via a plain ``import mixle.engines._xxx`` exactly as before; the ``*_kernels_available``
functions fall back to the isolated cache (extending ``mixle.engines.__path__``) when that copy could
not be made. Set ``MIXLE_KERNEL_CACHE_DIR`` to override the cache location.

The cache directory is verified privately-owned (0700, no symlink, this user only) before anything is
built into or imported from it -- a compiled extension is native code with no sandboxing, so a
predictably-named shared cache must not become a spot another user on the same machine could plant a
malicious build into (see :func:`_private_cache_base_dir`).
"""

from __future__ import annotations

import importlib.machinery
import os
import shutil
import stat
import sys
import sysconfig
import tempfile


def _uid_suffix() -> str:
    """A per-user filesystem suffix for the default cache location.

    Mirrors ``mixle.stats.compute.fused_codegen._default_cache_dir``'s same isolation rationale (that
    module cannot be imported from here -- the "engines are model-agnostic compute" import-linter
    contract forbids ``mixle.engines`` from depending on ``mixle.stats``): a single shared tmp directory
    would let another user on a multi-user machine pre-place a file that gets loaded here. A loaded
    compiled extension runs arbitrary native code with no sandboxing at all, which makes an attacker
    substitution here strictly worse than in the exec-a-generated-module case that helper guards against.
    """
    return f"_{os.getuid()}" if hasattr(os, "getuid") else ""


def _cache_base_dir() -> str:
    """The top-level kernel-cache directory: the trust boundary :func:`_private_cache_base_dir` verifies
    and everything else in this module nests beneath. Honors ``MIXLE_KERNEL_CACHE_DIR`` if set.
    """
    return os.environ.get("MIXLE_KERNEL_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), f"mixle_kernel_cache{_uid_suffix()}"
    )


def _kernel_cache_dir() -> str:
    """Isolated, user-writable directory the optional kernel extensions build into.

    Never inside the installed ``mixle`` package (``mixle/engines/`` on disk) -- see the module
    docstring (MXR-080-0155). Namespaced by interpreter ABI tag + platform so a build for one Python
    version/platform can never shadow, or be mistaken for, another's; the trailing ``mixle/engines``
    mirrors the real package layout so this directory can be appended directly to
    ``mixle.engines.__path__``. Pure path computation -- does not create or verify anything on disk (see
    :func:`_private_cache_base_dir`, which callers that actually write here or trust an existing build
    here must go through first).
    """
    tag = f"{sys.implementation.cache_tag}-{sysconfig.get_platform()}"
    return os.path.join(_cache_base_dir(), tag, "mixle", "engines")


def _owned_privately(path: str, *, require_dir: bool) -> bool:
    """True iff ``path`` is a real dir/file we own that no other user can write (no symlink, no
    group/other-write).

    Mirrors ``mixle.stats.compute.fused_codegen._owned_privately`` (kept as an independent copy, not an
    import -- the "engines are model-agnostic compute" import-linter contract forbids ``mixle.engines``
    from depending on ``mixle.stats``). ``lstat`` (not ``stat``) so a symlink can never pass as its --
    possibly attacker-owned -- target.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if require_dir and not stat.S_ISDIR(info.st_mode):
        return False
    if not require_dir and not stat.S_ISREG(info.st_mode):
        return False
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return False
    return not (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def _private_cache_base_dir() -> str | None:
    """:func:`_cache_base_dir`, created (0700) if missing and healed back to 0700 if we own it but an
    older run left it group/other-writable, or ``None`` if it already exists and is not privately ours
    (e.g. another user pre-planted a world-writable directory at the expected path).

    A compiled extension is native code with no sandboxing at all: introducing a shared, predictably-
    named tmp-dir cache (this module's fix for MXR-080-0155's read-only-package-install problem) must
    not also introduce a spot another user on a shared machine could plant a malicious build into.
    Mirrors ``mixle.stats.compute.fused_codegen._private_cache_dir``'s same rationale (independent copy,
    see :func:`_owned_privately`).
    """
    base = _cache_base_dir()
    try:
        os.makedirs(base, mode=0o700, exist_ok=True)
        info = os.lstat(base)
        owned = not hasattr(os, "getuid") or info.st_uid == os.getuid()
        if owned and stat.S_ISDIR(info.st_mode) and (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
            os.chmod(base, 0o700)  # heal a dir we own that an older/insecure run left too permissive
    except OSError:
        return None
    return base if _owned_privately(base, require_dir=True) else None


def _matches_running_abi(filename: str, modname: str) -> bool:
    """True iff ``filename`` (a bare filename, no directory) is a valid compiled-extension name for
    ``modname`` under the CURRENTLY RUNNING interpreter -- i.e. exactly ``modname`` followed by one of
    :data:`importlib.machinery.EXTENSION_SUFFIXES` for this interpreter/platform/ABI.

    A same-prefix file built for a different Python version, platform, or ABI (e.g.
    ``_bitpacked.cpython-311-darwin.so`` sitting next to a cpython-312 interpreter) does NOT match, even
    though a naive ``startswith``/``endswith`` glob would find it (MXR-080-0155).
    """
    return any(filename == modname + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def _build_kernel_extension(modname: str, pyx_filename: str, extra_compile_args: list[str], force: bool) -> str:
    """Cythonize + compile ``mixle.engines.<modname>`` from ``<pyx_filename>`` into the isolated kernel
    cache (:func:`_kernel_cache_dir`), then best-effort copy the result next to its ``.pyx`` source too.

    Returns the build command's own exact reported output path for this extension. Never globs a
    directory for "anything matching this prefix" afterward -- that can silently select a stale artifact
    left over from a previous Python version, platform, or failed build instead of the one just built
    (MXR-080-0155). The isolated cache copy is the canonical, always-attempted destination and is what
    this function returns and validates; the package-directory copy is a best-effort convenience for the
    common writable install (keeps a plain ``import mixle.engines.<modname>`` working there unchanged)
    and is silently skipped -- not an error -- when the package directory is not writable.
    """
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    src_dir = os.path.dirname(os.path.abspath(__file__))
    ext = Extension(
        f"mixle.engines.{modname}",
        [os.path.join(src_dir, pyx_filename)],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
    )

    if _private_cache_base_dir() is None:
        raise RuntimeError(
            f"refusing to build into {_cache_base_dir()!r}: it already exists and is not privately "
            "owned by this user (owner mismatch, symlink, or group/other-writable) -- set "
            "MIXLE_KERNEL_CACHE_DIR to a trusted location"
        )
    cache_dir = _kernel_cache_dir()  # .../<abi tag>/mixle/engines, nested under the now-verified base
    cache_root = os.path.dirname(os.path.dirname(cache_dir))  # .../<abi tag> -- strip "mixle/engines"
    os.makedirs(cache_root, exist_ok=True)

    # Cython's generated .c/.cpp normally lands next to the .pyx source; redirect it into the cache too
    # so a read-only package directory never needs a write for this intermediate step either.
    exts = cythonize(
        [ext],
        quiet=True,
        compiler_directives={"language_level": "3"},
        force=force,
        build_dir=os.path.join(cache_root, "_cythonize"),
    )
    dist = Distribution({"ext_modules": exts})
    cmd = dist.get_command_obj("build_ext")
    cmd.build_lib = cache_root
    cmd.build_temp = os.path.join(cache_root, "_build_tmp")
    cmd.inplace = 0
    cmd.ensure_finalized()
    cmd.run()

    # The build command's own authoritative answer for where THIS extension landed -- not a glob.
    built_path = cmd.get_ext_fullpath(ext.name)
    if not os.path.exists(built_path):
        raise RuntimeError(f"build_ext did not produce the expected {built_path!r} for {ext.name!r}")
    if not _matches_running_abi(os.path.basename(built_path), modname):
        expected = [modname + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES]
        raise RuntimeError(
            f"built {built_path!r} does not match the running interpreter's ABI (expected one of {expected!r})"
        )

    # Best-effort: also place a copy next to the .pyx source so a fresh process that imports e.g.
    # mixle.engines.extended directly (without ever going through build_kernels) still finds it the same
    # way it always has. Never required for compile_*_kernels() itself to succeed -- a read-only package
    # install must not make the BUILD fail, only this convenience copy (MXR-080-0155).
    try:
        shutil.copy2(built_path, os.path.join(src_dir, os.path.basename(built_path)))
    except OSError:
        pass

    return built_path


def _kernel_available(modname: str) -> bool:
    """True if ``mixle.engines.<modname>`` is importable -- either from the package directory itself
    (the common case) or from the isolated kernel cache (:func:`_kernel_cache_dir`), extending
    ``mixle.engines.__path__`` with the latter if needed.

    A build whose package-directory copy could not be made (read-only install) is still discoverable by
    callers that go through this function (MXR-080-0155's isolated-build-destination fix). Never raises:
    an absent or untrusted cache base (see :func:`_private_cache_base_dir`) is simply treated as
    unavailable rather than surfaced as an error, matching this function's plain boolean contract.
    """
    import importlib

    try:
        importlib.import_module(f"mixle.engines.{modname}")
        return True
    except ImportError:
        pass
    if _private_cache_base_dir() is None:
        return False  # absent, or exists and is not privately ours: never import from it
    cache_dir = _kernel_cache_dir()
    if not os.path.isdir(cache_dir):
        return False
    import mixle.engines as _engines_pkg

    if cache_dir not in _engines_pkg.__path__:
        _engines_pkg.__path__.append(cache_dir)
    try:
        importlib.import_module(f"mixle.engines.{modname}")
        return True
    except ImportError:
        return False


def compile_dd_kernels(force: bool = False) -> str:
    """Cythonize + compile ``_dd_kernels.pyx``; returns the built extension's exact path.

    Builds into an isolated cache directory and best-effort copies the result next to its source (see
    the module docstring). Raises ``ImportError`` if Cython/numpy build tooling is missing, or a build
    error otherwise.
    """
    return _build_kernel_extension("_dd_kernels", "_dd_kernels.pyx", ["-O3", "-ffp-contract=fast"], force)


def dd_kernels_available() -> bool:
    """True if the compiled FMA double-double kernels are importable."""
    return _kernel_available("_dd_kernels")


def compile_bitpacked_kernels(force: bool = False) -> str:
    """Cythonize + compile ``_bitpacked.pyx`` (popcount binary/ternary GEMM) into the isolated kernel
    cache; returns its exact path. See the module docstring for the read-only-package rationale.
    """
    args = ["-O3", "-mcpu=native"]  # enable the NEON/AVX hardware popcount
    return _build_kernel_extension("_bitpacked", "_bitpacked.pyx", args, force)


def bitpacked_kernels_available() -> bool:
    """True if the compiled popcount binary/ternary kernels are importable."""
    return _kernel_available("_bitpacked")


def compile_lns_kernel(force: bool = False) -> str:
    """Cythonize + compile ``_lns_kernel.pyx`` (the integer log-sum-exp tree fold) into the isolated
    kernel cache; returns its exact path. See the module docstring for the read-only-package rationale.
    """
    return _build_kernel_extension("_lns_kernel", "_lns_kernel.pyx", ["-O3", "-mcpu=native"], force)


def lns_kernel_available() -> bool:
    """True if the compiled integer log-sum-exp kernel is importable."""
    return _kernel_available("_lns_kernel")
