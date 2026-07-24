"""Optional kernel build/select/isolate logic (mixle.engines.build_kernels): MXR-080-0155 regressions.

Covers the three build_kernels-specific findings from the 0.8.0 exhaustive review: (1) ``-mcpu=native``
is not even a valid flag on every supported platform and ties a binary to the exact build CPU, (2) the
prior "first filename matching a prefix" selection could silently return a stale/wrong-ABI artifact
instead of the one just built, (3) building inside the installed package directory fails outright on a
read-only install even though the pure-Python fallback is fine -- plus the multi-user cache-poisoning
surface introduced by this fix's own isolated cache directory (mirroring the coverage
fused_cache_security_test.py has for the analogous mixle.stats.compute.fused_codegen cache).

Most tests are pure logic or filesystem-permission checks (no compiler needed). The handful that perform
a real compile are gated on the Cython/numpy/setuptools build tooling actually being importable, matching
this suite's existing dd_kernels_test.py/lns_kernel_test.py convention, and build only against the safe,
solely build_kernels-owned _dd_kernels.pyx (never _bitpacked.pyx/_lns_kernel.pyx, which other in-flight
work in this tree owns) so nothing here can race a concurrent build of those. The stale-artifact and
read-only-package tests additionally build from an isolated scratch copy of build_kernels.py + the .pyx
source (never the real installed mixle/engines/ directory) so a deliberately-planted stale file or a
chmod'd-read-only directory can never affect the real package.
"""

import importlib
import importlib.machinery
import importlib.util
import os
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

from mixle.engines import build_kernels
from mixle.engines.build_kernels import (
    _kernel_available,
    _kernel_cache_dir,
    _matches_running_abi,
    _native_popcount_flags,
    _owned_privately,
    compile_dd_kernels,
    dd_kernels_available,
)


def _cython_toolchain_available() -> bool:
    try:
        import Cython.Build  # noqa: F401
        import numpy  # noqa: F401
        from setuptools import Extension  # noqa: F401
    except ImportError:
        return False
    return True


HAS_CYTHON = _cython_toolchain_available()
_SKIP_NO_CYTHON = "Cython/numpy/setuptools build tooling not installed"


def _load_scratch_build_kernels_module(dest_dir: str):
    """Copy build_kernels.py + the safe, unshared _dd_kernels.pyx into ``dest_dir`` and import that copy
    as an independent module object (never registered as ``mixle.engines.build_kernels``).

    Lets a test exercise a real ``compile_dd_kernels()`` call whose "package directory" is ``dest_dir``
    -- e.g. making ``dest_dir`` read-only, or pre-seeding a stale artifact into its build output -- without
    ever touching the actual installed ``mixle/engines/`` directory, which other work in this tree may be
    concurrently modifying.
    """
    real_pkg_dir = os.path.dirname(os.path.abspath(build_kernels.__file__))
    shutil.copy2(os.path.join(real_pkg_dir, "build_kernels.py"), os.path.join(dest_dir, "build_kernels.py"))
    shutil.copy2(os.path.join(real_pkg_dir, "_dd_kernels.pyx"), os.path.join(dest_dir, "_dd_kernels.pyx"))
    modname = f"_scratch_build_kernels_{os.path.basename(dest_dir)}"
    spec = importlib.util.spec_from_file_location(modname, os.path.join(dest_dir, "build_kernels.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CacheDirTest(unittest.TestCase):
    """_kernel_cache_dir: isolated from the package, ABI-namespaced, overridable, side-effect-free."""

    def test_default_cache_dir_is_outside_the_installed_package(self):
        pkg_dir = os.path.dirname(os.path.abspath(build_kernels.__file__))
        cache_dir = _kernel_cache_dir()
        rel = os.path.relpath(cache_dir, pkg_dir)
        self.assertTrue(rel.startswith(os.pardir), f"{cache_dir!r} should not be inside {pkg_dir!r}")

    def test_cache_dir_mirrors_the_package_layout_for_path_extension(self):
        # so mixle.engines.__path__.append(cache_dir) makes `import mixle.engines._xxx` find it.
        self.assertTrue(_kernel_cache_dir().replace(os.sep, "/").endswith("mixle/engines"))

    def test_cache_dir_is_namespaced_by_interpreter_abi_and_platform(self):
        self.assertIn(sys.implementation.cache_tag, _kernel_cache_dir())

    def test_cache_dir_honors_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": tmp}):
                cache_dir = _kernel_cache_dir()
        self.assertTrue(cache_dir.startswith(tmp))

    def test_cache_dir_does_not_create_anything_as_a_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, "not_yet_created")
            with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": override}):
                cache_dir = _kernel_cache_dir()
            self.assertFalse(os.path.exists(override))
            self.assertFalse(os.path.exists(cache_dir))


class PrivateCacheBaseDirTest(unittest.TestCase):
    """MXR-080-0155's own new attack surface: a predictably-named shared cache directory must not become
    a spot another user on the same machine can plant a malicious compiled extension into. Mirrors
    fused_cache_security_test.py's coverage of the analogous mixle.stats.compute.fused_codegen cache.
    """

    def test_default_cache_base_is_per_user(self):
        base = build_kernels._cache_base_dir()
        if hasattr(os, "getuid"):
            self.assertIn(str(os.getuid()), base)

    def test_owned_privately_accepts_a_fresh_private_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o700)
            self.assertTrue(_owned_privately(tmp, require_dir=True))

    def test_owned_privately_rejects_group_or_other_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o777)
            self.assertFalse(_owned_privately(tmp, require_dir=True))

    def test_owned_privately_rejects_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "real_dir")
            os.mkdir(target, 0o700)
            link = os.path.join(tmp, "link_dir")
            os.symlink(target, link)
            self.assertFalse(_owned_privately(link, require_dir=True))

    def test_owned_privately_rejects_a_different_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o700)
            with mock.patch("os.getuid", return_value=os.getuid() + 999983, create=True):
                self.assertFalse(_owned_privately(tmp, require_dir=True))

    def test_private_cache_base_dir_creates_a_fresh_dir_as_0700(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, "fresh_cache_base")
            with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": override}):
                result = build_kernels._private_cache_base_dir()
            self.assertEqual(result, override)
            self.assertEqual(stat.S_IMODE(os.lstat(override).st_mode), 0o700)

    def test_private_cache_base_dir_heals_an_own_dir_left_group_writable(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, "healable_cache_base")
            os.mkdir(override, 0o777)
            os.chmod(override, 0o777)  # mkdir's mode is umask-limited; force it explicitly
            with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": override}):
                result = build_kernels._private_cache_base_dir()
            self.assertEqual(result, override)
            self.assertEqual(stat.S_IMODE(os.lstat(override).st_mode), 0o700)

    def test_private_cache_base_dir_refuses_a_dir_owned_by_someone_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, "attacker_cache_base")
            os.mkdir(override, 0o700)
            with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": override}):
                with mock.patch("os.getuid", return_value=os.getuid() + 999983, create=True):
                    result = build_kernels._private_cache_base_dir()
            self.assertIsNone(result)

    def test_compile_raises_when_the_cache_base_is_untrusted(self):
        with mock.patch("mixle.engines.build_kernels._private_cache_base_dir", return_value=None):
            with self.assertRaises(RuntimeError):
                compile_dd_kernels()

    def test_availability_check_degrades_to_false_when_the_cache_base_is_untrusted(self):
        # Must never raise -- an untrusted/absent cache base is just "not available", matching this
        # function's plain boolean contract. Uses a probe name never really built anywhere so the
        # (correctly trusted, package-directory) fast path can't short-circuit this check.
        with mock.patch("mixle.engines.build_kernels._private_cache_base_dir", return_value=None):
            self.assertFalse(_kernel_available("_build_kernels_probe_untrusted_base"))


class AbiMatchTest(unittest.TestCase):
    """_matches_running_abi: exact suffix matching against the RUNNING interpreter, not a loose glob."""

    def test_accepts_every_real_suffix_for_the_running_interpreter(self):
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            self.assertTrue(_matches_running_abi("_dd_kernels" + suffix, "_dd_kernels"))

    def test_rejects_a_different_python_versions_tag(self):
        # A same-prefix, same-generic-extension file that is NOT one of this interpreter's own suffixes
        # -- e.g. built for a different cpython minor version -- must not be accepted (MXR-080-0155).
        fake = "_dd_kernels.cpython-1-fakeplatform-fakeabi.so"
        self.assertNotIn(fake, ["_dd_kernels" + s for s in importlib.machinery.EXTENSION_SUFFIXES])
        self.assertFalse(_matches_running_abi(fake, "_dd_kernels"))

    def test_rejects_a_same_suffix_but_different_module_name(self):
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        self.assertFalse(_matches_running_abi("_bitpacked" + suffix, "_dd_kernels"))

    def test_rejects_extra_characters_between_name_and_suffix(self):
        # The bug this guards against: the pre-fix code matched via startswith(prefix) +
        # endswith((".so", ".pyd")), which a name like "_dd_kernels_OLD_BACKUP.so" would pass. Exact
        # suffix matching must reject it.
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        self.assertFalse(_matches_running_abi("_dd_kernels_OLD_BACKUP" + suffix, "_dd_kernels"))


class NativePopcountFlagsTest(unittest.TestCase):
    """_native_popcount_flags: a portable, per-architecture feature flag -- never -mcpu=native/-march=native."""

    def _flags_for(self, machine):
        with mock.patch("mixle.engines.build_kernels.platform.machine", return_value=machine):
            return _native_popcount_flags()

    def test_x86_64_gets_the_narrow_popcnt_feature_flag(self):
        # Empirically confirmed against this toolchain (clang, cross-targeted at x86_64-unknown-linux-gnu):
        # -mpopcnt compiles cleanly and lowers __builtin_popcountll to a real `popcnt` instruction, while
        # -mcpu=native is rejected outright ("unsupported option '-mcpu=' for target").
        self.assertEqual(self._flags_for("x86_64"), ["-mpopcnt"])

    def test_amd64_spelling_also_gets_it(self):
        self.assertEqual(self._flags_for("AMD64"), ["-mpopcnt"])

    def test_arm64_gets_no_extra_flag(self):
        # Hardware popcount is already in AArch64's base ISA (confirmed by inspecting emitted assembly
        # for __builtin_popcountll under -O3 with no arch flag at all: a plain `cnt` instruction).
        self.assertEqual(self._flags_for("arm64"), [])

    def test_aarch64_gets_no_extra_flag(self):
        self.assertEqual(self._flags_for("aarch64"), [])

    def test_unknown_machine_gets_no_isa_specific_flag(self):
        self.assertEqual(self._flags_for("riscv64"), [])

    def test_never_returns_a_native_flag(self):
        for machine in ("x86_64", "AMD64", "arm64", "aarch64", "riscv64", ""):
            flags = self._flags_for(machine)
            self.assertTrue(all("native" not in f for f in flags), flags)
            self.assertTrue(all(not f.startswith(("-mcpu=", "-march=")) for f in flags), flags)


class KernelAvailableTest(unittest.TestCase):
    """_kernel_available: plain package import first, isolated-cache fallback second, never fooled by an
    ABI-mismatched file sitting in the cache.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        import mixle.engines as _engines_pkg

        self._engines_pkg = _engines_pkg
        self._orig_path = list(_engines_pkg.__path__)
        self.addCleanup(self._restore_path)

    def _restore_path(self):
        self._engines_pkg.__path__[:] = self._orig_path

    def test_reports_false_when_nothing_is_built_anywhere(self):
        self.assertFalse(_kernel_available("_build_kernels_probe_absent"))

    def test_a_wrong_abi_file_in_the_cache_does_not_count_as_available(self):
        cache_dir = _kernel_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        fake = os.path.join(cache_dir, "_build_kernels_probe_wrong_abi.cpython-1-fakeplatform-fakeabi.so")
        with open(fake, "wb") as f:
            f.write(b"not a real extension")
        self.assertFalse(_kernel_available("_build_kernels_probe_wrong_abi"))

    def test_a_module_only_present_in_the_cache_is_found_via_path_extension(self):
        # A real (pure-Python, not compiled -- the __path__-extension + import mechanism doesn't care)
        # module placed only in the isolated cache, never the package directory, proving the fallback
        # path itself, independent of whether the payload is a compiled .so.
        cache_dir = _kernel_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        modname = "_build_kernels_probe_stand_in"
        with open(os.path.join(cache_dir, modname + ".py"), "w") as f:
            f.write("MARKER = 'loaded from isolated cache'\n")
        self.addCleanup(sys.modules.pop, f"mixle.engines.{modname}", None)
        self.assertTrue(_kernel_available(modname))
        stand_in = importlib.import_module(f"mixle.engines.{modname}")
        self.assertEqual(stand_in.MARKER, "loaded from isolated cache")


@unittest.skipUnless(HAS_CYTHON, _SKIP_NO_CYTHON)
class FreshBuildNegativeControlTest(unittest.TestCase):
    """Negative control: a normal, successful fresh build still produces a usable, correctly-selected
    artifact, and the writable real package directory still gets the backward-compatible convenience
    copy (extended.py's own plain ``import mixle.engines._dd_kernels`` must keep working unchanged).
    """

    def test_fresh_build_is_usable_correct_and_outside_the_package(self):
        with tempfile.TemporaryDirectory() as cache_tmp:
            with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": cache_tmp}):
                built_path = compile_dd_kernels(force=True)

            self.assertTrue(os.path.exists(built_path))
            pkg_dir = os.path.dirname(os.path.abspath(build_kernels.__file__))
            self.assertTrue(os.path.relpath(built_path, pkg_dir).startswith(os.pardir))
            self.assertTrue(_matches_running_abi(os.path.basename(built_path), "_dd_kernels"))

            # A compiled extension's exported init symbol (PyInit__dd_kernels) is fixed at compile time
            # from the Extension name's last dotted component, so the loader name must match it exactly
            # -- unlike a plain .py file, this cannot be an arbitrary probe-local name.
            spec = importlib.util.spec_from_file_location("_dd_kernels", built_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            import numpy as np

            hi, lo = mod.dd_dot_c(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
            self.assertEqual(float(hi), 11.0)
            self.assertEqual(float(lo), 0.0)

            # Backward compatibility: the writable real package directory still gets the convenience
            # copy under MIXLE_KERNEL_CACHE_DIR overrides too -- only the cache side is redirected.
            convenience_copy = os.path.join(pkg_dir, os.path.basename(built_path))
            self.assertTrue(os.path.exists(convenience_copy))
            self.assertTrue(dd_kernels_available())


@unittest.skipUnless(HAS_CYTHON, _SKIP_NO_CYTHON)
class StaleArtifactSelectionTest(unittest.TestCase):
    """MXR-080-0155: a same-prefix stale/wrong-ABI file must never be selected over the artifact the
    build command just produced.
    """

    def test_old_glob_logic_would_have_picked_the_stale_file_but_the_new_code_does_not(self):
        with tempfile.TemporaryDirectory() as scratch, tempfile.TemporaryDirectory() as cache_tmp:
            scratch_mod = _load_scratch_build_kernels_module(scratch)
            with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": cache_tmp}):
                cache_dir = scratch_mod._kernel_cache_dir()
                os.makedirs(cache_dir, exist_ok=True)
                stale_name = "_dd_kernels.cpython-1-stalefake-stalefake.so"
                stale_path = os.path.join(cache_dir, stale_name)
                stale_content = b"leftover from a different Python version, not a real extension"
                with open(stale_path, "wb") as f:
                    f.write(stale_content)

                # Reproduction of the PRE-FIX selection logic (build_kernels.py before MXR-080-0155):
                # `[f for f in os.listdir(here) if f.startswith(prefix) and f.endswith((".so", ".pyd"))][0]`.
                # With only the stale file present, this is exactly what it would have returned.
                old_style_matches = [
                    f for f in os.listdir(cache_dir) if f.startswith("_dd_kernels") and f.endswith((".so", ".pyd"))
                ]
                self.assertEqual(old_style_matches, [stale_name])  # the old logic's only candidate is the fake

                built_path = scratch_mod.compile_dd_kernels(force=True)

            self.assertNotEqual(os.path.basename(built_path), stale_name)
            self.assertTrue(scratch_mod._matches_running_abi(os.path.basename(built_path), "_dd_kernels"))
            self.assertTrue(os.path.exists(built_path))
            # The stale file is still sitting right there, proving the new code had to actively avoid it
            # (via the build command's own exact reported path) rather than the scenario just not arising.
            self.assertTrue(os.path.exists(stale_path))
            with open(stale_path, "rb") as f:
                self.assertEqual(f.read(), stale_content)


@unittest.skipUnless(HAS_CYTHON, _SKIP_NO_CYTHON)
class ReadOnlyPackageDirectoryTest(unittest.TestCase):
    """MXR-080-0155: a read-only package/import directory must not make the build fail."""

    def test_build_succeeds_and_lands_outside_a_read_only_scratch_package_dir(self):
        with tempfile.TemporaryDirectory() as scratch, tempfile.TemporaryDirectory() as cache_tmp:
            scratch_mod = _load_scratch_build_kernels_module(scratch)
            os.chmod(scratch, stat.S_IRUSR | stat.S_IXUSR)  # read + traverse only, no write
            try:
                with mock.patch.dict(os.environ, {"MIXLE_KERNEL_CACHE_DIR": cache_tmp}):
                    built_path = scratch_mod.compile_dd_kernels(force=True)
            finally:
                os.chmod(scratch, stat.S_IRWXU)  # restore so TemporaryDirectory can clean up

            self.assertTrue(os.path.exists(built_path))
            self.assertTrue(os.path.relpath(built_path, scratch).startswith(os.pardir))
            # The best-effort copy into the (read-only) scratch "package dir" must have been silently
            # skipped, not raised -- confirm no build artifact (.so/.pyd/.c) appeared there. (__pycache__
            # may legitimately be present: loading the scratch build_kernels.py copy itself, before the
            # chmod, writes its own bytecode cache -- unrelated to the compile_dd_kernels() call under
            # test.)
            leftovers = [
                f for f in os.listdir(scratch) if f not in ("_dd_kernels.pyx", "build_kernels.py", "__pycache__")
            ]
            self.assertEqual(leftovers, [])

            import numpy as np

            # A compiled extension's exported init symbol (PyInit__dd_kernels) is fixed at compile time
            # from the Extension name's last dotted component, so the loader name must match it exactly.
            spec = importlib.util.spec_from_file_location("_dd_kernels", built_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            hi, lo = mod.dd_sum_c(np.array([1e16, 1.0, -1e16, -1.0] * 100))
            self.assertLess(abs(float(hi) + float(lo)), 1e-6)  # true sum 0


if __name__ == "__main__":
    unittest.main()
