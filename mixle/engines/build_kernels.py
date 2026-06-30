"""Optionally compile the FMA double-double kernels (:mod:`mixle.engines._dd_kernels`).

mixle imports and runs fine WITHOUT any compiled extension (``mixle.engines.extended`` falls back to the
pure-numpy double-double path). Calling :func:`compile_dd_kernels` builds the optional accelerator in
place; afterwards ``dd_dot`` automatically uses it (~3x via hardware FMA). Requires Cython + a C compiler.
A future ``pyproject`` build-system hook can build this on ``pip install``; keeping it on-demand avoids a
hard compiler dependency.
"""

from __future__ import annotations

import os


def compile_dd_kernels(force: bool = False) -> str:
    """Cythonize + compile ``_dd_kernels.pyx`` in place; returns the built extension path.

    Raises ``ImportError`` if Cython/numpy build tooling is missing, or a build error otherwise.
    """
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    here = os.path.dirname(os.path.abspath(__file__))
    ext = Extension(
        "mixle.engines._dd_kernels",
        [os.path.join(here, "_dd_kernels.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffp-contract=fast"],
    )
    exts = cythonize([ext], quiet=True, compiler_directives={"language_level": "3"}, force=force)
    dist = Distribution({"ext_modules": exts})
    cmd = dist.get_command_obj("build_ext")
    cmd.inplace = 1
    cmd.ensure_finalized()
    cmd.run()
    built = [f for f in os.listdir(here) if f.startswith("_dd_kernels") and f.endswith((".so", ".pyd"))]
    return os.path.join(here, built[0]) if built else ""


def dd_kernels_available() -> bool:
    """True if the compiled FMA double-double kernels are importable."""
    try:
        import mixle.engines._dd_kernels  # noqa: F401

        return True
    except ImportError:
        return False


def compile_bitpacked_kernels(force: bool = False) -> str:
    """Cythonize + compile ``_bitpacked.pyx`` (popcount binary/ternary GEMM) in place; uses hardware popcount."""
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.dist import Distribution

    here = os.path.dirname(os.path.abspath(__file__))
    ext = Extension(
        "mixle.engines._bitpacked",
        [os.path.join(here, "_bitpacked.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-mcpu=native"],  # enable the NEON/AVX hardware popcount
    )
    exts = cythonize([ext], quiet=True, compiler_directives={"language_level": "3"}, force=force)
    dist = Distribution({"ext_modules": exts})
    cmd = dist.get_command_obj("build_ext")
    cmd.inplace = 1
    cmd.ensure_finalized()
    cmd.run()
    built = [f for f in os.listdir(here) if f.startswith("_bitpacked") and f.endswith((".so", ".pyd"))]
    return os.path.join(here, built[0]) if built else ""


def bitpacked_kernels_available() -> bool:
    """True if the compiled popcount binary/ternary kernels are importable."""
    try:
        import mixle.engines._bitpacked  # noqa: F401

        return True
    except ImportError:
        return False
