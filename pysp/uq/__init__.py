"""Deprecated shim: uncertainty quantification now lives in :mod:`pysp.doe`.

The sensitivity / propagation / calibration tools were folded into ``pysp.doe`` -- they are the
analysis half of the same "design and analysis of computer experiments" concern, and reuse that
package's quasi-Monte-Carlo sampling, GP surrogate, and kernels. Import from ``pysp.doe`` instead;
this module re-exports the same names for backward compatibility.
"""

from pysp.doe.calibrate import KOCalibration, calibrate
from pysp.doe.propagate import propagate, unscented_transform
from pysp.doe.sensitivity import morris_screening, sobol_indices

__all__ = [
    "sobol_indices",
    "morris_screening",
    "propagate",
    "unscented_transform",
    "calibrate",
    "KOCalibration",
]
