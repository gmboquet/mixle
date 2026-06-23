"""PDE/ODE physics forward-and-inverse stack for pysp.ppl.

The forward-model infrastructure (``_operator`` contracts, the ``ops`` math namespace, ``dynamics``
operators/integrators, ``pde_solve`` sparse/adjoint solves, ``inverse`` ODE forward models,
``multiphysics`` coupled systems, ``pde`` PDE-constrained state space) plus the concrete equations
(``wave``/``wave_pml`` acoustics, ``flow``/``spectral_flow`` Navier-Stokes, ``gas_dynamics`` Euler,
``schrodinger``, ``fem`` Poisson, ``shape`` level-set inversion). The public solvers are re-exported
from ``pysp.ppl``; this subpackage groups the implementation that was a flat wall of modules.
"""

# Importing `pde` fires its register_composite("PDEStateSpace", ..., fit_fn=pde_fit) so the
# PDE-constrained state-space family is fittable. (When this stack relocates to pysparkplug-pde, that
# package's __init__ does this import instead, and pysp core stays PDE-free via the fit_fn hook.)
from pysp.ppl.physics import pde  # noqa: F401
