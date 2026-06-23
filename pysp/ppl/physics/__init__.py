"""PDE/ODE physics forward-and-inverse stack for pysp.ppl.

The forward-model infrastructure (``_operator`` contracts, the ``ops`` math namespace, ``dynamics``
operators/integrators, ``pde_solve`` sparse/adjoint solves, ``inverse`` ODE forward models,
``multiphysics`` coupled systems, ``pde`` PDE-constrained state space) plus the concrete equations
(``wave``/``wave_pml`` acoustics, ``flow``/``spectral_flow`` Navier-Stokes, ``gas_dynamics`` Euler,
``schrodinger``, ``fem`` Poisson, ``shape`` level-set inversion). The public solvers are re-exported
from ``pysp.ppl``; this subpackage groups the implementation that was a flat wall of modules.
"""
