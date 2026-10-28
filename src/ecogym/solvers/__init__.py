"""Solvers: one registry of forecasting methods with typed parameter schemas.

Importing this package registers the framework solvers (``recipe_ridge@1``).
Domain packs register their own through ``register_solver``.
"""

from . import recipe as _recipe  # noqa: F401  (registers recipe_ridge@1)
from .base import Genome, Param, SolverContext, SolverError, SolverSpec, get_solver, register_solver, solver_specs

__all__ = [
    "Genome",
    "Param",
    "SolverContext",
    "SolverError",
    "SolverSpec",
    "get_solver",
    "register_solver",
    "solver_specs",
]
