"""Conservative scattering of two free planar magnetic dipoles.

See ``notes/derivation.md`` for the dimensionless reduction.
"""
from . import dynamics, locked, scatter, integrators, regularize, dimensional  # noqa: F401

__all__ = ["dynamics", "locked", "scatter", "integrators", "regularize", "dimensional"]
