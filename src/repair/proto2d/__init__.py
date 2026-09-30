"""The 2D repair prototype: one block, a damage front, an interface family, and
a rigid-body frictional-equilibrium LP for the moment the interface can carry.

    model      block, damage field, contact samples, the LP, the sound-wood grid
    families   interface families as polygons (plain cut, mortise-and-tenon,
               cross-sections of two MiGumi splices, each in two orientations)

Viewers live in examples/proto2d/; docs/figures/proposal.py renders Figure 1
from `model` alone. Nothing here depends on the rest of the package.
"""
from . import families, model
from .families import FAMILIES, Family, evaluate
from .model import Block, Damage, Grid, Splice, Statics

__all__ = ["Block", "Damage", "FAMILIES", "Family", "Grid", "Splice", "Statics",
           "evaluate", "families", "model"]
