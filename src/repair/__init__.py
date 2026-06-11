"""Wood-joint repair planning (Direction A: template prior + parametric fit)."""
from __future__ import annotations

__version__ = "0.1.0"

from . import (charts, config, damage, dataset, datagen, energy, evaluate,
               inference, optimizer, prior, rasterize, templates, viz)

__all__ = ["charts", "config", "damage", "dataset", "datagen", "energy",
           "evaluate", "inference", "optimizer", "prior", "rasterize",
           "templates", "viz"]
