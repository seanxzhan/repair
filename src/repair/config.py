"""Central constants shared across the pipeline."""
from __future__ import annotations

from pathlib import Path

# Repo root = two levels up from this file (src/repair/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "migumi-dataset" / "joints"
OUT_DIR = REPO_ROOT / "out"

# Rasterization grid for the CNN prior.
RASTER_RES = 128

# Manufacturability: minimum machinable feature size, in *normalized* member units
# (a member's longest bbox side is normalized to 1.0, so this is 5% of the member).
TOOL_RADIUS = 0.05

# Proxy-energy weights. sound_removed is the primary (minimum-intervention) term;
# structural/fabrication pull the other way (toward longer, cleaner interfaces).
ENERGY_WEIGHTS = dict(
    w_sound=1.0,
    w_struct=0.6,
    w_fab=0.4,
    w_grain=0.3,
    w_defect=2.0,
)

RNG_SEED = 0


def ensure_out() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR
