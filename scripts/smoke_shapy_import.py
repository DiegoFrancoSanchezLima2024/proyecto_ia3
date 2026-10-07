"""Smoke test de importación de SHAPY en Windows."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.body3d.shapy_compat import importar_shapy


if __name__ == "__main__":
    config, build_model = importar_shapy(ROOT)
    print(
        "SHAPY_IMPORT_OK",
        f"config={type(config).__name__}",
        f"builder={callable(build_model)}",
    )
