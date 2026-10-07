"""Construye y carga el checkpoint SHAPY sin procesar fotografías."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.body3d.shapy_runner import cargar_modelo_shapy


if __name__ == "__main__":
    _, _, report = cargar_modelo_shapy(ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=False))
