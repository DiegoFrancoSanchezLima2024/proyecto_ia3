"""Perfiles físicos aproximados extraídos de siluetas binarias."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def perfil_mascara(path: Path, estatura_cm: float, bins: int) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"Máscara vacía: {path}")
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    alto = max(1, y1 - y0 + 1)
    escala = float(estatura_cm) / alto
    centros = np.linspace(y0, y1, bins)
    widths = []
    for center in centros:
        lo = max(y0, int(round(center)) - 2)
        hi = min(y1 + 1, int(round(center)) + 3)
        stripe = mask[lo:hi]
        cols = np.where(stripe.any(axis=0))[0]
        widths.append((cols[-1] - cols[0] + 1) * escala if len(cols) else 0.0)
    area_fisica = float(mask.sum()) * escala * escala
    bbox_ancho = (x1 - x0 + 1) * escala
    centro_x = float(xs.mean() - x0) / max(1, x1 - x0 + 1)
    occupancy = float(mask.sum()) / max(1, alto * (x1 - x0 + 1))
    return np.asarray(
        [*widths, area_fisica / 1000.0, bbox_ancho, centro_x, occupancy],
        dtype=np.float32,
    )


def caracteristicas_dos_vistas(
    front_path: Path, left_path: Path, estatura_cm: float, bins: int
) -> np.ndarray:
    front = perfil_mascara(front_path, estatura_cm, bins)
    left = perfil_mascara(left_path, estatura_cm, bins)
    return np.concatenate(([estatura_cm], front, left)).astype(np.float32)
