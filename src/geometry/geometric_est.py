"""Baseline geometrico 2.5D para siluetas frontal/lateral.

Es una caracteristica auxiliar, no ground truth. Usa la altura visible del cuerpo
y el mayor segmento continuo de cada fila para no contar el espacio entre brazos.
"""

from __future__ import annotations

import numpy as np


# Fraccion de la estatura medida desde los pies. Son puntos iniciales para una
# ablacion; el modelo profundo aprende la correccion residual.
CIRCUMFERENCE_LEVELS = {
    "chest": 0.76,
    "waist": 0.63,
    "hip": 0.53,
    "thigh": 0.43,
    "calf": 0.17,
    "ankle": 0.06,
}

# Priors antropometricos debiles para medidas que una seccion horizontal simple
# no puede separar de forma estable en la pose BodyM.
HEIGHT_PRIORS = {
    "arm-length": 0.295,
    "forearm": 0.165,
    "wrist": 0.095,
    "bicep": 0.180,
    "leg-length": 0.445,
    "shoulder-to-crotch": 0.370,
}


def ramanujan_circumference(a: float, b: float) -> float:
    if a <= 0 or b <= 0 or a + b <= 0:
        return 0.0
    h = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h))))


def _largest_run(row: np.ndarray) -> float:
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return 0.0
    gaps = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps, indices.size - 1]
    return float(np.max(indices[ends] - indices[starts] + 1))


def extract_profile(silhouette: np.ndarray) -> np.ndarray:
    binary = silhouette > 128
    return np.asarray([_largest_run(row) for row in binary], dtype=np.float32)


def _body_bounds(silhouette: np.ndarray) -> tuple[int, int]:
    occupied = np.any(silhouette > 128, axis=1)
    rows = np.flatnonzero(occupied)
    if rows.size < 2:
        return 0, max(1, silhouette.shape[0] - 1)
    return int(rows[0]), int(rows[-1])


def _width_near(profile: np.ndarray, row: int, radius: int = 2) -> float:
    lo = max(0, row - radius)
    hi = min(len(profile), row + radius + 1)
    values = profile[lo:hi]
    values = values[values > 0]
    return float(np.median(values)) if values.size else 0.0


def estimate_measurements(
    front_sil: np.ndarray,
    left_sil: np.ndarray,
    height_cm: float,
    measures: list | None = None,
) -> dict[str, float]:
    measures = list(measures or [
        *CIRCUMFERENCE_LEVELS,
        *HEIGHT_PRIORS,
        "shoulder-breadth",
    ])
    if height_cm <= 0:
        return {name: 0.0 for name in measures}

    front_profile = extract_profile(front_sil)
    left_profile = extract_profile(left_sil)
    front_top, front_bottom = _body_bounds(front_sil)
    left_top, left_bottom = _body_bounds(left_sil)
    front_height = max(1, front_bottom - front_top + 1)
    left_height = max(1, left_bottom - left_top + 1)
    front_scale = height_cm / front_height
    left_scale = height_cm / left_height

    results: dict[str, float] = {}
    for name in measures:
        if name in CIRCUMFERENCE_LEVELS:
            fraction = CIRCUMFERENCE_LEVELS[name]
            front_row = int(round(front_bottom - fraction * front_height))
            left_row = int(round(left_bottom - fraction * left_height))
            width_cm = _width_near(front_profile, front_row) * front_scale
            depth_cm = _width_near(left_profile, left_row) * left_scale
            results[name] = ramanujan_circumference(width_cm / 2.0, depth_cm / 2.0)
        elif name == "shoulder-breadth":
            fraction = 0.82
            row = int(round(front_bottom - fraction * front_height))
            results[name] = _width_near(front_profile, row) * front_scale
        elif name in HEIGHT_PRIORS:
            results[name] = height_cm * HEIGHT_PRIORS[name]
        else:
            results[name] = 0.0
    return results


class EstimadorGeometrico:
    def __init__(self, target_measures: list):
        self.measures = list(target_measures)

    def __call__(
        self,
        front_batch: np.ndarray,
        left_batch: np.ndarray,
        heights: np.ndarray,
    ) -> np.ndarray:
        batch_size = front_batch.shape[0]
        output = np.zeros((batch_size, len(self.measures)), dtype=np.float32)
        for index in range(batch_size):
            front = front_batch[index, 0] if front_batch.ndim == 4 else front_batch[index]
            left = left_batch[index, 0] if left_batch.ndim == 4 else left_batch[index]
            front = ((front * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            left = ((left * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            estimate = estimate_measurements(front, left, float(heights[index]), self.measures)
            output[index] = [estimate[name] for name in self.measures]
        return output
