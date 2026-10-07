"""Operaciones métricas sobre una malla corporal alineada."""

from __future__ import annotations

import numpy as np


def altura_malla(vertices: np.ndarray, vertical_axis: int = 1) -> float:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2:
        raise ValueError("vertices debe tener forma (N, 3)")
    return float(np.ptp(vertices[:, vertical_axis]))


def escalar_malla_a_estatura(
    vertices: np.ndarray,
    height_cm: float,
    vertical_axis: int = 1,
) -> tuple[np.ndarray, float]:
    if height_cm <= 0:
        raise ValueError("height_cm debe ser positivo")
    current = altura_malla(vertices, vertical_axis)
    if current <= 1e-8:
        raise ValueError("la malla tiene altura nula")
    scale = (float(height_cm) / 100.0) / current
    return np.asarray(vertices, dtype=np.float64) * scale, scale


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        raise ValueError("la sección no contiene suficientes puntos")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def perimetro_seccion_horizontal(
    vertices: np.ndarray,
    faces: np.ndarray,
    level: float,
    vertical_axis: int = 1,
) -> float:
    """Perímetro convexo de la intersección malla/plano, en unidades de malla."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces debe tener forma (M, 3)")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces contiene índices fuera de rango")

    plane_axes = [axis for axis in range(3) if axis != vertical_axis]
    intersections = []
    eps = 1e-10
    for face in faces:
        triangle = vertices[face]
        signed = triangle[:, vertical_axis] - float(level)
        points = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = signed[first], signed[second]
            if abs(a) <= eps:
                points.append(triangle[first])
            if a * b < -eps:
                t = a / (a - b)
                points.append(triangle[first] + t * (triangle[second] - triangle[first]))
        for point in points:
            projected = point[plane_axes]
            if not any(np.linalg.norm(projected - existing) <= 1e-8 for existing in intersections):
                intersections.append(projected)
    if len(intersections) < 3:
        raise ValueError("el plano no produce una sección corporal cerrada")

    hull = _convex_hull(np.asarray(intersections))
    closed = np.vstack([hull, hull[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())

