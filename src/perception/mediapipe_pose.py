"""Detección anatómica ligera para encuadrar automáticamente una persona."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REQUIRED_BODY_GROUPS = {
    "head": (0,),
    "shoulders": (11, 12),
    "hips": (23, 24),
    "knees": (25, 26),
    "ankles": (27, 28),
}


def _value(landmark: Any, name: str, default: float) -> float:
    if isinstance(landmark, dict):
        return float(landmark.get(name, default))
    return float(getattr(landmark, name, default))


def puntos_a_caja(
    landmarks: list[Any],
    width: int,
    height: int,
    margin_ratio: float = 0.12,
    min_visibility: float = 0.25,
) -> np.ndarray:
    """Convierte landmarks normalizados en una caja XYXY con margen."""
    if width <= 0 or height <= 0:
        raise ValueError("La imagen debe tener dimensiones positivas")
    if not 0 <= margin_ratio <= 0.5:
        raise ValueError("margin_ratio debe estar entre 0 y 0.5")

    points = []
    for landmark in landmarks:
        visibility = _value(landmark, "visibility", 1.0)
        presence = _value(landmark, "presence", 1.0)
        x = _value(landmark, "x", float("nan"))
        y = _value(landmark, "y", float("nan"))
        if min(visibility, presence) >= min_visibility and np.isfinite([x, y]).all():
            points.append((x * width, y * height))
    if len(points) < 4:
        raise ValueError("MediaPipe no encontró suficientes puntos corporales")

    points_array = np.asarray(points, dtype=np.float32)
    x1, y1 = points_array.min(axis=0)
    x2, y2 = points_array.max(axis=0)
    span_y = max(float(y2 - y1), 0.20 * height)
    # Los landmarks laterales quedan cerca del eje del cuerpo. SAM necesita una
    # caja que también incluya el contorno, cabello y extremidades visibles.
    span_x = max(float(x2 - x1), 0.30 * span_y, 0.08 * width)
    center_x = float((x1 + x2) / 2.0)
    x1 = center_x - span_x / 2.0
    x2 = center_x + span_x / 2.0
    x_margin = margin_ratio * span_x
    y_margin = margin_ratio * span_y
    return np.asarray(
        [
            max(0.0, x1 - x_margin),
            max(0.0, y1 - y_margin),
            min(float(width), x2 + x_margin),
            min(float(height), y2 + y_margin),
        ],
        dtype=np.float32,
    )


def diagnostico_pose(landmarks: list[Any]) -> dict:
    """Resume visibilidad de articulaciones necesarias para una captura corporal."""
    if not landmarks:
        return {"mean_visibility": 0.0, "warnings": ["no se detectó una persona"]}
    visibilities = [_value(item, "visibility", 1.0) for item in landmarks]
    # En un perfil es normal que un lado del cuerpo oculte al otro. Basta con
    # que una articulación de cada par anatómico sea claramente visible.
    missing_groups = []
    for name, indices in REQUIRED_BODY_GROUPS.items():
        visible = [
            _value(landmarks[index], "visibility", 1.0)
            for index in indices
            if index < len(landmarks)
        ]
        if not visible or max(visible) < 0.5:
            missing_groups.append(name)
    warnings = []
    if missing_groups:
        warnings.append(
            "articulaciones principales poco visibles; revise cabeza, hombros, "
            "cadera, rodillas y tobillos"
        )
    return {
        "mean_visibility": round(float(np.mean(visibilities)), 4),
        "missing_required_groups": missing_groups,
        "warnings": warnings,
    }


class MediaPipePoseDetector:
    """Wrapper opcional de MediaPipe Pose Landmarker para una persona."""

    def __init__(self, model_path: Path):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe no está instalado en sastre-ia-perception. "
                "Ejecuta: python -m pip install mediapipe"
            ) from exc
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"No existe el modelo MediaPipe: {model_path}")
        self.mp = mp
        self.model_path = model_path.resolve()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, image: Image.Image) -> dict:
        rgb = np.asarray(image.convert("RGB"))
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.pose_landmarks:
            raise ValueError("MediaPipe no detectó una persona completa")
        landmarks = result.pose_landmarks[0]
        box = puntos_a_caja(landmarks, image.width, image.height)
        diagnostics = diagnostico_pose(landmarks)
        return {
            "box_xyxy": box.tolist(),
            "quality": diagnostics,
            "landmarks": [
                {
                    "x": round(_value(item, "x", 0.0), 6),
                    "y": round(_value(item, "y", 0.0), 6),
                    "z": round(_value(item, "z", 0.0), 6),
                    "visibility": round(_value(item, "visibility", 1.0), 6),
                }
                for item in landmarks
            ],
        }
