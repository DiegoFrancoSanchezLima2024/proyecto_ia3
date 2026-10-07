"""Pruebas ligeras que también corren en sastre-ia-perception."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.sam2_capture import (
    ROOT,
    diagnostico_mascara,
    resolver_ruta_proyecto,
    seleccionar_mejor_mascara,
    validar_caja,
)
from src.perception.mediapipe_pose import puntos_a_caja, diagnostico_pose


def test_sam2_box_validation_and_best_mask_selection():
    box = validar_caja([1, 2, 9, 18], width=10, height=20)
    assert box.tolist() == pytest.approx([1, 2, 9, 18])
    with pytest.raises(ValueError):
        validar_caja([1, 2, 11, 18], width=10, height=20)

    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 2:6, 2:6] = True
    masks[1, 1:7, 3:5] = True
    selected, score = seleccionar_mejor_mascara(masks, np.array([0.2, 0.9]))
    assert score == pytest.approx(0.9)
    assert np.array_equal(selected, masks[1])


def test_capture_mask_diagnostics_flags_cropped_person():
    cropped = np.zeros((64, 64), dtype=bool)
    cropped[:, 20:44] = True
    quality = diagnostico_mascara(cropped)
    assert any("borde" in warning for warning in quality["warnings"])


def test_relative_capture_paths_are_anchored_to_project():
    assert resolver_ruta_proyecto("photos/front.jpg") == (ROOT / "photos/front.jpg").resolve()


def test_mediapipe_landmarks_generate_clamped_person_box():
    landmarks = [
        {"x": 0.2, "y": 0.1, "visibility": 0.9},
        {"x": 0.8, "y": 0.9, "visibility": 0.9},
        {"x": 0.5, "y": 0.5, "visibility": 0.9},
        {"x": 0.6, "y": 0.7, "visibility": 0.9},
    ]
    box = puntos_a_caja(landmarks, width=1000, height=2000)
    assert box[0] < 200 and box[1] < 200
    assert box[2] > 800 and box[3] > 1800
    assert np.all(box >= 0)


def test_pose_diagnostics_warns_when_required_joints_are_missing():
    landmarks = [{"visibility": 0.9}] * 10
    quality = diagnostico_pose(landmarks)
    assert quality["missing_required_groups"]
    assert quality["warnings"]
