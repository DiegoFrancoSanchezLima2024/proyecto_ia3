from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.body3d.contracts import SolicitudCuerpo3D
from src.body3d.mesh_measurements import (
    perimetro_seccion_horizontal,
    altura_malla,
    escalar_malla_a_estatura,
)
from src.body3d.readiness import preparacion_cuerpo3d


def _box_mesh(width=2.0, height=4.0, depth=1.0):
    vertices = np.asarray(
        [
            [-width / 2, 0, -depth / 2], [width / 2, 0, -depth / 2],
            [width / 2, 0, depth / 2], [-width / 2, 0, depth / 2],
            [-width / 2, height, -depth / 2], [width / 2, height, -depth / 2],
            [width / 2, height, depth / 2], [-width / 2, height, depth / 2],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
            [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_mesh_scaling_uses_known_height_as_metric_anchor():
    vertices, _ = _box_mesh()
    scaled, scale = escalar_malla_a_estatura(vertices, 164.0)
    assert scale == pytest.approx(0.41)
    assert altura_malla(scaled) == pytest.approx(1.64)


def test_horizontal_section_recovers_known_box_perimeter():
    vertices, faces = _box_mesh(width=2.0, height=4.0, depth=1.0)
    assert perimetro_seccion_horizontal(vertices, faces, level=2.0) == pytest.approx(6.0)


def test_body3d_request_requires_three_views_and_plausible_metadata():
    request = SolicitudCuerpo3D(
        views={"front": Path("front.jpg")},
        height_cm=164,
        weight_kg=69,
        sex="male",
    )
    with pytest.raises(ValueError, match="left, right"):
        request.validate(require_files=False)


def test_readiness_recognizes_official_nested_asset_layout(tmp_path):
    model_root = tmp_path / "pretrained" / "body3d" / "models" / "smplx" / "models" / "smplx"
    model_root.mkdir(parents=True)
    for filename in ("SMPLX_MALE.npz", "SMPLX_FEMALE.npz", "SMPLX_NEUTRAL.npz"):
        (model_root / filename).touch()
    checkpoint = (
        tmp_path
        / "pretrained"
        / "body3d"
        / "trained_models"
        / "shapy"
        / "SHAPY_A"
        / "checkpoints"
        / "best_checkpoint"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    source = (
        tmp_path
        / "pretrained"
        / "body3d"
        / "shapy-master"
        / "regressor"
        / "human_shape"
        / "__init__.py"
    )
    source.parent.mkdir(parents=True)
    source.touch()

    status = preparacion_cuerpo3d(tmp_path)
    assert status["assets_ready"]
    assert status["assets"]["smplx_model_directory"]["available"]
    assert status["assets"]["shapy_checkpoint"]["available"]
    assert status["assets"]["shapy_source"]["available"]


def test_readiness_does_not_confuse_assets_with_runtime(tmp_path):
    status = preparacion_cuerpo3d(tmp_path)
    assert not status["assets_ready"]
    assert not status["ready"]
    assert "smplx_model_directory" in status["missing"]
