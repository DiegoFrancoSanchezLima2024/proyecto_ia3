"""Formato de cámara y lectura consistente de fotografías originales."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/calibration_station.json"


def leer_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path):
    # Los píxeles de puntos manuales y los intrínsecos usan esta misma orientación.
    with Image.open(path) as image:
        return cv2.cvtColor(np.array(ImageOps.exif_transpose(image).convert("RGB")), cv2.COLOR_RGB2BGR)


def dictionary(config):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"]))


def tablero_charuco(config):
    board = config["charuco"]
    return cv2.aruco.CharucoBoard(
        (board["squares_x"], board["squares_y"]), board["square_m"],
        board["marker_m"], dictionary(config),
    )


def matrices_camara(camera, image_size=None):
    matrix = np.asarray(camera["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(camera["dist_coeffs"], dtype=np.float64).reshape(-1)
    size = camera["image_size"]
    if (len(size) != 2 or any(not isinstance(n, int) or n <= 0 for n in size)
            or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()):
        raise ValueError("Perfil de cámara inválido")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)
            or not 0 <= matrix[0, 2] < size[0] or not 0 <= matrix[1, 2] < size[1]):
        raise ValueError("Intrínsecos de cámara inválidos")
    if image_size is not None and tuple(size) != tuple(image_size):
        raise ValueError(f"Resolución incompatible: perfil {size}, foto {list(image_size)}. No redimensionar.")
    return matrix, distortion


def reprojection_errors(object_points, image_points, rvec, tvec, matrix, distortion):
    projected = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - np.asarray(image_points).reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), errors
