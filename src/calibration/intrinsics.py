"""ChArUco con validación en fotos separadas y trazabilidad de entradas."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .common import matrices_camara, tablero_charuco, file_hash, read_image, reprojection_errors


def recolectar_vistas(folder, config, expected_size=None, seen=None):
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise ValueError(f"No hay fotos JPG/PNG en {folder}")
    detector = cv2.aruco.CharucoDetector(tablero_charuco(config))
    board = tablero_charuco(config)
    accepted, rejected = [], []
    seen = set() if seen is None else seen
    for path in files:
        image = read_image(path)
        size = (image.shape[1], image.shape[0])
        expected_size = size if expected_size is None else expected_size
        if size != expected_size:
            raise ValueError(f"Resoluciones mezcladas: {path.name}: {size}, esperado {expected_size}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"Foto duplicada (también se comprueba entre calibración y validación): {path.name}")
        seen.add(digest)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        count = 0 if ids is None else len(ids)
        if count < config["quality"]["min_charuco_corners"] or board.checkCharucoCornersCollinear(ids):
            rejected.append({"file": str(path.resolve()), "reason": "pocas esquinas o colineales", "corners": count})
            continue
        objects, pixels = board.matchImagePoints(corners, ids)
        accepted.append({"object_points": objects, "image_points": pixels,
                         "file": str(path.resolve()), "sha256": file_hash(path),
                         "corners": count, "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
    return accepted, rejected, expected_size, seen


def ajustar_camara(train, validation, size, config, camera_id):
    quality = config["quality"]
    if len(train) < quality["min_train_views"] or len(validation) < quality["min_validation_views"]:
        raise ValueError(f"Fotos válidas insuficientes: calibración={len(train)}, validación={len(validation)}. "
                         f"Mínimos {quality['min_train_views']} y {quality['min_validation_views']}; recomendado 20 y 6.")
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [v["object_points"] for v in train], [v["image_points"] for v in train], size, None, None,
    )
    profile = {"schema_version": 1, "camera_id": camera_id, "opencv_version": cv2.__version__,
               "image_size": list(size), "camera_matrix": matrix.tolist(), "dist_coeffs": distortion.ravel().tolist(),
               "orientation": "PIL.ImageOps.exif_transpose", "train_rms_px": float(rms),
               "metric_accuracy_validated": False, "board": config["charuco"], "dictionary": config["dictionary"]}
    matrices_camara(profile)
    reports = {}
    for split, views in (("calibration", train), ("validation", validation)):
        rows = []
        for i, view in enumerate(views):
            if split == "calibration":
                rvec, tvec = rvecs[i], tvecs[i]
            else:
                # Sólo ajusta pose en fotos nuevas; NO vuelve a estimar K ni distorsión.
                ok, rvec, tvec = cv2.solvePnP(view["object_points"], view["image_points"], matrix, distortion)
                if not ok:
                    raise ValueError("No se pudo calcular pose en validación")
            error, _ = reprojection_errors(view["object_points"], view["image_points"], rvec, tvec, matrix, distortion)
            rows.append({k: v for k, v in view.items() if k not in ("object_points", "image_points")} | {"rms_px": error})
        reports[split] = rows
    warnings = []
    if rms > quality["max_train_rms_px"]:
        warnings.append("Error de reproyección alto en calibración")
    if max(row["rms_px"] for row in reports["validation"]) > quality["max_validation_view_rms_px"]:
        warnings.append("Error alto en una o más fotos de validación")
    pixels = np.concatenate([v["image_points"].reshape(-1, 2) for v in train])
    span = np.ptp(pixels, axis=0) / np.asarray(size)
    if np.any(span < 0.5):
        warnings.append("Cobertura insuficiente: mover tablero hacia bordes y esquinas del encuadre")
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rvecs])
    diversity = float(np.degrees(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    if diversity < 15:
        warnings.append("Poca variedad de inclinaciones del tablero; repetir con distintas orientaciones")
    profile.update({"status": "needs_recapture" if warnings else "passed_reprojection_checks",
                    "warnings": warnings, "coverage_xy_fraction": span.tolist(),
                    "normal_diversity_degrees": diversity, "views": reports,
                    "quality_thresholds": quality,
                    "limitations": ["Umbrales iniciales de ingeniería, no certifican centímetros",
                                    "Misma lente, zoom, resolución, orientación y procesamiento en futuras fotos",
                                    "La escala física impresa y el montaje requieren verificación con regla"]})
    return profile


def calibrate(train_dir, validation_dir, config, camera_id):
    if Path(train_dir).resolve() == Path(validation_dir).resolve():
        raise ValueError("Calibración y validación necesitan carpetas de fotos distintas")
    train, rejected_train, size, seen = recolectar_vistas(train_dir, config)
    validation, rejected_validation, _, _ = recolectar_vistas(validation_dir, config, size, seen)
    profile = ajustar_camara(train, validation, size, config, camera_id)
    profile["rejected_images"] = rejected_train + rejected_validation
    return profile
