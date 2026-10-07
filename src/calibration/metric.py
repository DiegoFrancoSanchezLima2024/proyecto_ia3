"""Pose métrica del suelo y diagnóstico de segmentos estrictamente verticales.

No equivale a estimar estatura humana: corona y proyección de corona al suelo
requieren un detector y un modelo de postura, todavía no integrados aquí.
"""
from __future__ import annotations

import cv2
import numpy as np

from .common import matrices_camara, dictionary, reprojection_errors


def esquinas_objeto_piso(config):
    half = config["floor"]["marker_m"] / 2
    if half <= 0:
        raise ValueError("El lado de marcador debe ser positivo")
    result = {}
    for marker in config["floor"]["markers"]:
        x, y = marker["center_m"]
        yaw = np.radians(float(marker.get("yaw_degrees", 0.0)))
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]], dtype=np.float64)
        # TL, TR, BR, BL del patrón impreso; borde superior orientado a +Y.
        if marker["id"] in result:
            raise ValueError("IDs de suelo duplicados")
        local = np.array([[-half, half], [half, half], [half, -half], [-half, -half]], dtype=np.float64)
        xy = local @ rotation.T + [x, y]
        result[marker["id"]] = np.c_[xy, np.zeros(4)]
    return result


def resolver_pose_piso(objects, pixels, camera, config):
    matrix, distortion = matrices_camara(camera)
    objects = np.asarray(objects, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if (len(objects) != len(pixels) or len(objects) < 12 or not np.isfinite(objects).all()
            or not np.isfinite(pixels).all() or not np.allclose(objects[:, 2], 0)):
        raise ValueError("Correspondencias de suelo inválidas")
    solutions = cv2.solvePnPGeneric(objects, pixels, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rvec, tvec in zip(solutions[1], solutions[2]):
        rotation = cv2.Rodrigues(rvec)[0]
        center = (-rotation.T @ tvec).ravel()
        depth = (rotation @ objects.T + tvec)[2]
        if center[2] <= 0 or np.any(depth <= 0):
            continue
        rms, _ = reprojection_errors(objects, pixels, rvec, tvec, matrix, distortion)
        candidates.append((rms, rvec, tvec, center))
    if not candidates:
        raise ValueError("Pose imposible: cámara bajo el suelo o marcadores detrás de cámara")
    candidates.sort(key=lambda value: value[0])
    if len(candidates) > 1:
        delta = candidates[1][0] - candidates[0][0]
        center_distance = np.linalg.norm(candidates[1][3] - candidates[0][3])
        if delta < 0.15 and center_distance > 0.05:
            raise ValueError("Pose plana ambigua: ampliar marcadores en imagen o cambiar ligeramente el ángulo")
    _, rvec, tvec, _ = candidates[0]
    rvec, tvec = cv2.solvePnPRefineLM(objects, pixels, matrix, distortion, rvec.copy(), tvec.copy())
    rotation = cv2.Rodrigues(rvec)[0]
    center = (-rotation.T @ tvec).ravel()
    rms, errors = reprojection_errors(objects, pixels, rvec, tvec, matrix, distortion)
    if center[2] <= 0 or np.any((rotation @ objects.T + tvec)[2] <= 0):
        raise ValueError("Pose refinada físicamente inválida")
    if (rms > config["quality"]["max_floor_rms_px"]
            or errors.max() > config["quality"]["max_floor_corner_error_px"]):
        raise ValueError(f"Montaje/detección inconsistente: RMS={rms:.2f}px, máximo={errors.max():.2f}px")
    return {"rotation_world_to_camera": rotation.tolist(), "translation_world_to_camera_m": tvec.ravel().tolist(),
            "camera_center_world_m": center.tolist(), "rms_px": rms, "max_corner_error_px": float(errors.max()),
            "metric_accuracy_validated": False}


def _floor_detections(image, config):
    """Lectura de marcadores sin asumir una cámara calibrada ni escala válida."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary(config), params).detectMarkers(image)
    known = esquinas_objeto_piso(config)
    selected = []
    if ids is not None:
        selected = [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners) if int(i) in known]
    return selected, [] if ids is None else [int(i) for i in ids.ravel()]


def inspeccionar_marcadores_piso(image, config):
    """Preflight de visibilidad; NO estima pose, distancias o precisión métrica."""
    selected, all_ids = _floor_detections(image, config)
    expected = sorted(esquinas_objeto_piso(config))
    present = [i for i, _ in selected]
    duplicates = sorted({i for i in present if present.count(i) > 1})
    minimum = min((float(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1).min())
                   for _, p in selected), default=None)
    warnings = []
    if duplicates:
        warnings.append("IDs repetidos: usar una sola copia de cada marcador")
    if len(set(present)) < config["quality"]["min_floor_markers"]:
        warnings.append("Se requieren al menos 3 marcadores de suelo; intentar los 4")
    if minimum is not None and minimum < config["quality"]["min_marker_edge_px"]:
        warnings.append("Marcadores pequeños/achatados: ajustar encuadre o captura")
    return {"status": "needs_capture_adjustment" if warnings else "marker_visibility_passed",
            "scope": "visibility_only_not_metric_pose", "warnings": warnings,
            "marker_ids": sorted(set(present)), "missing_ids": sorted(set(expected) - set(present)),
            "duplicate_ids": duplicates, "other_marker_ids": sorted(set(all_ids) - set(expected)),
            "min_marker_edge_px": minimum,
            "image_size": [image.shape[1], image.shape[0]],
            "physical_layout_verified": False, "metric_accuracy_validated": False}


def detectar_piso(image, camera, config):
    matrices_camara(camera, (image.shape[1], image.shape[0]))
    if camera.get("status") != "passed_reprojection_checks":
        raise ValueError("El perfil no superó los controles de calibración")
    selected, _ = _floor_detections(image, config)
    known = esquinas_objeto_piso(config)
    selected_ids = [i for i, _ in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("ID repetido en foto; usar una sola copia de cada marcador")
    if len(selected) < config["quality"]["min_floor_markers"]:
        raise ValueError(f"Sólo {len(selected)} marcadores de suelo detectados. Se requieren al menos 3; intentar los 4.")
    min_edge = min(float(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1).min()) for _, p in selected)
    if min_edge < config["quality"]["min_marker_edge_px"]:
        raise ValueError(f"Marcador pequeño/achatado ({min_edge:.1f}px). Usar original de mayor resolución o ajustar captura.")
    objects = np.concatenate([known[i] for i, _ in selected])
    pixels = np.concatenate([p for _, p in selected])
    result = resolver_pose_piso(objects, pixels, camera, config)
    result.update({"marker_ids": selected_ids, "min_marker_edge_px": min_edge})
    return result


def altura_segmento_vertical(base_pixel, top_pixel, camera, pose):
    """Altura sin introducir longitud real, sólo para una vertical con base en Z=0.

    Marcar un tobillo y cabello humano NO satisface necesariamente este supuesto.
    """
    matrix, distortion = matrices_camara(camera)
    pixels = np.asarray([base_pixel, top_pixel], dtype=np.float64)
    width, height = camera["image_size"]
    if (pixels.shape != (2, 2) or not np.isfinite(pixels).all()
            or np.any(pixels < 0) or np.any(pixels[:, 0] >= width) or np.any(pixels[:, 1] >= height)):
        raise ValueError("Puntos fuera de la imagen original orientada")
    rotation = np.asarray(pose["rotation_world_to_camera"], dtype=float)
    translation = np.asarray(pose["translation_world_to_camera_m"], dtype=float).reshape(3)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError("Transformación de cámara inválida")
    center = -rotation.T @ translation
    if center[2] <= 0:
        raise ValueError("Cámara no está sobre el suelo")
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), matrix, distortion).reshape(2, 2)
    rays = np.c_[normalized, np.ones(2)] @ rotation
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    base_ray, top_ray = rays
    if base_ray[2] >= -1e-5:
        raise ValueError("La base no intersecta el suelo delante de cámara")
    base = center + (-center[2] / base_ray[2]) * base_ray
    system = np.column_stack((top_ray, -np.array([0., 0., 1.])))
    if np.linalg.cond(system) > 1000:
        raise ValueError("Geometría degenerada")
    (distance, estimate), _, _, _ = np.linalg.lstsq(system, base - center, rcond=None)
    residual = float(np.linalg.norm(center + distance * top_ray - (base + [0, 0, estimate])))
    if distance <= 0 or estimate <= 0 or residual > 0.02:
        raise ValueError("Puntos incompatibles con un segmento vertical sobre el suelo")
    return {"height_m": float(estimate), "height_cm": float(100 * estimate), "base_world_m": base.tolist(),
            "ray_vertical_residual_m": residual, "endpoint_source": "manual_pixels",
            "scope": "vertical_segment_diagnostic_not_automatic_body_height", "metric_accuracy_validated": False}
