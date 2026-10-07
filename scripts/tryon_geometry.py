"""Geometria auditable para la ruta experimental de vestidor de SATRE-IA.

No altera las medidas ni los moldes. Convierte la silueta SAM y los landmarks
frontales en mascaras de prenda y preajusta activos de catalogo antes de VTON.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt


def resize_and_crop_local(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    target_width, target_height = size
    if width / height < target_width / target_height:
        new_width = width
        new_height = width * target_height // target_width
    else:
        new_height = height
        new_width = height * target_width // target_height
    box = (
        (width - new_width) // 2,
        (height - new_height) // 2,
        (width + new_width) // 2,
        (height + new_height) // 2,
    )
    return image.crop(box).resize(size, Image.Resampling.LANCZOS)


def landmarks_after_crop(
    landmarks: list[dict], original_size: tuple[int, int], target_size: tuple[int, int]
) -> list[dict]:
    width, height = original_size
    target_width, target_height = target_size
    if width / height < target_width / target_height:
        crop_width = width
        crop_height = width * target_height // target_width
    else:
        crop_height = height
        crop_width = height * target_width // target_height
    x0, y0 = (width - crop_width) / 2, (height - crop_height) / 2
    transformed = []
    for landmark in landmarks:
        transformed.append(
            {
                **landmark,
                "x": (float(landmark["x"]) * width - x0) / crop_width,
                "y": (float(landmark["y"]) * height - y0) / crop_height,
            }
        )
    return transformed


def point(landmarks: list[dict], index: int, size: tuple[int, int]) -> tuple[float, float]:
    width, height = size
    return float(landmarks[index]["x"]) * width, float(landmarks[index]["y"]) * height


def _ordered_pair(first: tuple[float, float], second: tuple[float, float]):
    return (first, second) if first[0] <= second[0] else (second, first)


def _odd(value: int) -> int:
    value = max(3, value)
    return value if value % 2 else value + 1


def build_parsed_region_masks(
    size: tuple[int, int],
    landmarks: list[dict],
    sex: str,
    person_silhouette: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """Crea regiones por pose y las recorta contra la silueta SAM real.

    Esto no pretende ser un parser semantico de prendas. Es un parser regional
    determinista: pose para partes corporales + SAM para el contorno de persona.
    """

    width, height = size
    left_shoulder, right_shoulder = _ordered_pair(point(landmarks, 11, size), point(landmarks, 12, size))
    elbows = [point(landmarks, 13, size), point(landmarks, 14, size)]
    wrists = [point(landmarks, 15, size), point(landmarks, 16, size)]
    left_elbow, right_elbow = _ordered_pair(*elbows)
    left_wrist, right_wrist = _ordered_pair(*wrists)
    left_hip, right_hip = _ordered_pair(point(landmarks, 23, size), point(landmarks, 24, size))
    left_knee, right_knee = _ordered_pair(point(landmarks, 25, size), point(landmarks, 26, size))
    left_ankle, right_ankle = _ordered_pair(point(landmarks, 27, size), point(landmarks, 28, size))
    shoulder_width = max(right_shoulder[0] - left_shoulder[0], width * 0.12)
    torso_height = max(((left_hip[1] + right_hip[1]) - (left_shoulder[1] + right_shoulder[1])) / 2, height * 0.12)

    upper = Image.new("L", size, 0)
    draw = ImageDraw.Draw(upper)
    top_left = (left_shoulder[0] - shoulder_width * 0.12, left_shoulder[1] - height * 0.008)
    top_right = (right_shoulder[0] + shoulder_width * 0.12, right_shoulder[1] - height * 0.008)
    jacket_bottom = max(left_hip[1], right_hip[1]) + torso_height * 0.18
    draw.polygon(
        [
            top_left,
            top_right,
            (right_hip[0] + shoulder_width * 0.24, jacket_bottom),
            (left_hip[0] - shoulder_width * 0.24, jacket_bottom),
        ],
        fill=255,
    )
    # La región debe cubrir el volumen exterior de una manga de saco, no solo
    # el grosor del brazo desnudo. El sprite decide después el contorno final.
    sleeve_width = max(round(shoulder_width * 0.37), round(width * 0.048))
    draw.line([left_shoulder, left_elbow, left_wrist], fill=255, width=sleeve_width, joint="curve")
    draw.line([right_shoulder, right_elbow, right_wrist], fill=255, width=sleeve_width, joint="curve")
    # No perforar la máscara en la muñeca: la capa semántica de manos se
    # recompone al final. La perforación anterior cortaba el puño 15–23 px
    # antes de la muñeca y creaba una manga artificialmente corta.

    lower = Image.new("L", size, 0)
    draw = ImageDraw.Draw(lower)
    waist_y = min(left_hip[1], right_hip[1]) - torso_height * 0.08
    hip_left = left_hip[0] - shoulder_width * 0.16
    hip_right = right_hip[0] + shoulder_width * 0.16
    if sex == "female":
        knee_y = (left_knee[1] + right_knee[1]) / 2
        draw.polygon(
            [
                (hip_left, waist_y),
                (hip_right, waist_y),
                (right_knee[0] + shoulder_width * 0.18, knee_y),
                (left_knee[0] - shoulder_width * 0.18, knee_y),
            ],
            fill=255,
        )
    else:
        crotch_y = max(left_hip[1], right_hip[1]) + torso_height * 0.28
        draw.polygon(
            [(hip_left, waist_y), (hip_right, waist_y), (right_hip[0], crotch_y), (left_hip[0], crotch_y)],
            fill=255,
        )
        leg_width = max(round(shoulder_width * 0.34), round(width * 0.05))
        draw.line([left_hip, left_knee, left_ankle], fill=255, width=leg_width, joint="curve")
        draw.line([right_hip, right_knee, right_ankle], fill=255, width=leg_width, joint="curve")

    silhouette = person_silhouette.convert("L")
    if silhouette.size != size:
        silhouette = resize_and_crop_local(silhouette, size)
    silhouette = silhouette.point(lambda value: 255 if value >= 96 else 0)
    # Un saco estructurado puede sobresalir hasta 8 % del ancho de hombros por
    # lado. MaxFilter usa un diametro, por eso se convierte radio -> tamano.
    structural_radius = max(2, round(shoulder_width * 0.08))
    silhouette = silhouette.filter(ImageFilter.MaxFilter(_odd(structural_radius * 2 + 1)))
    upper = ImageChops.multiply(upper, silhouette)
    lower = ImageChops.multiply(lower, silhouette)
    upper = upper.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    lower = lower.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return upper, lower


def _rgba_from_catalog(asset: Image.Image) -> np.ndarray:
    if asset.mode == "RGBA":
        rgba = np.asarray(asset, dtype=np.uint8).copy()
        if np.any(rgba[:, :, 3] < 250):
            alpha = rgba[:, :, 3]
            foreground = (alpha > 0).astype(np.uint8)
            inward_distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 3)
            lightness = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB)[:, :, 0]
            core = np.logical_and(alpha >= 245, inward_distance > 12.0)
            if core.any():
                # El recorte de catálogo dejó una línea blanca de 1–4 px. Solo
                # se elimina brillo anómalo conectado al borde; una prenda
                # realmente clara conserva su borde porque su núcleo también
                # tendrá L alta y elevará este umbral adaptativo.
                bright_limit = min(248.0, float(np.percentile(lightness[core], 95)) + 30.0)
                contaminated = np.logical_and(inward_distance <= 12.0, lightness > bright_limit)
                alpha[contaminated] = 0
            # Alpha bleeding: el activo original contiene blanco incluso en
            # pixeles de borde casi transparentes. Sustituimos solo su RGB por
            # el color opaco mas cercano y conservamos exactamente su alpha.
            # Así el remuestreo no mezcla la prenda con un matte blanco.
            opaque = alpha >= 245
            partial = np.logical_and(alpha > 0, alpha < 245)
            if partial.any() and opaque.any():
                _, nearest = distance_transform_edt(~opaque, return_indices=True)
                rgba[:, :, :3][partial] = rgba[nearest[0][partial], nearest[1][partial], :3]
            return rgba
    rgb = np.asarray(asset.convert("RGB"), dtype=np.uint8)
    distance_from_white = np.max(255 - rgb, axis=2).astype(np.float32)
    alpha = np.clip((distance_from_white - 3.0) * 18.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return rgba


def _foreground_bbox(rgba: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(rgba[:, :, 3] > 16)
    if not len(xs):
        raise ValueError("El activo de prenda no contiene un primer plano detectable.")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _tight_foreground_box(rgba: np.ndarray, bounds) -> tuple[int, int, int, int]:
    """Recorta el espacio transparente interno de una pieza del catálogo."""
    x0, y0, x1, y1 = [int(round(value)) for value in bounds]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rgba.shape[1], x1), min(rgba.shape[0], y1)
    ys, xs = np.where(rgba[y0:y1, x0:x1, 3] > 16)
    if not len(xs):
        return x0, y0, x1, y1
    return x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max() + 1), y0 + int(ys.max() + 1)


def _paste_patch(
    canvas: np.ndarray,
    source: np.ndarray,
    source_box,
    destination_quad,
    feather_px: float = 0.0,
) -> None:
    x0, y0, x1, y1 = [int(round(value)) for value in source_box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(source.shape[1], x1), min(source.shape[0], y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    patch = source[y0:y1, x0:x1]
    src = np.float32([[0, 0], [patch.shape[1] - 1, 0], [patch.shape[1] - 1, patch.shape[0] - 1], [0, patch.shape[0] - 1]])
    dst = np.float32(destination_quad)
    matrix = cv2.getPerspectiveTransform(src, dst)
    patch_float = patch.astype(np.float32)
    patch_alpha = patch_float[:, :, 3] / 255.0
    patch_premultiplied = patch_float[:, :, :3] * patch_alpha[:, :, None]
    warped_rgb_premultiplied = cv2.warpPerspective(
        patch_premultiplied,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpPerspective(
        patch_alpha,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
    if feather_px > 0:
        warped_alpha = cv2.GaussianBlur(warped_alpha, (0, 0), feather_px)
        warped_rgb_premultiplied = cv2.GaussianBlur(
            warped_rgb_premultiplied, (0, 0), feather_px
        )
    src_alpha = warped_alpha[:, :, None]
    dst_alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
    dst_premultiplied = canvas[:, :, :3].astype(np.float32) * dst_alpha
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    out_premultiplied = warped_rgb_premultiplied + dst_premultiplied * (1.0 - src_alpha)
    out_rgb = np.divide(
        out_premultiplied,
        np.maximum(out_alpha, 1e-6),
        out=np.zeros_like(out_premultiplied),
        where=out_alpha > 1e-6,
    )
    canvas[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    canvas[:, :, 3:4] = np.clip(out_alpha * 255.0, 0, 255).astype(np.uint8)


def _extended_segment(start, end, start_px: float = 0.0, end_px: float = 0.0):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length
    return (
        (start[0] - ux * start_px, start[1] - uy * start_px),
        (end[0] + ux * end_px, end[1] + uy * end_px),
    )


def _limb_path_point(shoulder, elbow, wrist, t: float):
    """Interpola una manga continua que pasa exactamente por el codo."""
    if t <= 0.5:
        local = t * 2.0
        return (
            shoulder[0] * (1.0 - local) + elbow[0] * local,
            shoulder[1] * (1.0 - local) + elbow[1] * local,
        )
    local = (t - 0.5) * 2.0
    return (
        elbow[0] * (1.0 - local) + wrist[0] * local,
        elbow[1] * (1.0 - local) + wrist[1] * local,
    )


def _segment_quad(start, end, start_width: float, end_width: float):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(math.hypot(dx, dy), 1.0)
    nx, ny = -dy / length, dx / length
    return [
        (start[0] - nx * start_width / 2, start[1] - ny * start_width / 2),
        (start[0] + nx * start_width / 2, start[1] + ny * start_width / 2),
        (end[0] + nx * end_width / 2, end[1] + ny * end_width / 2),
        (end[0] - nx * end_width / 2, end[1] - ny * end_width / 2),
    ]


def _canvas(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    result = np.full((height, width, 4), 255, dtype=np.uint8)
    result[:, :, 3] = 0
    return result


def _finish(canvas: np.ndarray, transparent: bool = False) -> Image.Image:
    if transparent:
        return Image.fromarray(canvas, "RGBA")
    alpha = canvas[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(canvas[:, :, :3], 255)
    rgb = (canvas[:, :, :3] * alpha + white * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def prewarp_jacket(
    asset: Image.Image,
    size: tuple[int, int],
    landmarks: list[dict],
    jacket_bottom_hint: float | None = None,
    transparent: bool = False,
) -> Image.Image:
    rgba = _rgba_from_catalog(asset)
    x0, y0, x1, y1 = _foreground_bbox(rgba)
    garment_width, garment_height = x1 - x0, y1 - y0
    left_shoulder, right_shoulder = _ordered_pair(point(landmarks, 11, size), point(landmarks, 12, size))
    left_elbow, right_elbow = _ordered_pair(point(landmarks, 13, size), point(landmarks, 14, size))
    left_wrist, right_wrist = _ordered_pair(point(landmarks, 15, size), point(landmarks, 16, size))
    left_hip, right_hip = _ordered_pair(point(landmarks, 23, size), point(landmarks, 24, size))
    shoulder_width = max(right_shoulder[0] - left_shoulder[0], size[0] * 0.12)
    torso_height = max(((left_hip[1] + right_hip[1]) - (left_shoulder[1] + right_shoulder[1])) / 2, size[1] * 0.12)
    pose_bottom = max(left_hip[1], right_hip[1]) + torso_height * 0.18
    if jacket_bottom_hint is None:
        bottom_y = pose_bottom
    else:
        # La medida no puede desplazar el borde mas de 12 % del torso: la pose
        # sigue siendo el ancla principal y la metrica solo regulariza el largo.
        hint = min(max(jacket_bottom_hint, pose_bottom - torso_height * 0.12), pose_bottom + torso_height * 0.12)
        bottom_y = pose_bottom * 0.7 + hint * 0.3
    canvas = _canvas(size)
    body_box = (x0 + garment_width * 0.20, y0, x0 + garment_width * 0.80, y1)
    body_quad = [
        (left_shoulder[0] - shoulder_width * 0.10, left_shoulder[1]),
        (right_shoulder[0] + shoulder_width * 0.10, right_shoulder[1]),
        (right_hip[0] + shoulder_width * 0.24, bottom_y),
        (left_hip[0] - shoulder_width * 0.24, bottom_y),
    ]
    _paste_patch(canvas, rgba, body_box, body_quad)

    sleeves = [
        ((x0, x0 + garment_width * 0.31), left_shoulder, left_elbow, left_wrist),
        ((x0 + garment_width * 0.69, x1), right_shoulder, right_elbow, right_wrist),
    ]
    # Seis tiras conservan continuidad visual y aproximan la curva del brazo
    # mucho mejor que dos rectángulos rígidos unidos justo en el codo.
    sleeve_segments = 6
    source_top = y0 + garment_height * 0.10
    source_bottom = y0 + garment_height * 0.92
    for (source_left, source_right), shoulder, elbow, wrist in sleeves:
        for segment in range(sleeve_segments):
            t0 = segment / sleeve_segments
            t1 = (segment + 1) / sleeve_segments
            sy0 = source_top + (source_bottom - source_top) * t0
            sy1 = source_top + (source_bottom - source_top) * t1
            start = _limb_path_point(shoulder, elbow, wrist, t0)
            end = _limb_path_point(shoulder, elbow, wrist, t1)
            overlap = shoulder_width * 0.018
            start, end = _extended_segment(
                start, end, 0.0 if segment == 0 else overlap, 0.0 if segment == sleeve_segments - 1 else overlap
            )
            start_width = shoulder_width * (0.34 * (1.0 - t0) + 0.24 * t0)
            end_width = shoulder_width * (0.34 * (1.0 - t1) + 0.24 * t1)
            source_box = _tight_foreground_box(rgba, (source_left, sy0, source_right, sy1))
            _paste_patch(
                canvas,
                rgba,
                source_box,
                _segment_quad(start, end, start_width, end_width),
                feather_px=0.55,
            )
    return _finish(canvas, transparent=transparent)


def prewarp_lower(
    asset: Image.Image,
    size: tuple[int, int],
    landmarks: list[dict],
    sex: str,
    transparent: bool = False,
) -> Image.Image:
    rgba = _rgba_from_catalog(asset)
    x0, y0, x1, y1 = _foreground_bbox(rgba)
    garment_width, garment_height = x1 - x0, y1 - y0
    left_shoulder, right_shoulder = _ordered_pair(point(landmarks, 11, size), point(landmarks, 12, size))
    left_hip, right_hip = _ordered_pair(point(landmarks, 23, size), point(landmarks, 24, size))
    left_knee, right_knee = _ordered_pair(point(landmarks, 25, size), point(landmarks, 26, size))
    left_ankle, right_ankle = _ordered_pair(point(landmarks, 27, size), point(landmarks, 28, size))
    shoulder_width = right_shoulder[0] - left_shoulder[0]
    # MediaPipe ubica las articulaciones de cadera dentro del contorno corporal.
    # El minimo relativo al hombro evita pantalones artificialmente filiformes.
    hip_width = max(right_hip[0] - left_hip[0], shoulder_width * 0.82, size[0] * 0.09)
    canvas = _canvas(size)
    if sex == "female":
        knee_y = (left_knee[1] + right_knee[1]) / 2
        _paste_patch(
            canvas,
            rgba,
            (x0, y0, x1, y1),
            [
                (left_hip[0] - hip_width * 0.18, min(left_hip[1], right_hip[1])),
                (right_hip[0] + hip_width * 0.18, min(left_hip[1], right_hip[1])),
                (right_knee[0] + hip_width * 0.25, knee_y),
                (left_knee[0] - hip_width * 0.25, knee_y),
            ],
        )
    else:
        halves = [
            (x0, x0 + garment_width * 0.52, left_hip, left_knee, left_ankle),
            (x0 + garment_width * 0.48, x1, right_hip, right_knee, right_ankle),
        ]
        for sx0, sx1, hip, knee, ankle in halves:
            _paste_patch(
                canvas,
                rgba,
                (sx0, y0, sx1, y0 + garment_height * 0.58),
                _segment_quad(hip, knee, hip_width * 0.88, hip_width * 0.56),
            )
            _paste_patch(
                canvas,
                rgba,
                (sx0, y0 + garment_height * 0.42, sx1, y1),
                _segment_quad(knee, ankle, hip_width * 0.56, hip_width * 0.40),
            )
    return _finish(canvas, transparent=transparent)


def combine_catalog_conditions(upper: Image.Image, lower: Image.Image) -> Image.Image:
    upper_rgb, lower_rgb = np.asarray(upper.convert("RGB")), np.asarray(lower.convert("RGB"))
    lower_mask = np.max(255 - lower_rgb, axis=2) > 8
    result = upper_rgb.copy()
    result[lower_mask] = lower_rgb[lower_mask]
    return Image.fromarray(result, "RGB")


def load_person_silhouette(mask_path: str | Path, size: tuple[int, int]) -> Image.Image:
    return resize_and_crop_local(Image.open(mask_path).convert("L"), size)
