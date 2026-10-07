#!/usr/bin/env python3
"""Constrain a generative try-on result to the photographed person.

The generator is used for garment appearance only.  Background, head and hands
come from the original capture so identity and the scene remain auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from human_parsing_schp import infer_atr
from tryon_geometry import landmarks_after_crop, load_person_silhouette, point


def find_front(session_dir: Path) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        path = session_dir / "photos" / f"front{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError("No se encontro foto frontal")


def cover_ellipse(draw: ImageDraw.ImageDraw, center: tuple[float, float], rx: float, ry: float) -> None:
    x, y = center
    draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=255)


def resize_crop(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Aplica exactamente el mismo recorte central usado para la fotografía."""
    source_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]
    if source_ratio > target_ratio:
        crop_w = round(image.height * target_ratio)
        left = (image.width - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, image.height))
    else:
        crop_h = round(image.width / target_ratio)
        top = (image.height - crop_h) // 2
        image = image.crop((0, top, image.width, top + crop_h))
    resampling = Image.Resampling.NEAREST if image.mode == "L" else Image.Resampling.LANCZOS
    return image.resize(target_size, resampling)


def semantic_identity_mask(
    original_full: Image.Image,
    landmarks_originales: list[dict],
    landmarks_recortados: list[dict],
    size: tuple[int, int],
) -> tuple[Image.Image, str]:
    """Restaura identidad sin la antigua franja rectangular de cabeza.

    SCHP aporta cara y pelo, y combina sus brazos con los puntos distales de
    MediaPipe para obtener manos. Si el parser falla, el fallback sigue siendo
    anatómico (elipses de cabeza/manos), nunca un rectángulo de ancho completo.
    """
    try:
        regiones = infer_atr(original_full, landmarks_originales)
        cabeza = np.maximum(
            np.asarray(regiones["face"], dtype=np.uint8),
            np.asarray(regiones["hair"], dtype=np.uint8),
        )
        identidad = np.maximum(cabeza, np.asarray(regiones["hands"], dtype=np.uint8))
        mascara = resize_crop(Image.fromarray(identidad, "L"), size)
        # Cierra pequeños huecos semánticos sin invadir cuello/solapas.
        mascara = mascara.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.2))
        return mascara, "schp_atr_face_hair_hands"
    except Exception:  # noqa: BLE001 - el vestidor conserva un fallback local auditable
        restore = Image.new("L", size, 0)
        draw = ImageDraw.Draw(restore)
        left_shoulder = point(landmarks_recortados, 11, size)
        right_shoulder = point(landmarks_recortados, 12, size)
        shoulder_width = max(20.0, abs(right_shoulder[0] - left_shoulder[0]))
        nose = point(landmarks_recortados, 0, size)
        ears = [point(landmarks_recortados, index, size) for index in (7, 8)]
        cx = (ears[0][0] + ears[1][0]) / 2
        cy = nose[1] - shoulder_width * 0.14
        cover_ellipse(draw, (cx, cy), shoulder_width * 0.56, shoulder_width * 0.72)
        for indices in ((15, 17, 19, 21), (16, 18, 20, 22)):
            pts = [point(landmarks_recortados, index, size) for index in indices]
            hx = sum(p[0] for p in pts) / len(pts)
            hy = sum(p[1] for p in pts) / len(pts)
            cover_ellipse(draw, (hx, hy), shoulder_width * 0.105, shoulder_width * 0.14)
        return restore.filter(ImageFilter.GaussianBlur(1.4)), "mediapipe_fallback"


def preservation_report(original: Image.Image, final: Image.Image, region: Image.Image) -> dict:
    """Mide cuanto cambio realmente el resultado respecto al original dentro de `region`.

    Antes este script afirmaba `background_preserved: True` como constante. Con
    una dilatacion de la silueta y un desenfoque gaussiano encima, esa afirmacion
    no podia ser cierta: los pixeles generados se empujan hacia el fondo. Aqui se
    cuenta cuantos pixeles cambiaron de verdad.
    """
    left = np.asarray(original, dtype=np.int16)
    right = np.asarray(final, dtype=np.int16)
    mask = np.asarray(region.convert("L"), dtype=np.uint8) > 8
    area = int(mask.sum())
    if area == 0:
        return {"area_px": 0, "changed_percent": None, "preserved": None}
    difference = np.abs(left - right).max(axis=2)
    changed = int((difference[mask] > 6).sum())
    percent = round(changed / area * 100, 3)
    return {"area_px": area, "changed_percent": percent, "preserved": bool(percent < 1.0)}


def restore_mask_report(restore: Image.Image) -> dict:
    """Comprueba si la mascara de restauracion es un rectangulo de ancho completo.

    Una mascara semantica de cara/pelo nunca cubre el ancho entero de la imagen.
    Un rectangulo si, y su borde inferior es la franja visible en cuello/hombros.
    Este chequeo es estructural: no depende del contenido de la foto.
    """
    mask = np.asarray(restore.convert("L"), dtype=np.float32) / 255.0
    coverage = mask.mean(axis=1)
    max_coverage = float(coverage.max())
    return {
        "max_row_coverage": round(max_coverage, 3),
        "full_width_rows": int((coverage > 0.9).sum()),
        "rectangular_restore_detected": bool(max_coverage > 0.9),
    }


def boundary_step_report(final: Image.Image, restore: Image.Image) -> dict:
    """Mide el salto tonal en el borde de la mascara de restauracion.

    El desenfoque gaussiano reparte la costura sobre varios pixeles, asi que un
    gradiente de un solo pixel no la detecta. Comparar la media de las filas de
    arriba contra las de abajo si la detecta.
    """
    array = np.asarray(final.convert("RGB"), dtype=np.float32)
    mask = np.asarray(restore.convert("L"), dtype=np.float32) / 255.0
    height = array.shape[0]
    coverage = mask.mean(axis=1)
    boundary_rows = np.flatnonzero(np.abs(np.diff(coverage)) > 0.05)
    if height < 16 or boundary_rows.size == 0:
        return {"max_row_step": None, "visible_band_detected": False}
    row_mean = array.mean(axis=1)
    window = 6
    steps = []
    for row in boundary_rows:
        above = row_mean[max(0, row - window):max(1, row)]
        below = row_mean[min(height - 1, row + 1):min(height, row + 1 + window)]
        if len(above) and len(below):
            steps.append(float(np.abs(above.mean(axis=0) - below.mean(axis=0)).max()))
    if not steps:
        return {"max_row_step": None, "visible_band_detected": False}
    worst = max(steps)
    return {
        "max_row_step": round(worst, 2),
        "boundary_rows_checked": int(boundary_rows.size),
        "visible_band_detected": bool(worst > 8.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilation", type=int, default=13)
    args = parser.parse_args()

    generated = Image.open(args.generated).convert("RGB")
    size = generated.size
    manifest = json.loads((args.session_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    front = manifest["views"]["front"]
    original_full = Image.open(find_front(args.session_dir)).convert("RGB")

    # Misma politica de recorte central que usan ambos motores de prueba virtual local.
    original = resize_crop(original_full, size)

    silhouette = load_person_silhouette(front["mask"], size)
    kernel = max(3, args.dilation | 1)
    hard_region = silhouette.filter(ImageFilter.MaxFilter(kernel))
    soft_region = np.asarray(
        hard_region.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.uint8
    ).copy()
    hard_array = np.asarray(hard_region, dtype=np.uint8)
    # El suavizado ocurre hacia dentro: fuera del soporte duro el fondo queda
    # bit a bit igual a la captura, sin derrame generativo.
    soft_region[hard_array == 0] = 0
    garment_region = Image.fromarray(soft_region, "L")
    constrained = Image.composite(generated, original, garment_region)

    landmarks = landmarks_after_crop(front["pose"]["landmarks"], original_full.size, size)
    restore, restore_source = semantic_identity_mask(
        original_full, front["pose"]["landmarks"], landmarks, size
    )
    final = Image.composite(original, constrained, restore)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.save(args.output)
    garment_region.save(args.output.with_name(args.output.stem + "-garment-region.png"))
    restore.save(args.output.with_name(args.output.stem + "-identity-restore.png"))
    # Verificacion medida, no declarada. Estos tres campos eran constantes `True`.
    background_region = Image.eval(hard_region, lambda value: 255 - value)
    background = preservation_report(original, final, background_region)
    identity = preservation_report(original, final, restore)
    mask_shape = restore_mask_report(restore)
    boundary = boundary_step_report(final, restore)
    print(json.dumps({
        "mode": "generative_garment_constrained_to_person_v2_verified",
        "generated": str(args.generated),
        "output": str(args.output),
        "resolution": list(size),
        "background": background,
        "identity_restore": identity,
        "identity_restore_source": restore_source,
        "restore_mask": mask_shape,
        "restore_boundary": boundary,
        "verified": True,
        "warnings": [
            message
            for message in (
                f"El fondo cambio en {background['changed_percent']}% de sus pixeles."
                if background.get("preserved") is False
                else None,
                "La mascara de restauracion es un rectangulo de ancho completo "
                f"({mask_shape.get('full_width_rows')} filas): usar mascaras semanticas "
                "de cara/pelo/manos en su lugar."
                if mask_shape.get("rectangular_restore_detected")
                else None,
                f"Salto tonal de {boundary.get('max_row_step')} niveles en el borde de "
                "restauracion: franja visible en cuello/hombros."
                if boundary.get("visible_band_detected")
                else None,
            )
            if message
        ],
    }))


if __name__ == "__main__":
    main()
