"""Compositor 2D determinista y auditable para la vista frontal de SATRE-IA.

Ruta oficial local: sprite RGBA -> warp por pose -> iluminacion -> sombra ->
pliegues -> oclusiones -> grano -> QA. No usa difusion ni altera mediciones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "pattern-engine" / "assets" / "garments"

from tryon_geometry import (  # noqa: E402
    build_parsed_region_masks,
    landmarks_after_crop,
    load_person_silhouette,
    point,
    prewarp_jacket,
    prewarp_lower,
    resize_and_crop_local,
)
from human_parsing_schp import infer_atr  # noqa: E402


def select_assets(sex: str) -> tuple[Path, Path, list[str]]:
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    wanted = ["jacket", "trousers" if sex == "male" else "skirt"]
    chosen = []
    for category in wanted:
        record = next((item for item in manifest["items"] if item["sex"] == sex and item["category"] == category), None)
        if not record:
            raise ValueError(f"No hay sprite {category} para {sex}.")
        chosen.append(record)
    return ASSETS / chosen[0]["file"], ASSETS / chosen[1]["file"], [item["id"] for item in chosen]


def apply_region(rgba: Image.Image, region: Image.Image) -> Image.Image:
    array = np.asarray(rgba.convert("RGBA"), dtype=np.uint8).copy()
    region_array = np.asarray(region.convert("L").filter(ImageFilter.GaussianBlur(0.65)), dtype=np.float32) / 255.0
    array[:, :, 3] = (array[:, :, 3].astype(np.float32) * region_array).astype(np.uint8)
    return Image.fromarray(array, "RGBA")


def complete_sleeve_coverage(
    jacket: Image.Image, arm_mask: Image.Image, upper_region: Image.Image
) -> Image.Image:
    """Cierra vacíos entre la manga warpeada y el brazo semántico real."""
    layer = np.asarray(jacket.convert("RGBA"), dtype=np.uint8).copy()
    alpha = layer[:, :, 3]
    arms = np.asarray(arm_mask.convert("L"), dtype=np.uint8) > 32
    allowed = np.asarray(upper_region.convert("L"), dtype=np.uint8) > 32
    target = np.logical_and(arms, allowed)
    existing = alpha > 32
    missing = np.logical_and(target, alpha < 220)
    if not missing.any() or not existing.any():
        return jacket
    _, nearest = distance_transform_edt(~existing, return_indices=True)
    layer[:, :, :3][missing] = layer[nearest[0][missing], nearest[1][missing], :3]
    target_alpha = cv2.GaussianBlur((target.astype(np.uint8) * 255), (0, 0), 0.8)
    layer[:, :, 3] = np.maximum(alpha, target_alpha)
    return Image.fromarray(layer, "RGBA")


def match_luminance(
    rgba: Image.Image, photograph: Image.Image, reference_mask: Image.Image | None = None
) -> tuple[Image.Image, dict]:
    """Ajusta L con limites; conserva A/B para no convertir azul en tono de piel."""
    layer = np.asarray(rgba.convert("RGBA"), dtype=np.uint8).copy()
    photo = np.asarray(photograph.convert("RGB"), dtype=np.uint8)
    mask = layer[:, :, 3] > 32
    if mask.sum() < 100:
        return rgba, {"before_gap": None, "after_gap": None, "delta_l": 0.0}
    garment_lab = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    photo_lab = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_mask = np.asarray(reference_mask.convert("L")) > 32 if reference_mask is not None else mask
    if target_mask.sum() < 100:
        target_mask = mask
    source_mean = float(np.median(garment_lab[:, :, 0][mask]))
    target_mean = float(np.median(photo_lab[:, :, 0][target_mask]))
    # El color de la prenda es intencionalmente distinto. Solo trasladamos parte
    # de la exposicion local, con un limite estricto para conservar azul marino.
    delta = float(np.clip((target_mean - source_mean) * 0.35, -12.0, 12.0))
    before_gap = abs(target_mean - source_mean)
    garment_lab[:, :, 0][mask] = np.clip(garment_lab[:, :, 0][mask] + delta, 0, 255)
    after_mean = float(np.median(garment_lab[:, :, 0][mask]))
    rgb = cv2.cvtColor(garment_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    layer[:, :, :3] = rgb
    return Image.fromarray(layer, "RGBA"), {
        "before_gap": round(before_gap, 3), "after_gap": round(abs(target_mean - after_mean), 3), "delta_l": round(delta, 3)
    }


def apply_directional_lighting(
    rgba: Image.Image, photograph: Image.Image, reference_mask: Image.Image | None
) -> tuple[Image.Image, dict]:
    """Transfiere solo el plano suave de luz; no copia logos ni textura ajena."""
    layer = np.asarray(rgba.convert("RGBA"), dtype=np.uint8).copy()
    garment = layer[:, :, 3] > 32
    reference = np.asarray(reference_mask.convert("L")) > 32 if reference_mask is not None else garment
    if garment.sum() < 100 or reference.sum() < 100:
        return rgba, {"active": False, "horizontal_delta_l": 0.0, "vertical_delta_l": 0.0}
    photo = np.asarray(photograph.convert("RGB"), dtype=np.uint8)
    photo_l = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    height, width = photo_l.shape
    yy, xx = np.mgrid[0:height, 0:width]
    xn = xx.astype(np.float32) / max(width - 1, 1) * 2.0 - 1.0
    yn = yy.astype(np.float32) / max(height - 1, 1) * 2.0 - 1.0
    values = photo_l[reference]
    lo, hi = np.percentile(values, [10, 90])
    stable = np.logical_and(reference, np.logical_and(photo_l >= lo, photo_l <= hi))
    design = np.column_stack([xn[stable], yn[stable], np.ones(stable.sum(), dtype=np.float32)])
    coefficients, *_ = np.linalg.lstsq(design, photo_l[stable], rcond=None)
    field = coefficients[0] * xn + coefficients[1] * yn
    field -= float(np.median(field[garment]))
    field = np.clip(field * 0.42, -10.0, 10.0)
    lab = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0][garment] = np.clip(lab[:, :, 0][garment] + field[garment], 0, 255)
    layer[:, :, :3] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(layer, "RGBA"), {
        "active": True,
        "horizontal_delta_l": round(float(coefficients[0] * 2.0 * 0.42), 3),
        "vertical_delta_l": round(float(coefficients[1] * 2.0 * 0.42), 3),
    }


def add_folds(rgba: Image.Image, strength: float = 0.08) -> Image.Image:
    array = np.asarray(rgba.convert("RGBA"), dtype=np.uint8).copy()
    height, width = array.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    alpha = array[:, :, 3].astype(np.float32) / 255.0
    vertical = np.sin(xx / max(width, 1) * np.pi * 11 + yy / max(height, 1) * 1.7)
    center_folds = np.exp(-((xx - width * 0.5) / max(width * 0.28, 1)) ** 2)
    taper = np.clip((yy / max(height, 1) - 0.2) / 0.8, 0, 1)
    multiplier = 1.0 + vertical * center_folds * taper * strength
    array[:, :, :3] = np.clip(array[:, :, :3].astype(np.float32) * multiplier[:, :, None], 0, 255).astype(np.uint8)
    array[:, :, 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(array, "RGBA")


def shadow_from(layer: Image.Image, allowed_mask: Image.Image | None = None) -> Image.Image:
    alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(5))
    shifted = Image.new("L", layer.size, 0)
    shifted.paste(alpha, (3, 3))
    opacity = shifted.point(lambda value: round(value * 0.30))
    if allowed_mask is not None:
        opacity = Image.fromarray(
            cv2.bitwise_and(np.asarray(opacity), np.asarray(allowed_mask.convert("L"))), "L"
        )
    result = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    result.putalpha(opacity)
    return result


def occlusion_mask(size: tuple[int, int], landmarks: list[dict]) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    shoulder_width = abs(point(landmarks, 12, size)[0] - point(landmarks, 11, size)[0])
    hand_radius = max(8, shoulder_width * 0.12)
    for indices in ([15, 17, 19, 21], [16, 18, 20, 22]):
        pts = [point(landmarks, index, size) for index in indices]
        for x, y in pts:
            draw.ellipse((x - hand_radius, y - hand_radius, x + hand_radius, y + hand_radius), fill=255)
    # Cara y cabello aproximados por pose. Hasta integrar SCHP esto es un
    # fallback explicitamente registrado, no parsing semantico.
    eye_points = [point(landmarks, i, size) for i in (2, 5, 7, 8)]
    xs, ys = [p[0] for p in eye_points], [p[1] for p in eye_points]
    face_width = max(max(xs) - min(xs), shoulder_width * 0.28)
    face_cx = sum(xs) / len(xs)
    face_cy = sum(ys) / len(ys) + face_width * 0.18
    draw.ellipse((face_cx - face_width, face_cy - face_width * 1.35, face_cx + face_width, face_cy + face_width * 1.35), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(1.2))


def composite(
    photo: Image.Image,
    layers: list[Image.Image],
    occlusion: Image.Image,
    shadow_allowed: Image.Image | None = None,
) -> Image.Image:
    result = photo.convert("RGBA")
    for layer in layers:
        result = Image.alpha_composite(result, shadow_from(layer, shadow_allowed))
        result = Image.alpha_composite(result, layer)
    original = photo.convert("RGBA")
    result.paste(original, (0, 0), occlusion)
    array = np.asarray(result.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(20260911)
    noise = rng.normal(0, 2, array.shape[:2])[:, :, None]
    array = np.clip(array + noise * 0.05, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _edge_distance(alpha: np.ndarray) -> np.ndarray:
    binary = (alpha > 24).astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8))
    edge = np.logical_xor(binary > 0, eroded > 0)
    return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)


def _bright_halo_percent(layer: Image.Image) -> float:
    """Estima halos claros solo en el borde exterior semitransparente."""
    rgba = np.asarray(layer.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    interior = alpha > 245
    outer_edge = np.logical_and(alpha > 8, alpha < 180)
    if interior.sum() < 100 or outer_edge.sum() < 20:
        return 0.0
    lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1].astype(np.float32) - 128.0, lab[:, :, 2].astype(np.float32) - 128.0)
    threshold = min(245.0, float(np.percentile(lightness[interior], 90)) + 28.0)
    # Un brillo azul del propio tejido no es halo. El matte defectuoso es
    # simultáneamente claro y casi neutro (blanco/gris).
    suspicious = np.logical_and(lightness > threshold, chroma < 12.0)
    # Un borde muy corto puede convertir 5–10 píxeles inocuos en un porcentaje
    # enorme. El soporte mínimo evita esa inestabilidad sin ocultar un halo
    # sostenido alrededor de la prenda.
    support = max(int(outer_edge.sum()), 400)
    return float(np.logical_and(outer_edge, suspicious).sum() / support * 100)


def _fragmentation_percent(layer: Image.Image) -> float:
    """Área visible que quedó separada del componente principal del sprite."""
    binary = (np.asarray(layer.getchannel("A"), dtype=np.uint8) > 24).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    relevant = areas[areas >= max(12.0, areas.sum() * 0.001)]
    if relevant.size <= 1:
        return 0.0
    return float((relevant.sum() - relevant.max()) / max(relevant.sum(), 1.0) * 100)


def _arm_open_angle(landmarks: list[dict], size: tuple[int, int], indices: tuple[int, int, int]) -> float:
    shoulder = np.asarray(point(landmarks, indices[0], size), dtype=np.float32)
    elbow = np.asarray(point(landmarks, indices[1], size), dtype=np.float32)
    wrist = np.asarray(point(landmarks, indices[2], size), dtype=np.float32)
    direction = 0.35 * (elbow - shoulder) + 0.65 * (wrist - shoulder)
    return float(np.degrees(np.arctan2(abs(direction[0]), max(abs(direction[1]), 1e-6))))


def qa_metrics(
    layers: list[Image.Image], silhouette: Image.Image, occlusion: Image.Image,
    landmarks: list[dict], luminance: dict, semantic_ok: bool, sprite_pose: str,
) -> dict:
    combined = np.zeros((layers[0].height, layers[0].width), dtype=np.uint8)
    for layer in layers:
        combined = np.maximum(combined, np.asarray(layer.getchannel("A"), dtype=np.uint8))
    garment = combined > 24
    person = np.asarray(silhouette.convert("L")) > 64
    outside = float(np.logical_and(garment, ~person).sum() / max(garment.sum(), 1) * 100)
    overlap = float(np.logical_and(garment, np.asarray(occlusion) > 32).sum() / max(garment.sum(), 1) * 100)
    size = layers[0].size
    shoulder_width = abs(point(landmarks, 12, size)[0] - point(landmarks, 11, size)[0])
    jacket_alpha_u8 = np.asarray(layers[-1].getchannel("A"), dtype=np.uint8)
    jacket_alpha = jacket_alpha_u8 > 24
    # Extension estructural: se mide contra SAM original, no contra una mascara
    # ya dilatada. La distancia maxima admisible es 8 % del ancho de hombros.
    outside_distance = cv2.distanceTransform((~person).astype(np.uint8), cv2.DIST_L2, 3)
    outside_jacket = np.logical_and(jacket_alpha, ~person)
    structural_extension = float(outside_distance[outside_jacket].max()) if outside_jacket.any() else 0.0
    structural_limit = float(shoulder_width * 0.08)

    jacket_edge_distance = _edge_distance(jacket_alpha_u8)
    left_shoulder = point(landmarks, 11, size)
    right_shoulder = point(landmarks, 12, size)
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    head_points = [point(landmarks, i, size) for i in (0, 7, 8)]
    head_y = sum(p[1] for p in head_points) / len(head_points)
    neck_y = head_y + 0.72 * (shoulder_y - head_y)
    neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
    central = np.zeros_like(jacket_alpha, dtype=bool)
    x0 = max(0, round(neck_x - shoulder_width * 0.22))
    x1 = min(size[0], round(neck_x + shoulder_width * 0.22))
    y0 = max(0, round(neck_y - shoulder_width * 0.20))
    y1 = min(size[1], round(neck_y + shoulder_width * 0.25))
    central[y0:y1, x0:x1] = True
    jacket_edge = np.logical_and(jacket_alpha, cv2.erode(jacket_alpha.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    collar_edge_points = np.argwhere(np.logical_and(jacket_edge, central))
    if len(collar_edge_points):
        collar_gap = float(np.min(np.hypot(collar_edge_points[:, 1] - neck_x, collar_edge_points[:, 0] - neck_y)))
    else:
        collar_gap = float(shoulder_width)

    cuff_gaps = []
    for index in (15, 16):
        x, y = point(landmarks, index, size)
        xi, yi = int(np.clip(round(x), 0, size[0] - 1)), int(np.clip(round(y), 0, size[1] - 1))
        cuff_gaps.append(float(jacket_edge_distance[yi, xi]))
    cuff_gap = float(np.mean(cuff_gaps))
    collar_percent = collar_gap / max(shoulder_width, 1) * 100
    cuff_percent = cuff_gap / max(shoulder_width, 1) * 100
    halo_percent = max(_bright_halo_percent(layer) for layer in layers)
    jacket_fragmentation = _fragmentation_percent(layers[-1])
    arm_angles = [
        _arm_open_angle(landmarks, size, (11, 13, 15)),
        _arm_open_angle(landmarks, size, (12, 14, 16)),
    ]
    mean_arm_angle = float(np.mean(arm_angles))
    required_pose = "front-open" if mean_arm_angle >= 35.0 else "front-neutral"
    pose_compatible = required_pose == sprite_pose
    checks = {
        "structural_extension_within_limit": structural_extension <= structural_limit + 2.0,
        "collar_gap_percent_shoulder": collar_percent < 20.0,
        "cuff_gap_percent_shoulder": cuff_percent < 10.0,
        "occlusion_overlap_percent": overlap < 3.0,
        "luminance_gap_reduced": luminance["after_gap"] is not None and luminance["after_gap"] <= luminance["before_gap"],
        "semantic_parsing_active": semantic_ok,
        "bright_edge_halo_percent": halo_percent < 3.0,
        "jacket_fragmentation_percent": jacket_fragmentation < 1.0,
        "sprite_pose_compatible": pose_compatible,
    }
    return {
        # Ninguno de estos checks detecta el aspecto de pegatina, la discontinuidad
        # de manga ni la prenda original expuesta. Declarar "bueno" era un falso
        # positivo estructural: el estado ahora nombra lo que de verdad se verifico.
        "status": "tecnicamente_compuesto" if all(checks.values()) else "ajuste_dudoso",
        "status_scope": "geometria_y_continuidad_de_bordes_no_realismo",
        "checks": checks,
        "values": {
            "garment_area_outside_original_sam_percent": round(outside, 3),
            "structural_extension_px": round(structural_extension, 3),
            "structural_extension_limit_px": round(structural_limit, 3),
            "collar_visible_edge_gap_px": round(collar_gap, 3),
            "collar_visible_edge_gap_percent_shoulder": round(collar_percent, 3),
            "cuff_visible_edge_gap_px_mean": round(cuff_gap, 3),
            "cuff_visible_edge_gap_percent_shoulder": round(cuff_percent, 3),
            "occlusion_overlap_percent": round(overlap, 3),
            "shoulder_width_px": round(shoulder_width, 2),
            "bright_edge_halo_percent": round(halo_percent, 3),
            "jacket_fragmentation_percent": round(jacket_fragmentation, 3),
            "arm_open_angle_deg": [round(value, 2) for value in arm_angles],
            "arm_open_angle_deg_mean": round(mean_arm_angle, 2),
            "required_pose_variant": required_pose,
            "sprite_pose_variant": sprite_pose,
            "luminance": luminance,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--sex", choices=["male", "female"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    session = Path(args.session_dir).resolve()
    manifest = json.loads((session / "capture_manifest.json").read_text(encoding="utf-8"))
    front = manifest["views"]["front"]
    if not front.get("pose", {}).get("landmarks"):
        raise ValueError("La sesion no contiene landmarks frontales.")
    size = (args.width, args.height)
    source_photo = Image.open(front["photo"]).convert("RGB")
    photo = resize_and_crop_local(source_photo, size)
    landmarks = landmarks_after_crop(front["pose"]["landmarks"], source_photo.size, size)
    silhouette = load_person_silhouette(front["mask"], size)
    upper_region, lower_region = build_parsed_region_masks(size, landmarks, args.sex, silhouette)
    jacket_path, lower_path, asset_ids = select_assets(args.sex)
    jacket = prewarp_jacket(Image.open(jacket_path), size, landmarks, transparent=True)
    lower = prewarp_lower(Image.open(lower_path), size, landmarks, args.sex, transparent=True)
    jacket = apply_region(jacket, upper_region)
    lower = apply_region(lower, lower_region)
    parsing_error = None
    try:
        parsing = infer_atr(photo, landmarks)
        semantic_ok = True
        jacket = complete_sleeve_coverage(jacket, parsing["arms"], upper_region)
        # Para un saco se reemplazan los brazos vestidos; restaurar el brazo
        # original pondria la camiseta por encima de la manga. Solo se recuperan
        # manos, cuello, cara y pelo como capas anatomicas delanteras.
        occlusion_array = np.maximum.reduce([
            np.asarray(parsing[name].convert("L")) for name in ("hands", "neck", "face", "hair")
        ])
        occlusion = Image.fromarray(occlusion_array.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.8))
        shadow_allowed = parsing["shadow_allowed"]
        jacket_reference = parsing["upper_clothes"]
        lower_reference = parsing["lower_clothes"]
    except Exception as exc:
        semantic_ok = False
        parsing_error = f"{type(exc).__name__}: {exc}"
        parsing = {}
        occlusion = occlusion_mask(size, landmarks)
        shadow_allowed = None
        jacket_reference = None
        lower_reference = None
    # Los sprites de catálogo ya contienen textura. Los pliegues sintéticos se
    # mantienen muy suaves para no borrar solapas, botones ni costuras.
    jacket, jacket_light = match_luminance(add_folds(jacket, 0.025), photo, jacket_reference)
    lower, lower_light = match_luminance(add_folds(lower, 0.035), photo, lower_reference)
    jacket, jacket_direction = apply_directional_lighting(jacket, photo, jacket_reference)
    lower, lower_direction = apply_directional_lighting(lower, photo, lower_reference)
    luminance = {
        "before_gap": round((jacket_light["before_gap"] + lower_light["before_gap"]) / 2, 3),
        "after_gap": round((jacket_light["after_gap"] + lower_light["after_gap"]) / 2, 3),
        "delta_l": [jacket_light["delta_l"], lower_light["delta_l"]],
    }
    sprite_pose = "front-neutral"
    qa = qa_metrics([lower, jacket], silhouette, occlusion, landmarks, luminance, semantic_ok, sprite_pose)
    result = composite(photo, [lower, jacket], occlusion, shadow_allowed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output, optimize=True)
    jacket.save(output.parent / "layer_jacket.png")
    lower.save(output.parent / ("layer_trousers.png" if args.sex == "male" else "layer_skirt.png"))
    occlusion.save(output.parent / ("occlusion_semantic_refined.png" if semantic_ok else "occlusion_fallback.png"))
    if semantic_ok:
        for name in ("labels", "labels_color", "arms", "hands", "neck", "face", "hair", "shadow_allowed"):
            parsing[name].save(output.parent / f"schp_atr_{name}.png")
    metadata = {
        "schema_version": "1.0", "mode": "official_deterministic_layered_2d",
        "assets": asset_ids, "resolution": list(size), "warp": "piecewise_projective_pose_anchors",
        "parsing": "schp_atr_18_onnx_int8_cpu" if semantic_ok else "pose_fallback",
        "semantic_human_parsing": semantic_ok,
        "parsing_model_sha256": "4420d8db8c1f266967c89485786b01209f6d405f320fc0f87e8ced49392cefb5" if semantic_ok else None,
        "parsing_refinements": "ATR separa brazos/cara-pelo; manos y cuello se refinan con MediaPipe porque ATR no tiene esas clases separadas.",
        "parsing_error": parsing_error,
        "directional_lighting": {"jacket": jacket_direction, "lower": lower_direction},
        "layers": ["photo", "restricted_shadow", "lower", "restricted_shadow", "jacket", "hands_neck_face_hair", "grain"],
        "qa": qa,
        "notice": "Vista aproximada de estilo. No valida holgura, caida, comodidad ni reemplaza el molde metrico.",
    }
    (output.parent / "preview_2d_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
