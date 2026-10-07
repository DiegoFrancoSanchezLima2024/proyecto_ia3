"""Chequeos auditables de calidad para candidatos generativos de FASHN VTON.

No es un modelo nuevo: reutiliza la misma geometria por pose+silueta que ya
usa el compositor 2D determinista (`tryon_geometry.build_parsed_region_masks`)
para saber donde deberia estar el saco y el pantalon/falda, y aplica dos
heuristicas baratas sobre esa region sobre la imagen ya generada:

- cobertura de color de la prenda (compara contra el color dominante de la
  foto de referencia real que se le paso a FASHN).
- fuga de piel/anatomia dentro de la region del torso (el fallo observado en
  `male-jacket-masked-seed42.png`: busto/pecho alucinado).

Esto no reemplaza revision visual humana; es un filtro barato para descartar
candidatos claramente rotos antes de mostrarlos.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tryon_geometry import (
    build_parsed_region_masks,
    landmarks_after_crop,
    load_person_silhouette,
)


def _load_manifest(session_dir: Path) -> dict:
    return json.loads((session_dir / "capture_manifest.json").read_text(encoding="utf-8"))


def reference_garment_color(reference_path: Path) -> np.ndarray:
    """Color LAB mediano muestreado del torso de la foto de referencia de la prenda."""
    image = Image.open(reference_path).convert("RGB")
    width, height = image.size
    box = (round(width * 0.30), round(height * 0.22), round(width * 0.70), round(height * 0.45))
    patch = np.asarray(image.crop(box), dtype=np.uint8).reshape(-1, 3)
    lab = cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    # El recorte puede incluir fondo casi blanco del catalogo; se descarta por L alto.
    keep = lab[:, 0] < 235
    sample = lab[keep] if keep.any() else lab
    return np.median(sample, axis=0).astype(np.float32)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper) > 0


def _region_stats(
    region_mask: Image.Image, candidate_lab: np.ndarray, skin: np.ndarray, target_color: np.ndarray
) -> dict:
    region = np.asarray(region_mask.convert("L")) > 64
    area = int(region.sum())
    if area < 200:
        return {"area_px": area, "garment_color_coverage_percent": None, "skin_leak_percent": None}
    distance = np.linalg.norm(candidate_lab[region].astype(np.float32) - target_color, axis=1)
    coverage = float((distance < 26.0).sum() / area * 100)
    leak = float(np.logical_and(region, skin).sum() / area * 100)
    return {
        "area_px": area,
        "garment_color_coverage_percent": round(coverage, 2),
        "skin_leak_percent": round(leak, 2),
    }


def score_candidate(candidate_path: Path, session_dir: Path, sex: str, reference_path: Path) -> dict:
    manifest = _load_manifest(session_dir)
    front = manifest["views"]["front"]
    candidate = Image.open(candidate_path).convert("RGB")
    size = candidate.size

    source_photo = Image.open(front["photo"]).convert("RGB")
    landmarks = landmarks_after_crop(front["pose"]["landmarks"], source_photo.size, size)
    silhouette = load_person_silhouette(front["mask"], size)
    upper_region, lower_region = build_parsed_region_masks(size, landmarks, sex, silhouette)

    candidate_rgb = np.asarray(candidate, dtype=np.uint8)
    candidate_lab = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2LAB)
    skin = _skin_mask(candidate_rgb)
    target_color = reference_garment_color(reference_path)

    upper_stats = _region_stats(upper_region, candidate_lab, skin, target_color)
    lower_stats = _region_stats(lower_region, candidate_lab, skin, target_color)

    def value_or(stats: dict, key: str, default: float) -> float:
        value = stats.get(key)
        return default if value is None else value

    checks = {
        "upper_garment_color_ok": value_or(upper_stats, "garment_color_coverage_percent", 0.0) >= 35.0,
        "upper_skin_leak_ok": value_or(upper_stats, "skin_leak_percent", 0.0) <= 12.0,
        "lower_garment_color_ok": value_or(lower_stats, "garment_color_coverage_percent", 0.0) >= 30.0,
        "lower_skin_leak_ok": value_or(lower_stats, "skin_leak_percent", 0.0) <= 15.0,
    }
    score = (
        value_or(upper_stats, "garment_color_coverage_percent", 0.0)
        + value_or(lower_stats, "garment_color_coverage_percent", 0.0)
        - value_or(upper_stats, "skin_leak_percent", 0.0) * 2.0
        - value_or(lower_stats, "skin_leak_percent", 0.0) * 2.0
    )
    return {
        "candidate": str(candidate_path),
        # "bueno" afirmaba calidad. Estas heuristicas miden color de prenda y fuga
        # de piel; no miden realismo, largo, manos, fondo ni continuidad de manga.
        # El veredicto se limita a lo que realmente se comprobo.
        "verdict": "tecnicamente_compuesto" if all(checks.values()) else "sospechoso",
        "verdict_scope": "color_de_prenda_y_fuga_de_piel_unicamente",
        "score": round(score, 2),
        "checks": checks,
        "upper": upper_stats,
        "lower": lower_stats,
        "notice": "Heuristica de color/piel. NO evalua realismo global, largo hasta el tobillo, manos, preservacion de fondo ni linea cuello-solapa. No reemplaza revision visual.",
    }
