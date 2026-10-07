"""Normaliza los activos base en una biblioteca local de sprites RGBA.

No inventa variantes de pose: crea solamente las cuatro prendas que existen y
deja sus anclas declaradas en manifest.json para que el warp sea auditable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "pattern-engine" / "assets" / "garments"
CANVAS = (1024, 2048)

ITEMS = [
    {
        "id": "male-jacket-classic-navy-front-neutral",
        "category": "jacket",
        "sex": "male",
        "pose": "front-neutral",
        "source": ASSETS / "male-navy-jacket.png",
        "target": ASSETS / "jacket" / "classic-navy" / "front-neutral.png",
        "anchors": {
            "left_shoulder": [0.27, 0.18], "right_shoulder": [0.73, 0.18],
            "left_elbow": [0.18, 0.42], "right_elbow": [0.82, 0.42],
            "left_wrist": [0.12, 0.69], "right_wrist": [0.88, 0.69],
            "left_waist": [0.34, 0.55], "right_waist": [0.66, 0.55],
            "left_hip": [0.31, 0.78], "right_hip": [0.69, 0.78],
        },
    },
    {
        "id": "female-jacket-classic-navy-front-neutral",
        "category": "jacket",
        "sex": "female",
        "pose": "front-neutral",
        "source": ASSETS / "female-navy-jacket.png",
        "target": ASSETS / "jacket" / "classic-navy" / "front-neutral-female.png",
        "anchors": {
            "left_shoulder": [0.28, 0.18], "right_shoulder": [0.72, 0.18],
            "left_elbow": [0.18, 0.42], "right_elbow": [0.82, 0.42],
            "left_wrist": [0.12, 0.69], "right_wrist": [0.88, 0.69],
            "left_waist": [0.36, 0.55], "right_waist": [0.64, 0.55],
            "left_hip": [0.32, 0.78], "right_hip": [0.68, 0.78],
        },
    },
    {
        "id": "male-trousers-straight-navy-front-neutral",
        "category": "trousers",
        "sex": "male",
        "pose": "front-neutral",
        "source": ASSETS / "male-navy-trousers.png",
        "target": ASSETS / "trousers" / "straight-navy" / "front-neutral.png",
        "anchors": {
            "left_waist": [0.25, 0.08], "right_waist": [0.75, 0.08],
            "left_hip": [0.28, 0.18], "right_hip": [0.72, 0.18],
            "left_knee": [0.31, 0.57], "right_knee": [0.69, 0.57],
            "left_ankle": [0.34, 0.94], "right_ankle": [0.66, 0.94],
        },
    },
    {
        "id": "female-skirt-pencil-navy-midi-front",
        "category": "skirt",
        "sex": "female",
        "pose": "midi-front",
        "source": ASSETS / "female-navy-skirt.png",
        "target": ASSETS / "skirt" / "pencil-navy" / "midi-front.png",
        "anchors": {
            "left_waist": [0.30, 0.08], "right_waist": [0.70, 0.08],
            "left_hip": [0.24, 0.30], "right_hip": [0.76, 0.30],
            "left_knee": [0.28, 0.91], "right_knee": [0.72, 0.91],
        },
    },
]


def remove_white_background(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    distance = np.max(255 - rgb, axis=2).astype(np.float32)
    alpha = np.clip((distance - 3.0) * 18.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def normalize(image: Image.Image) -> Image.Image:
    rgba = remove_white_background(image)
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 16)
    if not len(xs):
        raise ValueError("Activo sin primer plano")
    crop = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)))
    max_width, max_height = int(CANVAS[0] * 0.86), int(CANVAS[1] * 0.88)
    scale = min(max_width / crop.width, max_height / crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    canvas.alpha_composite(crop, ((CANVAS[0] - crop.width) // 2, (CANVAS[1] - crop.height) // 2))
    return canvas


def main() -> None:
    records = []
    for item in ITEMS:
        item["target"].parent.mkdir(parents=True, exist_ok=True)
        normalize(Image.open(item["source"])).save(item["target"], optimize=True)
        records.append({
            "id": item["id"], "category": item["category"], "sex": item["sex"],
            "pose": item["pose"], "file": item["target"].relative_to(ASSETS).as_posix(),
            "width_px": CANVAS[0], "height_px": CANVAS[1],
            "background": "transparent", "anchors_normalized": item["anchors"],
            "source_provenance": "SATRE-IA local catalog asset",
        })
    manifest = {
        "schema_version": "1.0", "coordinate_system": "normalized_xy_top_left",
        "canvas_px": list(CANVAS), "items": records,
        "limitations": ["Solo pose frontal neutra", "No valida caida ni holgura de confeccion"],
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
