"""Segmenta las cuatro fotos con SAM 2.1 sin acoplarlo al entorno de medidas."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VIEWS = ("front", "left", "back", "right")
REQUIRED_VIEWS = ("front", "left")


def resolver_ruta_proyecto(value: str | Path) -> Path:
    """Evita que Hydra altere la interpretación de rutas relativas."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validar_caja(box: list[float], width: int, height: int) -> np.ndarray:
    """Valida una caja XYXY incluida en los límites de la fotografía."""
    if len(box) != 4 or not np.isfinite(np.asarray(box, dtype=np.float32)).all():
        raise ValueError("Cada caja debe contener cuatro números finitos: x1 y1 x2 y2")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Caja fuera de una imagen {width}x{height}: {[x1, y1, x2, y2]}"
        )
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def seleccionar_mejor_mascara(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Selecciona la propuesta SAM de mayor confianza y la normaliza a bool."""
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or masks.shape[0] != scores.size or scores.size == 0:
        raise ValueError("SAM devolvió máscaras/puntuaciones con dimensiones inválidas")
    index = int(np.argmax(scores))
    return masks[index].astype(bool), float(scores[index])


def diagnostico_mascara(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("La máscara debe ser HxW")
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    warnings = []
    foreground_ratio = float(mask.mean())
    vertical_coverage = float(rows.mean())
    horizontal_coverage = float(cols.mean())
    if foreground_ratio < 0.03:
        warnings.append("persona demasiado pequeña o máscara vacía")
    if foreground_ratio > 0.85:
        warnings.append("la máscara ocupa casi toda la fotografía")
    if vertical_coverage < 0.65:
        warnings.append("no se observa el cuerpo completo")
    if mask[0].any() or mask[-1].any():
        warnings.append("la persona toca el borde superior o inferior")
    return {
        "foreground_ratio": foreground_ratio,
        "vertical_coverage": vertical_coverage,
        "horizontal_coverage": horizontal_coverage,
        "warnings": warnings,
    }


class Sam2PersonSegmenter:
    """Wrapper mínimo sobre la API oficial de SAM 2 para una persona por foto."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_config: str,
        sam2_root: Path | None = None,
    ):
        if sam2_root is not None:
            sys.path.insert(0, str(Path(sam2_root).resolve()))
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 no está instalado. Sigue docs/PERCEPTION_SETUP.md en el "
                "entorno sastre-ia-perception."
            ) from exc

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No existe el checkpoint SAM 2: {checkpoint_path}")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_sam2(model_config, str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self.checkpoint_path = checkpoint_path.resolve()
        self.model_config = model_config
        self.last_peak_vram_gib = 0.0

    def segment(self, image: Image.Image, box: np.ndarray) -> tuple[np.ndarray, float]:
        rgb = np.asarray(image.convert("RGB")).copy()
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        amp = (
            self.torch.autocast("cuda", dtype=self.torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with self.torch.inference_mode(), amp:
            self.predictor.set_image(rgb)
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=True,
            )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self.last_peak_vram_gib = self.torch.cuda.max_memory_allocated(
                self.device
            ) / (1024**3)
        return seleccionar_mejor_mascara(masks, scores)


def load_boxes(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(payload).difference(VIEWS))
    if unknown:
        raise ValueError(f"Vistas desconocidas en boxes JSON: {unknown}")
    return payload


def segment_capture(
    segmenter: Sam2PersonSegmenter,
    photos: dict[str, Path],
    output_dir: Path,
    boxes: dict[str, list[float]] | None = None,
    pose_detector=None,
) -> dict:
    missing = sorted(set(REQUIRED_VIEWS).difference(photos))
    if missing:
        raise ValueError(f"Faltan fotografías: {missing}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boxes = boxes or {}
    manifest = {
        "schema_version": "1.0",
        "segmenter": "sam2.1",
        "checkpoint": str(segmenter.checkpoint_path),
        "model_config": segmenter.model_config,
        "device": str(segmenter.device),
        "views": {},
        "warnings": [],
    }

    for view in VIEWS:
        if view not in photos:
            continue
        photo_path = Path(photos[view])
        if not photo_path.is_file():
            raise FileNotFoundError(f"No existe la foto {view}: {photo_path}")
        with Image.open(photo_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        box_was_automatic = view not in boxes
        pose_result = None
        if box_was_automatic and pose_detector is not None:
            pose_result = pose_detector.detect(image)
            raw_box = pose_result["box_xyxy"]
        else:
            raw_box = boxes.get(view, [0.0, 0.0, float(width), float(height)])
        box = validar_caja(raw_box, width, height)
        mask, score = segmenter.segment(image, box)
        diagnostics = diagnostico_mascara(mask)
        if pose_result is not None:
            diagnostics["warnings"].extend(pose_result["quality"]["warnings"])
        if box_was_automatic and pose_detector is None:
            diagnostics["warnings"].append(
                "se usó toda la imagen como caja; revise visualmente la máscara"
            )
        mask_path = output_dir / f"{view}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        manifest["views"][view] = {
            "photo": str(photo_path.resolve()),
            "mask": str(mask_path.resolve()),
            "box_xyxy": box.tolist(),
            "box_was_automatic": box_was_automatic,
            "box_source": "mediapipe_pose" if pose_result is not None else (
                "full_image" if box_was_automatic else "manual"
            ),
            "pose": pose_result,
            "sam_score": score,
            "peak_vram_gib": round(segmenter.last_peak_vram_gib, 3),
            "quality": diagnostics,
        }
        manifest["warnings"].extend(
            f"{view}: {warning}" for warning in diagnostics["warnings"]
        )

    manifest_path = output_dir / "capture_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    for view in REQUIRED_VIEWS:
        parser.add_argument(f"--{view}", required=True)
    parser.add_argument("--right")
    parser.add_argument("--back")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
    )
    parser.add_argument("--sam2-root")
    parser.add_argument("--boxes-json")
    parser.add_argument("--pose-model")
    parser.add_argument("--output-dir", default="outputs/capture_masks")
    args = parser.parse_args()

    segmenter = Sam2PersonSegmenter(
        checkpoint_path=resolver_ruta_proyecto(args.checkpoint),
        model_config=args.model_config,
        sam2_root=resolver_ruta_proyecto(args.sam2_root) if args.sam2_root else None,
    )
    pose_detector = None
    if args.pose_model:
        from src.perception.mediapipe_pose import MediaPipePoseDetector

        pose_detector = MediaPipePoseDetector(resolver_ruta_proyecto(args.pose_model))
    try:
        manifest = segment_capture(
            segmenter=segmenter,
            photos={
                view: resolver_ruta_proyecto(getattr(args, view))
                for view in VIEWS
                if getattr(args, view) is not None
            },
            output_dir=resolver_ruta_proyecto(args.output_dir),
            boxes=load_boxes(
                resolver_ruta_proyecto(args.boxes_json) if args.boxes_json else None
            ),
            pose_detector=pose_detector,
        )
    finally:
        if pose_detector is not None:
            pose_detector.close()
    print(f"Máscaras guardadas: {resolver_ruta_proyecto(args.output_dir)}")
    for warning in manifest["warnings"]:
        print(f"[CALIDAD] {warning}")


if __name__ == "__main__":
    main()
