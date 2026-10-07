#!/usr/bin/env python3
"""Run one auditable local FASHN VTON 1.5 garment pass.

This runner intentionally handles one garment category at a time.  Chaining a
jacket and trousers is only enabled by the UI after each individual pass has
passed visual QA; the previous CatVTON experiment showed that blindly chaining
two generations can destroy the first garment and the person's identity.

Generates --num-candidates independent samples (sequential num_samples=1
calls with different seeds, not a batched num_samples=N call) and writes them
plus candidates.json next to --output. This script never judges which
candidate looks best; select_best_tryon_candidate.py does that separately.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
from PIL import Image

from fashn_vton import TryOnPipeline


def find_front_photo(session_dir: Path) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = session_dir / "photos" / f"front{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontro la foto frontal en {session_dir / 'photos'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--person-image", type=Path)
    source.add_argument("--session-dir", type=Path)
    parser.add_argument("--garment-image", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=Path("pretrained/fashn-vton-1.5"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("tops", "bottoms", "one-pieces"), required=True)
    parser.add_argument("--garment-photo-type", choices=("flat-lay", "model"), default="flat-lay")
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--masked", action="store_true", help="Use parser-derived masking instead of maskless mode")
    parser.add_argument(
        "--num-candidates", type=int, default=3,
        help="Independent generations (different seeds) to score later and pick the best from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    person_path = args.person_image or find_front_photo(args.session_dir)
    for path, label in ((person_path, "persona"), (args.garment_image, "prenda"), (args.weights_dir, "pesos")):
        if not path.exists():
            raise FileNotFoundError(f"No existe {label}: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("FASHN VTON requiere CUDA para esta demo local")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    started = time.perf_counter()
    pipeline = TryOnPipeline(
        weights_dir=str(args.weights_dir),
        device="cuda",
        preprocessing_device="cpu",
    )
    loaded_seconds = time.perf_counter() - started

    person_image = Image.open(person_path).convert("RGB")
    garment_image = Image.open(args.garment_image).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Llamadas secuenciales con num_samples=1, no un solo lote num_samples=N:
    # la corrida de referencia que funciono uso un pico de 2.72 GB de VRAM en
    # una tarjeta de 6 GB con lote de tamano 1, asi que agrupar N candidatos en
    # un lote arriesga quedarse sin memoria sin ganar tiempo real (el costo de
    # computo en GPU lo domina el bucle de muestreo de todos modos).
    candidates = []
    errors = []
    for index in range(max(1, args.num_candidates)):
        seed = args.seed + index
        torch.cuda.reset_peak_memory_stats()
        candidate_started = time.perf_counter()
        try:
            result = pipeline(
                person_image=person_image,
                garment_image=garment_image,
                category=args.category,
                garment_photo_type=args.garment_photo_type,
                num_samples=1,
                num_timesteps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=seed,
                segmentation_free=not args.masked,
            )
        except Exception as exc:  # noqa: BLE001 - keep other candidates alive on OOM/etc.
            torch.cuda.empty_cache()
            errors.append(f"candidato {index} (seed {seed}): {type(exc).__name__}: {exc}")
            continue
        candidate_seconds = time.perf_counter() - candidate_started
        candidate_path = args.output.with_name(f"{args.output.stem}-candidate-{index}{args.output.suffix}")
        result.images[0].save(candidate_path)
        candidates.append({
            "index": index,
            "path": str(candidate_path),
            "seed": seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "output_resolution": list(result.images[0].size),
            "inference_seconds": round(candidate_seconds, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        })
        torch.cuda.empty_cache()

    if not candidates:
        raise RuntimeError("Ningun candidato se genero correctamente: " + " | ".join(errors))

    manifest = {
        "engine": "FASHN VTON v1.5",
        "mode": "experimental_generative_tryon_multi_candidate",
        "person_image": str(person_path),
        "garment_image": str(args.garment_image),
        "category": args.category,
        "garment_photo_type": args.garment_photo_type,
        "segmentation_free": not args.masked,
        "num_candidates_requested": args.num_candidates,
        "model_load_seconds": round(loaded_seconds, 2),
        "candidates": candidates,
        "candidate_errors": errors,
        "notice": (
            "Resultado generativo experimental; no valida ajuste metrico, holgura ni caida fisica. "
            "El candidato final se elige aparte con select_best_tryon_candidate.py."
        ),
    }
    (args.output.parent / "candidates.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
