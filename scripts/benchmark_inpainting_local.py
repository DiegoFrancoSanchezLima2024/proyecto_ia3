#!/usr/bin/env python3
"""Mide Stable Diffusion Inpainting local con una carga reproducible.

Este benchmark no forma parte del vestidor final. Verifica que el checkpoint
FP16 puede cargarse sin red, que el offload funciona en la GPU disponible y
registra tiempo/VRAM para elegir entre recortes de 512 o 384 pixeles.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=Path, required=True)
    parser.add_argument("--imagen", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--tamano", type=int, choices=(384, 512), default=512)
    parser.add_argument("--pasos", type=int, default=20)
    parser.add_argument("--semilla", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    if not torch.cuda.is_available():
        raise RuntimeError("El benchmark requiere CUDA.")

    inicio_carga = time.perf_counter()
    tuberia = StableDiffusionInpaintPipeline.from_pretrained(
        args.modelo,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    tuberia.enable_model_cpu_offload()
    tuberia.enable_attention_slicing("max")
    segundos_carga = time.perf_counter() - inicio_carga

    imagen = Image.open(args.imagen).convert("RGB").resize(
        (args.tamano, args.tamano), Image.Resampling.LANCZOS
    )
    mascara = Image.new("L", imagen.size, 0)
    dibujo = ImageDraw.Draw(mascara)
    # Dos regiones inferiores similares a recortes de pie. Solo sirve para
    # someter al pipeline a una carga realista y comparable.
    y0 = round(args.tamano * 0.72)
    dibujo.rounded_rectangle(
        (round(args.tamano * 0.25), y0, round(args.tamano * 0.48), args.tamano - 4),
        radius=round(args.tamano * 0.04), fill=255,
    )
    dibujo.rounded_rectangle(
        (round(args.tamano * 0.52), y0, round(args.tamano * 0.75), args.tamano - 4),
        radius=round(args.tamano * 0.04), fill=255,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inicio_inferencia = time.perf_counter()
    resultado = tuberia(
        prompt=(
            "formal black leather Oxford dress shoes, realistic perspective, "
            "matching indoor lighting, feet standing naturally on the floor, photorealistic"
        ),
        negative_prompt=(
            "extra feet, extra legs, malformed shoes, floating shoes, sandals, "
            "sneakers, distorted toes"
        ),
        image=imagen,
        mask_image=mascara,
        width=args.tamano,
        height=args.tamano,
        num_inference_steps=args.pasos,
        guidance_scale=7.0,
        generator=torch.Generator(device="cpu").manual_seed(args.semilla),
    ).images[0]
    segundos_inferencia = time.perf_counter() - inicio_inferencia

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.save(args.salida)
    informe = {
        "modelo": str(args.modelo),
        "resolucion": [args.tamano, args.tamano],
        "pasos": args.pasos,
        "semilla": args.semilla,
        "precision": "fp16",
        "offload": "model_cpu_offload",
        "segundos_carga": round(segundos_carga, 2),
        "segundos_inferencia": round(segundos_inferencia, 2),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "gpu": torch.cuda.get_device_name(0),
        "salida": str(args.salida),
    }
    args.salida.with_suffix(".json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(informe, ensure_ascii=False))


if __name__ == "__main__":
    main()
