"""Mide lotes seguros para el regresor en la GPU local.

Uso:
  python scripts/benchmark_gpu.py --batches 8 12 16 20 24 32
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.regressor import RegresorMedidasCorporales


def benchmark(batch_size: int, image_size: int, steps: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = RegresorMedidasCorporales(
        n_measures=13,
        embed_dim=256,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
        pretrained_encoder=False,
        frozen_epochs=0,
        use_geometry=False,
    ).cuda().to(memory_format=torch.channels_last)
    model.on_epoch_start(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()
    front = torch.randn(
        batch_size, 1, image_size, image_size, device="cuda"
    ).contiguous(memory_format=torch.channels_last)
    left = torch.randn_like(front).contiguous(memory_format=torch.channels_last)
    meta = torch.randn(batch_size, 2, device="cuda")
    targets = torch.randn(batch_size, 13, device="cuda")

    elapsed = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.cuda.amp.autocast():
            prediction, _ = model([front, left], meta)
            loss = torch.nn.functional.smooth_l1_loss(prediction, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if step > 0:
            elapsed.append(time.perf_counter() - started)
    return {
        "batch": batch_size,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "seconds_per_step": sum(elapsed) / len(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32])
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--safety-gib", type=float, default=4.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA no esta disponible en este entorno")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("batch | alloc GiB | reserv GiB | s/step | samples/s | estado")
    safe = []
    results = []
    for batch in args.batches:
        try:
            result = benchmark(batch, args.image_size, args.steps)
            status = "SEGURO" if result["peak_reserved_gib"] <= args.safety_gib else "SIN_MARGEN"
            if status == "SEGURO":
                safe.append(batch)
                results.append(result)
            print(
                f"{batch:>5} | {result['peak_allocated_gib']:>9.2f} | "
                f"{result['peak_reserved_gib']:>10.2f} | "
                f"{result['seconds_per_step']:>6.3f} | "
                f"{batch / result['seconds_per_step']:>9.2f} | {status}"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:>5} |       OOM |        OOM |      - | NO_CABE")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    if safe:
        fastest = max(results, key=lambda item: item["batch"] / item["seconds_per_step"])
        print(f"Mayor lote con margen: {max(safe)}")
        print(
            f"Mayor throughput medido: batch {fastest['batch']} "
            f"({fastest['batch'] / fastest['seconds_per_step']:.2f} muestras/s)"
        )
        print("Selecciona entre ambos con un entrenamiento corto real; batch maximo no siempre es mas rapido.")
    else:
        print("Ningun lote evaluado dejo el margen configurado; prueba batch 4")


if __name__ == "__main__":
    main()
