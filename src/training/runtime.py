"""Utilidades compartidas de datos, precision y hardware."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def resolver_raiz_imagenes_crudas(root: Path, cfg: dict, split: str) -> Path:
    raw_root = root / cfg["paths"].get("bodym_root", "dataset/bodym")
    mapping = cfg["dataset"].get("raw_split_directories", {})
    default = "train" if split in {"train", "val", "calibration"} else split
    return raw_root / mapping.get(split, default)


def configurar_entorno_ejecucion(cfg: dict, device: torch.device) -> None:
    training = cfg["training"]
    if device.type != "cuda":
        return
    allow_tf32 = bool(training.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
    deterministic = bool(training.get("deterministic", False))
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def mover_imagen(tensor: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    tensor = tensor.to(device, non_blocking=device.type == "cuda")
    if channels_last and tensor.ndim == 4:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def geometria_normalizada(
    geo_estimator,
    front_cpu: torch.Tensor,
    left_cpu: torch.Tensor,
    meta_cpu: torch.Tensor,
    label_mean: np.ndarray,
    label_std: np.ndarray,
    height_mean: float,
    height_std: float,
    device: torch.device,
) -> torch.Tensor:
    """Calcula geometria en cm y la lleva al mismo espacio normalizado del target."""
    heights = meta_cpu[:, 0].numpy() * height_std + height_mean
    geometry_cm = geo_estimator(front_cpu.numpy(), left_cpu.numpy(), heights)
    geometry_norm = (geometry_cm - label_mean) / label_std
    return torch.as_tensor(geometry_norm, dtype=torch.float32, device=device)


def dataloader_kwargs(cfg: dict, device: torch.device, workers: int | None = None) -> dict:
    training = cfg["training"]
    workers = int(training.get("num_workers", 0) if workers is None else workers)
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True) and device.type == "cuda"),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    return kwargs
