"""Diagnostico del entorno principal y prueba CUDA minima."""

from __future__ import annotations

import importlib
import platform

import numpy
import pandas
import PIL
import sklearn
import timm
import torch
import torchvision
import yaml


print(f"Python:      {platform.python_version()}")
print(f"PyTorch:     {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"CUDA:        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU:         {properties.name}")
    print(f"VRAM:        {properties.total_memory / 2**30:.2f} GiB")
    print(f"Compute:     {properties.major}.{properties.minor}")
print(f"timm:        {timm.__version__}")
print(f"numpy:       {numpy.__version__}")
print(f"pandas:      {pandas.__version__}")
print(f"sklearn:     {sklearn.__version__}")
print(f"Pillow:      {PIL.__version__}")

for optional in ("smplx", "trimesh", "segment_anything"):
    status = "instalado" if importlib.util.find_spec(optional) else "opcional/no instalado"
    print(f"{optional:<12}{status}")

if torch.cuda.is_available():
    left = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    right = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    result = left @ right
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    del left, right, result
    torch.cuda.empty_cache()
print("=== ENTORNO PRINCIPAL OK ===")
