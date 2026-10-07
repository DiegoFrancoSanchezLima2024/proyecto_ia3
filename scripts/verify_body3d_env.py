"""Verificación exhaustiva y no destructiva del entorno Body3D."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.body3d.readiness import preparacion_cuerpo3d


def main() -> int:
    import numpy as np
    import smplx
    import torch

    status = preparacion_cuerpo3d(ROOT)
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "readiness": status,
        "smplx_models": {},
    }

    model_path = status["assets"]["smplx_model_directory"]["path"]
    if model_path and status["smplx_ready"]:
        for gender in ("male", "female", "neutral"):
            model = smplx.create(
                model_path,
                model_type="smplx",
                gender=gender,
                ext="npz",
                use_pca=False,
            )
            report["smplx_models"][gender] = {
                "vertices": int(model.get_num_verts()),
                "faces": int(len(model.faces)),
            }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status["ready"] and torch.cuda.is_available() else 2


if __name__ == "__main__":
    raise SystemExit(main())
