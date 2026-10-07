"""Preflight de dependencias y activos licenciados del módulo 3D."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


CORE_MODULES = {
    "torch": "torch",
    "numpy": "numpy",
    "smplx": "smplx",
}

# Importaciones transitivas que usa el inicializador oficial de SHAPY. Aunque
# algunas son de renderizado, ``human_shape.utils`` las importa al cargar el
# paquete y por tanto son obligatorias si se ejecuta el código sin modificar.
SHAPY_RUNTIME_MODULES = {
    "loguru": "loguru",
    "omegaconf": "omegaconf",
    "yacs": "yacs",
    "kornia": "kornia",
    "fvcore": "fvcore",
    "trimesh": "trimesh",
    "pyrender": "pyrender",
    "open3d": "open3d",
    "jpeg4py": "jpeg4py",
    "opencv": "cv2",
    "Pillow": "PIL",
    "matplotlib": "matplotlib",
    "torchvision": "torchvision",
}


def _modules_available(modules: dict[str, str]) -> dict[str, bool]:
    return {
        label: importlib.util.find_spec(module_name) is not None
        for label, module_name in modules.items()
    }


def _first_existing(candidates: list[Path], kind: str) -> Path | None:
    for candidate in candidates:
        if kind == "file" and candidate.is_file():
            return candidate
        if kind == "directory" and candidate.is_dir():
            return candidate
    return None


def preparacion_cuerpo3d(project_root: Path) -> dict:
    project_root = Path(project_root)
    body3d_root = project_root / "pretrained" / "body3d"
    model_candidates = [
        body3d_root / "models",
        body3d_root / "models" / "smplx" / "models",
    ]
    required_models = ("SMPLX_MALE.npz", "SMPLX_FEMALE.npz", "SMPLX_NEUTRAL.npz")
    model_root = next(
        (
            candidate
            for candidate in model_candidates
            if all((candidate / "smplx" / filename).is_file() for filename in required_models)
        ),
        None,
    )
    checkpoint = _first_existing(
        [
            body3d_root / "shapy.ckpt",
            body3d_root
            / "trained_models"
            / "shapy"
            / "SHAPY_A"
            / "checkpoints"
            / "best_checkpoint",
        ],
        "file",
    )
    dependencies = _modules_available(CORE_MODULES)
    shapy_runtime_dependencies = _modules_available(SHAPY_RUNTIME_MODULES)
    shapy_source_candidates = [
        project_root / "vendor" / "shapy",
        body3d_root / "shapy-master",
    ]
    shapy_source_root = next(
        (
            candidate
            for candidate in shapy_source_candidates
            if (candidate / "regressor" / "human_shape" / "__init__.py").is_file()
        ),
        None,
    )
    shapy_source = (
        importlib.util.find_spec("human_shape") is not None
        or shapy_source_root is not None
    )
    assets = {
        "smplx_model_directory": model_root is not None,
        "shapy_checkpoint": checkpoint is not None,
        "shapy_source": shapy_source,
    }
    core_ready = all(dependencies.values())
    assets_ready = all(assets.values())
    shapy_runtime_ready = all(shapy_runtime_dependencies.values())
    missing = [name for name, available in dependencies.items() if not available]
    missing.extend(name for name, available in assets.items() if not available)
    missing.extend(
        f"shapy_runtime:{name}"
        for name, available in shapy_runtime_dependencies.items()
        if not available
    )
    official_demo_windows_compatible = sys.platform != "win32"
    return {
        "ready": core_ready and assets_ready and shapy_runtime_ready,
        "assets_ready": assets_ready,
        "runtime_ready": core_ready and shapy_runtime_ready,
        "smplx_ready": core_ready and model_root is not None,
        "shapy_initializer_ready": (
            checkpoint is not None and shapy_source and shapy_runtime_ready
        ),
        "dependencies": dependencies,
        "shapy_runtime_dependencies": shapy_runtime_dependencies,
        "platform": {
            "name": sys.platform,
            "official_demo_compatible": official_demo_windows_compatible,
            "execution_strategy": (
                "official_demo"
                if official_demo_windows_compatible
                else "custom_windows_runner"
            ),
            "note": (
                None
                if official_demo_windows_compatible
                else "regressor/demo.py importa el modulo Unix 'resource'; "
                "en Windows se necesita un adaptador sin esa dependencia."
            ),
        },
        "assets": {
            "smplx_model_directory": {
                "path": str(model_root) if model_root else None,
                "available": model_root is not None,
                "models": list(required_models),
            },
            "shapy_checkpoint": {
                "path": str(checkpoint) if checkpoint else None,
                "available": checkpoint is not None,
                "size_bytes": checkpoint.stat().st_size if checkpoint else None,
            },
            "shapy_source": {
                "path": str(shapy_source_root) if shapy_source_root else None,
                "available": shapy_source,
            },
        },
        "missing": missing,
        "license_notice": (
            "Los modelos SMPL-X/SHAPY requieren aceptación y descarga manual de sus licencias; "
            "la presencia del código no concede uso comercial."
        ),
    }
