"""
src/utils/config.py
====================
Carga y acceso centralizado a la configuracion YAML.
"""
import yaml
from pathlib import Path


def _deep_merge(base: dict, override: dict) -> dict:
    """Combina configuraciones anidadas sin modificar los diccionarios fuente."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def cargar_config(path: str | Path = None, _seen: set[Path] | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Herencia circular de configuracion: {path}")
    seen.add(path)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(cargar_config(parent_path, seen), config)


def get_meas_cols(cfg: dict) -> list:
    return cfg["dataset"]["measurements"]


def get_device(cfg: dict) -> str:
    import torch
    req = cfg["hardware"].get("device", "cuda")
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
