"""Contratos verificables para el módulo corporal 3D."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


VALID_SEX_CODES = {"female", "male", "neutral"}


@dataclass(frozen=True)
class SolicitudCuerpo3D:
    views: dict[str, Path]
    height_cm: float
    weight_kg: float
    sex: str

    def validate(self, require_files: bool = True) -> None:
        required = {"front", "left", "right"}
        missing = sorted(required - set(self.views))
        if missing:
            raise ValueError(f"Faltan vistas 3D requeridas: {', '.join(missing)}")
        if not 120.0 <= float(self.height_cm) <= 230.0:
            raise ValueError("height_cm fuera del rango adulto admitido [120, 230]")
        if not 30.0 <= float(self.weight_kg) <= 250.0:
            raise ValueError("weight_kg fuera del rango admitido [30, 250]")
        if self.sex not in VALID_SEX_CODES:
            raise ValueError("sex debe ser female, male o neutral")
        if require_files:
            absent = [str(path) for path in self.views.values() if not Path(path).is_file()]
            if absent:
                raise FileNotFoundError(f"No existen vistas: {', '.join(absent)}")

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["views"] = {name: str(path) for name, path in self.views.items()}
        return payload

