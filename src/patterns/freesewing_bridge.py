"""Audita si una predicción tiene los datos requeridos por FreeSewing."""

from __future__ import annotations

import json
from pathlib import Path


def cargar_mapa_medidas(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Versión del mapa FreeSewing no compatible")
    return payload


def auditar_entradas_freesewing(prediction: dict, mapping: dict) -> dict:
    if "bespoke" not in prediction:
        raise ValueError("La predicción no contiene el contrato a medida")
    measurements = prediction["measurements"]
    reports = {}
    for garment_name in prediction["bespoke"]["garments"]:
        design = mapping["designs"][garment_name]
        values_mm = {}
        missing = []
        for target_name, source_name in design["required"].items():
            if source_name is None or source_name not in measurements:
                missing.append(target_name)
                continue
            source = measurements[source_name]
            value_cm = float(source.get("final_cm", source["estimate_cm"]))
            values_mm[target_name] = round(value_cm * 10.0, 1)
        reports[garment_name] = {
            "design": design["design"],
            "package": design["package"],
            "ready": not missing,
            "measurements_mm": values_mm,
            "missing_measurements": missing,
            "coverage": round(len(values_mm) / len(design["required"]), 3),
        }
    return {
        "engine": mapping["engine"],
        "ready": all(report["ready"] for report in reports.values()),
        "garments": reports,
        "units": "mm",
        "policy": "missing_measurements_are_never_filled_with_unvalidated_guesses",
    }

