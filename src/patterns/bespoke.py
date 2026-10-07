"""Convierte medidas corporales en un contrato de confección a medida.

Esta capa no asigna tallas comerciales. Conserva por separado las medidas del
cuerpo y las medidas terminadas de la prenda (cuerpo + holgura).
"""

from __future__ import annotations

import json
from pathlib import Path


TRUSTED_CUT_SOURCES = frozenset({"manual_tape", "calibrated_3d"})


def cargar_config_a_medida(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Versión de configuración de confección no compatible")
    return payload


def _measurement_value(measurement: dict) -> float:
    return float(measurement.get("final_cm", measurement["estimate_cm"]))


def _garment_plan(result: dict, garment_name: str, garment: dict) -> dict:
    measurements = result["measurements"]
    required = list(garment["required_measurements"])
    missing = [name for name in required if name not in measurements]
    available = [name for name in required if name in measurements]

    body = {
        name: round(_measurement_value(measurements[name]), 2) for name in available
    }
    finished = dict(body)
    for name, ease_cm in garment.get("ease_cm", {}).items():
        if name in body:
            finished[name] = round(body[name] + float(ease_cm), 2)

    unvalidated = [
        name
        for name in available
        if measurements[name].get("source", "model_estimate")
        not in TRUSTED_CUT_SOURCES
        or measurements[name].get("requires_manual_confirmation", False)
    ]
    outliers = [
        name
        for name in available
        if measurements[name].get("anthropometric_status") == "outlier"
    ]
    blockers = []
    if missing:
        blockers.append("missing_required_measurements")
    if result.get("decision") == "repeat_capture":
        blockers.append("capture_rejected")
    if outliers:
        blockers.append("anthropometric_outliers")
    if unvalidated:
        blockers.append("automatic_measurements_not_cut_validated")

    if missing or result.get("decision") == "repeat_capture":
        status = "blocked"
    elif blockers:
        status = "draft_ready"
    else:
        status = "cut_ready"

    return {
        "garment": garment_name,
        "fit_mode": "made_to_measure",
        "pattern_status": status,
        "body_measurements_cm": body,
        "finished_garment_measurements_cm": finished,
        "ease_cm": garment.get("ease_cm", {}),
        "seam_allowance_cm": float(garment["seam_allowance_cm"]),
        "hem_allowance_cm": float(garment["hem_allowance_cm"]),
        "pattern_pieces": list(garment["pattern_pieces"]),
        "required_measurements": required,
        "missing_measurements": missing,
        "unvalidated_for_cutting": unvalidated,
        "anthropometric_outliers": outliers,
        "blockers": blockers,
        "style_profile": garment["style_profile"],
    }


def aplicar_plan_a_medida(
    result: dict,
    gender_code: int,
    height_cm: float,
    config: dict,
) -> dict:
    """Añade prendas a medida y nunca crea etiquetas S/M/L."""
    sex = "male" if int(gender_code) == 1 else "female"
    garments = {
        name: _garment_plan(result, name, garment)
        for name, garment in config["sex_profiles"][sex]["garments"].items()
    }
    statuses = [garment["pattern_status"] for garment in garments.values()]
    if "blocked" in statuses:
        overall_status = "blocked"
    elif "draft_ready" in statuses:
        overall_status = "draft_ready"
    else:
        overall_status = "cut_ready"

    result.pop("sizing", None)
    result["bespoke"] = {
        "config_id": config["config_id"],
        "sex_profile": sex,
        "height_cm": float(height_cm),
        "fit_mode": "made_to_measure",
        "overall_pattern_status": overall_status,
        "garments": garments,
        "policy": (
            "No usa tallas comerciales. Las holguras se suman solo a la prenda; "
            "nunca modifican las medidas corporales."
        ),
        "cutting_rule": (
            "Un molde automático es borrador hasta que todas sus medidas críticas "
            "procedan de una fuente validada para corte."
        ),
    }
    return result

