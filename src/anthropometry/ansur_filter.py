"""Control de plausibilidad ANSUR II sin alterar las medidas predichas."""

from __future__ import annotations

import json
import math
from pathlib import Path


def cargar_referencia_antropometrica(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Versión de referencia antropométrica no compatible")
    return payload


def _expected_cm(model: dict, height_cm: float, weight_kg: float) -> float:
    coefficients = model["coefficients"]
    return (
        float(coefficients["intercept"])
        + float(coefficients["height_cm"]) * height_cm
        + float(coefficients["weight_kg"]) * weight_kg
    )


def aplicar_filtro_antropometrico(
    result: dict,
    gender_code: int,
    height_cm: float,
    weight_kg: float | None,
    reference: dict,
) -> dict:
    """Añade alertas y confianza; nunca modifica estimate_cm ni final_cm."""
    sex = "male" if int(gender_code) == 1 else "female"
    section = reference["sex_models"][sex]
    domain = section["input_domain_p01_p99"]
    reasons = []
    if weight_kg is None:
        reasons.append("weight_kg_missing")
    elif not math.isfinite(float(weight_kg)):
        reasons.append("weight_kg_invalid")
    if not domain["height_cm"][0] <= float(height_cm) <= domain["height_cm"][1]:
        reasons.append("height_outside_ansur_p01_p99")
    if weight_kg is not None and math.isfinite(float(weight_kg)):
        if not domain["weight_kg"][0] <= float(weight_kg) <= domain["weight_kg"][1]:
            reasons.append("weight_outside_ansur_p01_p99")

    active = not reasons
    checks = {}
    flagged = []
    if active:
        for name, model in section["measurements"].items():
            if name not in result["measurements"]:
                continue
            measurement = result["measurements"][name]
            observed = float(measurement.get("final_cm", measurement["estimate_cm"]))
            expected = _expected_cm(model, float(height_cm), float(weight_kg))
            lower = expected + float(model["residual_interval_cm"][0])
            upper = expected + float(model["residual_interval_cm"][1])
            source = measurement.get("source", "model_estimate")
            predicted_lower = measurement.get("lower_cm")
            predicted_upper = measurement.get("upper_cm")
            if (
                source != "manual_tape"
                and predicted_lower is not None
                and predicted_upper is not None
            ):
                comparison_interval = [float(predicted_lower), float(predicted_upper)]
            else:
                comparison_interval = [observed, observed]
            overlaps = (
                comparison_interval[1] >= lower and comparison_interval[0] <= upper
            )
            status = "plausible" if overlaps else "outlier"
            checks[name] = {
                "observed_cm": round(observed, 2),
                "observed_interval_cm": [round(v, 2) for v in comparison_interval],
                "expected_cm": round(expected, 2),
                "plausible_interval_cm": [round(lower, 2), round(upper, 2)],
                "status": status,
                "ansur_column": model["ansur_column"],
            }
            measurement["anthropometric_status"] = status
            if status == "outlier":
                flagged.append(name)
                measurement["requires_manual_confirmation"] = True

    result["anthropometric_check"] = {
        "reference_id": reference["reference_id"],
        "sex_model": sex,
        "active": active,
        "inactive_reasons": reasons,
        "flagged_measurements": flagged,
        "checks": checks,
        "policy": "confidence_and_plausibility_only_does_not_change_centimeters",
        "notice": (
            "ANSUR II es una muestra militar estadounidense. Se usa únicamente "
            "como control externo de plausibilidad, no como verdad individual ni "
            "como tabla de tallas."
        ),
    }
    if flagged and result.get("decision") not in {None, "repeat_capture"}:
        result["decision"] = "manual_confirmation"
    return result
