"""Filtro auditable de talla: recomienda ajuste sin alterar centímetros."""

from __future__ import annotations

import json
from pathlib import Path


SIZE_ORDER = ("XS", "S", "M", "L", "XL", "XXL")


def load_size_chart(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _classify(value: float, sizes: dict, measurement: str) -> str | None:
    available = [name for name in SIZE_ORDER if measurement in sizes.get(name, {})]
    for index, name in enumerate(available):
        lower, upper = sizes[name][measurement]
        if lower <= value < upper or (index == len(available) - 1 and value == upper):
            return name
    return None


def _interval_sizes(value: dict, sizes: dict, measurement: str) -> list[str]:
    lower = value.get("lower_cm")
    upper = value.get("upper_cm")
    if lower is None or upper is None:
        return []
    matches = []
    for name in SIZE_ORDER:
        if measurement not in sizes.get(name, {}):
            continue
        band_lower, band_upper = sizes[name][measurement]
        if float(upper) >= band_lower and float(lower) < band_upper:
            matches.append(name)
    return matches


def _confidence(measurements: dict, names: tuple[str, ...], sizes: dict) -> dict:
    scores = []
    evidence = []
    for name in names:
        value = measurements.get(name)
        if value is None:
            continue
        source = value.get("source", "model_estimate")
        score = 1.0 if source == "manual_tape" else 0.55
        anthropometric_status = value.get("anthropometric_status", "not_checked")
        if anthropometric_status == "outlier":
            score *= 0.5
        interval_sizes = _interval_sizes(value, sizes, name)
        if source != "manual_tape" and len(interval_sizes) > 1:
            score *= 0.75
        scores.append(score)
        evidence.append(
            {
                "measurement": name,
                "source": source,
                "size": _classify(
                    float(value.get("final_cm", value["estimate_cm"])), sizes, name
                ),
                "interval_sizes": interval_sizes,
                "anthropometric_status": anthropometric_status,
            }
        )
    score = sum(scores) / len(names) if names else 0.0
    label = "high" if score >= 0.85 else "medium" if score >= 0.60 else "low"
    return {"score": round(score, 2), "label": label, "evidence": evidence}


def _recommend_garment(
    measurements: dict,
    names: tuple[str, ...],
    sizes: dict,
    length_fit: str,
) -> dict:
    per_measurement = {}
    for name in names:
        if name not in measurements:
            continue
        value = measurements[name]
        final_cm = float(value.get("final_cm", value["estimate_cm"]))
        per_measurement[name] = _classify(final_cm, sizes, name)
    valid = [value for value in per_measurement.values() if value in SIZE_ORDER]
    alpha = max(valid, key=SIZE_ORDER.index) if valid else None
    alterations = []
    if len(set(valid)) > 1:
        alterations.append(
            "las medidas caen en tallas distintas; usar la mayor y entallar el molde"
        )
    return {
        "alpha_size": alpha,
        "length_fit": length_fit,
        "display_size": f"{alpha}-{length_fit.title()}" if alpha else None,
        "measurement_sizes": per_measurement,
        "confidence": _confidence(measurements, names, sizes),
        "alteration_notes": alterations,
    }


def apply_size_filter(
    result: dict,
    gender_code: int,
    height_cm: float,
    chart: dict,
) -> dict:
    """Añade recomendaciones; nunca modifica estimate_cm ni final_cm."""
    sex = "male" if int(gender_code) == 1 else "female"
    section = chart[sex]
    sizes = section["sizes"]
    bands = section["length_bands_cm"]
    if height_cm <= bands["short_max"]:
        length_fit = "short"
    elif height_cm <= bands["regular_max"]:
        length_fit = "regular"
    else:
        length_fit = "tall"

    measurements = result["measurements"]
    garments = {
        "jacket": _recommend_garment(
            measurements, ("chest", "waist"), sizes, length_fit
        )
    }
    lower_garment = "trousers" if sex == "male" else "skirt"
    garments[lower_garment] = _recommend_garment(
        measurements, ("waist", "hip"), sizes, length_fit
    )
    result["sizing"] = {
        "chart_id": chart["chart_id"],
        "sex_chart": sex,
        "height_fit": length_fit,
        "garments": garments,
        "notice": chart["notice"],
        "measurement_policy": "filter_only_does_not_change_centimeters",
    }
    return result
