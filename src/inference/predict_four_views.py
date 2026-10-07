"""Fusiona dos perfiles de una captura de cuatro máscaras sin reentrenar."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predict_measurements import PredictorDeMedidas, calidad_mascara
from src.anthropometry.ansur_filter import (
    aplicar_filtro_antropometrico,
    cargar_referencia_antropometrica,
)
from src.patterns.bespoke import aplicar_plan_a_medida, cargar_config_a_medida


CRITICAL_MEASUREMENTS = frozenset({"chest", "waist", "hip"})


def aplicar_confirmaciones_manuales(
    result: dict,
    confirmed_measurements: dict[str, float],
) -> dict:
    """Añade una capa asistida sin sobrescribir la predicción original."""
    measurements = result["measurements"]
    unknown = sorted(set(confirmed_measurements) - set(measurements))
    if unknown:
        raise ValueError(f"Medidas confirmadas desconocidas: {', '.join(unknown)}")

    normalized = {}
    for name, raw_value in confirmed_measurements.items():
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 < value < 300.0:
            raise ValueError(f"Valor confirmado inválido para {name}: {raw_value}")
        normalized[name] = round(value, 2)

    for name, prediction in measurements.items():
        estimate = float(prediction["estimate_cm"])
        if name in normalized:
            confirmed = normalized[name]
            prediction.update(
                {
                    "confirmed_cm": confirmed,
                    "final_cm": confirmed,
                    "source": "manual_tape",
                    "model_error_cm": round(estimate - confirmed, 2),
                    "absolute_error_cm": round(abs(estimate - confirmed), 2),
                    "requires_manual_confirmation": False,
                }
            )
        else:
            prediction.update(
                {
                    "confirmed_cm": None,
                    "final_cm": estimate,
                    "source": "model_estimate",
                    "model_error_cm": None,
                    "absolute_error_cm": None,
                }
            )

    confirmed_names = sorted(normalized)
    status = "complete" if len(confirmed_names) == len(measurements) else "partial"
    result["confirmation"] = {
        "status": status,
        "confirmed_count": len(confirmed_names),
        "total_measurements": len(measurements),
        "confirmed_measurements": confirmed_names,
        "protocol": "manual_tape",
    }
    if result["decision"] != "repeat_capture":
        result["decision"] = (
            "accepted_assisted" if status == "complete" else "manual_confirmation"
        )
    return result


def load_confirmed_measurements(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("El archivo de confirmaciones debe contener un objeto JSON")
    return payload


def aplicar_politica_captura(
    decision: str,
    capture_warnings: list[str],
    clothing_fit: str,
) -> tuple[str, list[str]]:
    """Impide aprobar centímetros cuando la captura no representa el cuerpo."""
    if clothing_fit not in {"tight", "loose", "unknown"}:
        raise ValueError("clothing_fit debe ser tight, loose o unknown")
    warnings = list(capture_warnings)
    if clothing_fit == "loose":
        warnings.append(
            "ropa holgada: la máscara representa la ropa y no el contorno corporal"
        )
        return "repeat_capture", warnings
    if capture_warnings:
        return "repeat_capture", warnings
    if clothing_fit == "unknown":
        warnings.append("ajuste de la ropa no confirmado")
        if decision == "accepted":
            decision = "manual_confirmation"
    return decision, warnings


def fusionar_predicciones_par(
    left: dict,
    right: dict,
    max_critical_disagreement_cm: float,
) -> dict:
    """Promedia puntos y conserva la unión de intervalos correlacionados."""
    if max_critical_disagreement_cm <= 0:
        raise ValueError("max_critical_disagreement_cm debe ser positivo")
    if left["measurements"].keys() != right["measurements"].keys():
        raise ValueError("Los pares no contienen las mismas medidas")

    measurements = {}
    critical_failures = []
    for name in left["measurements"]:
        left_value = left["measurements"][name]
        right_value = right["measurements"][name]
        disagreement = abs(
            float(left_value["estimate_cm"]) - float(right_value["estimate_cm"])
        )
        lower_values = [
            value
            for value in (left_value["lower_cm"], right_value["lower_cm"])
            if value is not None
        ]
        upper_values = [
            value
            for value in (left_value["upper_cm"], right_value["upper_cm"])
            if value is not None
        ]
        critical_disagreement = bool(
            name in CRITICAL_MEASUREMENTS
            and disagreement > max_critical_disagreement_cm
        )
        if critical_disagreement:
            critical_failures.append(name)
        measurements[name] = {
            "estimate_cm": round(
                (float(left_value["estimate_cm"]) + float(right_value["estimate_cm"]))
                / 2.0,
                2,
            ),
            "lower_cm": round(min(lower_values), 2) if lower_values else None,
            "upper_cm": round(max(upper_values), 2) if upper_values else None,
            "left_estimate_cm": float(left_value["estimate_cm"]),
            "right_estimate_cm": float(right_value["estimate_cm"]),
            "side_disagreement_cm": round(disagreement, 2),
            "requires_manual_confirmation": bool(
                left_value["requires_manual_confirmation"]
                or right_value["requires_manual_confirmation"]
                or critical_disagreement
            ),
        }

    quality_warnings = [
        *(f"frente+izquierda: {item}" for item in left["quality"]["warnings"]),
        *(f"frente+derecha: {item}" for item in right["quality"]["warnings"]),
    ]
    if quality_warnings or critical_failures:
        decision = "repeat_capture"
    elif any(
        value["requires_manual_confirmation"] for value in measurements.values()
    ):
        decision = "manual_confirmation"
    else:
        decision = "accepted"

    return {
        "decision": decision,
        "critical_disagreement_threshold_cm": max_critical_disagreement_cm,
        "threshold_calibration": "provisional_safety_threshold",
        "critical_disagreements": critical_failures,
        "quality_warnings": quality_warnings,
        "measurements": measurements,
    }


def predict_four_view_masks(
    exp_dir: Path,
    front_path: Path,
    left_path: Path,
    right_path: Path,
    height_cm: float,
    gender: float,
    weight_kg: float | None,
    back_path: Path | None = None,
    max_critical_disagreement_cm: float = 3.0,
    capture_manifest_path: Path | None = None,
    clothing_fit: str = "unknown",
    confirmed_measurements: dict[str, float] | None = None,
    bespoke_config_path: Path | None = None,
    anthropometric_reference_path: Path | None = None,
) -> dict:
    predictor = PredictorDeMedidas(exp_dir)
    front = predictor.load_mask(front_path)
    left = predictor.load_mask(left_path)
    right_mirrored = predictor.load_mask(right_path, mirror=True)

    left_result = predictor.predict_tensors(
        front, left, height_cm, gender, weight_kg
    )
    right_result = predictor.predict_tensors(
        front, right_mirrored, height_cm, gender, weight_kg
    )
    fusion = fusionar_predicciones_par(
        left_result, right_result, max_critical_disagreement_cm
    )

    back_quality = None
    if back_path is not None:
        back_quality = calidad_mascara(predictor.load_mask(back_path))
        fusion["quality_warnings"].extend(
            f"espalda: {item}" for item in back_quality["warnings"]
        )
        if back_quality["warnings"]:
            fusion["decision"] = "repeat_capture"

    capture_warnings = []
    if capture_manifest_path is not None:
        capture_manifest_path = Path(capture_manifest_path)
        if not capture_manifest_path.is_file():
            raise FileNotFoundError(
                f"No existe el manifiesto de captura: {capture_manifest_path}"
            )
        capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
        capture_warnings = [str(item) for item in capture_manifest.get("warnings", [])]
    fusion["decision"], policy_warnings = aplicar_politica_captura(
        fusion["decision"], capture_warnings, clothing_fit
    )
    fusion["quality_warnings"].extend(policy_warnings)

    result = {
        "schema_version": "1.0",
        "model": left_result["model"],
        "checkpoint_epoch": left_result["checkpoint_epoch"],
        "device": left_result["device"],
        "bias_correction_cm": left_result.get("bias_correction_cm", {}),
        "inputs": {
            "front_mask": str(front_path),
            "left_mask": str(left_path),
            "right_mask": str(right_path),
            "right_was_mirrored": True,
            "back_mask": str(back_path) if back_path is not None else None,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "gender_code": int(gender),
            "clothing_fit": clothing_fit,
            "capture_manifest": (
                str(capture_manifest_path) if capture_manifest_path is not None else None
            ),
        },
        "quality": {
            "front": left_result["quality"]["front"],
            "left": left_result["quality"]["side"],
            "right": right_result["quality"]["side"],
            "back": back_quality,
            "warnings": fusion.pop("quality_warnings"),
        },
        "pair_predictions": {
            "front_left": left_result["measurements"],
            "front_right": right_result["measurements"],
        },
        **fusion,
        "notice": (
            "La espalda se usa solo para calidad. El umbral de desacuerdo es "
            "provisional hasta calibrarlo; confirme medidas críticas antes de cortar."
        ),
    }
    if confirmed_measurements:
        aplicar_confirmaciones_manuales(result, confirmed_measurements)
    reference_path = anthropometric_reference_path or (
        ROOT / "configs" / "anthropometry" / "ansur2_reference.json"
    )
    if reference_path.is_file():
        aplicar_filtro_antropometrico(
            result,
            int(gender),
            height_cm,
            weight_kg,
            cargar_referencia_antropometrica(reference_path),
        )
    bespoke_path = bespoke_config_path or ROOT / "configs" / "bespoke_garments.json"
    aplicar_plan_a_medida(
        result, int(gender), height_cm, cargar_config_a_medida(bespoke_path)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="experiments/exp_006_weight_huber")
    parser.add_argument("--front-mask", required=True)
    parser.add_argument("--left-mask", required=True)
    parser.add_argument("--right-mask", required=True)
    parser.add_argument("--back-mask")
    parser.add_argument("--height-cm", required=True, type=float)
    parser.add_argument("--gender", required=True, type=float, choices=[0.0, 1.0])
    parser.add_argument("--weight-kg", type=float)
    parser.add_argument("--max-critical-disagreement-cm", type=float, default=3.0)
    parser.add_argument("--capture-manifest")
    parser.add_argument(
        "--clothing-fit",
        choices=["tight", "loose", "unknown"],
        default="unknown",
    )
    parser.add_argument("--output", default="outputs/four_view_prediction.json")
    parser.add_argument(
        "--confirmed-measurements",
        help="JSON opcional con medidas de cinta, por ejemplo {\"chest\": 98}",
    )
    parser.add_argument(
        "--bespoke-config", default="configs/bespoke_garments.json"
    )
    parser.add_argument(
        "--anthropometric-reference",
        default="configs/anthropometry/ansur2_reference.json",
    )
    args = parser.parse_args()

    result = predict_four_view_masks(
        exp_dir=ROOT / args.exp,
        front_path=Path(args.front_mask),
        left_path=Path(args.left_mask),
        right_path=Path(args.right_mask),
        back_path=Path(args.back_mask) if args.back_mask else None,
        height_cm=args.height_cm,
        gender=args.gender,
        weight_kg=args.weight_kg,
        max_critical_disagreement_cm=args.max_critical_disagreement_cm,
        capture_manifest_path=(
            Path(args.capture_manifest) if args.capture_manifest else None
        ),
        clothing_fit=args.clothing_fit,
        confirmed_measurements=(
            load_confirmed_measurements(Path(args.confirmed_measurements))
            if args.confirmed_measurements
            else None
        ),
        bespoke_config_path=ROOT / args.bespoke_config,
        anthropometric_reference_path=ROOT / args.anthropometric_reference,
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Decisión: {result['decision']}")
    print(f"{'Medida':<22} {'Modelo':>9} {'Final':>9} {'Fuente':>15}")
    print("-" * 59)
    for name, value in result["measurements"].items():
        final_value = value.get("final_cm", value["estimate_cm"])
        source = value.get("source", "model_estimate")
        print(
            f"{name:<22} {value['estimate_cm']:>7.2f}cm "
            f"{final_value:>7.2f}cm {source:>15}"
        )
    for warning in result["quality"]["warnings"]:
        print(f"[CALIDAD] {warning}")
    if result.get("anthropometric_check", {}).get("flagged_measurements"):
        print(
            "[ANTROPOMETRÍA] revisar: "
            + ", ".join(result["anthropometric_check"]["flagged_measurements"])
        )
    for garment, plan in result["bespoke"]["garments"].items():
        print(f"[A MEDIDA] {garment}: molde {plan['pattern_status']}")
    print(f"Guardado: {output_path}")


if __name__ == "__main__":
    main()
