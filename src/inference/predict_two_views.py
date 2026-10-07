"""Inferencia oficial de SATRE-IA con dos siluetas y estatura.

La ruta no acepta peso ni sexo como entradas del modelo. El peso se predice
como objetivo auxiliar y el tipo de conjunto solo se usa después, para decidir
qué moldes ofrecer en la interfaz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predict_measurements import PredictorDeMedidas
from src.inference.predict_profile_ensemble import PredictorPerfilesArboles
from src.inference.predict_four_views import (
    aplicar_confirmaciones_manuales,
    aplicar_politica_captura,
    load_confirmed_measurements,
)
from src.patterns.bespoke import aplicar_plan_a_medida, cargar_config_a_medida


def predecir_dos_vistas(
    exp_dir: Path,
    front_path: Path,
    left_path: Path,
    height_cm: float,
    capture_manifest_path: Path | None = None,
    clothing_fit: str = "unknown",
    garment_route: str = "male",
    confirmed_measurements: dict[str, float] | None = None,
    bespoke_config_path: Path | None = None,
    backend: str = "profiles",
) -> dict:
    if backend == "profiles":
        predictor = PredictorPerfilesArboles(exp_dir)
    elif backend == "cnn":
        predictor = PredictorDeMedidas(exp_dir)
    else:
        raise ValueError("backend debe ser profiles o cnn")
    base = predictor.predict_paths(front_path, left_path, height_cm)

    capture_warnings: list[str] = []
    if capture_manifest_path is not None:
        manifest = json.loads(Path(capture_manifest_path).read_text(encoding="utf-8"))
        capture_warnings = [str(item) for item in manifest.get("warnings", [])]

    quality_warnings = [*base["quality"]["warnings"]]
    decision = "manual_confirmation" if any(
        item["requires_manual_confirmation"]
        for item in base["measurements"].values()
    ) else "accepted"
    decision, policy_warnings = aplicar_politica_captura(
        decision, capture_warnings, clothing_fit
    )
    quality_warnings.extend(policy_warnings)

    if garment_route not in {"male", "female"}:
        raise ValueError("garment_route debe ser male o female")

    result = {
        "schema_version": "2.0",
        "decision": decision,
        "model": base["model"],
        "checkpoint_epoch": base["checkpoint_epoch"],
        "device": base["device"],
        "bias_correction_cm": base.get("bias_correction_cm", {}),
        "estimated_weight": base.get("estimated_weight"),
        "inputs": {
            "front_mask": str(front_path),
            "left_mask": str(left_path),
            "height_cm": height_cm,
            "garment_route": garment_route,
            "model_inputs": ["front_silhouette", "left_silhouette", "height_cm"],
            "clothing_fit": clothing_fit,
            "capture_manifest": str(capture_manifest_path)
            if capture_manifest_path is not None
            else None,
        },
        "quality": {
            "front": base["quality"]["front"],
            "left": base["quality"]["side"],
            "warnings": quality_warnings,
        },
        "measurements": base["measurements"],
        "notice": (
            "Estimación con dos siluetas BodyM y estatura. El peso también es "
            "estimado por la red; no fue escrito por el usuario. Confirme las "
            "medidas marcadas antes de cortar tela."
        ),
    }

    if confirmed_measurements:
        aplicar_confirmaciones_manuales(result, confirmed_measurements)

    # Esta elección no entra a la red: únicamente habilita saco+pantalón o
    # saco+falda en el módulo de patronaje.
    gender_code_for_patterns = 1 if garment_route == "male" else 0
    bespoke_path = bespoke_config_path or ROOT / "configs" / "bespoke_garments.json"
    aplicar_plan_a_medida(
        result,
        gender_code_for_patterns,
        height_cm,
        cargar_config_a_medida(bespoke_path),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="experiments/exp_008_perfiles_arboles")
    parser.add_argument("--backend", choices=["profiles", "cnn"], default="profiles")
    parser.add_argument("--front-mask", required=True)
    parser.add_argument("--left-mask", required=True)
    parser.add_argument("--height-cm", required=True, type=float)
    parser.add_argument("--capture-manifest")
    parser.add_argument("--clothing-fit", choices=["tight", "loose", "unknown"], default="unknown")
    parser.add_argument("--garment-route", choices=["male", "female"], default="male")
    parser.add_argument("--confirmed-measurements")
    parser.add_argument("--bespoke-config", default="configs/bespoke_garments.json")
    parser.add_argument("--output", default="outputs/two_view_prediction.json")
    args = parser.parse_args()

    result = predecir_dos_vistas(
        exp_dir=ROOT / args.exp,
        front_path=Path(args.front_mask),
        left_path=Path(args.left_mask),
        height_cm=args.height_cm,
        capture_manifest_path=Path(args.capture_manifest) if args.capture_manifest else None,
        clothing_fit=args.clothing_fit,
        garment_route=args.garment_route,
        confirmed_measurements=(
            load_confirmed_measurements(Path(args.confirmed_measurements))
            if args.confirmed_measurements
            else None
        ),
        bespoke_config_path=ROOT / args.bespoke_config,
        backend=args.backend,
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Decisión: {result['decision']}")
    if result.get("estimated_weight"):
        print(f"Peso estimado: {result['estimated_weight']['estimate_kg']:.2f} kg")
    for warning in result["quality"]["warnings"]:
        print(f"[CALIDAD] {warning}")
    print(f"Guardado: {output_path}")


if __name__ == "__main__":
    main()
