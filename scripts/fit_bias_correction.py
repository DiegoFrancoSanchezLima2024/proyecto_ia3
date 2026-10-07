"""Estima el sesgo sistematico por medida en el split de CALIBRACION.

El modelo campeon no solo se equivoca al azar: en algunas medidas se pasa
siempre para el mismo lado (la cintura, por ejemplo, sale ~2 cm de mas en
promedio). Ese corrimiento constante se puede restar sin reentrenar nada.

Regla que no se debe romper: el sesgo se mide SOLO en el split de
calibracion. Si se midiera en test, las metricas de test dejarian de ser
honestas porque el modelo habria visto las respuestas.

Uso:
    python -m src.evaluation.evaluate --exp-dir experiments/exp_006_weight_huber --split calibration
    python scripts/fit_bias_correction.py --exp-dir experiments/exp_006_weight_huber

Escribe experiments/<exp>/bias_correction.json, que
src/inference/predict_measurements.py carga solo si existe.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_subject_errors(csv_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Promedia las fotos de cada sujeto y devuelve el error con signo."""
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"El archivo no tiene filas: {csv_path}")

    measurements = [
        column[len("pred_") :]
        for column in rows[0]
        if column.startswith("pred_") and f"true_{column[len('pred_'):]}" in rows[0]
    ]

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row)

    errors: dict[str, list[float]] = {name: [] for name in measurements}
    for subject_rows in by_subject.values():
        for name in measurements:
            try:
                predicted = statistics.fmean(
                    float(row[f"pred_{name}"]) for row in subject_rows
                )
                actual = statistics.fmean(
                    float(row[f"true_{name}"]) for row in subject_rows
                )
            except (KeyError, ValueError):
                continue
            errors[name].append(predicted - actual)
    return measurements, errors


def fit(errors: dict[str, list[float]], min_cm: float, z: float) -> dict:
    """Se queda solo con los sesgos que se distinguen del ruido."""
    accepted: dict[str, float] = {}
    detail: dict[str, dict] = {}
    for name, values in errors.items():
        if len(values) < 10:
            continue
        mean = statistics.fmean(values)
        sem = statistics.stdev(values) / (len(values) ** 0.5)
        significant = abs(mean) > max(min_cm, z * sem)
        detail[name] = {
            "mean_signed_error_cm": round(mean, 3),
            "standard_error_cm": round(sem, 3),
            "n_subjects": len(values),
            "applied": significant,
        }
        if significant:
            accepted[name] = round(mean, 3)
    return {"bias_cm": accepted, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Por defecto <exp-dir>/predictions_calibration.csv",
    )
    parser.add_argument(
        "--min-cm",
        type=float,
        default=0.2,
        help="Sesgo minimo que vale la pena corregir",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=2.0,
        help="Cuantos errores estandar debe superar el sesgo para aplicarse",
    )
    args = parser.parse_args()

    exp_dir = args.exp_dir if args.exp_dir.is_absolute() else ROOT / args.exp_dir
    predictions = args.predictions or exp_dir / "predictions_calibration.csv"
    if not predictions.is_file():
        raise SystemExit(
            f"No existe {predictions}.\n"
            "Genera primero el split de calibracion con:\n"
            f"  python -m src.evaluation.evaluate --exp-dir {args.exp_dir} --split calibration"
        )
    if "test" in predictions.name:
        raise SystemExit(
            "Rechazado: el sesgo no se puede estimar sobre el split de test, "
            "eso contamina la evaluacion."
        )

    _, errors = load_subject_errors(predictions)
    payload = fit(errors, args.min_cm, args.z)
    payload["source"] = str(predictions.relative_to(ROOT)) if predictions.is_relative_to(ROOT) else str(predictions)
    payload["min_cm"] = args.min_cm
    payload["z"] = args.z

    output = exp_dir / "bias_correction.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Escrito: {output}")
    print(f"{'medida':<20} {'sesgo':>9} {'error est.':>11}  aplicado")
    print("-" * 55)
    for name, info in payload["detail"].items():
        mark = "si" if info["applied"] else "no"
        print(
            f"{name:<20} {info['mean_signed_error_cm']:>+7.2f}cm "
            f"{info['standard_error_cm']:>9.2f}cm  {mark}"
        )


if __name__ == "__main__":
    main()
