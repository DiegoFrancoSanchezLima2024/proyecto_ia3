"""Compara un regresor explicable de perfiles de silueta sobre BodyM.

Convierte cada máscara en anchos físicos aproximados usando la estatura como
escala: ancho_px / alto_corporal_px * estatura_cm. Así reduce la dependencia
del encuadre y de la distancia exacta de cámara.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.silhouette_profiles import caracteristicas_dos_vistas

OBJETIVOS = [
    "chest", "waist", "hip", "thigh", "calf", "ankle", "arm-length",
    "forearm", "wrist", "bicep", "shoulder-breadth", "leg-length",
    "shoulder-to-crotch", "weight_kg",
]
DIRECTORIOS = {
    "train": "train", "val": "train", "calibration": "train",
    "test": "testA", "test_wild": "testB",
}


def extraer_split(split: str, bins: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(ROOT / "dataset" / "bodym_clean" / split / "data.csv")
    raw = ROOT / "dataset" / "bodym" / DIRECTORIOS[split]
    features = []
    for row in frame.itertuples(index=False):
        filename = str(row.photo_id)
        if not filename.lower().endswith(".png"):
            filename += ".png"
        features.append(
            caracteristicas_dos_vistas(
                raw / "mask" / filename,
                raw / "mask_left" / filename,
                row.height_cm,
                bins,
            )
        )
    return (
        np.asarray(features, dtype=np.float32),
        frame[OBJETIVOS].to_numpy(dtype=np.float32),
        frame,
    )


def metricas(pred: np.ndarray, target: np.ndarray) -> dict:
    by_target = {
        name: float(mean_absolute_error(target[:, i], pred[:, i]))
        for i, name in enumerate(OBJETIVOS)
    }
    measures = OBJETIVOS[:-1]
    return {
        "mae_13_cm": float(np.mean([by_target[name] for name in measures])),
        "mae_weight_kg": by_target["weight_kg"],
        "bias_weight_kg": float(np.mean(pred[:, -1] - target[:, -1])),
        "by_target": by_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--trees", type=int, default=350)
    parser.add_argument("--min-leaf", type=int, default=3)
    parser.add_argument("--output-dir", default="experiments/exp_008_perfiles_arboles")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    datasets = {split: extraer_split(split, args.bins) for split in DIRECTORIOS}
    model = ExtraTreesRegressor(
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        max_features=0.85,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(datasets["train"][0], datasets["train"][1])
    report = {
        split: metricas(model.predict(values[0]), values[1])
        for split, values in datasets.items()
    }
    calibration_prediction = model.predict(datasets["calibration"][0])
    calibration_error = np.abs(calibration_prediction - datasets["calibration"][1])
    coverage = 0.90
    k = min(
        len(calibration_error) - 1,
        int(np.ceil((len(calibration_error) + 1) * coverage)) - 1,
    )
    q_hat = np.sort(calibration_error, axis=0)[k]
    joblib.dump(
        {
            "model": model,
            "bins": args.bins,
            "targets": OBJETIVOS,
            "model_inputs": ["front_silhouette", "left_silhouette", "height_cm"],
        },
        output / "model.joblib",
        compress=3,
    )
    np.save(output / "conformal_q_hat.npy", q_hat.astype(np.float32))
    (output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for split, values in report.items():
        print(
            f"{split:12} MAE13={values['mae_13_cm']:.3f} cm "
            f"peso={values['mae_weight_kg']:.3f} kg "
            f"sesgo_peso={values['bias_weight_kg']:+.3f} kg"
        )
    print(f"Guardado: {output}")


if __name__ == "__main__":
    main()
