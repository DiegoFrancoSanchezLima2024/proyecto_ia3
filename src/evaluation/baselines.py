"""Baselines que todo modelo visual de Sastre-IA debe superar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import agregar_predicciones_por_sujeto, calcular_metricas
from src.utils.config import cargar_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="reports/baselines.json")
    args = parser.parse_args()
    cfg = cargar_config(ROOT / args.config)
    data_root = ROOT / cfg["paths"]["data_root"]
    measures = cfg["dataset"]["measurements"]
    features = [cfg["dataset"]["height_col"], cfg["dataset"]["gender_col"]]
    weight_features = [*features, cfg["dataset"]["weight_col"]]
    train = pd.read_csv(data_root / "train" / "data.csv")
    train_y = train[measures].to_numpy(dtype=np.float32)
    train_x = train[features].to_numpy(dtype=np.float32)
    train_mean = train_y.mean(axis=0)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(train_x, train_y)
    ridge_with_weight = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge_with_weight.fit(train[weight_features].to_numpy(dtype=np.float32), train_y)

    report = {"measurements": measures, "features": features, "splits": {}}
    for split in ("val", "test", "test_wild"):
        frame = pd.read_csv(data_root / split / "data.csv")
        targets = frame[measures].to_numpy(dtype=np.float32)
        mean_predictions = np.broadcast_to(train_mean, targets.shape)
        ridge_predictions = ridge.predict(frame[features].to_numpy(dtype=np.float32))
        weight_predictions = ridge_with_weight.predict(
            frame[weight_features].to_numpy(dtype=np.float32)
        )
        report["splits"][split] = {
            "train_mean": calcular_metricas(mean_predictions, targets, measures),
            "height_gender_ridge": calcular_metricas(ridge_predictions, targets, measures),
            "height_gender_weight_ridge": calcular_metricas(
                weight_predictions,
                targets,
                measures,
            ),
        }
        subject_results = {"n_subjects": int(frame["subject_id"].nunique())}
        for name, predictions in {
            "train_mean": mean_predictions,
            "height_gender_ridge": ridge_predictions,
            "height_gender_weight_ridge": weight_predictions,
        }.items():
            values = frame[["subject_id"]].copy()
            for index, measure in enumerate(measures):
                values[f"true_{measure}"] = targets[:, index]
                values[f"pred_{measure}"] = predictions[:, index]
            aggregated = agregar_predicciones_por_sujeto(values, measures)
            subject_results[name] = calcular_metricas(
                aggregated[[f"pred_{measure}" for measure in measures]].to_numpy(),
                aggregated[[f"true_{measure}" for measure in measures]].to_numpy(),
                measures,
            )
        report["splits"][split]["_subject_aggregate"] = subject_results
        print(
            f"{split:<10} mean={report['splits'][split]['train_mean']['_global']['mae']:.3f}cm "
            f"ridge={report['splits'][split]['height_gender_ridge']['_global']['mae']:.3f}cm"
            f" ridge+weight={report['splits'][split]['height_gender_weight_ridge']['_global']['mae']:.3f}cm"
        )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Guardado: {output}")


if __name__ == "__main__":
    main()
