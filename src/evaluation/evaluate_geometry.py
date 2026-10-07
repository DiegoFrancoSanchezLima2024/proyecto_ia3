"""Evalua el estimador geometrico 2.5D sin entrenar una red neuronal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    agregar_predicciones_por_sujeto,
    calcular_metricas,
    imprimir_tabla_metricas,
)
from src.geometry.geometric_est import EstimadorGeometrico
from src.training.train import _dataset
from src.utils.config import cargar_config


def evaluate_geometry(cfg: dict, split: str) -> dict:
    train_ds = _dataset(cfg, "train")
    dataset = _dataset(cfg, split, train_ds)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("eval_batch_size", 32)),
        shuffle=False,
        num_workers=0,
    )
    estimator = EstimadorGeometrico(train_ds.meas_cols)
    predictions, targets = [], []
    for front, left, meta, _, raw_targets in loader:
        heights = meta[:, 0].numpy() * train_ds.height_std + train_ds.height_mean
        predictions.append(estimator(front.numpy(), left.numpy(), heights))
        targets.append(raw_targets.numpy())

    predictions_np = np.concatenate(predictions)
    targets_np = np.concatenate(targets)
    metrics = calcular_metricas(predictions_np, targets_np, train_ds.meas_cols)

    output = dataset.df.iloc[: len(predictions_np)][["subject_id", "photo_id"]].copy()
    for index, name in enumerate(train_ds.meas_cols):
        output[f"true_{name}"] = targets_np[:, index]
        output[f"pred_{name}"] = predictions_np[:, index]
    by_subject = agregar_predicciones_por_sujeto(output, train_ds.meas_cols)
    subject_predictions = by_subject[
        [f"pred_{name}" for name in train_ds.meas_cols]
    ].to_numpy()
    subject_targets = by_subject[
        [f"true_{name}" for name in train_ds.meas_cols]
    ].to_numpy()
    metrics["_subject_aggregate"] = {
        "n_subjects": int(len(by_subject)),
        "n_captures": int(len(output)),
        "metrics": calcular_metricas(
            subject_predictions, subject_targets, train_ds.meas_cols
        ),
    }

    imprimir_tabla_metricas(metrics, train_ds.meas_cols)
    print(
        f"[Geometry] Agregado por sujeto: n={len(by_subject)} "
        f"MAE={metrics['_subject_aggregate']['metrics']['_global']['mae']:.3f}cm"
    )
    report_path = ROOT / "reports" / f"geometric_baseline_{split}.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[Geometry] Guardado en {report_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--split", default="test", choices=["val", "calibration", "test", "test_wild"]
    )
    args = parser.parse_args()
    evaluate_geometry(cargar_config(ROOT / args.config), args.split)
