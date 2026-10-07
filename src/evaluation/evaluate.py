"""Evaluacion independiente sobre val, calibration, Test-A o Test-B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.bodym_dataset import ConjuntoDatosBodyM
from src.evaluation.metrics import (
    agregar_predicciones_por_sujeto,
    calcular_metricas,
    imprimir_tabla_metricas,
)
from src.geometry.geometric_est import EstimadorGeometrico
from src.models.regressor import RegresorMedidasCorporales
from src.training.runtime import (
    amp_context,
    configurar_entorno_ejecucion,
    dataloader_kwargs,
    mover_imagen,
    geometria_normalizada,
    resolver_raiz_imagenes_crudas,
)
from src.utils.config import cargar_config
from src.utils.visualize import graficar_bland_altman, graficar_tabla_metricas, graficar_curvas_entrenamiento


def evaluate(
    exp_dir: Path, split: str, cfg: dict, limit_batches: int | None = None
) -> dict:
    checkpoint_path = exp_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No se encontro el checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    measurements = list(checkpoint["meas_cols"])
    label_mean = np.asarray(checkpoint["label_mean"], dtype=np.float32)
    label_std = np.asarray(checkpoint["label_std"], dtype=np.float32)
    height_mean = float(checkpoint["height_mean"])
    height_std = float(checkpoint["height_std"])
    use_weight_meta = bool(checkpoint.get("use_weight_meta", False))
    use_gender_meta = bool(checkpoint.get("use_gender_meta", True))
    model_cfg = {**cfg["model"], **checkpoint.get("model_config", {})}
    views = checkpoint.get("dataset_views", cfg["dataset"].get("views", ["front", "left"]))

    device = torch.device(
        "cuda"
        if cfg["hardware"].get("device", "cuda") == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    configurar_entorno_ejecucion(cfg, device)
    print(
        f"[Eval] split={split} checkpoint_epoch={checkpoint['epoch']} "
        f"val_MAE={checkpoint['mae']:.3f}cm device={device}"
    )

    history_path = exp_dir / "history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        graficar_curvas_entrenamiento(history, save_path=str(exp_dir / "training_curves.png"))

    model = RegresorMedidasCorporales(
        n_measures=len(measurements),
        embed_dim=model_cfg["embed_dim"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        n_views=len(views),
        dropout=model_cfg["dropout"],
        pretrained_encoder=False,
        frozen_epochs=model_cfg.get("encoder_frozen_epochs", 5),
        adapt_batchnorm_epochs=model_cfg.get("adapt_batchnorm_epochs", 0),
        use_geometry=model_cfg.get("use_geometry", False),
        use_profile_features=model_cfg.get("use_profile_features", False),
        profile_bins=model_cfg.get("profile_bins", 32),
        meta_dim=int(
            checkpoint.get(
                "meta_dim", 1 + int(use_gender_meta) + int(use_weight_meta)
            )
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    if device.type == "cuda" and cfg["training"].get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    csv_path = ROOT / cfg["paths"]["data_root"] / split / "data.csv"
    view_dirs = cfg["dataset"].get("view_directories", {})
    dataset = ConjuntoDatosBodyM(
        data_csv=str(csv_path),
        img_root=str(resolver_raiz_imagenes_crudas(ROOT, cfg, split)),
        meas_cols=measurements,
        height_col=cfg["dataset"].get("height_col", "height_cm"),
        gender_col=cfg["dataset"].get("gender_col", "gender"),
        weight_col=cfg["dataset"].get("weight_col", "weight_kg"),
        use_weight_meta=use_weight_meta,
        use_gender_meta=use_gender_meta,
        img_size=cfg["dataset"]["img_size"],
        augment=False,
        label_mean=label_mean,
        label_std=label_std,
        height_mean=height_mean,
        height_std=height_std,
        weight_mean=checkpoint.get("weight_mean"),
        weight_std=checkpoint.get("weight_std"),
        strict_images=cfg["dataset"].get("strict_images", True),
        front_dir=view_dirs.get("front", "mask"),
        left_dir=view_dirs.get("left", "mask_left"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("eval_batch_size", 32)),
        shuffle=False,
        **dataloader_kwargs(cfg, device),
    )
    geometry = EstimadorGeometrico(measurements) if model_cfg.get("use_geometry") else None
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    channels_last = bool(cfg["training"].get("channels_last", True))
    all_predictions, all_targets = [], []

    with torch.inference_mode():
        for batch_index, (front_cpu, left_cpu, meta_cpu, _, raw_targets) in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            geometry_norm = None
            if geometry is not None:
                geometry_norm = geometria_normalizada(
                    geometry,
                    front_cpu,
                    left_cpu,
                    meta_cpu,
                    label_mean,
                    label_std,
                    height_mean,
                    height_std,
                    device,
                )
            front = mover_imagen(front_cpu, device, channels_last)
            left = mover_imagen(left_cpu, device, channels_last)
            meta = meta_cpu.to(device, non_blocking=device.type == "cuda")
            with amp_context(device, use_amp):
                delta, _ = model([front, left], meta, geometry_norm)
                prediction_norm = delta if geometry_norm is None else geometry_norm + delta
            prediction_cm = prediction_norm.float().cpu().numpy() * label_std + label_mean
            all_predictions.append(prediction_cm)
            all_targets.append(raw_targets.numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    results = calcular_metricas(predictions, targets, measurements)
    imprimir_tabla_metricas(results, measurements)
    graficar_tabla_metricas(results, measurements, save_path=str(exp_dir / f"metrics_{split}.png"))
    for measurement in ("chest", "waist", "hip"):
        if measurement in measurements:
            index = measurements.index(measurement)
            graficar_bland_altman(
                predictions[:, index],
                targets[:, index],
                measurement,
                save_path=str(exp_dir / f"ba_{measurement}_{split}.png"),
            )

    metadata_columns = ["subject_id", "photo_id", "height_cm"]
    if "gender" in dataset.df.columns:
        metadata_columns.append("gender")
    if use_weight_meta:
        metadata_columns.append(cfg["dataset"].get("weight_col", "weight_kg"))
    output = dataset.df.iloc[: len(predictions)][metadata_columns].copy()
    for index, measurement in enumerate(measurements):
        output[f"true_{measurement}"] = targets[:, index]
        output[f"pred_{measurement}"] = predictions[:, index]
        output[f"abs_error_{measurement}"] = np.abs(predictions[:, index] - targets[:, index])

    subject_output = agregar_predicciones_por_sujeto(output, measurements)
    subject_predictions = subject_output[
        [f"pred_{measurement}" for measurement in measurements]
    ].to_numpy()
    subject_targets = subject_output[
        [f"true_{measurement}" for measurement in measurements]
    ].to_numpy()
    subject_results = calcular_metricas(subject_predictions, subject_targets, measurements)
    results["_subject_aggregate"] = {
        "n_subjects": int(len(subject_output)),
        "n_captures": int(len(output)),
        "metrics": subject_results,
    }
    print(
        f"[Eval] Agregado por sujeto: n={len(subject_output)} "
        f"MAE={subject_results['_global']['mae']:.3f}cm"
    )

    conformal_path = exp_dir / "conformal_q_hat.npy"
    if conformal_path.is_file():
        q_hat = np.load(conformal_path)
        coverage = ((targets >= predictions - q_hat) & (targets <= predictions + q_hat)).mean(axis=0)
        results["_conformal"] = {
            "target": float(cfg["conformal"]["coverage"]),
            "coverage_by_measure": dict(zip(measurements, coverage.tolist())),
            "mean_coverage": float(coverage.mean()),
        }
        subject_coverage = (
            (subject_targets >= subject_predictions - q_hat)
            & (subject_targets <= subject_predictions + q_hat)
        ).mean(axis=0)
        results["_subject_aggregate"]["conformal"] = {
            "target": float(cfg["conformal"]["coverage"]),
            "coverage_by_measure": dict(zip(measurements, subject_coverage.tolist())),
            "mean_coverage": float(subject_coverage.mean()),
        }
        for index, measurement in enumerate(measurements):
            output[f"lower_{measurement}"] = predictions[:, index] - q_hat[index]
            output[f"upper_{measurement}"] = predictions[:, index] + q_hat[index]
            subject_output[f"lower_{measurement}"] = subject_predictions[:, index] - q_hat[index]
            subject_output[f"upper_{measurement}"] = subject_predictions[:, index] + q_hat[index]

    (exp_dir / f"results_{split}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    output.to_csv(exp_dir / f"predictions_{split}.csv", index=False)
    subject_output.to_csv(exp_dir / f"predictions_{split}_by_subject.csv", index=False)
    print(f"[Eval] Resultados guardados en {exp_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="experiments/exp_001_baseline")
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "calibration", "test", "test_wild"],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    evaluate(
        ROOT / args.exp,
        args.split,
        cargar_config(ROOT / args.config),
        limit_batches=args.limit_batches,
    )
