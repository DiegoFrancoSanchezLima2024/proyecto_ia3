"""Entrenamiento reproducible del baseline antropometrico de Sastre-IA."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.bodym_dataset import ConjuntoDatosBodyM
from src.evaluation.metrics import calcular_metricas, imprimir_tabla_metricas
from src.geometry.geometric_est import EstimadorGeometrico
from src.models.regressor import RegresorMedidasCorporales
from src.training.conformal import CalibradorConforme
from src.training.losses import PerdidaTotal
from src.training.runtime import (
    amp_context,
    configurar_entorno_ejecucion,
    dataloader_kwargs,
    mover_imagen,
    geometria_normalizada,
    resolver_raiz_imagenes_crudas,
)
from src.utils.config import cargar_config


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(cfg: dict) -> torch.device:
    requested = cfg["hardware"].get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[WARN] CUDA no esta disponible; se usara CPU.")
    return torch.device("cpu")


def _dataset(cfg: dict, split: str, train_stats: ConjuntoDatosBodyM | None = None) -> ConjuntoDatosBodyM:
    data_csv = ROOT / cfg["paths"]["data_root"] / split / "data.csv"
    if not data_csv.is_file():
        raise FileNotFoundError(
            f"No existe {data_csv}. Ejecuta primero: python src/data/clean_bodym.py"
        )
    view_dirs = cfg["dataset"].get("view_directories", {})
    shared = {}
    if train_stats is not None:
        shared = {
            "label_mean": train_stats.label_mean,
            "label_std": train_stats.label_std,
            "height_mean": train_stats.height_mean,
            "height_std": train_stats.height_std,
        }
        if train_stats.use_weight_meta:
            shared.update(
                weight_mean=train_stats.weight_mean,
                weight_std=train_stats.weight_std,
            )
    return ConjuntoDatosBodyM(
        data_csv=str(data_csv),
        img_root=str(resolver_raiz_imagenes_crudas(ROOT, cfg, split)),
        meas_cols=cfg["dataset"]["measurements"],
        height_col=cfg["dataset"].get("height_col", "height_cm"),
        gender_col=cfg["dataset"].get("gender_col", "gender"),
        weight_col=cfg["dataset"].get("weight_col", "weight_kg"),
        use_weight_meta=cfg["dataset"].get("use_weight_meta", False),
        use_gender_meta=cfg["dataset"].get("use_gender_meta", True),
        img_size=cfg["dataset"]["img_size"],
        augment=split == "train" and cfg["augmentation"].get("enabled", True),
        augmentation_config=cfg.get("augmentation", {}),
        strict_images=cfg["dataset"].get("strict_images", True),
        front_dir=view_dirs.get("front", "mask"),
        left_dir=view_dirs.get("left", "mask_left"),
        **shared,
    )


def pesos_balanceo_sujetos(frame) -> torch.Tensor:
    """Asigna a cada foto 1/n_fotos para igualar la masa de cada sujeto."""
    if "subject_id" not in frame.columns:
        raise ValueError("El muestreo balanceado requiere subject_id")
    if frame.empty or frame["subject_id"].isna().any():
        raise ValueError("subject_id contiene valores vacios")
    counts = frame["subject_id"].value_counts()
    weights = frame["subject_id"].map(lambda value: 1.0 / counts[value]).to_numpy(copy=True)
    return torch.as_tensor(weights, dtype=torch.double)


def build_dataloaders(cfg: dict, device: torch.device):
    train_ds = _dataset(cfg, "train")
    val_ds = _dataset(cfg, "val", train_ds)
    calibration_ds = _dataset(cfg, "calibration", train_ds)
    common = dataloader_kwargs(cfg, device)
    sampler = None
    if cfg["training"].get("subject_balanced_sampling", False):
        generator = torch.Generator().manual_seed(int(cfg["project"]["seed"]))
        sampler = WeightedRandomSampler(
            pesos_balanceo_sujetos(train_ds.df),
            num_samples=len(train_ds),
            replacement=True,
            generator=generator,
        )
        print(
            f"  muestreo balanceado: {train_ds.df['subject_id'].nunique()} sujetos, "
            f"{len(train_ds)} fotos/epoca"
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **common,
    )
    eval_batch = int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"]))
    val_loader = DataLoader(val_ds, batch_size=eval_batch, shuffle=False, **common)
    calibration_loader = DataLoader(
        calibration_ds, batch_size=eval_batch, shuffle=False, **common
    )
    return train_loader, val_loader, calibration_loader, train_ds


def _geometry_for_batch(
    geo_estimator,
    front_cpu,
    left_cpu,
    meta_cpu,
    train_ds,
    device,
):
    if geo_estimator is None:
        return None
    return geometria_normalizada(
        geo_estimator,
        front_cpu,
        left_cpu,
        meta_cpu,
        train_ds.label_mean,
        train_ds.label_std,
        train_ds.height_mean,
        train_ds.height_std,
        device,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    geo_estimator,
    scaler,
    device,
    cfg,
    train_ds,
    max_batches: int | None = None,
) -> float:
    model.train()
    accumulation = max(1, int(cfg["training"].get("grad_accumulation", 1)))
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    channels_last = bool(cfg["training"].get("channels_last", True))
    total_steps = min(len(loader), max_batches) if max_batches else len(loader)
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (front_cpu, left_cpu, meta_cpu, label_cpu, _) in enumerate(loader):
        if step >= total_steps:
            break
        geometry = _geometry_for_batch(
            geo_estimator, front_cpu, left_cpu, meta_cpu, train_ds, device
        )
        front = mover_imagen(front_cpu, device, channels_last)
        left = mover_imagen(left_cpu, device, channels_last)
        meta = meta_cpu.to(device, non_blocking=device.type == "cuda")
        label = label_cpu.to(device, non_blocking=device.type == "cuda")

        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, total_steps - group_start)
        with amp_context(device, use_amp):
            delta, log_var = model([front, left], meta, geometry)
            prediction = delta if geometry is None else geometry + delta
            losses = criterion(prediction, label, log_var)
            scaled_loss = losses["total"] / group_size

        scaler.scale(scaled_loss).backward()
        end_group = (step + 1) % accumulation == 0 or step + 1 == total_steps
        if end_group:
            scaler.unscale_(optimizer)
            if cfg["training"].get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(losses["total"].detach())
    return running_loss / max(1, total_steps)


@torch.no_grad()
def predict_loader(
    model,
    loader,
    geo_estimator,
    device,
    cfg,
    train_ds,
    criterion=None,
    max_batches: int | None = None,
):
    model.eval()
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    channels_last = bool(cfg["training"].get("channels_last", True))
    all_predictions, all_targets = [], []
    running_loss = 0.0
    batches = 0

    for step, (front_cpu, left_cpu, meta_cpu, label_cpu, raw_label) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        geometry = _geometry_for_batch(
            geo_estimator, front_cpu, left_cpu, meta_cpu, train_ds, device
        )
        front = mover_imagen(front_cpu, device, channels_last)
        left = mover_imagen(left_cpu, device, channels_last)
        meta = meta_cpu.to(device, non_blocking=device.type == "cuda")
        label = label_cpu.to(device, non_blocking=device.type == "cuda")
        with amp_context(device, use_amp):
            delta, log_var = model([front, left], meta, geometry)
            prediction = delta if geometry is None else geometry + delta
            if criterion is not None:
                running_loss += float(criterion(prediction, label, log_var)["total"])

        prediction_cm = prediction.float().cpu().numpy() * train_ds.label_std + train_ds.label_mean
        all_predictions.append(prediction_cm)
        all_targets.append(raw_label.numpy())
        batches += 1

    if not all_predictions:
        raise RuntimeError("El DataLoader no produjo lotes")
    return (
        np.concatenate(all_predictions),
        np.concatenate(all_targets),
        running_loss / max(1, batches),
    )


def construir_optimizador(model: RegresorMedidasCorporales, cfg: dict):
    backbone = list(model.encoder.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    task_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": task_parameters,
                "lr": float(cfg["training"]["learning_rate"]),
                "name": "task_heads",
            },
            {
                "params": backbone,
                "lr": float(cfg["training"].get("encoder_learning_rate", 3e-5)),
                "name": "encoder",
            },
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )


def build_scheduler(optimizer, cfg: dict):
    epochs = int(cfg["training"]["epochs"])
    warmup = min(int(cfg["training"].get("warmup_epochs", 0)), max(0, epochs - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup), eta_min=1e-6
    )
    if warmup == 0:
        return cosine
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.20, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[linear, cosine], milestones=[warmup]
    )


def resolve_measurement_weights(cfg: dict, measurements: list[str]) -> list[float]:
    configured = cfg["training"].get("measurement_weights", {}) or {}
    unknown = sorted(set(configured) - set(measurements))
    if unknown:
        raise ValueError(f"Pesos configurados para medidas desconocidas: {unknown}")
    weights = [float(configured.get(name, 1.0)) for name in measurements]
    if not all(np.isfinite(value) and value > 0 for value in weights):
        raise ValueError("Todos los pesos de medidas deben ser finitos y mayores que cero")
    return weights


def resolver_mae_seleccion(metrics: dict, cfg: dict, measurements: list[str]) -> float:
    """Calcula el criterio de checkpoint sin mezclar auxiliares con medidas.

    Un experimento multitarea puede predecir peso en kg junto a longitudes en cm.
    El peso ayuda a aprender forma, pero no debe decidir por si solo cual checkpoint
    produce las mejores medidas corporales.
    """
    selected = list(cfg["training"].get("selection_measurements") or measurements)
    unknown = sorted(set(selected) - set(measurements))
    if unknown:
        raise ValueError(
            f"Medidas de seleccion desconocidas: {', '.join(unknown)}"
        )
    if not selected:
        raise ValueError("selection_measurements no puede estar vacio")
    return float(np.mean([float(metrics[name]["mae"]) for name in selected]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exp", default="exp_001_baseline")
    parser.add_argument("--epochs", type=int, help="Sobrescribe epochs para una prueba corta")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--use-geometry", action="store_true")
    parser.add_argument("--loss", choices=["heteroscedastic_huber", "huber"])
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg = cargar_config(config_path)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.no_pretrained:
        cfg["model"]["encoder_pretrained"] = False
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.use_geometry:
        cfg["model"]["use_geometry"] = True
    if args.loss:
        cfg["training"]["loss"] = args.loss

    device = select_device(cfg)
    set_seed(cfg["project"]["seed"], cfg["training"].get("deterministic", False))
    configurar_entorno_ejecucion(cfg, device)
    print(f"[Train] Dispositivo: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {properties.total_memory / 2**30:.2f} GiB")

    exp_dir = ROOT / cfg["paths"]["experiments"] / args.exp
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print("[Train] Validando datos...")
    train_loader, val_loader, calibration_loader, train_ds = build_dataloaders(cfg, device)
    print(
        f"  train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"calibration={len(calibration_loader.dataset)} medidas={train_ds.n_measures}"
    )

    model = RegresorMedidasCorporales(
        n_measures=train_ds.n_measures,
        embed_dim=cfg["model"]["embed_dim"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        n_views=len(cfg["dataset"].get("views", ["front", "left"])),
        dropout=cfg["model"]["dropout"],
        pretrained_encoder=cfg["model"]["encoder_pretrained"],
        frozen_epochs=cfg["model"]["encoder_frozen_epochs"],
        adapt_batchnorm_epochs=cfg["model"].get("adapt_batchnorm_epochs", 0),
        use_geometry=cfg["model"]["use_geometry"],
        use_profile_features=cfg["model"].get("use_profile_features", False),
        profile_bins=cfg["model"].get("profile_bins", 32),
        meta_dim=(
            1
            + int(train_ds.use_gender_meta)
            + int(train_ds.use_weight_meta)
        ),
    ).to(device)
    if device.type == "cuda" and cfg["training"].get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)

    geometry = (
        EstimadorGeometrico(train_ds.meas_cols) if cfg["model"].get("use_geometry") else None
    )
    optimizer = construir_optimizador(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = PerdidaTotal(
        huber_delta=cfg["training"].get("huber_delta_normalized", 1.0),
        lambda_coherence=cfg["training"].get("lambda_coherence", 0.0),
        mode=cfg["training"].get("loss", "heteroscedastic_huber"),
        measurement_weights=resolve_measurement_weights(cfg, train_ds.meas_cols),
    ).to(device)
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    best_mae = float("inf")
    early_stopping_best_mae = float("inf")
    epochs_without_improvement = 0
    history = []
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.on_epoch_start(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            geometry,
            scaler,
            device,
            cfg,
            train_ds,
            args.limit_train_batches,
        )
        predictions, targets, val_loss = predict_loader(
            model,
            val_loader,
            geometry,
            device,
            cfg,
            train_ds,
            criterion,
            args.limit_val_batches,
        )
        metrics = calcular_metricas(predictions, targets, train_ds.meas_cols)
        scheduler.step()
        elapsed = time.perf_counter() - started
        peak_gib = (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        )
        selection_mae = resolver_mae_seleccion(metrics, cfg, train_ds.meas_cols)
        current = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mae": metrics["_global"]["mae"],
            "selection_mae": selection_mae,
            "pct_2cm": metrics["_global"]["pct_2cm"],
            "peak_vram_gib": peak_gib,
            "elapsed_seconds": elapsed,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(current)
        print(
            f"Epoch {epoch:03d}/{epochs} train={train_loss:.4f} val={val_loss:.4f} "
            f"MAE={current['mae']:.3f} seleccion={selection_mae:.3f} "
            f"±2={current['pct_2cm']:.1f}% "
            f"VRAM_peak={peak_gib:.2f}GiB tiempo={elapsed:.1f}s"
        )

        min_delta = float(cfg["training"].get("early_stopping_min_delta", 0.0))
        # Guardar siempre el minimo real. ``min_delta`` solo decide si una
        # mejora es suficientemente grande para reiniciar la paciencia.
        if selection_mae < best_mae:
            best_mae = selection_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "mae": best_mae,
                    "metrics": metrics,
                    "label_mean": train_ds.label_mean.tolist(),
                    "label_std": train_ds.label_std.tolist(),
                    "height_mean": train_ds.height_mean,
                    "height_std": train_ds.height_std,
                    "weight_mean": train_ds.weight_mean,
                    "weight_std": train_ds.weight_std,
                    "use_weight_meta": train_ds.use_weight_meta,
                    "use_gender_meta": train_ds.use_gender_meta,
                    "meta_dim": (
                        1
                        + int(train_ds.use_gender_meta)
                        + int(train_ds.use_weight_meta)
                    ),
                    "meas_cols": train_ds.meas_cols,
                    "model_config": cfg["model"],
                    "dataset_views": cfg["dataset"].get("views", ["front", "left"]),
                },
                checkpoint_dir / "best_model.pt",
            )
            print(f"  [SAVED] best_model.pt MAE={best_mae:.3f} cm")

        if selection_mae < early_stopping_best_mae - min_delta:
            early_stopping_best_mae = selection_mae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == epochs:
            imprimir_tabla_metricas(metrics, train_ds.meas_cols)

        patience = int(cfg["training"].get("early_stopping_patience", 0))
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"[EarlyStopping] Sin mejora >= {min_delta:.3f} cm durante "
                f"{patience} epocas."
            )
            break

    (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[Train] Mejor MAE de validacion: {best_mae:.3f} cm")

    if args.skip_calibration:
        print("[Conformal] Omitida por --skip-calibration")
        return

    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    cal_predictions, cal_targets, _ = predict_loader(
        model, calibration_loader, geometry, device, cfg, train_ds
    )
    calibrator = CalibradorConforme(
        coverage=cfg["conformal"]["coverage"],
        unreliable_threshold=cfg["conformal"]["unreliable_threshold"],
    )
    calibrator.calibrate(cal_predictions, cal_targets)
    calibrator.save(str(exp_dir / "conformal_q_hat.npy"))
    coverage = calibrator.empirical_coverage(cal_predictions, cal_targets)
    (exp_dir / "calibration_metrics.json").write_text(
        json.dumps(
            {
                "target_coverage": cfg["conformal"]["coverage"],
                "empirical_coverage": dict(zip(train_ds.meas_cols, coverage.tolist())),
                "q_hat_cm": dict(zip(train_ds.meas_cols, calibrator.q_hat.tolist())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
