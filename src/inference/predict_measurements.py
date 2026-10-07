"""Predice medidas desde una máscara frontal y una lateral."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.bodym_dataset import TransformacionSiluetaPareada
from src.geometry.geometric_est import EstimadorGeometrico
from src.models.regressor import RegresorMedidasCorporales
from src.training.runtime import amp_context, configurar_entorno_ejecucion, mover_imagen, geometria_normalizada
from src.utils.config import cargar_config


def calidad_mascara(mask: torch.Tensor) -> dict:
    """Calcula controles simples de ocupación y encuadre de una máscara."""
    foreground = (mask * 0.5 + 0.5) > 0.5
    occupied_rows = foreground.any(dim=-1).any(dim=0)
    occupied_cols = foreground.any(dim=-2).any(dim=0)
    foreground_ratio = float(foreground.float().mean())
    vertical_coverage = float(occupied_rows.float().mean())
    horizontal_coverage = float(occupied_cols.float().mean())
    warnings = []
    if foreground_ratio < 0.03:
        warnings.append("silueta demasiado pequena o vacia")
    if foreground_ratio > 0.85:
        warnings.append("mascara ocupa casi toda la imagen")
    if vertical_coverage < 0.65:
        warnings.append("no se observa el cuerpo completo")
    return {
        "foreground_ratio": foreground_ratio,
        "vertical_coverage": vertical_coverage,
        "horizontal_coverage": horizontal_coverage,
        "warnings": warnings,
    }


def _load_mask(path: Path, transform: TransformacionSiluetaPareada) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"No existe la máscara: {path}")
    with Image.open(path) as image:
        return transform(image.convert("L"), transform.sample_params())


class PredictorDeMedidas:
    """Carga una vez el campeón y permite inferir varios pares de vistas."""

    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir)
        cfg_path = self.exp_dir / "config_resolved.yaml"
        checkpoint_path = self.exp_dir / "checkpoints" / "best_model.pt"
        if not cfg_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Experimento incompleto: {self.exp_dir}")

        self.cfg = cargar_config(cfg_path)
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.use_weight_meta = bool(self.checkpoint.get("use_weight_meta", False))
        # Los checkpoints historicos siempre usaban genero. Los nuevos modelos
        # BodyM de dos vistas aprenden sin ese dato y solo reciben estatura.
        self.use_gender_meta = bool(self.checkpoint.get("use_gender_meta", True))
        self.weight_target_name = self.cfg["dataset"].get("weight_col", "weight_kg")
        self.model_cfg = {
            **self.cfg["model"],
            **self.checkpoint.get("model_config", {}),
        }
        self.measurements = list(self.checkpoint["meas_cols"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        configurar_entorno_ejecucion(self.cfg, self.device)

        self.model = RegresorMedidasCorporales(
            n_measures=len(self.measurements),
            embed_dim=self.model_cfg["embed_dim"],
            n_heads=self.model_cfg["n_heads"],
            n_layers=self.model_cfg["n_layers"],
            n_views=len(self.checkpoint.get("dataset_views", ["front", "left"])),
            dropout=self.model_cfg["dropout"],
            pretrained_encoder=False,
            frozen_epochs=self.model_cfg.get("encoder_frozen_epochs", 5),
            adapt_batchnorm_epochs=self.model_cfg.get("adapt_batchnorm_epochs", 0),
            use_geometry=self.model_cfg.get("use_geometry", False),
            use_profile_features=self.model_cfg.get("use_profile_features", False),
            profile_bins=self.model_cfg.get("profile_bins", 32),
            meta_dim=int(
                self.checkpoint.get(
                    "meta_dim",
                    1 + int(self.use_gender_meta) + int(self.use_weight_meta),
                )
            ),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.channels_last = bool(self.cfg["training"].get("channels_last", True))
        if self.device.type == "cuda" and self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

        self.transform = TransformacionSiluetaPareada(
            self.cfg["dataset"]["img_size"], augment=False
        )
        self.label_mean = np.asarray(
            self.checkpoint["label_mean"], dtype=np.float32
        )
        self.label_std = np.asarray(
            self.checkpoint["label_std"], dtype=np.float32
        )
        q_path = self.exp_dir / "conformal_q_hat.npy"
        self.q_hat = (
            np.load(q_path)
            if q_path.is_file()
            else np.full(len(self.measurements), np.nan, dtype=np.float32)
        )
        self.unreliable_threshold = float(
            self.cfg["conformal"].get("unreliable_threshold", 5.0)
        )
        self.bias_cm = self._load_bias_correction()

    def _load_bias_correction(self) -> np.ndarray:
        """Sesgo medio por medida, estimado en el split de calibracion.

        El archivo lo genera scripts/fit_bias_correction.py y NUNCA debe
        calcularse sobre el split de test: eso inflaria las metricas.
        Si no existe, la correccion es cero y el comportamiento es el previo.
        """
        bias = np.zeros(len(self.measurements), dtype=np.float32)
        bias_path = self.exp_dir / "bias_correction.json"
        if not bias_path.is_file():
            return bias
        payload = json.loads(bias_path.read_text(encoding="utf-8"))
        table = payload.get("bias_cm", payload)
        for index, name in enumerate(self.measurements):
            value = table.get(name)
            if value is not None:
                bias[index] = float(value)
        return bias

    def load_mask(self, path: Path, mirror: bool = False) -> torch.Tensor:
        mask = _load_mask(Path(path), self.transform)
        return torch.flip(mask, dims=(-1,)) if mirror else mask

    def _metadata(
        self, height_cm: float, gender: float | None, weight_kg: float | None
    ) -> torch.Tensor:
        if not 120.0 <= height_cm <= 230.0:
            raise ValueError("height_cm debe estar entre 120 y 230 cm")
        if self.use_gender_meta and gender not in {0.0, 1.0}:
            raise ValueError("Este modelo requiere gender 0 o 1, igual que en BodyM")
        if self.use_weight_meta:
            if weight_kg is None:
                raise ValueError("Este modelo requiere --weight-kg")
            if not 30.0 <= weight_kg <= 250.0:
                raise ValueError("weight_kg debe estar entre 30 y 250 kg")

        meta_values = [
            (height_cm - float(self.checkpoint["height_mean"]))
            / float(self.checkpoint["height_std"])
        ]
        if self.use_gender_meta:
            meta_values.append(float(gender))
        if self.use_weight_meta:
            meta_values.append(
                (weight_kg - float(self.checkpoint["weight_mean"]))
                / float(self.checkpoint["weight_std"])
            )
        return torch.tensor([meta_values], dtype=torch.float32)

    def predict_tensors(
        self,
        front_cpu: torch.Tensor,
        side_cpu: torch.Tensor,
        height_cm: float,
        gender: float | None = None,
        weight_kg: float | None = None,
    ) -> dict:
        front_quality = calidad_mascara(front_cpu)
        side_quality = calidad_mascara(side_cpu)
        quality_warnings = [
            *(f"frontal: {item}" for item in front_quality["warnings"]),
            *(f"lateral: {item}" for item in side_quality["warnings"]),
        ]
        if (
            abs(
                front_quality["vertical_coverage"]
                - side_quality["vertical_coverage"]
            )
            > 0.10
        ):
            quality_warnings.append(
                "las vistas tienen encuadres verticales incompatibles"
            )

        meta_cpu = self._metadata(height_cm, gender, weight_kg)
        front_batch = front_cpu.unsqueeze(0)
        side_batch = side_cpu.unsqueeze(0)
        geometry_norm = None
        if self.model_cfg.get("use_geometry", False):
            geometry_norm = geometria_normalizada(
                EstimadorGeometrico(self.measurements),
                front_batch,
                side_batch,
                meta_cpu,
                self.label_mean,
                self.label_std,
                float(self.checkpoint["height_mean"]),
                float(self.checkpoint["height_std"]),
                self.device,
            )

        front = mover_imagen(front_batch, self.device, self.channels_last)
        side = mover_imagen(side_batch, self.device, self.channels_last)
        meta = meta_cpu.to(self.device)
        use_amp = bool(
            self.cfg["training"].get("amp", True) and self.device.type == "cuda"
        )
        with torch.inference_mode(), amp_context(self.device, use_amp):
            delta, _ = self.model([front, side], meta, geometry_norm)
            prediction_norm = delta if geometry_norm is None else geometry_norm + delta

        prediction = (
            prediction_norm.float().cpu().numpy()[0] * self.label_std
            + self.label_mean
        )
        # El modelo sobreestima de forma sistematica algunas medidas (cintura,
        # sobre todo). Restar ese sesgo medido en calibracion baja el error sin
        # reentrenar. Los intervalos conformales se calibraron sobre las
        # predicciones sin corregir, asi que siguen siendo validos (mas anchos
        # de lo necesario, nunca mas angostos).
        prediction = prediction - self.bias_cm
        measures = {}
        estimated_weight = None
        for index, name in enumerate(self.measurements):
            radius = float(self.q_hat[index])
            has_interval = bool(np.isfinite(radius))
            if name == self.weight_target_name and not self.use_weight_meta:
                estimated_weight = {
                    "estimate_kg": round(float(prediction[index]), 2),
                    "lower_kg": round(float(prediction[index] - radius), 2)
                    if has_interval
                    else None,
                    "upper_kg": round(float(prediction[index] + radius), 2)
                    if has_interval
                    else None,
                    "requires_manual_confirmation": bool(
                        not has_interval or radius > 8.0
                    ),
                    "source": "modelo_multitarea_bodym",
                }
                continue
            measures[name] = {
                "estimate_cm": round(float(prediction[index]), 2),
                "lower_cm": round(float(prediction[index] - radius), 2)
                if has_interval
                else None,
                "upper_cm": round(float(prediction[index] + radius), 2)
                if has_interval
                else None,
                "requires_manual_confirmation": bool(
                    not has_interval or radius > self.unreliable_threshold
                ),
            }
        return {
            "model": self.exp_dir.name,
            "checkpoint_epoch": int(self.checkpoint["epoch"]),
            "height_cm": height_cm,
            "gender_code": int(gender) if gender is not None else None,
            "estimated_weight": estimated_weight,
            "uses_weight": self.use_weight_meta,
            "uses_gender": self.use_gender_meta,
            "device": str(self.device),
            "quality": {
                "front": front_quality,
                "side": side_quality,
                "warnings": quality_warnings,
            },
            "measurements": measures,
            "bias_correction_cm": {
                name: round(float(self.bias_cm[index]), 3)
                for index, name in enumerate(self.measurements)
                if name != self.weight_target_name and self.bias_cm[index] != 0.0
            },
            "notice": (
                "Estimación asistida; confirme las medidas marcadas antes de cortar tela."
            ),
        }

    def predict_paths(
        self,
        front_path: Path,
        side_path: Path,
        height_cm: float,
        gender: float | None = None,
        weight_kg: float | None = None,
        mirror_side: bool = False,
    ) -> dict:
        front_cpu = self.load_mask(front_path)
        side_cpu = self.load_mask(side_path, mirror=mirror_side)
        result = self.predict_tensors(
            front_cpu, side_cpu, height_cm, gender, weight_kg
        )
        result["side_mirrored"] = bool(mirror_side)
        return result


def predict(
    exp_dir: Path,
    front_path: Path,
    side_path: Path,
    height_cm: float,
    gender: float | None = None,
    weight_kg: float | None = None,
) -> dict:
    return PredictorDeMedidas(exp_dir).predict_paths(
        front_path, side_path, height_cm, gender, weight_kg
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="experiments/exp_002_huber")
    parser.add_argument("--front-mask", required=True)
    parser.add_argument("--side-mask", required=True)
    parser.add_argument("--height-cm", required=True, type=float)
    parser.add_argument("--gender", type=float, choices=[0.0, 1.0])
    parser.add_argument("--weight-kg", type=float)
    parser.add_argument("--output", default="outputs/measurement_prediction.json")
    args = parser.parse_args()
    result = predict(
        ROOT / args.exp,
        Path(args.front_mask),
        Path(args.side_mask),
        args.height_cm,
        args.gender,
        args.weight_kg,
    )
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Modelo: {result['model']} | epoca {result['checkpoint_epoch']} | {result['device']}")
    print(f"{'Medida':<22} {'Estimacion':>10} {'Intervalo 90%':>22} {'Confirmar':>10}")
    print("-" * 68)
    for name, value in result["measurements"].items():
        interval = (
            f"[{value['lower_cm']:.2f}, {value['upper_cm']:.2f}]"
            if value["lower_cm"] is not None
            else "no calibrado"
        )
        confirm = "SI" if value["requires_manual_confirmation"] else "no"
        print(f"{name:<22} {value['estimate_cm']:>8.2f}cm {interval:>22} {confirm:>10}")
    if result.get("estimated_weight"):
        weight = result["estimated_weight"]
        print(f"Peso estimado por el modelo: {weight['estimate_kg']:.2f} kg")
    for warning in result["quality"]["warnings"]:
        print(f"[CALIDAD] {warning}")
    print(f"Guardado: {output_path}")


if __name__ == "__main__":
    main()
