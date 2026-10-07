"""Inferencia del ensamble oficial basado en perfiles de silueta BodyM."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from src.features.silhouette_profiles import caracteristicas_dos_vistas
from src.inference.predict_measurements import calidad_mascara
from src.data.bodym_dataset import TransformacionSiluetaPareada


class PredictorPerfilesArboles:
    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir)
        payload = joblib.load(self.exp_dir / "model.joblib")
        self.model = payload["model"]
        self.bins = int(payload["bins"])
        self.targets = list(payload["targets"])
        self.q_hat = np.load(self.exp_dir / "conformal_q_hat.npy")
        if len(self.targets) != len(self.q_hat):
            raise ValueError("El modelo y sus intervalos conformales no coinciden")
        self.transform_quality = TransformacionSiluetaPareada([320, 240], augment=False)

    def predict_paths(
        self,
        front_path: Path,
        side_path: Path,
        height_cm: float,
        gender: float | None = None,
        weight_kg: float | None = None,
        mirror_side: bool = False,
    ) -> dict:
        if not 120.0 <= height_cm <= 230.0:
            raise ValueError("height_cm debe estar entre 120 y 230 cm")
        if gender is not None or weight_kg is not None:
            raise ValueError("Este modelo no acepta sexo ni peso como entradas")
        if mirror_side:
            raise ValueError("La ruta BodyM oficial espera el perfil izquierdo")
        features = caracteristicas_dos_vistas(
            Path(front_path), Path(side_path), height_cm, self.bins
        )
        prediction = self.model.predict(features[None, :])[0]
        front_tensor = self.transform_quality(
            Image.open(front_path).convert("L"),
            self.transform_quality.sample_params(),
        )
        side_tensor = self.transform_quality(
            Image.open(side_path).convert("L"),
            self.transform_quality.sample_params(),
        )
        front_quality = calidad_mascara(front_tensor)
        side_quality = calidad_mascara(side_tensor)
        warnings = [
            *(f"frontal: {item}" for item in front_quality["warnings"]),
            *(f"lateral: {item}" for item in side_quality["warnings"]),
        ]
        measurements = {}
        estimated_weight = None
        for index, name in enumerate(self.targets):
            estimate = float(prediction[index])
            radius = float(self.q_hat[index])
            if name == "weight_kg":
                estimated_weight = {
                    "estimate_kg": round(estimate, 2),
                    "lower_kg": round(estimate - radius, 2),
                    "upper_kg": round(estimate + radius, 2),
                    "requires_manual_confirmation": radius > 8.0,
                    "source": "extra_trees_perfiles_bodym",
                }
            else:
                measurements[name] = {
                    "estimate_cm": round(estimate, 2),
                    "lower_cm": round(estimate - radius, 2),
                    "upper_cm": round(estimate + radius, 2),
                    "requires_manual_confirmation": radius > 5.0,
                }
        return {
            "model": self.exp_dir.name,
            "checkpoint_epoch": None,
            "height_cm": height_cm,
            "estimated_weight": estimated_weight,
            "uses_weight": False,
            "uses_gender": False,
            "device": "cpu",
            "quality": {
                "front": front_quality,
                "side": side_quality,
                "warnings": warnings,
            },
            "measurements": measurements,
            "bias_correction_cm": {},
        }
