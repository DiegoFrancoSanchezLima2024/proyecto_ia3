"""
src/training/conformal.py
==========================
Calibracion conformal split para intervalos de prediccion garantizados.
Garantiza cobertura empirica al nivel indicado (e.g., 90%).
"""
import numpy as np
from pathlib import Path
import json


class CalibradorConforme:
    """
    Split Conformal Prediction para regresion con incertidumbre.
    Produce intervalos de la forma [pred - q, pred + q] por medida.
    """

    def __init__(self, coverage: float = 0.90, unreliable_threshold: float = 5.0):
        assert 0 < coverage < 1, "coverage debe estar en (0,1)"
        self.coverage   = coverage
        self.threshold  = unreliable_threshold
        self.q_hat      = None   # [N_measures] quantiles de calibracion

    def calibrate(self, preds: np.ndarray, targets: np.ndarray):
        """
        Calibra con el conjunto de validacion.
        preds:   [N, M] predicciones en cm
        targets: [N, M] valores reales en cm
        """
        residuals = np.abs(preds - targets)   # [N, M]
        n = residuals.shape[0]
        # Estadistico de orden conformal finito: k=ceil((n+1)*(1-alpha)).
        # Evita la interpolacion lineal de np.quantile, que puede subcubrir.
        alpha    = 1.0 - self.coverage
        k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        self.q_hat = np.sort(residuals, axis=0)[k - 1]  # [M]
        print(
            f"[Conformal] Calibrado con {n} muestras, k={k}, "
            f"cobertura objetivo={self.coverage:.0%}"
        )
        print(f"  q_hat (cm): {np.round(self.q_hat, 3)}")

    def predict_intervals(self, preds: np.ndarray) -> tuple:
        """
        Retorna (lower, upper, is_reliable) para cada prediccion.
        preds: [N, M]
        """
        assert self.q_hat is not None, "Debe calibrar antes de predecir."
        lower = preds - self.q_hat        # [N, M]
        upper = preds + self.q_hat        # [N, M]
        # Flagear como 'no confiable' si el intervalo es demasiado amplio
        is_reliable = (self.q_hat <= self.threshold)  # [M] bool
        return lower, upper, is_reliable

    def empirical_coverage(self, preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Cobertura empirica por medida (debe ser >= self.coverage)."""
        lower, upper, _ = self.predict_intervals(preds)
        covered = (targets >= lower) & (targets <= upper)
        return covered.mean(axis=0)  # [M]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_hat)
        print(f"[Conformal] Guardado en {path}")

    def load(self, path: str):
        self.q_hat = np.load(path)
        print(f"[Conformal] Cargado desde {path}")
