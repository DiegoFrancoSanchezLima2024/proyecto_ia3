"""
tests/test_metrics.py
======================
Tests de correctitud para las metricas de evaluacion.
Ejecutar: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    agregar_predicciones_por_sujeto,
    bland_altman,
    calcular_metricas,
)


MEAS_NAMES = ["chest", "waist", "hip"]


def test_perfect_predictions():
    """MAE = 0 si pred == target."""
    preds   = np.array([[100.0, 80.0, 95.0], [102.0, 82.0, 97.0]])
    targets = preds.copy()
    results = calcular_metricas(preds, targets, MEAS_NAMES)
    assert results["_global"]["mae"] == pytest.approx(0.0, abs=1e-6)
    assert results["chest"]["mae"]   == pytest.approx(0.0, abs=1e-6)


def test_known_mae():
    """MAE correcto para errores conocidos."""
    preds   = np.array([[101.0, 80.0, 95.0]])
    targets = np.array([[100.0, 80.0, 95.0]])
    results = calcular_metricas(preds, targets, MEAS_NAMES)
    assert results["chest"]["mae"] == pytest.approx(1.0, abs=1e-6)
    assert results["waist"]["mae"] == pytest.approx(0.0, abs=1e-6)


def test_pct_2cm():
    """% dentro de ±2cm correcto."""
    # 3 predicciones: error de 1, 3, 2 cm
    preds   = np.array([[101.0], [103.0], [102.0]])
    targets = np.array([[100.0], [100.0], [100.0]])
    results = calcular_metricas(preds, targets, ["chest"])
    # 2 de 3 estan dentro de ±2cm (errores de 1 y 2)
    assert results["chest"]["pct_2cm"] == pytest.approx(2/3 * 100, abs=1e-3)


def test_bias_sign():
    """El sesgo (bias) tiene el signo correcto."""
    preds   = np.array([[105.0]])
    targets = np.array([[100.0]])
    results = calcular_metricas(preds, targets, ["chest"])
    assert results["chest"]["bias"] > 0  # sobreestimacion


def test_bland_altman_basic():
    """Bland-Altman retorna las claves correctas."""
    pred = np.array([100.0, 101.0, 99.0])
    ref  = np.array([100.0, 100.0, 100.0])
    ba   = bland_altman(pred, ref)
    assert "bias"      in ba
    assert "loa_upper" in ba
    assert "loa_lower" in ba
    assert ba["loa_upper"] > ba["loa_lower"]


def test_global_is_mean_of_measures():
    """El MAE global es la media de los MAEs por medida."""
    preds   = np.array([[101.0, 102.0, 103.0]])
    targets = np.zeros_like(preds)
    results = calcular_metricas(preds, targets, MEAS_NAMES)
    expected_global = np.mean([results[n]["mae"] for n in MEAS_NAMES])
    assert results["_global"]["mae"] == pytest.approx(expected_global, abs=1e-6)


def test_aggregate_predictions_by_subject_averages_repeated_captures():
    frame = pd.DataFrame(
        {
            "subject_id": ["a", "a", "b"],
            "true_chest": [100.0, 100.0, 90.0],
            "pred_chest": [98.0, 102.0, 91.0],
        }
    )
    result = agregar_predicciones_por_sujeto(frame, ["chest"])
    assert result["subject_id"].tolist() == ["a", "b"]
    assert result["n_captures"].tolist() == [2, 1]
    assert result["pred_chest"].tolist() == pytest.approx([100.0, 91.0])
    assert result["abs_error_chest"].tolist() == pytest.approx([0.0, 1.0])


def test_aggregate_predictions_by_subject_rejects_inconsistent_targets():
    frame = pd.DataFrame(
        {
            "subject_id": ["a", "a"],
            "true_chest": [100.0, 101.0],
            "pred_chest": [99.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="inconsistentes"):
        agregar_predicciones_por_sujeto(frame, ["chest"])


if __name__ == "__main__":
    test_perfect_predictions()
    test_known_mae()
    test_pct_2cm()
    test_bias_sign()
    test_bland_altman_basic()
    test_global_is_mean_of_measures()
    print("\n[OK] Todos los tests de metricas pasaron.")
