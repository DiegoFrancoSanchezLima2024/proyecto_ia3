"""
src/evaluation/metrics.py
==========================
Metricas completas de evaluacion antropometrica.
Todas las metricas se calculan POR MEDIDA, no como promedio global.
"""
import numpy as np
import pandas as pd
from scipy import stats


def calcular_metricas(
    preds: np.ndarray,      # [N, M] predicciones en cm
    targets: np.ndarray,    # [N, M] valores reales en cm
    meas_names: list,       # nombres de las M medidas
) -> dict:
    """
    Computa MAE, RMSE, mediana, sesgo, P90 y porcentajes por tolerancia.

    Returns:
        dict con metricas por medida y resumen global.
    """
    assert preds.shape == targets.shape
    errors = preds - targets          # [N, M] con signo
    abs_err = np.abs(errors)          # [N, M] absoluto

    results = {}
    for i, name in enumerate(meas_names):
        e = errors[:, i]
        ae = abs_err[:, i]
        results[name] = {
            "mae":         float(ae.mean()),
            "rmse":        float(np.sqrt((e**2).mean())),
            "median_ae":   float(np.median(ae)),
            "p90_ae":      float(np.percentile(ae, 90)),
            "bias":        float(e.mean()),
            "pct_1cm":     float((ae <= 1.0).mean() * 100),
            "pct_2cm":     float((ae <= 2.0).mean() * 100),
            "pct_3cm":     float((ae <= 3.0).mean() * 100),
            "std":         float(e.std()),
        }

    # Resumen global (media de cada metrica sobre todas las medidas)
    results["_global"] = {
        "mae":       float(np.mean([results[n]["mae"]    for n in meas_names])),
        "rmse":      float(np.mean([results[n]["rmse"]   for n in meas_names])),
        "pct_1cm":   float(np.mean([results[n]["pct_1cm"]for n in meas_names])),
        "pct_2cm":   float(np.mean([results[n]["pct_2cm"]for n in meas_names])),
        "pct_3cm":   float(np.mean([results[n]["pct_3cm"]for n in meas_names])),
        "p90_ae":    float(np.mean([results[n]["p90_ae"] for n in meas_names])),
    }
    return results


def agregar_predicciones_por_sujeto(
    frame: pd.DataFrame,
    meas_names: list,
    target_tolerance: float = 1e-4,
) -> pd.DataFrame:
    """Promedia capturas repetidas y devuelve una prediccion por persona."""
    if "subject_id" not in frame.columns:
        raise ValueError("Se requiere la columna subject_id")
    if frame.empty:
        raise ValueError("No hay predicciones para agregar")

    true_cols = [f"true_{name}" for name in meas_names]
    pred_cols = [f"pred_{name}" for name in meas_names]
    missing = [name for name in true_cols + pred_cols if name not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas para agregar por sujeto: {missing}")

    grouped = frame.groupby("subject_id", sort=False)
    spread = grouped[true_cols].max() - grouped[true_cols].min()
    if float(spread.to_numpy().max()) > target_tolerance:
        raise ValueError("Un sujeto tiene medidas reales inconsistentes entre capturas")

    result = grouped[true_cols].first()
    result[pred_cols] = grouped[pred_cols].mean()
    result.insert(0, "n_captures", grouped.size())
    result = result.reset_index()
    for name in meas_names:
        result[f"abs_error_{name}"] = np.abs(
            result[f"pred_{name}"] - result[f"true_{name}"]
        )
    return result


def bland_altman(
    method1: np.ndarray,  # [N] predicciones
    method2: np.ndarray,  # [N] referencias (cinta metrica manual)
) -> dict:
    """Analisis de Bland-Altman para una medida."""
    diff  = method1 - method2
    mean_ = (method1 + method2) / 2
    bias  = float(diff.mean())
    sd    = float(diff.std())
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    return {
        "bias": bias, "sd": sd,
        "loa_upper": loa_upper, "loa_lower": loa_lower,
        "mean_comparison": mean_,
        "differences": diff,
    }


def imprimir_tabla_metricas(results: dict, meas_names: list):
    """Imprime tabla de metricas en consola."""
    header = f"{'Medida':<22} {'MAE':>6} {'RMSE':>6} {'Med':>6} {'P90':>6} {'%±1cm':>7} {'%±2cm':>7} {'%±3cm':>7} {'Bias':>7}"
    print("\n" + "="*75)
    print(header)
    print("-"*75)
    for name in meas_names:
        r = results.get(name, {})
        print(f"{name:<22} {r.get('mae',0):>6.2f} {r.get('rmse',0):>6.2f} "
              f"{r.get('median_ae',0):>6.2f} {r.get('p90_ae',0):>6.2f} "
              f"{r.get('pct_1cm',0):>6.1f}% {r.get('pct_2cm',0):>6.1f}% "
              f"{r.get('pct_3cm',0):>6.1f}% "
              f"{r.get('bias',0):>+7.2f}")
    print("-"*75)
    g = results.get("_global", {})
    print(f"{'GLOBAL':<22} {g.get('mae',0):>6.2f} {g.get('rmse',0):>6.2f} "
          f"{'':>6} {g.get('p90_ae',0):>6.2f} "
          f"{g.get('pct_1cm',0):>6.1f}% {g.get('pct_2cm',0):>6.1f}% "
          f"{g.get('pct_3cm',0):>6.1f}%")
    print("="*75)
