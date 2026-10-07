#!/usr/bin/env python3
"""Evalúa el artefacto exp_008 sin reentrenarlo ni alterar sus pesos."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import r2_score

from benchmark_regresor_perfiles import OBJETIVOS, extraer_split


RAIZ = Path(__file__).resolve().parents[1]
DIRECTORIO = RAIZ / "experiments" / "exp_008_perfiles_arboles"


def metricas(prediccion: np.ndarray, real: np.ndarray, q_hat: np.ndarray) -> dict:
    error = prediccion - real
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    r2 = r2_score(real, prediccion, multioutput="raw_values")
    cobertura = np.mean(np.abs(error) <= q_hat[None, :], axis=0)
    por_objetivo = {
        nombre: {
            "mae": float(mae[indice]),
            "rmse": float(rmse[indice]),
            "r2": float(r2[indice]),
            "sesgo": float(np.mean(error[:, indice])),
            "cobertura_conformal_90": float(cobertura[indice]),
            "radio_intervalo": float(q_hat[indice]),
        }
        for indice, nombre in enumerate(OBJETIVOS)
    }
    return {
        "muestras": int(len(real)),
        "mae_medio_13_cm": float(np.mean(mae[:-1])),
        "rmse_medio_13_cm": float(np.mean(rmse[:-1])),
        "r2_medio_13": float(np.mean(r2[:-1])),
        "mae_peso_kg": float(mae[-1]),
        "rmse_peso_kg": float(rmse[-1]),
        "r2_peso": float(r2[-1]),
        "cobertura_media_13": float(np.mean(cobertura[:-1])),
        "por_objetivo": por_objetivo,
    }


def main() -> None:
    paquete = joblib.load(DIRECTORIO / "model.joblib")
    modelo = paquete["model"]
    bins = int(paquete["bins"])
    q_hat = np.load(DIRECTORIO / "conformal_q_hat.npy")
    reporte = {}
    for split in ("test", "test_wild"):
        caracteristicas, objetivos, _ = extraer_split(split, bins)
        reporte[split] = metricas(modelo.predict(caracteristicas), objetivos, q_hat)
        resumen = reporte[split]
        print(
            f"{split}: n={resumen['muestras']} "
            f"MAE13={resumen['mae_medio_13_cm']:.3f} cm "
            f"RMSE13={resumen['rmse_medio_13_cm']:.3f} cm "
            f"R2={resumen['r2_medio_13']:.3f} "
            f"cobertura={resumen['cobertura_media_13']:.3f}"
        )
    salida = DIRECTORIO / "metrics_extended.json"
    salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {salida}")


if __name__ == "__main__":
    main()
