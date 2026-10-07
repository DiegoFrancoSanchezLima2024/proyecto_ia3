"""Construye una referencia compacta y reproducible desde ANSUR II público."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Solo equivalencias suficientemente cercanas para un control de plausibilidad.
# No incluimos longitudes con definiciones anatómicas distintas entre datasets.
MEASUREMENT_MAP = {
    "chest": "chestcircumference",
    "waist": "waistcircumference",
    "hip": "buttockcircumference",
    "thigh": "thighcircumference",
    "calf": "calfcircumference",
    "ankle": "anklecircumference",
    "wrist": "wristcircumference",
    "shoulder-breadth": "biacromialbreadth",
}

EXPECTED_ROWS = {"male": 4082, "female": 1986}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_sex(path: Path, sex: str) -> dict:
    frame = pd.read_csv(path, encoding="latin-1")
    if len(frame) != EXPECTED_ROWS[sex]:
        raise ValueError(f"{sex}: {len(frame)} filas; esperadas {EXPECTED_ROWS[sex]}")
    required = {"stature", "weightkg", *MEASUREMENT_MAP.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{sex}: faltan columnas: {', '.join(missing)}")

    # El manual ANSUR expresa longitudes en milímetros y weightkg en décimas de kg.
    height = pd.to_numeric(frame["stature"], errors="coerce") / 10.0
    weight = pd.to_numeric(frame["weightkg"], errors="coerce") / 10.0
    domain = {
        "height_cm": [round(float(v), 2) for v in height.quantile([0.01, 0.99])],
        "weight_kg": [round(float(v), 2) for v in weight.quantile([0.01, 0.99])],
    }
    models = {}
    for target, column in MEASUREMENT_MAP.items():
        values = pd.to_numeric(frame[column], errors="coerce") / 10.0
        valid = pd.concat([height, weight, values], axis=1).dropna()
        x = np.column_stack(
            [np.ones(len(valid)), valid.iloc[:, 0], valid.iloc[:, 1]]
        )
        y = valid.iloc[:, 2].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ coefficients
        interval = np.quantile(residuals, [0.025, 0.975])
        models[target] = {
            "ansur_column": column,
            "n": int(len(valid)),
            "coefficients": {
                "intercept": round(float(coefficients[0]), 8),
                "height_cm": round(float(coefficients[1]), 8),
                "weight_kg": round(float(coefficients[2]), 8),
            },
            "residual_interval_cm": [round(float(v), 4) for v in interval],
            "population_quantiles_cm": {
                name: round(float(value), 2)
                for name, value in zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    values.quantile([0.05, 0.25, 0.5, 0.75, 0.95]),
                )
            },
        }
    return {
        "rows": int(len(frame)),
        "input_domain_p01_p99": domain,
        "measurements": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="dataset/ansur2/raw")
    parser.add_argument(
        "--output", default="configs/anthropometry/ansur2_reference.json"
    )
    args = parser.parse_args()
    raw = ROOT / args.raw_dir
    paths = {
        "male": raw / "ANSUR_II_MALE_Public.csv",
        "female": raw / "ANSUR_II_FEMALE_Public.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "schema_version": "1.0",
        "reference_id": "ansur2_public_height_weight_ols_v1",
        "purpose": "external_plausibility_filter_not_training_ground_truth",
        "units": {"measurements": "cm", "weight": "kg"},
        "source": {
            "official_page": "https://ph.health.mil/topics/workplacehealth/ergo/Pages/Anthropometric-Database.aspx",
            "public_release": "ANSUR II 2012 public CSV files",
            "acquisition_mirror": "https://github.com/senihberkay/US-Army-ANSUR-II",
            "local_files": {
                sex: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for sex, path in paths.items()
            },
        },
        "fit": {
            "predictors": ["height_cm", "weight_kg"],
            "method": "ordinary_least_squares",
            "residual_interval": "empirical_95_percent",
        },
        "sex_models": {sex: fit_sex(path, sex) for sex, path in paths.items()},
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Referencia creada: {output}")
    print("Filas:", {sex: model["rows"] for sex, model in payload["sex_models"].items()})


if __name__ == "__main__":
    main()
