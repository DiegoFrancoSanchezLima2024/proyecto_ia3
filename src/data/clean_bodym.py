"""Construye splits BodyM reproducibles sin alterar las etiquetas de test.

El conjunto oficial de test se conserva como evaluacion externa. Train, validacion
y calibracion conformal se separan por sujeto a partir del split oficial de train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BODYM_DIR = ROOT / "dataset" / "bodym"
OUT_DIR = ROOT / "dataset" / "bodym_clean"
REPORT_DIR = ROOT / "reports"

MEASUREMENT_COLUMNS = [
    "chest",
    "waist",
    "hip",
    "thigh",
    "calf",
    "ankle",
    "arm-length",
    "forearm",
    "wrist",
    "bicep",
    "shoulder-breadth",
    "leg-length",
    "shoulder-to-crotch",
]


def white_ratio(path: Path) -> float:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
        return float((array > 128).mean())
    except (OSError, ValueError):
        return -1.0


def image_status(path: Path, min_ratio: float, max_ratio: float) -> str:
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "zero_bytes"
    ratio = white_ratio(path)
    if ratio < 0:
        return "corrupt"
    if ratio < min_ratio:
        return f"too_dark({ratio:.4f})"
    if ratio > max_ratio:
        return f"too_white({ratio:.4f})"
    return "ok"


def normalize_gender(series: pd.Series) -> pd.Series:
    mapping = {
        "female": 0,
        "f": 0,
        "mujer": 0,
        "woman": 0,
        "0": 0,
        "male": 1,
        "m": 1,
        "hombre": 1,
        "man": 1,
        "1": 1,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        unknown = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Valores de genero no reconocidos: {unknown}")
    return normalized.astype(np.int64)


def load_official_split(
    name: str,
    source_dir: Path,
    min_ratio: float,
    max_ratio: float,
) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_csv(source_dir / "hwg_metadata.csv")
    measurements = pd.read_csv(source_dir / "measurements.csv")
    photo_map = pd.read_csv(source_dir / "subject_to_photo_map.csv")
    for frame in (metadata, measurements, photo_map):
        frame.columns = frame.columns.str.strip()

    required_meta = {"subject_id", "gender", "height_cm", "weight_kg"}
    required_measurements = {"subject_id", *MEASUREMENT_COLUMNS}
    required_map = {"subject_id", "photo_id"}
    for label, frame, required in (
        ("metadata", metadata, required_meta),
        ("measurements", measurements, required_measurements),
        ("photo map", photo_map, required_map),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Faltan columnas en {name}/{label}: {missing}")

    data = metadata.merge(measurements, on="subject_id", how="inner", validate="one_to_one")
    data = data.merge(photo_map[["subject_id", "photo_id"]], on="subject_id", how="inner")
    data["gender"] = normalize_gender(data["gender"])

    numeric_cols = ["height_cm", "weight_kg", *MEASUREMENT_COLUMNS]
    numeric = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    valid_numeric = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    valid_positive = (numeric.to_numpy(dtype=np.float64) > 0).all(axis=1)

    statuses = []
    valid_images = []
    for photo_id in data["photo_id"]:
        filename = str(photo_id)
        if not filename.lower().endswith(".png"):
            filename += ".png"
        front = image_status(source_dir / "mask" / filename, min_ratio, max_ratio)
        left = image_status(source_dir / "mask_left" / filename, min_ratio, max_ratio)
        statuses.append({"photo_id": photo_id, "front": front, "left": left})
        valid_images.append(front == "ok" and left == "ok")

    keep = valid_numeric & valid_positive & np.asarray(valid_images, dtype=bool)
    clean = data.loc[keep, [
        "subject_id",
        "photo_id",
        "height_cm",
        "weight_kg",
        "gender",
        *MEASUREMENT_COLUMNS,
    ]].copy()

    report = {
        "source": str(source_dir),
        "rows_merged": int(len(data)),
        "rows_valid": int(len(clean)),
        "invalid_numeric": int((~valid_numeric).sum()),
        "non_positive_numeric": int((~valid_positive).sum()),
        "invalid_images": int((~np.asarray(valid_images, dtype=bool)).sum()),
        "image_failures": [status for status in statuses if status["front"] != "ok" or status["left"] != "ok"][:50],
    }
    print(
        f"[{name}] merge={len(data)} validas={len(clean)} "
        f"invalidas_imagen={report['invalid_images']} invalidas_numericas={report['invalid_numeric']}"
    )
    return clean.reset_index(drop=True), report


def split_by_subject(
    data: pd.DataFrame,
    val_ratio: float,
    calibration_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if val_ratio <= 0 or calibration_ratio <= 0 or val_ratio + calibration_ratio >= 1:
        raise ValueError("val_ratio y calibration_ratio deben ser positivos y sumar menos de 1")

    subjects = data["subject_id"].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    n_cal = max(1, int(round(len(subjects) * calibration_ratio)))
    val_subjects = set(subjects[:n_val])
    cal_subjects = set(subjects[n_val:n_val + n_cal])

    splits = {
        "val": data[data["subject_id"].isin(val_subjects)].reset_index(drop=True),
        "calibration": data[data["subject_id"].isin(cal_subjects)].reset_index(drop=True),
        "train": data[
            ~data["subject_id"].isin(val_subjects.union(cal_subjects))
        ].reset_index(drop=True),
    }
    subject_sets = {name: set(frame["subject_id"].unique()) for name, frame in splits.items()}
    assert subject_sets["train"].isdisjoint(subject_sets["val"])
    assert subject_sets["train"].isdisjoint(subject_sets["calibration"])
    assert subject_sets["val"].isdisjoint(subject_sets["calibration"])
    return splits


def save_split(name: str, data: pd.DataFrame) -> None:
    output = OUT_DIR / name
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "data.csv", index=False)
    print(f"  {name:<12} {len(data):>5} fotos / {data['subject_id'].nunique():>4} sujetos")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    parser.add_argument("--min-white-ratio", type=float, default=0.03)
    parser.add_argument("--max-white-ratio", type=float, default=0.95)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    official_train, train_report = load_official_split(
        "train", BODYM_DIR / "train", args.min_white_ratio, args.max_white_ratio
    )
    train_splits = split_by_subject(
        official_train, args.val_ratio, args.calibration_ratio, args.seed
    )
    for name in ("train", "val", "calibration"):
        save_split(name, train_splits[name])

    test, test_report = load_official_split(
        "test", BODYM_DIR / "testA", args.min_white_ratio, args.max_white_ratio
    )
    test_wild, wild_report = load_official_split(
        "test_wild", BODYM_DIR / "testB", args.min_white_ratio, args.max_white_ratio
    )
    save_split("test", test)
    save_split("test_wild", test_wild)

    stats = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "calibration_ratio": args.calibration_ratio,
        "measurements": MEASUREMENT_COLUMNS,
        "splits": {
            name: {
                "n_samples": int(len(frame)),
                "n_subjects": int(frame["subject_id"].nunique()),
            }
            for name, frame in {**train_splits, "test": test, "test_wild": test_wild}.items()
        },
        "integrity": {
            "train": train_report,
            "test": test_report,
            "test_wild": wild_report,
        },
    }
    (REPORT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Reporte: {REPORT_DIR / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
