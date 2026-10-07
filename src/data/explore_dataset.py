"""
src/data/explore_dataset.py
============================
EDA del Amazon Body Measurements Dataset (BodyM).
Muestra estadisticas de metadata, medidas corporales y siluetas de ejemplo.

Ejecutar desde la raiz del proyecto:
    python src/data/explore_dataset.py

Requiere: pip install pandas matplotlib numpy Pillow
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# --- Rutas ------------------------------------------------------------------
# src/data/explore_dataset.py  ->  parents[2] = raiz del proyecto
ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dataset" / "bodym"
TRAIN_DIR = DATA_ROOT / "train"
TESTA_DIR = DATA_ROOT / "testA"
TESTB_DIR = DATA_ROOT / "testB"
REPORTS   = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

print(f"[INFO] Raiz del proyecto : {ROOT}")
print(f"[INFO] Dataset en        : {DATA_ROOT}")

# --- Carga CSVs -------------------------------------------------------------
print("\nCargando CSVs...")
train_meta = pd.read_csv(TRAIN_DIR / "hwg_metadata.csv")
train_meas = pd.read_csv(TRAIN_DIR / "measurements.csv")
train_map  = pd.read_csv(TRAIN_DIR / "subject_to_photo_map.csv")
testA_meta = pd.read_csv(TESTA_DIR / "hwg_metadata.csv")
testB_meta = pd.read_csv(TESTB_DIR / "hwg_metadata.csv")

print("=" * 60)
print(f"TRAIN  metadata: {train_meta.shape}  | measurements: {train_meas.shape}")
print(f"TEST A metadata: {testA_meta.shape}")
print(f"TEST B metadata: {testB_meta.shape}")
print(f"\nColumnas hwg_metadata : {train_meta.columns.tolist()}")
print(f"Columnas measurements : {train_meas.columns.tolist()}")
print(f"\nEstadisticas metadata:\n{train_meta.describe()}")
print(f"\nEstadisticas measurements:\n{train_meas.describe()}")

# --- Detectar columnas clave ------------------------------------------------
height_col = next((c for c in train_meta.columns if "height" in c.lower()), None)
weight_col = next((c for c in train_meta.columns if "weight" in c.lower()), None)
gender_col = next((c for c in train_meta.columns
                   if "gender" in c.lower() or "sex" in c.lower()), None)

if gender_col:
    print(f"\nDistribucion genero ({gender_col}):\n{train_meta[gender_col].value_counts()}")

# --- Figura 1: Distribuciones y conteos -------------------------------------
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Amazon Body Measurements -- EDA", fontsize=16, fontweight="bold")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

if height_col:
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(train_meta[height_col].dropna(), bins=40, color="steelblue", edgecolor="white")
    ax.set_title("Altura"); ax.set_xlabel(height_col)

if weight_col:
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(train_meta[weight_col].dropna(), bins=40, color="coral", edgecolor="white")
    ax.set_title("Peso"); ax.set_xlabel(weight_col)

if gender_col:
    ax = fig.add_subplot(gs[0, 2])
    counts = train_meta[gender_col].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color=["#4A90D9", "#E86C6C"])
    ax.set_title("Genero")

ax = fig.add_subplot(gs[0, 3])
bars = ax.bar(
    ["Train", "TestA", "TestB"],
    [len(train_meta), len(testA_meta), len(testB_meta)],
    color=["#2ecc71", "#e67e22", "#9b59b6"]
)
ax.set_title("Sujetos por Split")
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
            str(int(b.get_height())), ha="center", va="bottom", fontsize=9)

# Histogramas de medidas corporales (hasta 8 columnas numericas)
skip = {height_col, weight_col}
meas_cols = [c for c in train_meas.select_dtypes(include=np.number).columns
             if c not in skip][:8]

for i, col in enumerate(meas_cols[:4]):
    ax = fig.add_subplot(gs[1, i])
    ax.hist(train_meas[col].dropna(), bins=35, color="#8e44ad", edgecolor="white")
    ax.set_title(col[:18]); ax.tick_params(labelsize=7)

for i, col in enumerate(meas_cols[4:8]):
    ax = fig.add_subplot(gs[2, i])
    ax.hist(train_meas[col].dropna(), bins=35, color="#16a085", edgecolor="white")
    ax.set_title(col[:18]); ax.tick_params(labelsize=7)

out1 = REPORTS / "eda_report.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\n[OK] {out1} guardado")
plt.show()

# --- Figura 2: Muestra de siluetas ------------------------------------------
mask_dir      = TRAIN_DIR / "mask"
mask_left_dir = TRAIN_DIR / "mask_left"

if mask_dir.exists():
    imgs   = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    sample = random.sample(imgs, min(6, len(imgs)))
    fig2, axes = plt.subplots(2, 6, figsize=(18, 7))
    fig2.suptitle("Siluetas -- Frontal (arriba) / Lateral (abajo)", fontsize=13, fontweight="bold")
    for j, fname in enumerate(sample):
        axes[0, j].imshow(
            Image.open(mask_dir / fname).convert("L"), cmap="gray"
        )
        axes[0, j].axis("off")
        axes[0, j].set_title(fname[:8], fontsize=7)
        lp = mask_left_dir / fname
        if lp.exists():
            axes[1, j].imshow(Image.open(lp).convert("L"), cmap="gray")
        axes[1, j].axis("off")
    plt.tight_layout()
    out2 = REPORTS / "silhouettes_sample.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"[OK] {out2} guardado")
    plt.show()
else:
    print("[WARN] No se encontro carpeta mask/ -- se omite visualizacion de siluetas")

print("\n[OK] EDA completado.")
