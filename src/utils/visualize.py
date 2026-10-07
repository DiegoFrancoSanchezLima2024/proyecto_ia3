"""
src/utils/visualize.py
=======================
Utilidades de visualizacion para el proyecto Sastre-IA.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def plot_silhouettes(front: np.ndarray, left: np.ndarray, title: str = "", save_path: str = None):
    """Muestra silueta frontal y lateral lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 10))
    axes[0].imshow(front, cmap="gray"); axes[0].set_title("Frontal"); axes[0].axis("off")
    axes[1].imshow(left,  cmap="gray"); axes[1].set_title("Lateral"); axes[1].axis("off")
    if title: fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_width_profile(profile_cm: np.ndarray, height_cm: float, title: str = "", save_path: str = None):
    """Grafica el perfil de ancho corporal en cm."""
    y = np.linspace(0, height_cm, len(profile_cm))
    fig, ax = plt.subplots(1, 1, figsize=(4, 10))
    ax.fill_betweenx(y, -profile_cm / 2, profile_cm / 2, alpha=0.6, color="steelblue")
    ax.plot(-profile_cm / 2, y, "steelblue", lw=1.5)
    ax.plot( profile_cm / 2, y, "steelblue", lw=1.5)
    ax.set_xlabel("Ancho (cm)"); ax.set_ylabel("Altura (cm)")
    ax.invert_yaxis()
    if title: ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def graficar_tabla_metricas(results: dict, meas_names: list, save_path: str = None):
    """Tabla visual de metricas por medida."""
    rows = []
    for name in meas_names:
        r = results.get(name, {})
        rows.append([name, f"{r.get('mae',0):.2f}", f"{r.get('rmse',0):.2f}",
                     f"{r.get('pct_2cm',0):.1f}%", f"{r.get('bias',0):+.2f}"])
    headers = ["Medida", "MAE (cm)", "RMSE (cm)", "%±2cm", "Sesgo (cm)"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.4 + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    # Colorear MAE: verde si <2cm, amarillo si <3cm, rojo si >3cm
    for i, row in enumerate(rows):
        mae = float(row[1])
        color = "#c8e6c9" if mae < 2 else ("#fff9c4" if mae < 3 else "#ffcdd2")
        for j in range(len(headers)):
            tbl[(i + 1, j)].set_facecolor(color)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def graficar_bland_altman(pred: np.ndarray, ref: np.ndarray, measure: str, save_path: str = None):
    """Grafica Bland-Altman para una medida."""
    diff = pred - ref
    mean = (pred + ref) / 2
    bias = diff.mean(); sd = diff.std()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean, diff, alpha=0.5, s=20, color="steelblue")
    ax.axhline(bias, color="red", lw=2, label=f"Sesgo = {bias:.2f} cm")
    ax.axhline(bias + 1.96 * sd, color="orange", lw=1.5, ls="--", label=f"+1.96sd = {bias+1.96*sd:.2f}")
    ax.axhline(bias - 1.96 * sd, color="orange", lw=1.5, ls="--", label=f"-1.96sd = {bias-1.96*sd:.2f}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Media (pred + ref) / 2  (cm)"); ax.set_ylabel("Diferencia pred - ref  (cm)")
    ax.set_title(f"Bland-Altman — {measure}", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def graficar_curvas_entrenamiento(history: list, save_path: str = None):
    """Curvas de entrenamiento desde el history.json."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_loss"]   for h in history], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, [h["mae"] for h in history], color="coral")
    axes[1].axhline(2.0, ls="--", color="green", label="Objetivo 2cm")
    axes[1].set_title("MAE global (cm)"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    axes[2].plot(epochs, [h["pct_2cm"] for h in history], color="steelblue")
    axes[2].axhline(80, ls="--", color="green", label="Objetivo 80%")
    axes[2].set_title("% dentro de ±2cm"); axes[2].legend(); axes[2].set_xlabel("Epoch")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
