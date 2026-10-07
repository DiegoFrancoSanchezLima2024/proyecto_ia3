"""
src/geometry/silhouette.py
===========================
Extraccion de perfiles anatomicos desde siluetas binarias.
Complementa geometric_est.py con funciones de bajo nivel.
"""
import numpy as np
from PIL import Image
from pathlib import Path


def load_silhouette(path: Path, size: int = None) -> np.ndarray:
    """Carga silueta como array uint8 HxW. Opcional: resize."""
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.NEAREST)
    return np.array(img)


def binarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convierte imagen gris a binaria 0/255."""
    return (img > threshold).astype(np.uint8) * 255


def body_bbox(binary: np.ndarray):
    """
    Retorna (row_min, row_max, col_min, col_max) del cuerpo.
    Asume fondo negro, cuerpo blanco.
    """
    rows = np.any(binary > 128, axis=1)
    cols = np.any(binary > 128, axis=0)
    if not rows.any():
        return 0, binary.shape[0], 0, binary.shape[1]
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return int(r_min), int(r_max), int(c_min), int(c_max)


def crop_to_body(binary: np.ndarray, margin: int = 5) -> np.ndarray:
    """Recorta la imagen al bbox del cuerpo con un margen."""
    r0, r1, c0, c1 = body_bbox(binary)
    H, W = binary.shape
    r0 = max(0, r0 - margin)
    r1 = min(H, r1 + margin)
    c0 = max(0, c0 - margin)
    c1 = min(W, c1 + margin)
    return binary[r0:r1, c0:c1]


def width_profile(binary: np.ndarray) -> np.ndarray:
    """
    Perfil de ancho horizontal por fila (pixeles con cuerpo).
    Retorna array [H] con el ancho en px para cada fila.
    """
    mask = binary > 128
    widths = np.zeros(binary.shape[0], dtype=np.float32)
    for i, row in enumerate(mask):
        cols = np.where(row)[0]
        if len(cols) >= 2:
            widths[i] = float(cols[-1] - cols[0] + 1)
    return widths


def depth_profile(binary: np.ndarray) -> np.ndarray:
    """
    Igual que width_profile pero para la silueta lateral.
    El 'depth' se infiere del ancho lateral.
    """
    return width_profile(binary)


def smooth_profile(profile: np.ndarray, window: int = 5) -> np.ndarray:
    """Suaviza el perfil con media movil."""
    kernel = np.ones(window) / window
    return np.convolve(profile, kernel, mode="same")


def normalize_to_height(profile: np.ndarray, height_cm: float) -> np.ndarray:
    """
    Convierte un perfil de pixeles a cm usando la altura total.
    Asume que el cuerpo ocupa la altura total de la imagen.
    """
    H = len(profile)
    if H == 0 or height_cm <= 0:
        return profile
    scale = height_cm / H   # cm por pixel
    return profile * scale


def find_anatomical_row(profile: np.ndarray, fraction: float) -> int:
    """
    Retorna el indice de fila correspondiente a una fraccion de la altura.
    fraction=0.0 = cabeza, fraction=1.0 = pies.
    """
    H = len(profile)
    return int(np.clip(H * fraction, 0, H - 1))


def extract_width_at_fraction(
    profile: np.ndarray,
    fraction: float,
    height_cm: float,
    window_rows: int = 3,
) -> float:
    """
    Extrae el ancho medio (en cm) en una fraccion de la altura corporal.
    Promedia sobre window_rows filas para robustez.
    """
    H = len(profile)
    row = find_anatomical_row(profile, fraction)
    r0 = max(0, row - window_rows)
    r1 = min(H, row + window_rows + 1)
    px_mean = profile[r0:r1].mean()
    scale = height_cm / H
    return float(px_mean * scale)
