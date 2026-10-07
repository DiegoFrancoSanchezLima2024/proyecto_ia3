"""Dataset PyTorch estricto para las siluetas frontal y lateral de BodyM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class TransformacionSiluetaPareada:
    """Transformacion ligera sin albumentations y sincronizada entre vistas."""

    def __init__(
        self,
        img_size: int | Sequence[int],
        augment: bool,
        config: Optional[Mapping] = None,
    ):
        cfg = dict(config or {})
        if isinstance(img_size, Sequence) and not isinstance(img_size, (str, bytes)):
            if len(img_size) != 2:
                raise ValueError("img_size debe ser un entero o [alto, ancho]")
            self.resize_size = [int(img_size[0]), int(img_size[1])]
        else:
            side = int(img_size)
            self.resize_size = [side, side]
        if min(self.resize_size) <= 0:
            raise ValueError("Las dimensiones de img_size deben ser positivas")
        self.augment = bool(augment)
        self.rotation_limit = float(cfg.get("rotation_limit", 4.0))
        self.morph_kernel = int(cfg.get("morphology_kernel", 3))
        self.morph_p = float(cfg.get("morphology_probability", 0.20))
        self.noise_std = float(cfg.get("gaussian_noise_std", 0.015))
        self.noise_p = float(cfg.get("gaussian_noise_probability", 0.20))
        self.blur_p = float(cfg.get("blur_probability", 0.15))

    def sample_params(self) -> dict:
        if not self.augment:
            return {"angle": 0.0, "morph": None, "noise": False, "blur": False}
        morph = None
        if random.random() < self.morph_p:
            morph = random.choice(("dilate", "erode"))
        return {
            "angle": random.uniform(-self.rotation_limit, self.rotation_limit),
            "morph": morph,
            "noise": random.random() < self.noise_p,
            "blur": random.random() < self.blur_p,
        }

    def __call__(self, image: Image.Image, params: dict) -> torch.Tensor:
        image = TF.rotate(
            image,
            angle=params["angle"],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        image = TF.resize(
            image,
            self.resize_size,
            interpolation=InterpolationMode.NEAREST,
        )
        tensor = (TF.pil_to_tensor(image).float() / 255.0 > 0.5).float()

        kernel = max(1, self.morph_kernel)
        if kernel % 2 == 0:
            kernel += 1
        if params["morph"] == "dilate":
            tensor = F.max_pool2d(tensor.unsqueeze(0), kernel, 1, kernel // 2).squeeze(0)
        elif params["morph"] == "erode":
            tensor = 1.0 - F.max_pool2d(
                (1.0 - tensor).unsqueeze(0), kernel, 1, kernel // 2
            ).squeeze(0)

        if params["blur"]:
            tensor = TF.gaussian_blur(tensor, kernel_size=[3, 3], sigma=[0.1, 1.0])
        if params["noise"] and self.noise_std > 0:
            tensor = (tensor + torch.randn_like(tensor) * self.noise_std).clamp_(0.0, 1.0)

        return (tensor - 0.5) / 0.5


class ConjuntoDatosBodyM(Dataset):
    """Carga pares de vistas, metadatos y objetivos normalizados.

    Las imagenes faltantes son un error. Devolver una imagen negra ocultaba rutas
    incorrectas y producia metricas aparentemente validas sobre datos inexistentes.
    """

    def __init__(
        self,
        data_csv: str,
        img_root: str,
        meas_cols: list,
        height_col: str,
        gender_col: str,
        weight_col: str = "weight_kg",
        use_weight_meta: bool = False,
        use_gender_meta: bool = True,
        img_size: int | Sequence[int] = 320,
        augment: bool = False,
        augmentation_config: Optional[Mapping] = None,
        label_mean: Optional[np.ndarray] = None,
        label_std: Optional[np.ndarray] = None,
        height_mean: Optional[float] = None,
        height_std: Optional[float] = None,
        weight_mean: Optional[float] = None,
        weight_std: Optional[float] = None,
        strict_images: bool = True,
        front_dir: str = "mask",
        left_dir: str = "mask_left",
    ):
        self.data_csv = Path(data_csv)
        self.df = pd.read_csv(self.data_csv)
        self.img_root = Path(img_root)
        self.meas_cols = list(meas_cols)
        self.height_col = height_col
        self.gender_col = gender_col
        self.weight_col = weight_col
        self.use_weight_meta = bool(use_weight_meta)
        self.use_gender_meta = bool(use_gender_meta)
        if self.use_weight_meta and weight_col in self.meas_cols:
            raise ValueError(
                "weight_col no puede ser entrada y objetivo a la vez; produciria fuga de datos"
            )
        self.front_dir = front_dir
        self.left_dir = left_dir
        self.transform = TransformacionSiluetaPareada(img_size, augment, augmentation_config)

        required = {"photo_id", height_col, *self.meas_cols}
        if self.use_gender_meta:
            required.add(gender_col)
        if self.use_weight_meta:
            required.add(weight_col)
        missing_cols = sorted(required.difference(self.df.columns))
        if missing_cols:
            raise ValueError(f"Columnas ausentes en {self.data_csv}: {missing_cols}")

        numeric_cols = [height_col, *self.meas_cols]
        if self.use_gender_meta:
            numeric_cols.append(gender_col)
        if self.use_weight_meta:
            numeric_cols.append(weight_col)
        numeric = self.df[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        invalid_rows = ~np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
        if invalid_rows.any():
            raise ValueError(
                f"{int(invalid_rows.sum())} filas contienen NaN/inf en {self.data_csv}. "
                "Regenera los splits con src/data/clean_bodym.py."
            )

        labels = numeric[self.meas_cols].to_numpy(dtype=np.float32)
        self.label_mean = np.asarray(
            label_mean if label_mean is not None else labels.mean(axis=0), dtype=np.float32
        )
        self.label_std = np.asarray(
            label_std if label_std is not None else labels.std(axis=0), dtype=np.float32
        )
        self.label_std = np.maximum(self.label_std, 1e-6)

        heights = numeric[height_col].to_numpy(dtype=np.float32)
        self.height_mean = float(height_mean if height_mean is not None else heights.mean())
        self.height_std = max(
            float(height_std if height_std is not None else heights.std()), 1e-6
        )

        self.weight_mean = None
        self.weight_std = None
        if self.use_weight_meta:
            weights = numeric[weight_col].to_numpy(dtype=np.float32)
            self.weight_mean = float(
                weight_mean if weight_mean is not None else weights.mean()
            )
            self.weight_std = max(
                float(weight_std if weight_std is not None else weights.std()), 1e-6
            )

        if strict_images:
            self._validate_image_paths()

    @staticmethod
    def _photo_filename(value) -> str:
        photo = str(value).strip()
        return photo if photo.lower().endswith(".png") else f"{photo}.png"

    def _paths_for_row(self, row: pd.Series) -> tuple[Path, Path]:
        photo = self._photo_filename(row["photo_id"])
        return self.img_root / self.front_dir / photo, self.img_root / self.left_dir / photo

    def _validate_image_paths(self) -> None:
        missing = []
        for _, row in self.df.iterrows():
            for path in self._paths_for_row(row):
                if not path.is_file():
                    missing.append(str(path))
                    if len(missing) >= 5:
                        break
            if len(missing) >= 5:
                break
        if missing:
            examples = "\n  - ".join(missing)
            raise FileNotFoundError(
                f"Faltan imagenes para {self.data_csv}. Primeros casos:\n  - {examples}"
            )

    @staticmethod
    def _load_gray(path: Path) -> Image.Image:
        if not path.is_file():
            raise FileNotFoundError(f"No existe la silueta requerida: {path}")
        with Image.open(path) as image:
            return image.convert("L").copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        front_path, left_path = self._paths_for_row(row)
        params = self.transform.sample_params()
        front_t = self.transform(self._load_gray(front_path), params)
        left_t = self.transform(self._load_gray(left_path), params)

        height = float(row[self.height_col])
        meta = [(height - self.height_mean) / self.height_std]
        if self.use_gender_meta:
            meta.append(float(row[self.gender_col]))
        if self.use_weight_meta:
            weight = float(row[self.weight_col])
            meta.append((weight - self.weight_mean) / self.weight_std)
        meta_t = torch.tensor(meta, dtype=torch.float32)

        raw_label = row[self.meas_cols].to_numpy(dtype=np.float32)
        label_norm = (raw_label - self.label_mean) / self.label_std
        return (
            front_t,
            left_t,
            meta_t,
            torch.from_numpy(label_norm.astype(np.float32)),
            torch.from_numpy(raw_label),
        )

    @property
    def n_measures(self) -> int:
        return len(self.meas_cols)
