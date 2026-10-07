"""Inferencia local SCHP-ATR ONNX INT8 y mascaras utiles para composicion.

ATR ofrece clases semanticas de cara, pelo, brazos, piernas y prendas. No
ofrece manos, cuello ni divisiones superior/inferior del brazo; estas se
derivan de ATR + landmarks MediaPipe y se declaran como refinamientos.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "pretrained" / "schp-atr" / "onnx" / "schp-atr-18-int8-static.onnx"


def _pose_point(landmarks: list[dict], index: int, size: tuple[int, int]) -> tuple[float, float]:
    return float(landmarks[index]["x"]) * size[0], float(landmarks[index]["y"]) * size[1]


def infer_atr(photo: Image.Image, landmarks: list[dict], model_path: Path = DEFAULT_MODEL) -> dict[str, Image.Image]:
    import onnxruntime as ort

    if not model_path.exists():
        raise FileNotFoundError(f"Falta el modelo local SCHP-ATR: {model_path}")
    rgb = np.asarray(photo.convert("RGB"), dtype=np.uint8)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.406, 0.456, 0.485], dtype=np.float32)
    std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    tensor = ((resized - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    labels_small = logits.argmax(axis=0).astype(np.uint8)
    labels = cv2.resize(labels_small, photo.size, interpolation=cv2.INTER_NEAREST)
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51], [255, 85, 0],
        [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85], [85, 51, 0], [52, 86, 128],
        [0, 128, 0], [0, 0, 255], [51, 170, 221], [0, 255, 255], [85, 255, 170], [170, 255, 85],
    ], dtype=np.uint8)
    labels_color = Image.fromarray(palette[labels], "RGB")

    def mask(ids: tuple[int, ...]) -> Image.Image:
        return Image.fromarray((np.isin(labels, ids) * 255).astype(np.uint8), "L")

    arms = mask((14, 15))
    face_neck = mask((11,))
    hair = mask((1, 2))
    upper = mask((4, 7))
    lower = mask((5, 6, 7, 8))
    legs = mask((12, 13))
    shoes = mask((9, 10))

    width, height = photo.size
    shoulder_width = abs(_pose_point(landmarks, 12, photo.size)[0] - _pose_point(landmarks, 11, photo.size)[0])
    hand_seed = Image.new("L", photo.size, 0)
    draw = ImageDraw.Draw(hand_seed)
    radius = max(8, shoulder_width * 0.15)
    for indices in ((15, 17, 19, 21), (16, 18, 20, 22)):
        for index in indices:
            x, y = _pose_point(landmarks, index, photo.size)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    # La mano es la porcion distal de la clase brazo alrededor de los puntos
    # de mano. Se amplia levemente porque MediaPipe Pose no dibuja dedos.
    hands = Image.fromarray(
        cv2.bitwise_and(np.asarray(arms), np.asarray(hand_seed.filter(ImageFilter.MaxFilter(9)))), "L"
    ).filter(ImageFilter.MaxFilter(5))

    ear_y = max(_pose_point(landmarks, 7, photo.size)[1], _pose_point(landmarks, 8, photo.size)[1])
    shoulder_y = (_pose_point(landmarks, 11, photo.size)[1] + _pose_point(landmarks, 12, photo.size)[1]) / 2
    neck_split = ear_y + (shoulder_y - ear_y) * 0.56
    rows = np.arange(height)[:, None]
    face_neck_array = np.asarray(face_neck)
    face = Image.fromarray(np.where(rows < neck_split, face_neck_array, 0).astype(np.uint8), "L")
    neck = Image.fromarray(np.where(rows >= neck_split, face_neck_array, 0).astype(np.uint8), "L")
    shadow_allowed = Image.fromarray(
        np.maximum.reduce([np.asarray(upper), np.asarray(lower), np.asarray(legs)]), "L"
    )
    lighting_reference = Image.fromarray(np.maximum(np.asarray(upper), np.asarray(lower)), "L")
    return {
        "labels": Image.fromarray(labels, "L"), "labels_color": labels_color, "arms": arms, "hands": hands,
        "face": face, "neck": neck, "hair": hair, "upper_clothes": upper,
        "lower_clothes": lower, "legs": legs, "shoes": shoes,
        "shadow_allowed": shadow_allowed, "lighting_reference": lighting_reference,
    }
