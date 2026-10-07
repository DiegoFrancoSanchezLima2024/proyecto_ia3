"""Ejecuta SHAPY sobre una captura multivista ya segmentada por Sastre-IA.

El script consume el ``capture_manifest.json`` producido por la etapa de
percepcion. Usa las cajas MediaPipe/SAM2, estima una forma por vista y fusiona
las mallas canonicas con una media robusta. No inventa medidas de sastreria:
esas se calculan posteriormente sobre la malla escalada y validada.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.body3d.mesh_measurements import escalar_malla_a_estatura
from src.body3d.shapy_runner import cargar_modelo_shapy


def _image_tensor(
    photo_path: Path,
    box_xyxy: list[float],
    crop_size: int,
    padding: float = 1.15,
) -> torch.Tensor:
    """Recorta una caja cuadrada conservando proporciones y normaliza RGB."""

    bgr = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo abrir la foto: {photo_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1) * float(padding)
    left = center_x - side / 2.0
    top = center_y - side / 2.0

    # OpenCV aplica relleno negro fuera de la imagen, igual que el recorte
    # afín usado por el código original de SHAPY/ExPose.
    scale = crop_size / side
    transform = np.asarray(
        [[scale, 0.0, -left * scale], [0.0, scale, -top * scale]],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        rgb,
        transform,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def _robust_fuse(arrays: list[np.ndarray]) -> np.ndarray:
    """Mediana por coordenada: tolera una vista lateral menos estable."""

    if not arrays:
        raise ValueError("No hay predicciones para fusionar")
    return np.median(np.stack(arrays, axis=0), axis=0)


def run_capture(
    manifest_path: Path,
    height_cm: float,
    output_dir: Path,
    weight_kg: float | None = None,
    sex: str = "neutral",
    device: str = "cuda",
) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest.get("views", {})
    if not views:
        raise ValueError("El manifiesto no contiene vistas")
    if sex not in {"female", "male", "neutral"}:
        raise ValueError("sex debe ser female, male o neutral")
    if weight_kg is not None and not 30.0 <= weight_kg <= 250.0:
        raise ValueError("weight_kg fuera del rango [30, 250]")

    network, config, load_report = cargar_modelo_shapy(PROJECT_ROOT, device=device)
    target_device = torch.device(load_report["device"])
    crop_size = int(config.datasets.pose.transforms.crop_size)

    vertices_by_view: list[np.ndarray] = []
    betas_by_view: list[np.ndarray] = []
    view_reports = {}
    faces = None

    with torch.inference_mode():
        for view_name, view in views.items():
            photo_path = Path(view["photo"])
            image = _image_tensor(
                photo_path,
                view["box_xyxy"],
                crop_size=crop_size,
            ).to(target_device)
            output = network(image, targets=None, compute_losses=False)
            stage = output[output["stage_keys"][-1]]

            vertices = stage["v_shaped"][0].detach().cpu().numpy()
            betas = stage["betas"][0].detach().cpu().numpy()
            vertices_by_view.append(vertices)
            betas_by_view.append(betas)
            if faces is None:
                stage_faces = stage["faces"]
                faces = (
                    stage_faces.detach().cpu().numpy()
                    if torch.is_tensor(stage_faces)
                    else np.asarray(stage_faces)
                )
            quality = view.get("pose", {}).get("quality", {})
            view_reports[view_name] = {
                "photo": str(photo_path),
                "mean_visibility": quality.get("mean_visibility"),
                "sam_score": view.get("sam_score"),
                "betas": betas.astype(float).round(6).tolist(),
            }

    fused_vertices = _robust_fuse(vertices_by_view)
    fused_betas = _robust_fuse(betas_by_view)
    scaled_vertices, metric_scale = escalar_malla_a_estatura(
        fused_vertices, height_cm=height_cm
    )
    per_vertex_spread_cm = np.linalg.norm(
        np.stack(vertices_by_view) - fused_vertices[None, ...], axis=2
    ) * metric_scale * 100.0

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "body_shape.npz"
    mesh_path = output_dir / "body_shape.ply"
    report_path = output_dir / "shapy_result.json"
    np.savez_compressed(
        npz_path,
        vertices_m=scaled_vertices.astype(np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        betas=fused_betas.astype(np.float32),
        height_cm=np.float32(height_cm),
    )

    import trimesh

    mesh = trimesh.Trimesh(
        vertices=scaled_vertices,
        faces=np.asarray(faces),
        process=False,
    )
    mesh.export(mesh_path)

    preview_path = output_dir / "body_shape_preview.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7, 7), facecolor="white")
    points = scaled_vertices[::2] * 100.0
    for axis, horizontal, title in (
        (axes[0], 0, "Frente"),
        (axes[1], 2, "Perfil"),
    ):
        axis.scatter(points[:, horizontal], points[:, 1], s=0.28, c="#243449")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Sastre-IA · forma 3D experimental · {height_cm:.0f} cm")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "schema_version": "1.0",
        "status": "experimental_shape_estimate",
        "manifest": str(manifest_path),
        "height_cm": float(height_cm),
        "weight_kg": None if weight_kg is None else float(weight_kg),
        "sex": sex,
        "metadata_usage": {
            "height": "metric_scale",
            "weight": "used_by_exp_006_not_directly_by_shapy",
            "sex": "used_by_exp_006_and_garment_selection_not_directly_by_neutral_shapy",
        },
        "view_count": len(vertices_by_view),
        "fusion": "coordinate_median",
        "crop_size": crop_size,
        "metric_scale": float(metric_scale),
        "cross_view_median_vertex_spread_cm": float(np.median(per_vertex_spread_cm)),
        "cross_view_p95_vertex_spread_cm": float(np.percentile(per_vertex_spread_cm, 95)),
        "betas": fused_betas.astype(float).round(6).tolist(),
        "views": view_reports,
        "model": load_report,
        "artifacts": {
            "mesh": str(mesh_path),
            "npz": str(npz_path),
            "preview": str(preview_path),
        },
        "limitations": [
            "La ropa holgada puede sesgar la forma corporal.",
            "La malla esta escalada por estatura; aun no es una garantia de +/-2 cm.",
            "Las medidas finales requieren secciones anatomicas y calibracion con el dataset.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--height-cm", type=float, required=True)
    parser.add_argument("--weight-kg", type=float)
    parser.add_argument(
        "--sex", choices=("female", "male", "neutral"), default="neutral"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_capture(
        args.manifest,
        height_cm=args.height_cm,
        output_dir=args.output_dir,
        weight_kg=args.weight_kg,
        sex=args.sex,
        device=args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
