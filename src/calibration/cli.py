"""Ejecutar desde la raíz: python -m src.calibration.cli --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .common import DEFAULT_CONFIG, ROOT, file_hash, read_image, leer_json, write_json
from .intrinsics import calibrate
from .kit import generar_kit
from .metric import detectar_piso, inspeccionar_marcadores_piso, altura_segmento_vertical


def main():
    parser = argparse.ArgumentParser(description="Sastre-IA: calibración experimental sin datos corporales")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    kit = commands.add_parser("kit", help="Generar cinco páginas A4: ChArUco y cuatro ArUco")
    kit.add_argument("--output", type=Path, default=ROOT / "outputs/calibration_kit")
    kit.add_argument("--force", action="store_true")
    fit = commands.add_parser("calibrate", help="Calibrar lente con fotos del tablero y validación separada")
    fit.add_argument("--train", type=Path, default=ROOT / "dataset/local/calibration/camera01/train")
    fit.add_argument("--validation", type=Path, default=ROOT / "dataset/local/calibration/camera01/validation")
    fit.add_argument("--camera-id", default="camera01")
    fit.add_argument("--output", type=Path, default=ROOT / "outputs/calibration/camera01.json")
    fit.add_argument("--force", action="store_true")
    inspect = commands.add_parser("inspect-floor", help="Comprobar visibilidad de ArUco sin aprobar lente ni medir distancias")
    inspect.add_argument("--image", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--force", action="store_true")
    for name, help_text in (("floor", "Detectar estación métrica en una foto original"),
                            ("vertical-check", "Diagnóstico manual de una varilla vertical; NO estatura humana automática")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--image", type=Path, required=True)
        command.add_argument("--camera", type=Path, default=ROOT / "outputs/calibration/camera01.json")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--force", action="store_true")
        if name == "vertical-check":
            command.add_argument("--pose", type=Path, required=True)
            command.add_argument("--points", type=Path, required=True,
                                 help='JSON {"base_pixel": [x,y], "top_pixel": [x,y]} en foto original orientada')
    args = parser.parse_args()
    try:
        config = leer_json(args.config)
        if args.command != "kit" and args.output.exists() and not args.force:
            raise ValueError("El archivo de salida existe. Use otra ruta o --force para reemplazar sólo ese resultado.")
        if args.command == "kit":
            result = generar_kit(config, args.output, args.force)
            for folder in ("train", "validation"):
                (ROOT / "dataset/local/calibration/camera01" / folder).mkdir(parents=True, exist_ok=True)
        elif args.command == "calibrate":
            result = calibrate(args.train, args.validation, config, args.camera_id)
            result["station_config_sha256"] = file_hash(args.config)
            write_json(args.output, result)
            result = {k: result[k] for k in ("status", "train_rms_px", "warnings", "metric_accuracy_validated")}
            result["output"] = str(args.output.resolve())
        elif args.command == "inspect-floor":
            image = read_image(args.image)
            result = inspeccionar_marcadores_piso(image, config)
            result.update({"image_sha256": file_hash(args.image),
                           "station_config_sha256": file_hash(args.config)})
            write_json(args.output, result)
            result["output"] = str(args.output.resolve())
        else:
            camera = leer_json(args.camera)
            if camera.get("station_config_sha256") != file_hash(args.config):
                raise ValueError("La configuración no coincide con la usada para calibrar. No mezclar kits/perfiles.")
            image = read_image(args.image)
            if args.command == "floor":
                result = detectar_piso(image, camera, config)
                result.update({"image_sha256": file_hash(args.image), "camera_sha256": file_hash(args.camera),
                               "image_size": [image.shape[1], image.shape[0]], "station_config_sha256": file_hash(args.config)})
            else:
                pose = leer_json(args.pose)
                if (pose.get("image_sha256") != file_hash(args.image)
                        or pose.get("camera_sha256") != file_hash(args.camera)
                        or pose.get("station_config_sha256") != file_hash(args.config)):
                    raise ValueError("La pose debe corresponder a esta misma foto, cámara y estación")
                points = leer_json(args.points)
                result = altura_segmento_vertical(points["base_pixel"], points["top_pixel"], camera, pose)
                result.update({"image_sha256": file_hash(args.image), "camera_sha256": file_hash(args.camera),
                               "pose_sha256": file_hash(args.pose), "points_sha256": file_hash(args.points)})
            write_json(args.output, result)
            result["output"] = str(args.output.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if result.get("status") in {"needs_recapture", "needs_capture_adjustment"}:
            return 2
        return 0
    except (ValueError, KeyError, OSError, cv2.error) as error:
        parser.exit(2, f"No se puede continuar: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
