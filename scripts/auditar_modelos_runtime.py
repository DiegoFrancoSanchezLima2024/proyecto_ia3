"""Genera una auditoría reproducible de los modelos de SATRE-IA.

La auditoría distingue cuatro hechos que antes se confundían con facilidad:

1. el artefacto está descargado;
2. existe código capaz de cargarlo;
3. forma parte de la ruta oficial de medición;
4. dejó evidencia de ejecución en una sesión local.

No carga modelos pesados ni modifica una sesión. Solo inspecciona artefactos,
código, métricas y la trazabilidad JSON ya guardada por el prototipo.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]


def existe(ruta: str) -> bool:
    return (RAIZ / ruta).exists()


def contiene(ruta: str, texto: str) -> bool:
    archivo = RAIZ / ruta
    return archivo.is_file() and texto in archivo.read_text(encoding="utf-8")


def leer_json(ruta: Path) -> dict:
    try:
        return json.loads(ruta.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def ultima_sesion_medida() -> tuple[Path | None, dict, dict]:
    raiz_sesiones = RAIZ / "outputs" / "ui_sessions"
    candidatas: list[tuple[float, Path, dict, dict]] = []
    if raiz_sesiones.is_dir():
        for sesion in raiz_sesiones.iterdir():
            prediccion = leer_json(sesion / "prediction.json")
            captura = leer_json(sesion / "capture_manifest.json")
            if prediccion and captura:
                candidatas.append(
                    ((sesion / "prediction.json").stat().st_mtime, sesion, prediccion, captura)
                )
    if not candidatas:
        return None, {}, {}
    _, sesion, prediccion, captura = max(candidatas, key=lambda item: item[0])
    return sesion, prediccion, captura


def entrada_modelo(
    nombre: str,
    rol: str,
    estado: str,
    artefactos: list[str],
    evidencia: list[str],
    observacion: str,
) -> dict:
    return {
        "nombre": nombre,
        "rol": rol,
        "estado": estado,
        "artefactos_presentes": all(existe(ruta) for ruta in artefactos),
        "artefactos": artefactos,
        "evidencia": evidencia,
        "observacion": observacion,
    }


def construir_auditoria() -> dict:
    registro = leer_json(RAIZ / "configs" / "modelos_runtime.json")
    medicion = registro.get("medicion_oficial", {})
    regresion = medicion.get("regresion", {})
    sesion, prediccion, captura = ultima_sesion_medida()
    vistas = captura.get("views", {})
    modelo_sesion = prediccion.get("model")
    segmentador_sesion = captura.get("segmenter")
    fuente_cajas = sorted(
        {
            vista.get("box_source")
            for vista in vistas.values()
            if isinstance(vista, dict) and vista.get("box_source")
        }
    )
    resultados_008 = leer_json(RAIZ / "experiments" / "exp_008_perfiles_arboles" / "results.json")
    prueba_008 = resultados_008.get("test", {})
    prueba_libre_008 = resultados_008.get("test_wild", {})

    oficial = (
        regresion.get("nombre") == "exp_008_perfiles_arboles"
        and regresion.get("backend") == "profiles"
        and contiene("scripts/run_assisted_demo.ps1", "configs/modelos_runtime.json")
        and contiene("scripts/run_assisted_demo.ps1", "src/perception/sam2_capture.py")
    )
    body3d_omitido = contiene("scripts/run_assisted_demo.ps1", "body3d = $null")
    fashn_condicional = contiene(
        "pattern-engine/server.mjs", "SATRE_ENABLE_EXPERIMENTAL_TRYON === '1'"
    )

    modelos = [
        entrada_modelo(
            "MediaPipe Pose Landmarker Lite",
            "Localizar cuerpo y producir cajas/landmarks para ambas fotos",
            "activo_oficial",
            ["pretrained/mediapipe/pose_landmarker_lite.task"],
            [
                "scripts/run_assisted_demo.ps1 -> src/perception/sam2_capture.py --pose-model",
                f"última sesión: box_source={','.join(fuente_cajas) or 'sin evidencia'}",
            ],
            "No calcula centímetros; guía la segmentación y la vista anatómica.",
        ),
        entrada_modelo(
            "SAM 2.1 Hiera Small",
            "Extraer las siluetas frontal y lateral",
            "activo_oficial",
            ["pretrained/sam2/sam2.1_hiera_small.pt"],
            [
                "scripts/run_assisted_demo.ps1 -> src/perception/sam2_capture.py",
                f"última sesión: segmenter={segmentador_sesion or 'sin evidencia'}",
            ],
            "Produce máscaras; no estima medidas por sí solo.",
        ),
        entrada_modelo(
            "exp_008_perfiles_arboles (ExtraTrees)",
            "Estimar peso y 13 medidas desde dos siluetas y estatura",
            "activo_oficial",
            [
                "experiments/exp_008_perfiles_arboles/model.joblib",
                "experiments/exp_008_perfiles_arboles/conformal_q_hat.npy",
                "experiments/exp_008_perfiles_arboles/results.json",
            ],
            [
                "run_assisted_demo.ps1 -> predict_two_views.py --backend profiles",
                f"última sesión: model={modelo_sesion or 'sin evidencia'}",
                f"test MAE 13 medidas={prueba_008.get('mae_13_cm')} cm; peso={prueba_008.get('mae_weight_kg')} kg",
                f"test libre MAE 13 medidas={prueba_libre_008.get('mae_13_cm')} cm; peso={prueba_libre_008.get('mae_weight_kg')} kg",
            ],
            "Es el único regresor autorizado por la ruta oficial actual.",
        ),
        entrada_modelo(
            "SCHP-ATR ONNX INT8",
            "Parsing semántico para compositor 2D e identidad del vestidor",
            "activo_bajo_demanda_no_mide",
            ["pretrained/schp-atr/onnx/schp-atr-18-int8-static.onnx"],
            ["server.mjs /api/tryon/layered -> build_layered_tryon_preview.py -> infer_atr"],
            "Se ejecuta en CPU en las rutas visuales; nunca modifica las medidas.",
        ),
        entrada_modelo(
            "FASHN VTON 1.5 + DWPose + parser FASHN",
            "Vestidor generativo realista experimental",
            "opcional_desactivado_por_defecto" if fashn_condicional else "estado_incierto",
            [
                "pretrained/fashn-vton-1.5/model.safetensors",
                "pretrained/fashn-vton-1.5/dwpose/yolox_l.onnx",
                "pretrained/fashn-vton-1.5/dwpose/dw-ll_ucoco_384.onnx",
            ],
            ["server.mjs /api/tryon -> run_fashn_vton_local.py"],
            "Solo se habilita con SATRE_ENABLE_EXPERIMENTAL_TRYON=1; no valida ajuste ni moldes.",
        ),
        entrada_modelo(
            "Stable Diffusion v1.5 Inpainting",
            "Correcciones locales planificadas de calzado, falda y bajos",
            "verificado_no_integrado",
            [
                "pretrained/stable-diffusion-inpainting/unet/diffusion_pytorch_model.fp16.safetensors",
                "pretrained/stable-diffusion-inpainting/vae/diffusion_pytorch_model.fp16.safetensors",
                "pretrained/stable-diffusion-inpainting/text_encoder/model.fp16.safetensors",
            ],
            [
                "scripts/benchmark_inpainting_local.py",
                "benchmark 512x512: 8.49 s, pico 1.911 GB en RTX 3050 6 GB",
            ],
            "Checkpoint listo, pero todavía no forma parte de /api/tryon; no debe contarse como activo en la demo.",
        ),
        entrada_modelo(
            "SHAPY + SMPL-X",
            "Reconstrucción corporal 3D experimental",
            "instalado_no_conectado" if body3d_omitido else "estado_incierto",
            [
                "pretrained/body3d/trained_models/shapy/SHAPY_A/checkpoints/best_checkpoint",
                "pretrained/body3d/models/smplx/models/smplx/SMPLX_NEUTRAL.npz",
            ],
            ["run_assisted_demo.ps1 registra body3d=null y usa -SkipBody3D"],
            "Ocupa varios GB, pero no aporta nada a la salida oficial actual.",
        ),
        entrada_modelo(
            "exp_007_multitarea_sin_peso (CNN)",
            "Ablación de dos vistas",
            "ablacion_rechazada",
            ["experiments/exp_007_multitarea_sin_peso/checkpoints/best_model.pt"],
            ["docs/SELECCION_MODELO_DOS_VISTAS.md"],
            "Fue superado claramente por exp_008 y no debe presentarse como modelo activo.",
        ),
        entrada_modelo(
            "exp_001 a exp_006",
            "Historial de entrenamiento y comparación",
            "historico_no_oficial",
            ["experiments/exp_006_weight_huber/checkpoints/best_model.pt"],
            ["predict_four_views.py conserva exp_006 como ruta antigua de cuatro vistas"],
            "Son evidencia de investigación/ablación; la demo de dos fotos no los ejecuta.",
        ),
        entrada_modelo(
            "FreeSewing Jaeger/Charlie/Penelope",
            "Trazar saco, pantalón y falda desde medidas",
            "activo_patronaje_no_ia",
            ["pattern-engine/node_modules/@freesewing/jaeger/package.json"],
            ["pattern-engine/generate.mjs y garments.mjs"],
            "Es un motor paramétrico, no un modelo de IA ni un predictor de medidas.",
        ),
    ]

    return {
        "schema_version": "1.0",
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "ruta_oficial_consistente": oficial,
        "politica": (
            "La ruta oficial usa únicamente MediaPipe Pose, SAM2.1 y exp_008. "
            "Los demás componentes tienen roles separados y no deben contarse como modelos de medición."
        ),
        "ultima_sesion_con_trazabilidad": str(sesion.relative_to(RAIZ)) if sesion else None,
        "modelos": modelos,
    }


def a_markdown(auditoria: dict) -> str:
    filas = []
    for modelo in auditoria["modelos"]:
        presente = "sí" if modelo["artefactos_presentes"] else "NO"
        filas.append(
            f"| {modelo['nombre']} | `{modelo['estado']}` | {presente} | {modelo['rol']} |"
        )
    return "\n".join(
        [
            "# Auditoría reproducible de modelos",
            "",
            f"Generada: `{auditoria['generado_utc']}`",
            "",
            auditoria["politica"],
            "",
            f"Última sesión trazable: `{auditoria['ultima_sesion_con_trazabilidad']}`",
            "",
            "| Componente | Estado real | Artefactos | Función |",
            "|---|---|---:|---|",
            *filas,
            "",
            "## Regla de interpretación",
            "",
            "Un peso descargado no cuenta como modelo usado. Para afirmar uso en la demo debe existir "
            "una llamada desde la ruta correspondiente y, cuando aplica, procedencia guardada en la sesión.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="outputs/auditoria_modelos_runtime.json")
    parser.add_argument("--markdown", default="docs/AUDITORIA_MODELOS_RUNTIME.md")
    args = parser.parse_args()

    auditoria = construir_auditoria()
    salida_json = RAIZ / args.json
    salida_md = RAIZ / args.markdown
    salida_json.parent.mkdir(parents=True, exist_ok=True)
    salida_md.parent.mkdir(parents=True, exist_ok=True)
    salida_json.write_text(
        json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    salida_md.write_text(a_markdown(auditoria), encoding="utf-8")
    print(f"Ruta oficial consistente: {auditoria['ruta_oficial_consistente']}")
    print(f"JSON: {salida_json}")
    print(f"Markdown: {salida_md}")


if __name__ == "__main__":
    main()
