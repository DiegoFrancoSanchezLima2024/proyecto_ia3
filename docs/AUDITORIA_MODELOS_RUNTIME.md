# Auditoría reproducible de modelos

Generada: `2026-09-23T00:21:16.217959+00:00`

La ruta oficial usa únicamente MediaPipe Pose, SAM2.1 y exp_008. Los demás componentes tienen roles separados y no deben contarse como modelos de medición.

Última sesión trazable: `outputs\ui_sessions\7b775d63-2f28-40f0-b38f-6d3122993052`

| Componente | Estado real | Artefactos | Función |
|---|---|---:|---|
| MediaPipe Pose Landmarker Lite | `activo_oficial` | sí | Localizar cuerpo y producir cajas/landmarks para ambas fotos |
| SAM 2.1 Hiera Small | `activo_oficial` | sí | Extraer las siluetas frontal y lateral |
| exp_008_perfiles_arboles (ExtraTrees) | `activo_oficial` | sí | Estimar peso y 13 medidas desde dos siluetas y estatura |
| SCHP-ATR ONNX INT8 | `activo_bajo_demanda_no_mide` | sí | Parsing semántico para compositor 2D e identidad del vestidor |
| FASHN VTON 1.5 + DWPose + parser FASHN | `opcional_desactivado_por_defecto` | sí | Vestidor generativo realista experimental |
| Stable Diffusion v1.5 Inpainting | `verificado_no_integrado` | sí | Correcciones locales planificadas de calzado, falda y bajos |
| SHAPY + SMPL-X | `instalado_no_conectado` | sí | Reconstrucción corporal 3D experimental |
| exp_007_multitarea_sin_peso (CNN) | `ablacion_rechazada` | sí | Ablación de dos vistas |
| exp_001 a exp_006 | `historico_no_oficial` | sí | Historial de entrenamiento y comparación |
| FreeSewing Jaeger/Charlie/Penelope | `activo_patronaje_no_ia` | sí | Trazar saco, pantalón y falda desde medidas |

## Regla de interpretación

Un peso descargado no cuenta como modelo usado. Para afirmar uso en la demo debe existir una llamada desde la ruta correspondiente y, cuando aplica, procedencia guardada en la sesión.
