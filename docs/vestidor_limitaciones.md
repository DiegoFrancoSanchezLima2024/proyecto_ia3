# Vestidor experimental: alcance y limitaciones

## Estado del pipeline

La vista de apariencia usa FASHN VTON 1.5 en modo sin máscara, seguida por
restauración semántica de fondo, rostro, cabello y manos mediante SCHP-ATR.
La salida es una visualización experimental y no una simulación física del
ajuste, la holgura o la caída del tejido.

## Calzado

FASHN solo expone las categorías `tops`, `bottoms` y `one-pieces`; no ofrece
una categoría de calzado. El overlay 2D de zapatos fue descartado y eliminado
del runtime porque producía apariencia de pegatina. La interfaz no debe afirmar
que FASHN generó zapatos.

La corrección prevista es inpainting localizado con Stable Diffusion v1.5
Inpainting. Cada pie se delimitará con MediaPipe Pose:

- izquierdo: tobillo 27, talón 29 y punta 31;
- derecho: tobillo 28, talón 30 y punta 32.

## Ablaciones descartadas

### FASHN con máscara nativa

La prueba controlada a 30 pasos tardó 88.55 s y consumió 2.719 GB de VRAM.
Persistieron restos de la ropa inferior, aparecieron zapatos defectuosos y el
fondo cambió más que con el modo libre. No se usa en la demo.

### Dos pasadas usando la misma referencia completa

La secuencia `bottoms -> tops` alteró la primera prenda y recuperó colores y
gráficos incorrectos. No se usa con las referencias actuales. Solo debe volver
a evaluarse con flat-lays independientes de saco, pantalón y falda.

## Inpainting local

Checkpoint verificado: `pretrained/stable-diffusion-inpainting`, descargado en
FP16 y en formato `safetensors`. Benchmark local reproducible:

| Parámetro | Resultado |
|---|---:|
| GPU | RTX 3050 Laptop 6 GB |
| Recorte | 512 x 512 px |
| Pasos | 20 |
| Offload | `model_cpu_offload` |
| Carga | 0.51 s |
| Inferencia | 8.49 s |
| VRAM pico | 1.911 GB |

El tiempo es inferior a 30 s, por lo que no se reduce a 384 x 384. Este
benchmark comprueba viabilidad y memoria; no constituye todavía validación de
calidad sobre todas las poses.

## Perfiles de demostración

- FASHN rápido: 30 pasos, alrededor de 90 s por generación.
- FASHN de alta calidad: 45 pasos; requiere validación de tiempo antes de usarlo
  en vivo y no es el perfil predeterminado.
- Inpainting: recortes locales de 512 x 512, semilla fija y QA independiente.

## Reglas de comunicación

- Etiqueta obligatoria: **Visualización experimental; no simulación física**.
- No afirmar que la imagen valida medidas, holgura o caída.
- No reentrenar FASHN ni incorporar LoRA antes de agotar el postprocesado.
- No reactivar overlays 2D ni el modo enmascarado nativo.
