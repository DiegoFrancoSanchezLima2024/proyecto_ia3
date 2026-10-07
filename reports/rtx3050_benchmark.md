# Benchmark local - RTX 3050 Laptop 6 GB

Fecha: 2026-09-02  
PyTorch: 2.3.1 + CUDA 12.1  
Entrada: dos siluetas de 320 x 320  
Modelo: EfficientNet-B0, ultimo bloque descongelado, Transformer y AdamW FP16

| Batch | Pico asignado | Pico reservado | Segundos/paso | Muestras/s |
|---:|---:|---:|---:|---:|
| 8 | 1.27 GiB | 1.34 GiB | 0.894 | 8.95 |
| 12 | 1.87 GiB | 2.01 GiB | 1.425 | 8.42 |
| 16 | 2.45 GiB | 2.63 GiB | 1.960 | 8.16 |
| 20 | 3.07 GiB | 3.17 GiB | 2.594 | 7.71 |
| 24 | 3.62 GiB | 3.99 GiB | 3.108 | 7.72 |
| 32 | 4.81 GiB | 5.20 GiB | 4.260 | 7.51 |

Decision inicial: batch fisico 16 y acumulacion 2. Batch 24 tambien cabe, pero no
fue mas rapido por muestra. Batch 32 deja solo 0.8 GiB y no es recomendable en
Windows durante una demo. Antes del entrenamiento definitivo se compararan 8, 16
y 24 durante tres epocas reales porque la carga de imagenes puede cambiar el
throughput observado con datos sinteticos.

Se probo tambien concatenar frontal/lateral para ejecutar una sola llamada del
encoder. Fue descartado: batch 16 subio aproximadamente de 1.96 a 2.14 s/paso y
reservo mas memoria. El modelo conserva el procesamiento secuencial de vistas.

## SAM 2.1 Small

Fecha: 2026-09-03  
Entorno: `sastre-ia-perception`, PyTorch 2.5.1 + CUDA 12.1  
Checkpoint: `sam2.1_hiera_small.pt` (184.416.285 bytes)  
Prueba: cuatro inferencias secuenciales de 720 x 960, FP16, una imagen por vez

| Modelo | Pico asignado por vista | Resultado |
|---|---:|---|
| SAM 2.1 Hiera Small | **0,583 GiB** | Ejecuta correctamente en Windows y RTX 3050 6 GB |

Esta es una prueba funcional, no una métrica de segmentación: se usaron siluetas
de BodyM como entrada para verificar instalación, CUDA y generación del
manifiesto. La siguiente evaluación debe utilizar RGB real y una caja ajustada a
la persona; una caja de imagen completa queda marcada para revisión.
