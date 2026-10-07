# Análisis de `exp_005_subject_balanced_huber`

Fecha: 2026-09-03  
Checkpoint: época 25, MAE de validación 1.815 cm  
Decisión: **rechazado; `exp_002_huber` continúa como modelo principal**

## Comparación por persona

| Métrica | `exp_002` | `exp_005` | Cambio |
|---|---:|---:|---:|
| Test-A global | 1.609 cm | 1.727 cm | +0.118 cm |
| Test-B global | 1.855 cm | 1.942 cm | +0.087 cm |
| Test-B pecho | 3.748 cm | 4.026 cm | +0.278 cm |
| Test-B cintura | 3.967 cm | 4.509 cm | +0.542 cm |
| Test-B cadera | 3.217 cm | 3.370 cm | +0.153 cm |

El muestreo balanceado tampoco mejora equidad entre los dos grupos codificados
por `gender`:

| Grupo | `exp_002` global | `exp_005` global |
|---|---:|---:|
| 0 (243 sujetos) | 1.835 cm | 1.918 cm |
| 1 (157 sujetos) | 1.884 cm | 1.980 cm |

En cintura del grupo 1, el error aumenta de 4.460 a 5.210 cm. La hipótesis de
que sujetos con muchas fotos dominaban el error queda refutada en este ensayo.
Además, el muestreo aleatorio incrementó el tiempo estable por época de
aproximadamente 60 a 77 segundos.

## Decisión de ingeniería

No continuar con variaciones de pérdida, perfiles, geometría rígida, ensambles o
muestreo sobre el mismo conjunto sin una nueva fuente de información. Se congela
`exp_002_huber` como checkpoint de la demo. El siguiente trabajo debe ser:

1. inferencia reproducible desde dos máscaras o fotografías;
2. control de calidad de pose y encuadre;
3. validación local por sujeto contra cinta métrica;
4. usar esos errores para decidir entre calibración local, segmentación mejorada
   o reconstrucción corporal 3D.
