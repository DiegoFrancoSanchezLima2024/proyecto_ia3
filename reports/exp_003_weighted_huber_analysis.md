# Análisis de `exp_003_weighted_huber`

Fecha: 2026-09-03  
Checkpoint: época 27, MAE de validación 1.754 cm  
Decisión: **rechazado; conservar `exp_002_huber`**

## Resultado

| Métrica por persona | `exp_002` | `exp_003` | Cambio |
|---|---:|---:|---:|
| Test-A global | 1.609 cm | 1.723 cm | +0.114 cm |
| Test-B global | 1.855 cm | 1.907 cm | +0.052 cm |
| Test-B pecho | 3.748 cm | 4.157 cm | +0.409 cm |
| Test-B cintura | 3.967 cm | 4.557 cm | +0.590 cm |
| Test-B cadera | 3.217 cm | 3.258 cm | +0.041 cm |
| Test-B cobertura conformal | 88.4 % | 87.5 % | -0.9 pp |

La ponderación mejoró validación pero no generalizó. Además introdujo sesgos
positivos importantes en Test-A: +1.29 cm en pecho, +3.97 cm en cintura y
+2.23 cm en cadera por captura. La hipótesis de que bastaba con aumentar el peso
de los contornos queda refutada para esta configuración.

Una corrección de sesgo estimada solo con el conjunto de calibración también se
descartó: cambió el MAE por persona de Test-A de 1.609 a 1.614 cm y el de Test-B
de 1.855 a 1.843 cm. La ganancia no es consistente entre dominios.

## Ablación geométrica

El estimador geométrico 2.5D aislado obtuvo 16.623 cm de MAE por persona en
Test-A. Las secciones horizontales fijas confunden pose, brazos y niveles
anatómicos. No debe integrarse al siguiente entrenamiento en su estado actual.

## Próxima hipótesis

`exp_004_profiles_huber` conserva el CNN y Huber de `exp_002`, pero añade perfiles
de ancho frontal y lateral en 32 bandas verticales. La hipótesis es que devuelve
información espacial que se pierde con el promedio global de EfficientNet, sin
usar niveles anatómicos rígidos ni elevar de forma material la VRAM.
