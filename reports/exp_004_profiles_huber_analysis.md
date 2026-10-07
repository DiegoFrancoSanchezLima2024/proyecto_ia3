# Análisis de `exp_004_profiles_huber`

Fecha: 2026-09-03  
Checkpoint: época 16, MAE de validación 1.838 cm  
Decisión: **rechazado; conservar `exp_002_huber`**

## Comparación por persona

| Métrica | `exp_002` | `exp_004` | Cambio |
|---|---:|---:|---:|
| Test-A global | 1.609 cm | 1.702 cm | +0.093 cm |
| Test-B global | 1.855 cm | 1.984 cm | +0.129 cm |
| Test-B pecho | 3.748 cm | 4.310 cm | +0.562 cm |
| Test-B cintura | 3.967 cm | 4.515 cm | +0.548 cm |
| Test-B cadera | 3.217 cm | 3.343 cm | +0.126 cm |

Los perfiles de ocupación por fila no aportaron generalización. Es probable que
mezclen brazos y torso y que su posición vertical cambie con pose y encuadre.
Esta rama no debe activarse en producción ni combinarse por ensamble: los
ensambles exploratorios con `exp_002` también empeoraron Test-B.

## Siguiente hipótesis

El conjunto de entrenamiento tiene 1,411 sujetos y 4,337 fotos. La mediana es
2 fotos por sujeto, pero el máximo es 27. El entrenamiento actual por captura
sobrerrepresenta hasta 13.5 veces a algunas personas. `exp_005` mantiene la
arquitectura y Huber de `exp_002`, pero asigna a cada foto peso de muestreo
inverso al número de capturas de su sujeto. Cada sujeto aporta así la misma masa
probabilística mientras se conservan 4,337 muestras por época.
