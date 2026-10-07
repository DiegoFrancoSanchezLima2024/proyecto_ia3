# Análisis de exp_006_weight_huber

Fecha: 2026-09-03

## Veredicto

**PROMOVIDO** como modelo campeón asistido por peso.

El mejor checkpoint es el de la época 38, seleccionado únicamente con
validación (MAE 1.390 cm). Test-A y Test-B se evaluaron después del entrenamiento
y no participaron en la selección.

## Resultados por sujeto

| Modelo | Test-A | Test-B no controlado |
|---|---:|---:|
| Ridge con estatura, sexo y peso | 1.494 cm | 1.651 cm |
| exp_002 sin peso | 1.609 cm | 1.855 cm |
| **exp_006 con peso** | **1.242 cm** | **1.421 cm** |

En Test-B, exp_006 mejora 23.4% respecto a exp_002 y 13.9% respecto al Ridge con
peso. Por tanto, la señal visual sí aporta información adicional después de
controlar estatura, sexo y peso.

## Medidas críticas por sujeto

| Medida | exp_002 Test-B | exp_006 Test-B | Mejora relativa |
|---|---:|---:|---:|
| Pecho | 3.748 cm | **2.841 cm** | 24.2% |
| Cintura | 3.967 cm | **3.087 cm** | 22.2% |
| Cadera | 3.217 cm | **2.027 cm** | 37.0% |

La cintura continúa siendo la medida más difícil. Estos MAE no autorizan corte
directo de tela: debe mantenerse confirmación manual para las medidas críticas.

## Incertidumbre conformal

Objetivo nominal: 90%.

| Split | Cobertura por captura | Cobertura por sujeto |
|---|---:|---:|
| Test-A | 91.3% | 93.2% |
| Test-B | 88.4% | 89.2% |

La cobertura de Test-B queda 0.8 puntos porcentuales por debajo del objetivo por
sujeto. Es una desviación pequeña, pero confirma cambio de dominio. La cobertura
más baja por sujeto en Test-B corresponde a `shoulder-to-crotch` (83.0%), seguida
por `leg-length` (86.0%) y `arm-length` (86.5%). Antes de usar los intervalos en
personas reales conviene recalibrarlos con un conjunto local separado.

## Error por población en Test-B

| Grupo IMC | Sujetos | MAE global | MAE medio pecho/cintura/cadera |
|---|---:|---:|---:|
| Bajo peso | 18 | 1.233 cm | 2.378 cm |
| Normal | 219 | 1.286 cm | 2.312 cm |
| Sobrepeso | 93 | 1.400 cm | 2.637 cm |
| Obesidad | 70 | **1.918 cm** | **3.804 cm** |

No aparece una diferencia material por sexo (1.423 frente a 1.418 cm), pero sí
una degradación clara en el grupo de obesidad. El siguiente entrenamiento no
debe optimizarse a ciegas: primero hay que revisar ejemplos extremos, cobertura
de pesos/IMC en train y calidad de sus siluetas.

El P90 del MAE global por sujeto es 2.046 cm; existe un caso extremo de 7.426 cm.
Esto justifica control de calidad de captura y detección de fuera de distribución.

## Uso de hardware

- Pico con encoder congelado: 1.06 GiB en la primera época.
- Pico después de descongelar el último bloque: 2.46 GiB.
- La RTX 3050 de 6 GB tiene margen suficiente; aumentar arquitectura no es la
  prioridad mientras el problema principal sea dominio y colas de error.

Los avisos de falta de Flash Attention y deprecación del scheduler no invalidan
el resultado. El primero afecta rendimiento, no predicciones; el segundo proviene
de la API de PyTorch usada internamente y debe limpiarse en mantenimiento futuro.

## Decisión de ingeniería

1. Congelar exp_006 como referencia para el modo `height + weight + gender`.
2. No reemplazarlo todavía por SMPL-X, RTMPose o un backbone más grande.
3. Construir la entrada desde fotos: segmentación y control de pose/encuadre.
4. Recoger un piloto local medido y cuantificar el cambio BodyM -> cámara real.
5. Recalibrar intervalos conformales exclusivamente con sujetos locales que no
   se usen para entrenamiento.
6. Investigar después peso estimado out-of-fold para ofrecer un modo sin balanza.

## Artefactos

- `experiments/exp_006_weight_huber/checkpoints/best_model.pt`
- `experiments/exp_006_weight_huber/results_test.json`
- `experiments/exp_006_weight_huber/results_test_wild.json`
- `experiments/exp_006_weight_huber/predictions_test_by_subject.csv`
- `experiments/exp_006_weight_huber/predictions_test_wild_by_subject.csv`
- `experiments/exp_006_weight_huber/conformal_q_hat.npy`
