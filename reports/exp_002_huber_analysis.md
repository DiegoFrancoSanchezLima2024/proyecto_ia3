# Análisis de `exp_002_huber`

Fecha: 2026-09-03  
Checkpoint evaluado: época 20, MAE de validación 1.770 cm  
Decisión: **adoptar Huber como nuevo baseline principal**

## Comparación global

| Split y unidad | `exp_001` heterocedástico | `exp_002` Huber | Cambio |
|---|---:|---:|---:|
| Validación por captura | 1.778 cm | 1.770 cm | -0.008 cm |
| Test-A por captura | 1.858 cm | 1.830 cm | -0.028 cm |
| Test-A por persona | 1.706 cm | 1.609 cm | -0.097 cm |
| Test-B por captura | 2.090 cm | 2.049 cm | -0.041 cm |
| Test-B por persona | 1.961 cm | 1.855 cm | -0.106 cm |

La mejora por persona es 5.7 % en Test-A y 5.4 % en Test-B. La repetición del
resultado en ambos conjuntos respalda que no es solo una mejora de validación.

## Test-B consolidado por persona

| Medida | Heterocedástico | Huber | Cambio |
|---|---:|---:|---:|
| Pecho | 4.15 cm | 3.75 cm | -0.40 cm |
| Cintura | 4.58 cm | 3.97 cm | -0.61 cm |
| Cadera | 3.52 cm | 3.22 cm | -0.31 cm |
| Muslo | 2.37 cm | 2.17 cm | -0.20 cm |
| Bíceps | 1.66 cm | 1.50 cm | -0.16 cm |
| Largo de brazo | 1.07 cm | 1.16 cm | +0.09 cm |
| Hombro-entrepierna | 1.30 cm | 1.54 cm | +0.24 cm |

Huber cumple el objetivo del experimento: mejora los contornos prioritarios. El
costo en hombro-entrepierna debe vigilarse en el siguiente experimento.

## Incertidumbre

La cobertura conformal por persona sube de 87.4 % a 88.4 % en Test-B, todavía
por debajo del objetivo nominal de 90 %. Los semianchos para pecho, cintura y
cadera continúan alrededor de 6.98, 8.14 y 5.91 cm; el modelo todavía requiere
confirmación manual antes de generar un patrón de corte.

## Incidencia corregida

La época 27 obtuvo 1.764 cm, pero el checkpoint conservado fue la época 20 con
1.770 cm. La causa era que `early_stopping_min_delta=0.01` controlaba tanto la
paciencia como el guardado. El código ahora:

- guarda cualquier nuevo mínimo real;
- usa `min_delta` solamente para reiniciar la paciencia.

No se recomienda reentrenar `exp_002`: la diferencia no guardada es 0.006 cm y
no cambia la conclusión experimental.

## Conclusión

`exp_002_huber` debe usarse como modelo de referencia para la siguiente
ablación. Aún no se aprueba corte automático: pecho, cintura y cadera permanecen
por encima de la tolerancia deseada para sastrería.
