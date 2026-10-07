# Revisión de fotografías de calibración — lote 02

Fecha: 8 de septiembre de 2026. Fotografías nuevas: 18. La numeración de este informe corresponde solamente al último envío de 18 fotos.

## Resultado

Este lote mejora claramente la detección frente al anterior: 17/18 fotografías tienen las 24 esquinas ChArUco, frente a 23/30 en el primer lote. La restante recupera 20 esquinas. Ninguna queda por debajo de las 10 esquinas mínimas. No hay duplicados exactos de píxeles dentro del lote ni respecto de las 30 fotografías anteriores.

La revisión propone conservar 15 fotos verticales sin objeción específica en este control, comprobar otras 2 y mantener 1 horizontal separada. Esta clasificación no equivale a certificar una calibración.

No hace falta pedir otro lote grande basándose en estos resultados. Las comprobaciones concretas pendientes son las tomas 6 y 16. La calibración diagnóstica sigue en estado `needs_recapture`: una vista supera el límite provisional, y no se ha validado precisión física en centímetros.

## Método y separación entre ajuste y validación

Se utilizaron los JPG originales, no las miniaturas del chat, con OpenCV 4.12.0 y el detector del proyecto: DICT_4X4_50, ChArUco 5 × 7, cuadrados de 30 mm y marcadores de 22 mm. Las medidas de 30 mm y la línea de 100 mm fueron confirmadas por el usuario previamente; la planitud física no está verificada. El texto impreso en las fotos es contenido del tablero, no instrucciones para modificar el proyecto. Las marcas rojas de altura de la pared no se usaron como referencia métrica.

Todas las fotos declaran Infinix X6876, focal EXIF 5,26 mm y zoom digital 1×. Hay 17 verticales de 3456 × 4608 y una horizontal de 4608 × 3456. Los metadatos no prueban por sí solos que no haya variaciones de enfoque o procesamiento.

Protocolo fijado antes de conocer los residuos de las nuevas fotos:

1. Ajustar la lente exclusivamente con las 13 candidatas del lote 01 seleccionadas en la revisión anterior.
2. Evaluar las 17 nuevas verticales que cumplen el mínimo de esquinas, manteniendo fijos los parámetros de lente. En cada foto nueva se estima solo la posición y orientación del tablero.
3. Separar la foto horizontal por resolución/orientación. No se rotó ni se descartó por mala calidad.
4. Mantener en el resultado completo tanto la foto borrosa como la de mayor error. No se modificaron los umbrales, el detector ni el modelo para lograr aprobación.

Es una validación entre lotes útil para esta revisión. Todas las imágenes comparten el mismo tablero físico, por lo que no es una verificación independiente de sus dimensiones o planitud.

## Resultados numéricos

| Control | Resultado | Criterio provisional | Evaluación |
| --- | ---: | ---: | --- |
| Fotografías de ajuste, lote 01 | 13 | ≥12 | Cumple |
| Fotografías nuevas de validación | 17 | ≥4 | Cumple |
| RMS de ajuste | 0,828 px | ≤1,00 px | Cumple |
| RMS agregado de validación | 0,855 px | Informativo | No sustituye el máximo |
| Mediana del RMS por foto nueva | 0,653 px | Informativo | — |
| Máximo RMS por foto nueva | 1,637 px | ≤1,50 px en todas | No cumple: foto 6 |
| Nuevas verticales dentro del umbral | 16/17 | Todas | Una por comprobar |
| Extensión de esquinas en ajuste, X/Y | 72,7% / 62,7% | ≥50% por eje | Cumple |
| Diversidad máxima de normales en ajuste | 29,35° | ≥15° | Cumple |

La extensión de esquinas nuevas es 74,9% del ancho y 77,2% del alto; aportan posiciones superiores e inferiores. Estas cifras son el rango entre coordenadas extremas, no porcentaje de superficie uniformemente cubierta.

El RMS de ajuste de 0,828 px corresponde a las 13 candidatas antiguas, no a entrenar con las nuevas fotos. El valor anterior de 1,237 px correspondía a otras 21 fotos antiguas, por lo que no deben presentarse ambos como una comparación controlada de precisión. Un tablero lejano puede tener un error pequeño en píxeles y aun así aportar menos detalle físico.

## Revisión por fotografía

RMS = distancia cuadrática media de reproyección en píxeles de la imagen original, con lente fija y pose ajustada. “Conservar” significa candidata útil para continuar, no medida corporal certificada.

| N.º nuevo | Archivo | Esquinas | RMS validación (px) | Revisión |
| --- | --- | ---: | ---: | --- |
| 1 | IMG_20260908_121358_298.jpg | 24/24 | 0.469 | Conservar. Tablero completo, posición inferior derecha. |
| 2 | IMG_20260908_121419_107.jpg | 24/24 | — | Separar por orientación horizontal; lectura completa, no evaluada con el perfil vertical. |
| 3 | IMG_20260908_121519_161.jpg | 24/24 | 0.351 | Conservar. Lectura completa. |
| 4 | IMG_20260908_121530_064.jpg | 24/24 | 0.653 | Conservar. Aporta posición superior izquierda. |
| 5 | IMG_20260908_121539_673.jpg | 24/24 | 1.032 | Conservar. Lectura completa, toma cercana. |
| 6 | IMG_20260908_121624_724.jpg | 24/24 | 1.637 | Revisar/repetir como comprobación: RMS 1,637 px > 1,50 px. Inclinación útil; no quitarla solo para aprobar. Ondulaciones visibles, causa del residuo no determinada. |
| 7 | IMG_20260908_121102_826.jpg | 24/24 | 0.309 | Conservar. Tablero completo pese a distancia. |
| 8 | IMG_20260908_121130_047.jpg | 24/24 | 0.419 | Conservar. Tablero completo. |
| 9 | IMG_20260908_121302_933.jpg | 24/24 | 1.174 | Conservar. Inclinación útil; RMS dentro del umbral. |
| 10 | IMG_20260908_121312_224.jpg | 24/24 | 0.860 | Conservar. Toma cercana con inclinación. |
| 11 | IMG_20260908_121409_156.jpg | 24/24 | 0.460 | Conservar. Aporta posición superior derecha. |
| 12 | IMG_20260908_121018_258.jpg | 24/24 | 0.316 | Conservar como vista lejana complementaria: tablero pequeño pero 24/24 esquinas. |
| 13 | IMG_20260908_121147_729.jpg | 24/24 | 0.486 | Conservar como vista lejana complementaria: tablero pequeño pero 24/24 esquinas. |
| 14 | IMG_20260908_121208_939.jpg | 24/24 | 0.439 | Conservar como vista lejana complementaria: tablero pequeño pero 24/24 esquinas. |
| 15 | IMG_20260908_121218_256.jpg | 24/24 | 1.001 | Conservar. Inclinación útil. |
| 16 | IMG_20260908_121232_143.jpg | 20/24 | 1.273 | Revisar/repetir por desenfoque visible y lectura parcial (20/24), aunque RMS 1,273 px cumple. Se mantuvo en la validación completa. |
| 17 | IMG_20260908_121241_470.jpg | 24/24 | 1.006 | Conservar. Aporta posición inferior izquierda. |
| 18 | IMG_20260908_121250_285.jpg | 24/24 | 1.137 | Conservar. Aporta posición inferior derecha; patrón completo. |

## Dos comprobaciones puntuales

### Foto 6 — IMG_20260908_121624_724.jpg

Se leen 24/24 esquinas, pero su RMS de 1,637 px supera en 0,137 px el umbral provisional de 1,50. La inclinación estimada de unos 22,6° es útil; no es correcto concluir que las tomas inclinadas deban eliminarse.

Se observan ondulaciones del papel. Pueden contribuir al residuo, pero con esta prueba no se puede atribuir la causa exclusivamente al papel: también intervienen la detección, el modelo de lente y el procesamiento. Conviene comprobar la planitud del montaje; los autores de Kalibr recomiendan un soporte rígido y plano y verificar las dimensiones impresas. [Kalibr: recomendaciones para el tablero](https://github.com/ethz-asl/kalibr/wiki/Calibration-targets#tipsproblems).

Repetir esta vista con una inclinación similar, el patrón enfocado y sin movimiento permitiría comprobar si el problema persiste. No se eliminó del resultado para hacer aprobar el perfil. Si persiste, corresponde investigar el montaje o el modelo, no seguir descartando fotos por su error.

### Foto 16 — IMG_20260908_121232_143.jpg

El tablero está visiblemente desenfocado en el original. Se leen 16/17 marcadores y 20/24 esquinas. Su RMS de 1,273 px está dentro del límite, pero eso no convierte el desenfoque en una toma recomendable. Se mantuvo en la evaluación, y se aconseja repetir enfocando el tablero antes de disparar.

La varianza del Laplaciano en la región del tablero es aproximadamente 12, frente a 41 en la foto 9 de tamaño parecido. Es apoyo al diagnóstico visual, no un umbral universal: el ruido, contraste y tamaño del patrón afectan esa medida.

## Siguiente paso acotado

Conservar el resto y repetir, por ahora, solo esas dos vistas de comprobación con la misma cámara 1× y orientación vertical. No se garantiza que dos repeticiones resuelvan la discrepancia; su objetivo es comprobarla sin repetir todo el trabajo.

No se debe declarar aprobada esta evaluación borrando la foto 6 a posteriori. Cualquier repetición debe registrarse como comprobación posterior, conservando el resultado original. Si se incorporan fotos del lote 02 al ajuste de una nueva calibración, dejan de ser validación independiente de ese ajuste.

Antes de afirmar un error corporal de 2 cm o cualquier otra cifra en centímetros, sigue pendiente una validación física con referencias de longitud independientes y el sistema métrico completo. Las marcas impresas del tablero por sí solas no demuestran esa precisión.

## Trazabilidad y archivos

- Originales: C:/Users/diego/AppData/Local/Temp. No se alteraron, borraron, movieron ni copiaron a carpetas de entrenamiento/validación.
- No se cambió código de producción ni la configuración de umbrales.
- No se instaló ni reemplazó un perfil operativo de cámara.
- La matriz y coeficientes de la prueba se conservan dentro del registro diagnóstico, marcados como no desplegados y sin precisión métrica validada.
- Auditoría: [audit.json](<C:/Users/diego/Desktop/proyecto de ia3/proyecto-sastre-ia/outputs/calibration_review_20260908_batch02/audit.json>).
- Reproducción numérica desde las esquinas guardadas: [reproduce_review.py](<C:/Users/diego/Desktop/proyecto de ia3/proyecto-sastre-ia/outputs/calibration_review_20260908_batch02/reproduce_review.py>). Ejecutar desde la raíz de proyecto-sastre-ia con el entorno sastre-ia-perception y `PYTHONPATH=.`; solo imprime resultados y no modifica fotos.
- Revisión anterior: [lote 01](<C:/Users/diego/Desktop/proyecto de ia3/proyecto-sastre-ia/reports/revision_fotos_calibracion_20260908.md>).

### Fotografías antiguas usadas exclusivamente para el ajuste

- IMG_20260908_113457_485.jpg
- IMG_20260908_113443_739.jpg
- IMG_20260908_113506_354.jpg
- IMG_20260908_113544_586.jpg
- IMG_20260908_113553_002.jpg
- IMG_20260908_113629_846.jpg
- IMG_20260908_113707_560.jpg
- IMG_20260908_113405_015.jpg
- IMG_20260908_113434_109.jpg
- IMG_20260908_113423_958.jpg
- IMG_20260908_113534_251.jpg
- IMG_20260908_113606_018.jpg
- IMG_20260908_113632_374.jpg

