# Revisión del suelo — 8 de septiembre, 14:12

El usuario confirma que las hojas están acostadas completamente en el piso, con los soportes retirados. Las nuevas fotos son visualmente compatibles con esa descripción y ya no muestran los soportes triangulares anteriores. Esto resuelve la duda sobre la inclinación del montaje; no constituye una medición instrumental de planitud o distancias.

| Original | IDs detectados | Menor lado proyectado |
| --- | --- | --- |
| IMG_20260908_141204_625.jpg | 30, 31, 32, 33 | 324,26 px |
| IMG_20260908_141212_805.jpg | 30, 31, 32, 33 | 366,11 px |

Ambas pasan `inspect-floor`, sin IDs faltantes, duplicados ni desconocidos. La segunda es la referencia preferida de este montaje por el mayor tamaño mínimo de los marcadores, no por una certificación de nitidez o precisión métrica. Los JSON de visibilidad y hashes están en `outputs/calibration/suelo_visibilidad_20260908_141204_625.json` y `outputs/calibration/suelo_visibilidad_20260908_141212_805.json`.

Pendiente de comprobación física con cinta/regla: cuadrados negros de 16 × 16 cm, centros separados 84 cm lateralmente y 64 cm en profundidad, hojas alineadas y sin giro individual. No se dedujeron esas distancias de la foto ni de las baldosas. El campo `physical_layout_verified` permanece falso porque la geometría completa aún no está verificada.

No hacen falta más fotos iguales para demostrar visibilidad. La siguiente acción del usuario es comprobar esas dimensiones y comunicar cualquier diferencia. Sigue pendiente cerrar la validación de lente (`needs_recapture`, máximo 1,637 px frente al límite 1,50 px); no se ejecutó pose métrica ni se estimaron centímetros. Después de resolver la lente y verificar el montaje se podrá ensayar una varilla vertical antes de medir cuerpos.

No se modificaron originales, parámetros ni umbrales. No se aplicó aumento de datos.
