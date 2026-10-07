# Revisión de 30 fotos ChArUco — Infinix GT 30

Fecha: 2026-09-08. Estado: **material parcialmente aprovechable; calibración no aprobada todavía**.

## Resultado principal

- 30 originales examinados, sin modificarlos, rotarlos ni moverlos.
- Todos registran Infinix X6876, focal EXIF 5.26 mm y zoom digital 1×. El EXIF no demuestra que el enfoque ni el procesamiento interno sean constantes.
- 23 fotos: 24/24 esquinas. Otras 4: 10–20 esquinas. Tres no alcanzan el mínimo de 10.
- 24 fotos verticales de 3456×4608; 6 horizontales de 4608×3456. No hay duplicados exactos de píxeles dentro de este lote.
- Las 21 verticales con suficientes esquinas se usaron sólo para un diagnóstico matemático, no para generar un perfil de cámara utilizable.
- Escala impresa confirmada por el usuario: cuadrados 3×3 cm y línea 10 cm. La planitud del soporte no está verificada físicamente.

## Qué dice la geometría

El ajuste exploratorio a las 21 verticales produce **RMS 1.237 px**, por encima del umbral provisional del proyecto de **1.0 px**. Estos son píxeles de reproyección, no centímetros de error de altura.

También se dejó fuera una foto cada vez, se estimó la lente con las restantes y se comprobó la foto omitida ajustando sólo su pose. El máximo fue **2.434 px**. Este análisis leave-one-view-out es exploratorio: no reemplaza un conjunto de validación nuevo y separado. No se asignaron automáticamente las últimas fotos del mensaje a validación.

Hay variedad geométrica real: separación máxima de normales estimadas **51.9°**. No es correcto decir que faltan todas las inclinaciones. Sin embargo, varias tomas son casi frontales y algunas inclinadas tienen contraluz, desenfoque o residuo elevado.

Las esquinas observadas cubren aproximadamente **76.2 % del ancho y 64.4 % del alto** por extensión mínima–máxima conjunta, no como porcentaje de área cubierta uniformemente. La zona superior del encuadre está menos representada (primera esquina aproximadamente al 20 % de su altura).

Al retirar distintas fotos, fx varía de **3361.0 a 3505.7 px**. Es una sensibilidad al conjunto, no un intervalo de confianza ni una traducción a error corporal.

No se relajaron umbrales, no se recalibró con una selección para afirmar que pasó y no se modificó el código del calibrador.

## Problemas observados y solución

1. **Movimiento/desenfoque:** especialmente fotos 3, 12, 18 y 19. Apoyar teléfono y tablero; esperar a que ambos estén inmóviles y enfocados; usar temporizador de 2 s si ayuda.
2. **Contraluz:** fotos 22, 23 y 30 frente a la ventana. Colocar la ventana al costado o detrás del fotógrafo para iluminar la cara impresa; no detrás del tablero. No intentar arreglar estas fotos con IA, enfoque artificial o cambios de brillo para la calibración.
3. **Poca luz en otras tomas:** el EXIF llega a ISO 5684 y exposición de 0.060 s. Son condiciones con riesgo de ruido/movimiento, no un rechazo automático por ISO. La varianza del Laplaciano también aumenta con ruido; no se usa aquí como umbral universal de nitidez.
4. **Orientaciones mezcladas:** seis imágenes son horizontales. No son intrínsecamente malas, pero el programa actual exige la misma resolución/orientación. Preservarlas aparte. Una integración futura podría tratar rotaciones correctamente con sus coordenadas/intrínsecos; no estirar o girar arbitrariamente archivos para saltar la validación.
5. **Papel ondulado:** se ven arrugas y ondulaciones. Pegar toda la superficie sobre soporte rígido y plano, no solamente las esquinas; comprobar a ras de la superficie que no haya bolsas. Las imágenes no permiten cuantificar por sí solas cuánto de cada residuo viene del papel, lente, enfoque o ruido.

Cambiar de habitación, fondo o mueble no es necesario. Lo importante es cambiar la posición y orientación del **mismo tablero plano respecto a la cámara**, manteniendo nitidez y modo de captura.

## Revisión por fotografía

Los números corresponden al orden de las 30 imágenes enviadas. **Candidata no significa foto certificada ni cámara calibrada**. No borrar ninguna original.

| N.º | Archivo | Esquinas /24 | Orientación | Acción | RMS LOO (px) |
|---:|---|---:|---|---|---:|
| 1 | IMG_20260907_173515_754 (1).jpg | 24 | Vertical | Revisar / repetir | 2.234 |
| 2 | IMG_20260908_113356_226.jpg | 24 | Horizontal | Separar: horizontal | — |
| 3 | IMG_20260908_113359_253.jpg | 0 | Vertical | Repetir: detección insuficiente | — |
| 4 | IMG_20260908_113412_390.jpg | 24 | Horizontal | Separar: horizontal | — |
| 5 | IMG_20260908_113416_588.jpg | 24 | Horizontal | Separar: horizontal | — |
| 6 | IMG_20260908_113457_485.jpg | 24 | Vertical | Conservar como candidata | 1.120 |
| 7 | IMG_20260908_113443_739.jpg | 24 | Vertical | Conservar como candidata | 1.171 |
| 8 | IMG_20260908_113506_354.jpg | 24 | Vertical | Conservar como candidata | 0.996 |
| 9 | IMG_20260908_113544_586.jpg | 24 | Vertical | Conservar como candidata | 0.818 |
| 10 | IMG_20260908_113553_002.jpg | 24 | Vertical | Conservar como candidata | 0.395 |
| 11 | IMG_20260908_113603_871.jpg | 11 | Vertical | Revisar / repetir | 0.942 |
| 12 | IMG_20260908_113627_762.jpg | 20 | Vertical | Revisar / repetir | 2.072 |
| 13 | IMG_20260908_113629_846.jpg | 24 | Vertical | Conservar como candidata | 1.038 |
| 14 | IMG_20260908_113707_560.jpg | 24 | Vertical | Conservar como candidata | 1.076 |
| 15 | IMG_20260908_113636_132.jpg | 24 | Vertical | Revisar / repetir | 1.682 |
| 16 | IMG_20260908_113710_813.jpg | 24 | Vertical | Revisar / repetir | 1.724 |
| 17 | IMG_20260908_113713_633.jpg | 24 | Horizontal | Separar: horizontal | — |
| 18 | IMG_20260908_113725_381.jpg | 20 | Vertical | Revisar / repetir | 2.434 |
| 19 | IMG_20260908_113715_841.jpg | 5 | Vertical | Repetir: detección insuficiente | — |
| 20 | IMG_20260908_113728_058.jpg | 24 | Horizontal | Separar: horizontal | — |
| 21 | IMG_20260908_113730_018.jpg | 24 | Horizontal | Separar: horizontal | — |
| 22 | IMG_20260908_113741_338.jpg | 10 | Vertical | Revisar / repetir | 1.145 |
| 23 | IMG_20260908_113748_785.jpg | 24 | Vertical | Revisar / repetir | 2.156 |
| 24 | IMG_20260908_113405_015.jpg | 24 | Vertical | Conservar como candidata | 0.642 |
| 25 | IMG_20260908_113434_109.jpg | 24 | Vertical | Conservar como candidata | 1.080 |
| 26 | IMG_20260908_113423_958.jpg | 24 | Vertical | Conservar como candidata | 0.676 |
| 27 | IMG_20260908_113534_251.jpg | 24 | Vertical | Conservar como candidata | 0.832 |
| 28 | IMG_20260908_113606_018.jpg | 24 | Vertical | Conservar como candidata | 0.315 |
| 29 | IMG_20260908_113632_374.jpg | 24 | Vertical | Conservar como candidata | 0.682 |
| 30 | IMG_20260908_113745_144.jpg | 0 | Vertical | Repetir: detección insuficiente | — |

### Candidatas para conservar (13, aún no certificadas)

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

### Observaciones concretas en las 8 fotos de revisión

- Foto 1: Revisar: foto inicial; residuo geométrico alto en el conjunto.
- Foto 11: Repetir preferiblemente: desenfoque visible; sólo 11 esquinas.
- Foto 12: Repetir preferiblemente: movimiento visible; 20 esquinas y residuo alto.
- Foto 15: Repetir preferiblemente: bordes suaves y residuo alto.
- Foto 16: Revisar: inclinación útil, pero residuo geométrico alto; comprobar planitud.
- Foto 18: Repetir: desenfoque visible y residuo geométrico alto.
- Foto 22: Repetir: contraluz fuerte, sólo 10 esquinas; no basta alcanzar el mínimo.
- Foto 23: Repetir: inclinación útil, pero tablero oscuro a contraluz y residuo alto.

## Próximo lote propuesto: 8 + 6 fotos nuevas

No es necesario rehacer las 30 por número. Primero corregir **planitud, iluminación y estabilidad**. Las 13 candidatas se podrán comparar con tomas nuevas; si el tablero anterior estaba deformado, su utilidad deberá reevaluarse, no asumirse.

1. Mantener cámara principal, modo normal, zoom 1× y vertical 3456×4608. Limpiar lente. No cambiar a modo retrato, alta resolución o video. Evitar tomas extremadamente cercanas y cambios grandes de enfoque; comprobar después a la distancia real de la demo.
2. Con una sola ubicación bien iluminada y sin ventana detrás del tablero, tomar **8 nuevas de calibración**: inclinaciones izquierda, derecha, arriba y abajo, dos por dirección (aprox. 15° y 30°). Repartir posiciones, incluyendo parte superior e inferior del encuadre. El cartón se inclina como puerta/tapa, sin doblarse; no basta girar la hoja dentro del plano de la imagen.
3. Tomar **6 adicionales para validación**, distintas y reservadas desde el inicio. Variar posición y ángulo; no copiar fotos de calibración ni elegir únicamente las más fáciles para validar.
4. Transferir originales sin compresión, recorte, rotación manual, filtros ni mejora con IA. Guardar las 8 en dataset/local/calibration/camera01/train y las 6 en validation, o adjuntarlas indicando los grupos.
5. Revisar el nuevo lote antes de combinarlo con las candidatas. Si no pasa, reportar el fallo; no ajustar los umbrales para obtener aprobación.

Las fotos originales de esta revisión permanecen en la carpeta temporal recibida. **No se copiaron a train/validation** y no se creó camera01.json. Los resultados numéricos y coordenadas detectadas se conservan en [audit.json](../outputs/calibration_review_20260908_batch01/audit.json), para no depender sólo de una descripción visual.

La siguiente etapa sigue siendo validar la lente y luego una referencia vertical real. No se ha demostrado todavía error de estatura ≤2 cm ni se necesitan datos corporales en esta revisión.

## Referencias técnicas

- [OpenCV: calibración ChArUco y vistas múltiples](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html).
- [ETH Zürich / Kalibr: planitud del soporte y verificación del tamaño impreso](https://github.com/ethz-asl/kalibr/wiki/Calibration-targets#tipsproblems). Se cita para buenas prácticas del soporte; no se cambió ChArUco por AprilGrid.

