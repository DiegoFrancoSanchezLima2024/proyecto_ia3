# Comprobación de lente: cinco fotos nuevas

Se evaluaron los cinco originales completos, sin filtrar por resultado y sin reajustar la matriz de cámara ni la distorsión del perfil diagnóstico anterior. Se ajustó únicamente la pose de cada tablero, usando el mismo detector y error RMS que la validación existente.

| Foto | Esquinas | RMS (px) | Inclinación estimada respecto de vista frontal |
| --- | --- | --- | --- |
| IMG_20260908_143712_289.jpg | 24 | 0,724 | 3,83° |
| IMG_20260908_143859_040.jpg | 24 | 0,436 | 3,90° |
| IMG_20260908_143819_447.jpg | 24 | 0,850 | 6,86° |
| IMG_20260908_143923_941.jpg | 24 | 0,502 | 2,10° |
| IMG_20260908_143642_513.jpg | 24 | 0,351 | 3,96° |

Todas pasan el límite por vista de 1,50 px. Las cinco sirven como comprobación adicional; no se descarta ninguna. El RMS no se interpreta como centímetros ni como puntuación de nitidez. No se editaron ni copiaron las imágenes en este análisis.

La evidencia favorable se limita a vistas casi frontales. Las inclinaciones estimadas son de 2,1 a 6,9 grados, con diversidad de normales de 5,37 grados. No reproducen la vista oblicua anterior de unos 22,6 grados que tuvo RMS 1,637 px. No se atribuye causalidad a las ondulaciones sin más evidencia ni se borra el fallo anterior. El perfil continúa `needs_recapture`; no se habilita pose métrica.

Siguiente captura focalizada: dos vistas oblicuas desde lados opuestos, aproximadamente 20–30 grados respecto de la vista perpendicular, conservando el tablero rígido y plano y las condiciones de cámara. Si el tablero queda en el piso, desplazar la cámara lateralmente y apuntar hacia él, en vez de seguir directamente encima. No es girar el teléfono dentro del plano de la foto. Mantener tablero completo, enfocado y con tamaño suficiente. Son una prueba de seguimiento, no una nueva calibración completa.

Reproducción: `scripts/review_charuco_20260908_1436.py`; resultados y hashes en `outputs/calibration_review_20260908_batch03/audit.json`. El script rechaza sobrescribir una auditoría existente.
