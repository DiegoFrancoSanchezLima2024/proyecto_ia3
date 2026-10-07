# Diagnóstico de cierre de calibración

Se evaluaron cuatro originales con el perfil congelado. Todos tienen 24 esquinas; dos archivos contienen exactamente los mismos píxeles, por lo que hay tres observaciones distintas.

| Archivo | RMS px | Inclinación estimada |
| --- | --- | --- |
| IMG_20260908_150537_475.jpg | 3,173 | 25,87° |
| IMG_20260908_150557_515.jpg | 2,345 | 19,93° |
| IMG_20260908_150523_600.jpg | 3,173 | Duplicado de 150537_475 |
| IMG_20260908_150428_350.jpg | 2,153 | 24,10° |

Las tres observaciones distintas superan 1,50 px. Los ángulos sí son adecuados para el seguimiento solicitado. No se atribuye el fallo a que el usuario no haya comprendido la captura.

## Pruebas exploratorias realizadas

- Refinamiento adicional de esquinas en ventana de 5 px, recalibrando solamente las 13 fotos originales de ajuste: máximo nuevo 3,321 px; no mejora.
- Reajuste usando las 30 fotos anteriores (13 de ajuste más 17 que dejan de ser validación en este experimento): máximo nuevo 2,882 px; las tres vistas nuevas siguen superando 1,50 px. No se usaron las nuevas para ajustar.
- Una homografía libre sobre puntos corregidos con la distorsión anterior deja RMS de 1,905–2,018 px. Es una prueba más flexible, no una cámara física ni demostración de causa.

Estas comparaciones son exploratorias. No autorizan escoger parámetros por el mejor resultado y llamar independiente a esta misma evaluación. No se cambió el perfil operativo, ni umbrales, ni originales. Las coordenadas corregidas se usaron solo numéricamente; no se editaron fotos.

## Conclusión y próximo control

No se logró una calibración validada con estas alternativas. La causa no está aislada: planitud/impresión del patrón, localización de esquinas, enfoque/procesamiento y modelo de lente siguen siendo candidatos. Las ondulaciones visibles del papel justifican comprobar primero el patrón físicamente, no afirmar que sea la única causa.

Antes de más fotos: apoyar una regla recta sobre el área impresa en ambas direcciones y comprobar huecos o abombamiento sin presionar; revisar que varios cuadrados midan 30 mm tanto horizontal como verticalmente. Si está ondulado, montar una impresión sin arrugas sobre una placa realmente plana y rígida. Kalibr recomienda planitud y verificar las dimensiones impresas: https://github.com/ethz-asl/kalibr/wiki/Calibration-targets#tipsproblems

Una nueva sesión controlada debería mantener lente 1× y resolución vertical, sin cambios de modo/zoom, y separar antes del ajuste las fotos destinadas a validación. El mínimo implementado es 12 vistas de ajuste y 4 de validación; la recomendación del proyecto es 20 y 6. No se promete que un número de fotos garantice aprobar. Después de superar el control de reproyección todavía se necesita validación métrica con una longitud conocida.

Resultados: `outputs/calibration_review_20260908_batch05/audit.json` y `audit_expanded.json`. Script: `scripts/review_charuco_20260908_1504.py`. Estado conservado: `needs_recapture`.
