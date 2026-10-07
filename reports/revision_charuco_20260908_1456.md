# Seguimiento oblicuo — 8 de septiembre, 14:56

Se detectan 24 esquinas ChArUco y 17 marcadores en las cuatro imágenes originales. No se reajustó la lente ni se modificaron fotos.

| Foto | Evaluación | Inclinación estimada | RMS por vista |
| --- | --- | --- | --- |
| IMG_20260908_145647_101.jpg | No pasa límite de 1,50 px | 21,12° | 1,874 px |
| IMG_20260908_145738_734.jpg | Solo detección: horizontal 4608 × 3456 | No evaluada | No evaluado |
| IMG_20260908_145638_221.jpg | Pasa límite | 12,59° | 1,232 px |
| IMG_20260908_145744_191.jpg | Solo detección: horizontal 4608 × 3456 | No evaluada | No evaluado |

La primera captura sí alcanza la inclinación solicitada. Su fallo aporta evidencia de que la discrepancia persiste en una nueva vista oblicua: no corresponde afirmar que el usuario tomó un ángulo incorrecto. Visualmente se aprecia menor nitidez en la primera, pero no se ha aislado la contribución de desenfoque/movimiento, planitud del tablero o modelo de lente al residuo. Detectar todas las esquinas no acredita su localización precisa.

Las dos horizontales no son fotografías inútiles; se mantienen fuera de la evaluación del perfil vertical por la diferencia de coordenadas/resolución. No se giraron ni se aplicó sin más la matriz vertical a ellas.

El perfil diagnóstico permanece `needs_recapture`. No se borra el fallo anterior ni se habilitan medidas en centímetros. No se solicita otra tanda indiscriminada: el siguiente trabajo técnico es investigar la discrepancia con las observaciones existentes (localización de esquinas, planitud y ajuste de lente), conservando separación entre ajuste y validación. Si se cambia el modelo usando estos resultados, estas tomas dejan de ser una prueba independiente del nuevo modelo.

Auditoría: `outputs/calibration_review_20260908_batch04/audit.json`. Reproducción: `scripts/review_charuco_20260908_1456.py` (rechaza sobrescribir). Originales intactos; sin aumento de datos.
