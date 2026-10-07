# Análisis de captura real `demo_001`

## Veredicto

La tubería técnica funciona de extremo a extremo: recibe cuatro fotografías reales, segmenta a la persona con SAM2.1, extrae descriptores de las siluetas y estima medidas con el modelo `exp_006_weight_huber`.

La captura `demo_001` queda **rechazada para generar moldes de corte** (`decision: repeat_capture`). Las estimaciones se conservan únicamente como resultado experimental, no como medidas listas para confección.

## Datos de entrada

- Sexo del modelo: mujer (`gender=0`)
- Estatura declarada: 153 cm
- Peso declarado: 43 kg
- Vistas: frontal, perfil izquierdo, posterior y perfil derecho
- Segmentador: SAM2.1 Hiera Small
- Predictor antropométrico: `exp_006_weight_huber`

## Resultado de segmentación

| Vista | Puntaje SAM2 | Pico de VRAM | Control de calidad |
|---|---:|---:|---|
| Frontal | 0.9570 | 0.583 GiB | Sin advertencia geométrica |
| Izquierda | 0.9644 | 0.583 GiB | La persona toca el borde superior o inferior |
| Posterior | 0.9756 | 0.583 GiB | Sin advertencia geométrica |
| Derecha | 0.9673 | 0.583 GiB | Sin advertencia geométrica |

SAM2 segmentó correctamente el contorno **visible**, pero una máscara no puede recuperar el cuerpo oculto por prendas holgadas. En estas fotografías el contorno contiene la sudadera, el short ancho, el cabello y las pantuflas.

## Estimaciones experimentales

| Medida crítica | Estimación | Intervalo conservador | Confirmación manual |
|---|---:|---:|---|
| Pecho | 85.39 cm | 79.60–91.19 cm | Sí |
| Cintura | 71.79 cm | 65.49–78.11 cm | Sí |
| Cadera | 86.25 cm | 82.23–90.27 cm | No según el modelo |

Los desacuerdos entre los perfiles izquierdo y derecho fueron pequeños, pero esto no demuestra exactitud métrica. Ambos perfiles comparten el mismo sesgo de ropa y el modelo también utiliza estatura y peso como información previa. La validación real requiere comparar estas salidas con mediciones de cinta métrica que el modelo no haya visto.

## Motivos del rechazo

1. La ropa holgada impide observar pecho, cintura, cadera, muslos y entrepierna reales.
2. El cabello cubre parte del torso en vistas posterior y laterales.
3. Las pantuflas alteran el límite inferior y la estimación de longitudes.
4. La persona queda demasiado cerca del borde en la vista izquierda.
5. La posición de brazos no es completamente uniforme entre las cuatro vistas.

## Protocolo para `demo_002`

1. Usar camiseta/top ceñido y leggings o short elástico ceñido, sin chaqueta ni sudadera.
2. Recoger completamente el cabello para despejar cuello, hombros, espalda y pecho.
3. Tomar las fotos descalza o con medias finas.
4. Mantener una pose A: pies a la anchura de caderas y brazos separados 20–30 grados del torso.
5. Usar exactamente la misma pose, distancia, zoom y altura de cámara en las cuatro vistas.
6. Colocar la cámara aproximadamente a la altura de la cintura/cadera, recta y sin inclinación.
7. Dejar margen visible sobre la cabeza, bajo los pies y a ambos lados.
8. Preferir una pared lisa y buena iluminación frontal uniforme.
9. No usar modo retrato ni lente gran angular; preferir la cámara principal en 1x.
10. Mantener la estatura (153 cm), el peso (43 kg) y registrar medidas de cinta para evaluar el error.

## Criterio antes de generar moldes

No se deben producir piezas de saco o falda a escala de corte hasta que la captura pase los controles de calidad y pecho, cintura y cadera sean confirmados con cinta o queden dentro del umbral definido para la demo. El sistema puede mostrar una previsualización experimental, pero debe distinguir claramente entre `estimado`, `confirmado` y `apto para patrón`.
