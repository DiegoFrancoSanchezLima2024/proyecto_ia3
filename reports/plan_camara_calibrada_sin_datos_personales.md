# Sastre-IA: revisión y plan de cámara calibrada sin medidas introducidas

Fecha: 2026-09-04. Estado: propuesta de ingeniería sustentada en código, resultados guardados y fuentes primarias; todavía no implementada ni validada en cámara real.

## Objetivo y decisión

Obtener estatura y medidas corporales a partir de 3–4 fotos de cuerpo completo, sin solicitar estatura, peso ni sexo como entradas del predictor. La calibración del equipo y las dimensiones de los marcadores son configuración del sistema. La persona solo sigue instrucciones de captura y elige la prenda.

Ruta recomendada: percepción preentrenada + cámara/suelo calibrados + escala automática + estimación de forma multivista. La RTX 3050 de 6 GB permite desarrollar el núcleo ligero; modelos 3D grandes se compararán primero en un entorno independiente o Colab. No se ha medido aún su consumo local.

Este plan reemplaza la recomendación anterior de pedir estatura/peso para la demo. Los experimentos históricos siguen siendo referencias válidas de un problema asistido, pero no demuestran el objetivo automático.

## 1. Auditoría del proyecto

### Hallazgos confirmados

- `configs/default.yaml`: estatura como ancla, 13 salidas que excluyen estatura, `use_geometry: false`.
- `configs/exp_006_weight_huber.yaml`: `use_weight_meta: true`.
- `src/inference/predict_measurements.py`: requiere altura y sexo; exige peso si el checkpoint lo usa.
- `src/data/bodym_dataset.py`: introduce estatura/sexo y opcionalmente peso en los metadatos. Redimensiona imágenes completas a un cuadrado; no calcula calibración, escala o distancia.
- `src/models/regressor.py`: exige al menos dos metadatos. No basta eliminar los argumentos del comando para disponer de un modelo autónomo.
- `src/inference/predict_four_views.py`: dos inferencias frente/perfil; espalda solo para calidad. No es un ajuste corporal conjunto de cuatro vistas.
- `src/perception/sam2_capture.py`: las cajas de la primera demo se proporcionaron manualmente; el modo sin caja usa toda la imagen como prompt. Falta un detector automático de personas. El ajuste de ropa también se declara manualmente.
- `src/evaluation/metrics.py`: promedia capturas repetidas por sujeto; el resultado por sujeto no equivale al de una sola sesión.

El README comunica 1.421 cm globales en Test-B. Los resultados guardados de `exp_006` por sujeto son:

| Medida | MAE | P90 del error absoluto | Casos dentro de ±2 cm |
|---|---:|---:|---:|
| Pecho | 2.841 cm | 5.749 cm | 44.50% |
| Cintura | 3.087 cm | 5.881 cm | 41.75% |
| Cadera | 2.027 cm | 4.295 cm | 60.00% |

Fuente local: `experiments/exp_006_weight_huber/results_test_wild.json`, sección `_subject_aggregate.metrics`. Hay 400 sujetos y 1160 pares de capturas, 2.9 pares por sujeto en promedio y hasta 11. El MAE por captura es 2.930/3.247/2.082 cm para pecho/cintura/cadera. Un par BodyM ya contiene dos vistas; no confundir número de pares con número de fotos.

La mejora con peso es real para ese protocolo, pero no predice el rendimiento al retirar peso y altura reales. Tampoco corresponde interpretar el promedio de 13 medidas como una tolerancia garantizada para confección.

`exp_004` y `exp_005` no mejoraron el baseline. Eso no demuestra que balanceo o perfiles sean inútiles en general; demuestra que esas variantes concretas no funcionaron en esos ensayos. No hay evidencia de que la capacidad de VRAM sea la causa principal del error.

### Correcciones a recomendaciones anteriores

1. Una homografía del fondo es válida para puntos de su plano, no para todo el volumen de una persona situada delante. Se necesita geometría de cámara/suelo y profundidad o restricciones corporales.
2. Concordancia entre vistas mide consistencia, no exactitud. Cuatro vistas pueden compartir un sesgo de escala.
3. El puntaje de SAM2 no es precisión antropométrica y una máscara correcta de ropa holgada no revela el cuerpo oculto.
4. Un pico de 0.583 GiB registrado con PyTorch es memoria de tensores asignados medida por ese proceso, no todo el consumo de GPU. Medir también memoria reservada y consumo total.
5. La rama geométrica está desactivada en el campeón: sus resultados provienen de la red con metadatos. Las fórmulas de escala presentes en otras funciones no prueban que ese checkpoint use reconstrucción geométrica.

## 2. Captura métrica sin preguntar datos corporales

### Configuración una vez por cámara/modo

Calibrar matriz intrínseca K y distorsión con ChArUco o tablero de ajedrez. Como punto de partida operativo, obtener 20–30 imágenes nítidas del tablero cubriendo el campo visual y diferentes inclinaciones. Separar algunas imágenes para comprobar reproyección. Esa cantidad no garantiza calidad por sí sola.

Guardar resolución, orientación, lente, modo de captura y enfoque/zoom cuando puedan fijarse. Una captura de navegador puede tener recorte y estabilización diferentes de la aplicación nativa. Si cambian lente, recorte, zoom o geometría de imagen, revisar la calibración; si solo cambia resolución de forma conocida, transformar K correctamente.

Instalar un tablero rígido de marcadores de dimensiones verificadas en el suelo, alrededor de la zona de captura, con suficientes marcadores visibles pese a los pies. Estimar pose del tablero con `solvePnP` y conocer el plano del suelo en coordenadas métricas. Verificar escala con longitudes y posiciones que no se usaron para ajustar la calibración.

Esto sigue el modelo de cámara y estimación de pose documentados por [OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html). La calibración intrínseca por sí sola no determina la distancia absoluta a una persona.

### Captura de cada persona

- Cámara fija sobre soporte. Como rango inicial a ensayar: aproximadamente 2.5–4 m, según campo visual y espacio, con cuerpo y marcadores completos. La distancia efectiva se calcula; el usuario no la escribe.
- Persona gira cerca del centro marcado: frente, perfil izquierdo, espalda y perfil derecho. Si solo hay tres fotos, frente/perfil/espalda, con menor redundancia.
- Mantener postura A cómoda, cuerpo erguido, ropa ceñida, cabello recogido y pies visibles sin calzado grueso.
- Detección automática de persona → articulaciones → segmentación → validación de encuadre/postura. No pedir cajas ni sexo/peso/altura.
- Registrar pequeños desplazamientos y pose por vista. La rotación humana entre fotos impide tratar la secuencia como una escena rígida convencional de fotogrametría.

### Estatura automática

Corregir distorsión y proyectar rayos de puntos de contacto del pie al plano del suelo para estimar posición y distancia. Combinar el contorno superior de la cabeza, orientación vertical y pose corporal para estimar la altura; refinar junto a la malla y las demás vistas.

El tobillo de un detector no es el contacto con el suelo; el punto superior del cabello no es necesariamente el vértex anatómico. El pie y la cabeza no están obligatoriamente en la misma vertical. Deben tratarse explícitamente esas incertidumbres y rechazar postura inclinada o puntos ocultos.

En la aproximación frontal H≈h_px·Z/f_y, un error de distancia de 5 cm a 3 m implica aproximadamente 2.55 cm de error de altura para 153 cm, aun con contorno perfecto. Es un ejemplo geométrico, no un resultado del prototipo. Aumentar resolución o GPU no corrige ese sesgo.

Objetivo provisional: P90 del error de estatura ≤2 cm en el protocolo controlado, publicando todos los fallos y tasa de rechazo. Alcanzarlo debe demostrarse frente a referencias independientes; MAE ≤2 cm no implica que todos los casos estén dentro de ±2 cm.

## 3. Arquitectura y modelos

### Núcleo local

1. Detector de persona ligero compatible con RTMPose, por ejemplo RTMDet-tiny; seleccionar y medir el checkpoint concreto.
2. RTMPose-m WholeBody para pose/pies y control de calidad, preferentemente con ONNX Runtime en un entorno aislado. Sus puntos no equivalen automáticamente a puntos de sastrería. [Repositorio oficial](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).
3. SAM2.1 Small existente, con cajas automáticas. Conservar RGB y máscaras de resolución original para metrología; usar recortes reducidos solo para redes y registrar transformaciones a coordenadas originales.
4. OpenCV para calibración, escala y distancia sobre CPU.
5. Estimador ligero de proporciones BodyM y/o ajuste 3D calibrado, comparados con el mismo protocolo automático.

### Comparación 3D acotada

| Candidato | Papel | Decisión |
|---|---|---|
| SHAPY | Baseline preentrenado de forma y medidas | Prioritario por alineación con antropometría; verificar dependencias y VRAM |
| CameraHMR, variante BEDLAM2/SMPL-X | Inicialización con perspectiva y entrenamiento sintético reciente | Comparador prioritario en Colab o GPU local si el benchmark cabe |
| SMPLer-X-S | Inicialización más pequeña | Respaldo si los dos anteriores impiden ejecución local |
| SAM 3D Body | Comparador de reconstrucción con representación MHR | Posterior; necesita adaptador de medidas/topología y benchmark de memoria |
| Metric3D Small / Depth Pro | Profundidad monocular auxiliar | Ablación opcional, no condición de éxito del MVP |

[SHAPY](https://github.com/muelea/shapy) ofrece regresión desde imagen y mediciones virtuales. [CameraHMR](https://camerahmr.is.tue.mpg.de/) modela perspectiva y dispone de [inferencia BEDLAM2 con SMPL-X](https://github.com/pixelite1201/CameraHMR/blob/master/docs/bedlam2.md). Son candidatos, no ganadores demostrados en Sastre-IA.

[SMPLer-X](https://github.com/MotrixLab/SMPLer-X) ofrece checkpoints de distintos tamaños. [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) produce MHR, no SMPL-X directamente. [Metric3D](https://github.com/YvanYin/Metric3D) y [Depth Pro](https://github.com/apple/ml-depth-pro) estiman profundidad métrica; las fuentes consultadas no establecen una garantía de ±2 cm de estatura para nuestra captura.

### Ajuste conjunto

Una sola forma corporal compartida, con pose y ubicación independientes por fotografía. Cámaras calibradas; escala respaldada por el entorno. Optimizar reproyección de articulaciones, siluetas, contacto con suelo, alineación vertical y un prior corporal. Inicializar con una red preentrenada y congelar sus pesos; optimizar parámetros corporales de baja dimensión por persona.

La silueta de ropa debe recibir un tratamiento robusto: forzar la malla a llenar una sudadera sobreestima el cuerpo. Con solo tres o cuatro vistas persisten ambigüedades de superficie, concavidades y tejido oculto. Medir la malla en pose canónica con definiciones anatómicas explícitas y comprobar cuánto cambia bajo inicializaciones y perturbaciones de captura.

No promediar vértices de mallas con poses distintas, ni considerar una malla visualmente convincente como validación métrica.

## 4. Datos públicos y uso correcto

| Recurso | Aporta | Uso y límite |
|---|---|---|
| BodyM, ya descargado | Siluetas frontal/lateral y medidas reales | Aprender proporciones y comparar antropometría; no entrega RGB ni calibración de nuestra cámara |
| HBW de SHAPY | Imágenes reales y forma de referencia | Evaluación externa; test GT no público, usar validación accesible/protocolo oficial |
| BEDLAM/BEDLAM2 | RGB sintético, cuerpo y cámaras de referencia | Ensayar geometría y forma; aprovechar pesos y subconjunto, no descargar todo |
| THuman2.1 | Escaneos y ajustes SMPL-X | Opcional: vistas sintéticas; superficie vestida y fitting no son cinta corporal exacta |
| COCO-WholeBody | Anotaciones de pose | Aprovechar preentrenamiento de RTMPose; no etiquetas de perímetros |

[BodyM](https://diw9s4r18eiyi.cloudfront.net/bodym/) publica medidas obtenidas con escaneo y distingue Test-A/Test-B. [HBW](https://github.com/muelea/shapy/blob/master/documentation/DATA.md) requiere seguir su protocolo de acceso/evaluación. [BEDLAM](https://bedlam.is.tue.mpg.de/) y [BEDLAM2](https://bedlam2.is.tuebingen.mpg.de/) permiten evaluar cuerpo y cámara; BEDLAM2 completo publica 11 TB de imágenes, por lo que descargarlo entero no es razonable para este proyecto. [THuman2.1](https://github.com/ytrock/THuman2.0-Dataset) documenta transformaciones de escala/traslación de sus fittings: deben respetarse.

Mantener las licencias y condiciones de acceso de cada recurso. Los modelos SMPL-X y varios datasets requieren registro. Las imágenes RGB de un dataset no pueden emparejarse artificialmente con medidas de otro. Separar sujetos/formas entre entrenamiento y evaluación, incluso cuando se rendericen muchas vistas.

No hace falta crear un dataset propio de entrenamiento. La calibración necesita imágenes del tablero; la demostración de precisión necesita unas personas medidas solo para evaluación. Tres a cinco voluntarios con capturas repetidas sirven para un piloto, no para concluir precisión poblacional. Si el modelo ya entrenó con un benchmark, no presentarlo como generalización a datos inéditos.

## 5. Experimentos propuestos, en orden

Los identificadores siguientes describen experimentos futuros; no existen aún como configuraciones ejecutables.

| Etapa | Hipótesis | Prueba y criterio |
|---|---|---|
| E0: auditoría congelada | El baseline asistido no representa el modo automático | Reportar por captura y sujeto, por medida, y ablaciones de imagen/metadatos en validación |
| C1: escala/distancia | Cámara+suelo permiten escala sin estatura | Objetos verticales y posiciones independientes; medir error en cm a varias distancias |
| C2: estatura humana | Pose+segmentación+calibración permiten estatura | Captura ciega repetida; P90, sesgo, rango y tasa de rechazo |
| E7: proporciones sin metadatos personales | Las siluetas aportan forma sin peso/sexo/altura como entrada | Predecir r_i=m_i/H; salida final m_i=r_i·H_estimado por cámara |
| E8: inicialización 3D | Un modelo preentrenado estima mejor forma que E7 | Comparar SHAPY y CameraHMR con presupuesto y entradas iguales |
| E9: ajuste multivista | Forma compartida mejora sobre monocular | Ablación 1/2/3/4 vistas, misma escala y sujetos |
| E10: sastrería | Las medidas estimadas producen patrón verificable | Comprobar dimensiones, correspondencia de costuras, holgura y toile/prototipo textil |

Para E7 la estatura real de BodyM sirve para construir la etiqueta de proporción durante entrenamiento, no se entrega como entrada visual/metadato. Conservar aspecto y un protocolo consistente de recorte. La estatura automática es calculada por el módulo de cámara y aplicada después. Así no pedimos datos a la persona ni entrenamos a la red a deducir centímetros de imágenes sin escala.

BodyM por sí solo no permite evaluar C1/C2. En BodyM, usar estatura real al reconstruir centímetros constituye un ensayo de escala perfecta que debe etiquetarse como tal. Añadir perturbaciones de escala basadas en errores medidos de C1/C2 para estudiar sensibilidad; no afirmar que esto sustituye una prueba completa RGB→cm.

En E8/E9 guardar pérdidas por vista, residuo en píxeles, escala, mediciones finales y pico de memoria. Reproyección baja también puede coexistir con medidas incorrectas: comparar siempre con referencias independientes.

Seleccionar configuraciones con validación, repetir candidatos finales con varias semillas y usar bootstrap por sujeto cuando corresponda. Test-A/Test-B ya han informado decisiones en múltiples ensayos; declararlos benchmarks de desarrollo histórico y reservar una evaluación externa para el cierre. Los intervalos actuales de BodyM no conservan automáticamente cobertura al añadir escala estimada, fusionar vistas o cambiar de cámara.

## 6. Uso de RTX 3050 6 GB

- Presupuesto inicial: batch 1 para fotos reales y una tarea pesada residente por vez.
- Ejecutar detector/pose, segmentador y reconstructor por etapas; guardar resultados intermedios y liberar modelos. Procesos separados cuando las dependencias o memoria retenida lo aconsejen.
- Geometría, detección de marcadores y reportes en CPU; redes en GPU. AMP en inferencia tras comparar con FP32; ajuste geométrico y parámetros sensibles inicialmente en FP32.
- En entrenamiento ligero, conservar EfficientNet-B0 como control, AMP y acumulación ya disponibles. Ensayar 320 frente a 448/512 solo después de corregir encuadre/escala, midiendo exactitud y tiempo.
- Cachear máscaras/características cuando el encoder esté congelado; regenerar caché al cambiar encoder o preprocesamiento.
- Ajuste multivista a resolución moderada con refinamiento de contornos si el benchmark muestra beneficio. No mantener cuatro grafos pesados innecesariamente.
- Medir tiempo total, latencia por etapa, `max_memory_allocated`, memoria reservada y consumo total con monitor NVIDIA. Objetivo operativo aproximado ≤5 GiB totales para dejar margen al escritorio, sujeto a medición.
- Colab para comparar redes grandes y generar parámetros iniciales precomputados. La demo local solo se declara autónoma cuando funciona completa sin ese servicio.

Los registros previos informan 2.46 GiB de pico en entrenamiento de exp_006. Ocupar los 6 GB no es una métrica de calidad. No hay garantía todavía de que CameraHMR/SHAPY quepan con toda su tubería en la GPU local.

## 7. Integración con patronaje y defensa

Primero validar estatura, pecho, cintura, cadera, hombros y longitudes necesarias. BodyM no cubre por sí solo todas las entradas de saco, pantalón y falda: elaborar una correspondencia entre cada medida corporal, definición anatómica y fórmula de patrón. No equiparar automáticamente `leg-length` con entrepierna o `shoulder-to-crotch` con tiro de pantalón.

Usar [GarmentCode](https://github.com/maria-korosteleva/GarmentCode) como motor paramétrico y revisar [GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements) como herramienta de medidas sobre malla. La compatibilidad de topología y landmarks requiere adaptación. Estos proyectos no garantizan que cualquier saco formal salga validado sin trabajo de patronaje.

Separar medida corporal, holgura de diseño y margen de costura. Mantener alcance: saco/pantalón y saco/falda; validar primero una falda sencilla y luego pantalón y saco. La vista frontal 2D puede renderizar la malla/prenda; la apariencia generada por IA no demuestra ajuste ni caída física exacta. Verificar patrón a escala 1:1 y realizar al menos una prueba textil antes de afirmar confección validada.

Entregable inmediato recomendado: módulo de calibración con manifest de cámara, prueba de escala a distintas posiciones y estatura automática con imágenes anotadas. Solo después ejecutar el siguiente entrenamiento antropométrico. Las fotos actuales sirven para depurar percepción; no tienen la referencia métrica necesaria para validar C1/C2.
