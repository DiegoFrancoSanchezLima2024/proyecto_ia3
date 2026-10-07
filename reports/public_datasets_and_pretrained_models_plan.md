# Plan definitivo sin dataset propio de entrenamiento

Fecha de revisión: 2026-09-03

## Veredicto ejecutivo

Sí es viable terminar la tesis sin recolectar un dataset propio para entrenar.
No es viable afirmar precisión en personas reales sin comparar al menos la demo
contra medidas tomadas con cinta. Esa comparación no es un nuevo dataset de
entrenamiento: es una validación de caso real y puede hacerse con 3 a 5
voluntarios, o como mínimo con la persona de la defensa.

La mejor ruta para Sastre-IA no es mezclar inmediatamente DeepFashion, BEDLAM y
BodyM en una sola red. Los datos no están emparejados:

- BodyM tiene siluetas, estatura, peso, sexo y medidas corporales reales, pero no
  las fotografías RGB originales.
- DeepFashion-MultiModal tiene RGB, ropa, parsing, keypoints y DensePose, pero no
  medidas corporales en centímetros.
- BEDLAM tiene RGB sintético, cámara, máscaras y SMPL-X, pero no las medidas de
  cinta de BodyM ni la misma distribución de personas reales.

Por tanto, concatenar directamente sus características produciría un diseño que
no se puede entrenar de extremo a extremo con supervisión métrica. La solución
correcta es modular y conserva `exp_006_weight_huber` como campeón métrico.

## Arquitectura recomendada

```text
4 fotos RGB controladas
  frente | izquierda | espalda | derecha
                    |
                    v
        RTMPose-s/m: control de captura
  cuerpo completo, A-pose, inclinación, simetría y visibilidad
                    |
                    v
       SAM 2.1 small: máscara de la persona
                    |
        +-----------+-----------+
        |                       |
        v                       v
 frente + izquierda      frente + derecha reflejada
        |                       |
        +----------+------------+
                   v
      exp_006 + estatura + peso + sexo
                   |
                   v
      dos predicciones + desacuerdo + intervalos
                   |
                   v
       13 medidas corporales en centímetros
                   |
        +----------+-----------+
        |                      |
        v                      v
 reglas de patronaje      malla 3D de apoyo
 con holguras             SHAPY/SAM 3D Body
        |                      |
        v                      v
 SVG/PDF/DXF 1:1       visualización, no árbitro métrico
```

La espalda se conserva en el MVP para control de calidad y visualización 3D. No
debe anunciarse que `exp_006` aprendió cuatro vistas: BodyM solo permite entrenar
el par frontal/lateral. El perfil derecho sí puede reflejarse para producir una
segunda inferencia comparable.

## Selección de datasets públicos

| Prioridad | Dataset | Anotaciones útiles | Uso correcto | Decisión |
|---|---|---|---|---|
| P0 | BodyM | 8.978 pares de siluetas de 2.505 sujetos, estatura, peso, sexo y 14 medidas; splits Train/Test-A/Test-B | Entrenamiento y evaluación métrica | Ya usado; mantener como fuente principal |
| P1 | HBW de SHAPY | Fotos reales y escaneo 3D de 35 sujetos | Benchmark externo de forma 3D; no aumentar train | Solicitar acceso si el calendario lo permite |
| P2 | BEDLAM | RGB sintético, profundidad, máscaras, cámara, keypoints y SMPL-X | Pruebas de pose/forma y futura aumentación sintética | Usar solo un subconjunto o parámetros SMPL-X; no descargar todo ahora |
| P2 | THuman2.1 | 2.500 escaneos y ajustes SMPL-X | Renderizar cuatro vistas canónicas y estudiar forma | Solicitar acceso; no bloquear el MVP |
| P3 | DeepFashion-MultiModal | 44.096 RGB; 12.701 con parsing manual y keypoints; DensePose extraído | Evaluar/fine-tunear percepción de personas vestidas | No descargar para el primer prototipo |
| P3 | AGORA/HuMMan | Forma, pose, cámaras y datos multivista a gran escala | Investigación de HPS | Excesivos para el tiempo y almacenamiento disponibles |

### Por qué DeepFashion no es P0

Sus 24 clases describen principalmente prendas y accesorios (`top`, `outer`,
`skirt`, `dress`, `pants`, etc.). No son el contorno corporal desnudo y no hay
pecho, cintura o cadera reales. DensePose fue extraído por otro modelo, no
anotado como verdad métrica. Un SAM 2.1 ya preentrenado permite construir la
máscara del usuario sin volver a entrenar con las 44.096 imágenes.

DeepFashion solo se justifica si una evaluación concreta demuestra que SAM 2.1
falla con ropa ceñida o fondos de la demo. En ese caso puede utilizarse como
benchmark de parsing, no como reemplazo de BodyM.

### Por qué BEDLAM no es el estimador final

BEDLAM es excelente para pose y malla: publica 10.450 secuencias sintéticas,
profundidad, máscaras separadas y parámetros SMPL-X. Sin embargo, el BEDLAM
original usa solo 271 formas corporales distintas. Muchos fotogramas no equivalen
a muchos cuerpos. Además, una circunferencia calculada sobre SMPL-X debe usar la
misma definición anatómica que BodyM y el método de sastrería.

Si se utiliza, la descarga mínima debe ser parámetros SMPL-X y metadatos. Con
ellos se pueden renderizar frontal/laterales y calcular medidas sintéticas. Las
imágenes completas y secuencias a 30 fps no son necesarias para `exp_006`.

## Modelos preentrenados y decisión

| Función | Modelo | Papel | RTX 3050 6 GB |
|---|---|---|---|
| Segmentación | SAM 2.1 Hiera Small (Tiny como respaldo) | RGB a máscara con caja de persona | Sí, `batch=1`, FP16 y entorno separado |
| Pose y calidad | RTMPose-s o RTMPose-m | Rechazar mala pose/encuadre; no estimar volumen | Sí; preferible ONNX/TensorRT o MMPose separado |
| Centímetros | `exp_006_weight_huber` | Regresión principal de 13 medidas | Sí; ya medido con pico de 2,46 GiB |
| Baseline métrico 3D | SHAPY | Comparar pecho/cintura/cadera y producir SMPL-X | Colab o entorno legado separado |
| Malla moderna | SAM 3D Body | Reconstrucción visual y verificador de consistencia | Colab recomendado; modelos de 631M/840M parámetros |
| Literatura 2025 | Focused Human Body Measurement | Referencia ISO 8559 basada en SMPLer-X | No línea principal: repositorio inmaduro y checkpoint no documentado |

SAM 3D Body es actualmente una opción fuerte para reconstrucción monocular y
acepta prompts de máscara/keypoints, pero devuelve una malla MHR y sus métricas
publicadas evalúan pose/superficie, no error de cinta de las 13 medidas de
Sastre-IA. No debe reemplazar a `exp_006` sin un benchmark local comparable.

Para obtener una malla coherente con las medidas predichas hay dos líneas de
investigación posteriores:

1. SHAPY A2S puede generar parámetros SMPL-X a partir de estatura, pecho,
   cintura y cadera.
2. A2B/SMPL-Anthropometry permite transformar medidas a forma y volver a medir
   una malla, pero A2B espera muchas más medidas que las 13 actuales. No debe
   rellenarse lo faltante con valores inventados.

## Qué aporta precisión de verdad

Cambiar EfficientNet por un modelo enorme no ataca el mayor error observado. La
precisión real depende primero de:

1. **Protocolo de captura:** ropa ceñida, pies y cabeza visibles, cámara nivelada,
   A-pose constante, fondo contrastado y distancia estable.
2. **Escala:** pedir estatura y peso. Una foto monocular sin referencia no permite
   recuperar escala métrica de forma identificable. Si se desea estimar estatura,
   añadir un marcador impreso o calibración de suelo/cámara.
3. **Dos perfiles:** ejecutar frente+izquierda y frente+derecha reflejada. Si las
   medidas críticas discrepan, repetir captura en vez de promediar a ciegas.
4. **Detección fuera de distribución:** advertir cuando el IMC, máscara, pose o
   predicciones estén lejos del rango de entrenamiento.
5. **Incertidumbre:** mostrar intervalo, no un único número con falsa precisión.
6. **Definiciones:** separar medida corporal, medida de prenda terminada,
   holgura de vestir y margen de costura.

Los resultados actuales justifican conservar `exp_006`: MAE por sujeto de
1,242 cm en Test-A y 1,421 cm en Test-B. En Test-B, pecho/cintura/cadera tienen
2,841/3,087/2,027 cm de MAE. El grupo con obesidad sube a 3,804 cm de media en
esas tres medidas; por ello el sistema debe advertir y pedir confirmación manual
en casos de alto riesgo.

## Protocolo de demo real sin crear un dataset propio

La demo debe ejecutar la inferencia antes de mirar las medidas manuales:

1. Registrar consentimiento y no conservar las fotos después de la defensa si
   no son necesarias.
2. Introducir sexo/modelo corporal, estatura y peso medidos.
3. Capturar frente, izquierda, espalda y derecha con guía en pantalla.
4. Ejecutar RTMPose y rechazar automáticamente capturas incompletas o fuera de
   pose.
5. Segmentar con SAM 2.1; mostrar las cuatro máscaras para inspección.
6. Inferir con los dos pares laterales; mostrar promedio robusto, desacuerdo e
   intervalos conformales.
7. Solo después, medir con cinta pecho, cintura y cadera siguiendo el mismo
   protocolo anatómico; reportar error absoluto sin ocultar fallos.
8. Permitir corrección manual antes de enviar medidas a patronaje.

Con una sola persona esto es un estudio de caso, no una validación estadística.
Con 3 a 5 voluntarios ya se puede demostrar repetibilidad básica tomando cada
medida dos veces. Ninguna de esas fotografías debe incorporarse a train.

## Criterios de aprobación por etapa

| Etapa | Criterio mínimo |
|---|---|
| Captura | 100% del cuerpo visible, una sola persona y pose dentro de tolerancias documentadas |
| Máscara | Sin huecos grandes ni pérdida de cabeza, brazos, torso o piernas; revisión visual guardada |
| Doble perfil | Desacuerdo de pecho/cintura/cadera por debajo de un umbral calibrado con Test-B o repetir foto |
| Modelo métrico | Mantener Test-B por sujeto <= 1,421 cm para no degradar al campeón |
| Incertidumbre | Cobertura nominal evaluada por sujeto; recalibrar en real, no con el sujeto de demostración |
| Demo | Publicar predicción, cinta y error absoluto de las medidas críticas |
| Patronaje | No cortar sin confirmación manual de medidas críticas y validación geométrica del molde |

El umbral de desacuerdo no debe elegirse por intuición. Primero se simularán dos
laterales a partir de Test-B mediante perturbaciones/reflejos y se fijará el
percentil que mejor detecte capturas degradadas.

## Ejecución por hitos

### Hito 1 — Adaptador foto a máscara

- Crear un entorno `sastre-ia-perception`; no actualizar el entorno estable.
- Integrar RTMPose-s/m y SAM 2.1 Small.
- Guardar RGB, máscara, keypoints, puntuaciones de calidad y versión de modelo en
  un manifiesto reproducible.
- Probar primero con 4 fotos de una persona, sin reentrenar nada.

### Hito 2 — Inferencia de dos perfiles

- Normalizar cada máscara exactamente como BodyM.
- Ejecutar `exp_006` con izquierda y derecha reflejada.
- Añadir agregación robusta, desacuerdo por medida, intervalos y estado
  `aceptado/repetir/confirmación manual`.

### Hito 3 — Validación de caso real

- Medir 3 a 5 voluntarios solo para evaluación.
- Reportar MAE y error por medida; no hacer fine-tuning con ellos.
- Si el error real excede Test-B, corregir captura/segmentación antes de cambiar
  la arquitectura del regresor.

### Hito 4 — 3D de apoyo

- Ejecutar SAM 3D Body o SHAPY en Colab.
- Escalar con estatura conocida y comparar pecho/cintura/cadera virtuales contra
  `exp_006`.
- Mantener el 3D como visualización si no supera el benchmark métrico.

### Hito 5 — Aumentación sintética, solo si hace falta

- Usar SMPL/SMPL-X para generar formas difíciles, priorizando extremos de IMC,
  como el ABS del artículo de BodyM.
- Armonizar cada plano/landmark de medida con las definiciones del proyecto.
- Comparar `exp_006` contra `exp_007_synthetic` con los mismos splits oficiales.
- Promover únicamente si mejora Test-B y las medidas críticas sin perder
  cobertura de incertidumbre.

## Qué ejecutar primero

No descargar todavía DeepFashion completo, BEDLAM completo ni HuMMan. El orden
óptimo es:

1. Congelar checkpoint, métricas y configuración de `exp_006`.
2. Implementar el adaptador de fotos con RTMPose + SAM 2.1.
3. Implementar doble perfil y rechazo por desacuerdo.
4. Hacer la prueba real medida.
5. Recién entonces decidir si el cuello de botella es segmentación, dominio o
   regresión; seleccionar SHAPY/SAM 3D Body/sintético según la evidencia.

## Fuentes primarias

- BodyM: https://diw9s4r18eiyi.cloudfront.net/bodym/
- BMnet/ABS: https://www.amazon.science/publications/human-body-measurement-estimation-with-adversarial-augmentation
- DeepFashion-MultiModal: https://github.com/yumingj/DeepFashion-MultiModal
- BEDLAM: https://bedlam.is.tue.mpg.de/
- THuman2.1: https://github.com/ytrock/THuman2.0-Dataset
- SAM 2: https://github.com/facebookresearch/sam2
- RTMPose: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- SAM 3D Body: https://github.com/facebookresearch/sam-3d-body
- SHAPY: https://github.com/muelea/shapy
- Focused Human Body Measurement: https://openaccess.thecvf.com/content/CVPR2025/html/Chen_A_Focused_Human_Body_Model_for_Accurate_Anthropometric_Measurements_Extraction_CVPR_2025_paper.html
- A2B Human Mesh: https://github.com/kaulquappe23/a2b_human_mesh
- SMPL-Anthropometry: https://github.com/DavidBoja/SMPL-Anthropometry

