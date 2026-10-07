# Decisión técnica después de exp_002–exp_005

Fecha: 2026-09-03

## Conclusión ejecutiva

La estrategia híbrida es correcta, pero el siguiente experimento no debe ser
RTMPose sobre las máscaras de BodyM. El cuello de botella actual es volumen
corporal (pecho, cintura y cadera), no localización de articulaciones. Además,
la copia local de BodyM solo contiene siluetas; un estimador de pose preentrenado
con RGB sufriría cambio de dominio y no puede evaluarse limpiamente sin las fotos
originales.

La omisión más importante era `weight_kg`. El artículo original de BMnet usa
silueta frontal, silueta lateral, estatura, peso y sexo. Nuestra ablación local
confirma que peso es una señal de tamaño/volumen extremadamente fuerte.

## Evidencia local

Las cifras siguientes son MAE en centímetros, agregadas por sujeto para no dar
más importancia a personas con más fotografías:

| Modelo | Validación (303) | Test-A (87) | Test-B no controlado (400) |
|---|---:|---:|---:|
| Ridge: estatura + sexo | 3.072 | 3.227 | 3.753 |
| Ridge: estatura + sexo + peso | **1.588** | **1.494** | **1.651** |
| CNN multivista exp_002, sin peso | — | — | 1.855 |

En Test-B, el Ridge con peso logra pecho 3.314 cm, cintura 3.592 cm y cadera
2.427 cm. Exp_002 logra 3.748, 3.967 y 3.217 cm respectivamente. Esto no prueba
que el Ridge sea el producto final: prueba que el modelo visual debe recibir o
estimar peso y que todo modelo complejo debe superar este control sencillo.

Resultados completos: `reports/baselines_weight_ablation.json`.

## Arquitectura corregida del MVP

```text
captura guiada + control de calidad
        |-- estatura conocida o referencia métrica visible
        |-- peso ingresado (modo preciso) o ausente (modo básico)
        v
segmentación de persona -> máscaras canónicas frontal/lateral
        v
EfficientNet-B0 compartido + fusión multivista
        + [estatura, sexo, peso]
        v
13 medidas + intervalos conformales
        v
reglas de patronaje deterministas + holguras
        v
SVG/PDF/DXF 1:1 + previsualización frontal separada
```

No se debe presentar la imagen de virtual try-on como validación del ajuste de
la prenda. Es una visualización; la validez del molde se comprueba con medidas,
reglas geométricas, prototipo en tela y evaluación de un patronista.

## Captura de cuatro vistas sin prometer datos inexistentes

1. Capturar frente, perfil izquierdo, espalda y perfil derecho con pose, ropa
   ceñida, distancia y encuadre guiados.
2. Para el primer modelo científico usar frente + perfil izquierdo, porque es lo
   que BodyM permite entrenar y comparar.
3. Usar espalda y perfil derecho para control de calidad. El perfil derecho puede
   reflejarse y formar un segundo par de inferencia; promediar ambos pares solo
   después de medir que reduce error en el conjunto local.
4. Incorporar cuatro vistas como entradas aprendidas únicamente después de crear
   datos locales medidos con las cuatro vistas.

Una fotografía monocular sin objeto de tamaño conocido no determina escala
métrica. Para estimar también la estatura hay que añadir una referencia (p. ej.
marcador impreso medido), calibrar cámara/suelo o pedir la estatura. En el MVP,
pedir estatura es la opción más defendible.

## Modelos preentrenados: decisión por módulo

| Módulo | Elección | Uso correcto en Sastre-IA | Decisión RTX 3050 6 GB |
|---|---|---|---|
| Segmentación | SAM 2.1 tiny/small | RGB a máscara, con caja/clic de persona | Sí, entorno separado; no mejora por sí mismo la regresión sobre máscaras ya limpias |
| Pose/QC | RTMPose-s o RTMPose-m | Rechazar brazos pegados, pose torcida, cuerpo cortado; normalizar encuadre | Sí; no usar todavía como sustituto de profundidad o volumen |
| Medidas | EfficientNet-B0 multivista + metadatos | Aprender residuo visual sobre prior antropométrico | Sí; es la línea principal |
| Forma 3D | HMR2/4DHumans o SMPL-X fitting | Demo de malla e investigación; escalar con estatura | Después; no hacer depender el MVP de esto |
| Virtual try-on | CatVTON | Previsualización frontal de saco/falda/pantalón | Colab recomendado; el repositorio declara ~8 GB a 1024x768, por encima de 6 GB |
| Moldes | GarmentCode/PyGarment + plantillas propias | Componentes paramétricos, costuras y simulación | Sí, CPU/entorno separado; hay que implementar y validar el bloque sastre específico |
| Exportación 2D | OpenPattern o generador SVG propio | Curvas, pinzas, piquetes y salida 1:1 | Sí; útil para prototipo, no contiene automáticamente un saco formal validado |

Descartes por ahora:

- Sapiens2: el menor modelo por tarea es grande y su entorno moderno obligaría a
  migrar el stack estable; no resuelve directamente medidas métricas.
- Humanet: sirve como referencia académica, pero su stack SMPL/CUDA es antiguo.
- DeepFashion-MultiModal: útil para parsing, pose y apariencia, pero no contiene
  objetivos métricos para entrenar circunferencias.
- RTMPose extraído de las siluetas de BodyM: experimento de alto riesgo por cambio
  de dominio y baja información incremental.

## Plan experimental inmediato

### Exp_006: peso real ingresado

- Entradas: frontal, lateral, estatura, sexo y peso.
- Misma arquitectura y Huber de exp_002 para que la ablación cambie una sola cosa.
- Estadísticas de peso calculadas solo en train.
- Criterio de promoción: Test-B por sujeto menor de 1.651 cm (Ridge con peso) y sin
  empeorar pecho/cintura/cadera. Si no supera al Ridge, no se añade complejidad.
- Reportar también un modo sin peso usando exp_002; no inventar el peso faltante.

### Exp_007: peso estimado sin fuga de información

Solo si el producto debe funcionar sin balanza:

1. Entrenar una cabeza auxiliar para peso con train.
2. Generar predicciones de peso out-of-fold para cada sujeto de train.
3. Entrenar el regresor de medidas con peso predicho, no con peso real perfecto.
4. En validación/test usar exclusivamente el peso estimado.
5. Comparar tres columnas: sin peso, peso estimado y peso real.

Entrenar con peso real y luego sustituirlo por una predicción ruidosa produciría
una evaluación optimista. Como alternativa simple, inyectar ruido calibrado al
peso real durante train, aunque out-of-fold es metodológicamente más sólido.

### Datos locales

Recoger al menos un piloto de 30–50 participantes, con cuatro vistas, estatura,
peso y medidas tomadas dos veces por un patronista. Separar siempre por persona.
Guardar consentimiento, protocolo de cámara y versión de las reglas de medición.
Para una tesis, este conjunto local es más valioso que añadir otra red sin datos
equivalentes al uso real.

## Patronaje restringido y demostrable

Alcance recomendado:

- Hombre: saco de una fila y pantalón formal recto.
- Mujer: saco de una fila y falda recta formal.
- Una tela de traje configurada por preset; entretela/forro quedan documentados,
  pero fuera del cálculo inicial si no se cortan en la demo.

Cada patrón debe registrar: medidas fuente, holguras, margen de costura, línea de
hilo, doblez, piquetes, nombre/cantidad de piezas y escala. Antes de decir
"molde listo", verificar cierre de costuras emparejadas y dimensiones con pruebas
unitarias, exportar a escala 1:1 y validar al menos una toile física.

GarmentCode es la mejor base de investigación para patrones paramétricos y
simulación, pero no reemplaza las fórmulas de un método de sastrería ni la revisión
de un experto. OpenPattern es útil para construir/exportar geometría 2D y ofrece
bases de falda y pantalón; el saco formal requerirá una plantilla propia.

## Ejecución siguiente

```powershell
conda activate sastre-ia
python -m pytest tests -q
python src/training/train.py --config configs/exp_006_weight_huber.yaml `
  --exp exp_006_weight_huber
python src/evaluation/evaluate.py --config configs/exp_006_weight_huber.yaml `
  --exp experiments/exp_006_weight_huber --split test
python src/evaluation/evaluate.py --config configs/exp_006_weight_huber.yaml `
  --exp experiments/exp_006_weight_huber --split test_wild
```

## Fuentes primarias

- BodyM y protocolo: https://diw9s4r18eiyi.cloudfront.net/bodym/
- Artículo BMnet: https://assets.amazon.science/42/62/b1b5b44f4b5eaccf8c7d2475f734/human-body-measurement-estimation-with-adversarial-augmentation.pdf
- SAM 2: https://github.com/facebookresearch/sam2
- RTMPose: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- 4DHumans/HMR2: https://github.com/shubham-goel/4D-Humans
- SHAPY/HBW: https://shapy.is.tue.mpg.de/datasets.html
- GarmentCode: https://github.com/maria-korosteleva/GarmentCode
- OpenPattern: https://openpattern.readthedocs.io/en/latest/
- CatVTON: https://github.com/Zheng-Chong/CatVTON
- DressCode: https://github.com/aimagelab/dress-code
