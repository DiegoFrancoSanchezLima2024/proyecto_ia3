# Brief Fase 2-3: medición y vestidor — con correcciones al ROADMAP_TECNICO

Fecha: 2026-09-15 · Complementa y **corrige en dos puntos** a `docs/ROADMAP_TECNICO.md`.
Todo lo de aquí está verificado contra el código y los datos reales del repositorio,
no contra suposiciones. Cada afirmación indica cómo comprobarla.

---

## 0. El número que reencuadra el proyecto

Se midió cuánto aporta **todo** el pipeline visual (MediaPipe + SAM 2.1 +
EfficientNet-B0 + Transformer) comparándolo contra una regresión lineal trivial
que solo usa **estatura, peso y sexo — sin mirar ni una foto**.

Evaluación honesta: 87 sujetos de `predictions_test.csv`, agregados por sujeto,
validación cruzada 2-fold disjunta por sujeto para la línea base.

| | MAE agregado (13 medidas) |
|---|---|
| Modelo completo con fotos | **1.242 cm** |
| Fórmula lineal sin fotos (estatura+peso+sexo) | **1.544 cm** |
| **Lo que aportan las fotos** | **0.30 cm (19 %)** |

Por medida, la ganancia del pipeline visual sobre la fórmula sin fotos:

| Medida | Modelo | Sin fotos | Gana | |
|---|---|---|---|---|
| calf | 0.99 | 1.70 | +0.71 | +42 % |
| hip | 1.54 | 2.13 | +0.59 | +28 % |
| leg-length | 1.55 | 2.04 | +0.49 | +24 % |
| ankle | 0.65 | 0.84 | +0.18 | +22 % |
| bicep | 1.02 | 1.29 | +0.27 | +21 % |
| forearm | 0.60 | 0.75 | +0.15 | +20 % |
| chest | 2.48 | 3.06 | +0.58 | +19 % |
| shoulder-to-crotch | 1.12 | 1.38 | +0.27 | +19 % |
| arm-length | 1.02 | 1.24 | +0.22 | +18 % |
| waist | 2.60 | 2.90 | +0.31 | +11 % |
| wrist | 0.50 | 0.55 | +0.05 | +8 % |
| shoulder-breadth | 0.65 | 0.69 | +0.05 | +7 % |
| thigh | 1.43 | 1.49 | +0.06 | +4 % |

**Lectura honesta:** los modelos no son relleno — toda medida mejora — pero el
pipeline visual completo compra 0.30 cm. El grueso de la precisión viene de tres
números que el usuario teclea. Esto no invalida el proyecto: lo reencuadra. Y es
una contribución metodológica fuerte, porque casi nadie publica esta ablación.

Reproducir: el script está en el historial de la sesión; son ~90 líneas de Python
puro sobre `experiments/exp_006_weight_huber/predictions_test.csv`.

---

## 1. Correcciones al ROADMAP_TECNICO

### 1.1 CORRECCIÓN IMPORTANTE — la Fase 3 punto 11 apunta a código muerto

El roadmap afirma que el modelo devuelve el prior antropométrico porque la
arquitectura es residual (`m_hat = m_geo + Δm`) y `HEIGHT_PRIORS` da fracciones
fijas de estatura. **Eso no aplica al modelo en producción.**

Evidencia:

```
experiments/exp_006_weight_huber/config_resolved.yaml:61:  use_geometry: false
src/training/train.py:198:  prediction = delta if geometry is None else geometry + delta
src/evaluation/evaluate.py:145: prediction_norm = delta if geometry_norm is None else ...
```

Con `use_geometry: false`, `geometry is None`, así que `prediction = delta`: la red
predice la medida normalizada **directamente**. `HEIGHT_PRIORS` nunca se suma.
Además `src/inference/predict_four_views.py` (la ruta activa) **no importa**
`EstimadorGeometrico`; solo lo hace `predict_measurements.py`, que no se usa.

La coincidencia aritmética del roadmap (48.68 − 1.19 = 47.49) es casual.

**Consecuencia para el agente:** modificar `HEIGHT_PRIORS` en
`src/geometry/geometric_est.py` **no cambia nada** en la ruta activa. Si el agente
ejecuta esa instrucción tal cual, gasta días sin mover una métrica.

Lo que sí es cierto en espíritu: `arm-length` solo gana 0.22 cm (18 %) sobre la
fórmula sin fotos, así que en buena medida sí está siguiendo la media condicionada
por estatura. El arreglo correcto está en §3.2, y es otro.

### 1.2 CONFIRMADO — el MAE agregado esconde pecho y cintura

Verificado en `experiments/exp_006_weight_huber/results_test.json`:

| Medida | MAE | % dentro de 2 cm | Sesgo |
|---|---|---|---|
| waist | 2.83 cm | **38.3 %** | +2.00 |
| chest | 2.62 cm | **44.7 %** | −0.65 |
| hip | 1.72 cm | 64.2 % | +1.03 |
| wrist / ankle / forearm | 0.5–0.7 cm | 96–98 % | ~0 |

Las dos medidas que definen un saco son las peores del conjunto, y nunca
cumplieron ±2 cm ni siquiera dentro de distribución. El agregado de 1.24 cm está
arrastrado por muñeca, tobillo y antebrazo. **Reportar siempre por medida.**

### 1.3 PARCIAL — los defectos del vestidor: el diagnóstico es correcto, el archivo no

El roadmap culpa a `scripts/tryon_geometry.py:156`
(`ImageChops.multiply(lower, silhouette)`). Verificado: esa línea vive dentro de
`build_parsed_region_masks`, que **solo** usan `build_layered_tryon_preview.py`
(el compositor de sprites 2D) y `tryon_quality_checks.py`. **La ruta generativa
FASHN no la toca.**

El recorte equivalente en la ruta real está en `scripts/composite_tryon_identity.py`:

```python
130: garment_region = silhouette.filter(ImageFilter.MaxFilter(kernel)).filter(GaussianBlur(2.0))
131: constrained = Image.composite(generated, original, garment_region)
```

Fuera de `garment_region` se restaura la **foto original**. Y el servidor pasa
`--dilation 3` (`pattern-engine/server.mjs:204`), no el default 13 del script: el
kernel real es 3, o sea ~1 px de holgura. Todo lo que el generador dibujó más allá
del cuerpo desnudo + 1 px se descarta. **Ese es el efecto pegatina y la razón por
la que un hombro estructurado de saco no puede existir.** El diagnóstico del
roadmap es correcto; el archivo a tocar es este.

La franja negra **sí** está donde dice el roadmap y sigue presente:

```python
143: draw.rectangle((0, 0, size[0], head_bottom), fill=255)
```

---

## 2. Veredicto por modelo: qué trabaja y qué es peso muerto

| Componente | VRAM | ¿Trabaja? | Veredicto |
|---|---|---|---|
| MediaPipe Pose | 0 (CPU) | Sí: valida pose/encuadre, da landmarks | Mantener. Subir a Heavy si se usan landmarks para medir (§3.2) |
| SAM 2.1 | ~1 GB | Sí: produce las siluetas que come el CNN | Mantener, pero su aporte está acotado por los 0.30 cm |
| EfficientNet-B0 + Transformer | ~1.5 GB | Sí, pero aporta 0.30 cm sobre la fórmula trivial | Mantener arquitectura. **No cambiar a un encoder mayor**: el cuello de botella es señal, no capacidad |
| Filtro ANSUR II | 0 | Sí: marca atípicos | Mantener |
| Conformal | 0 | Sí, ya conectado en Fase 1 | Mantener |
| SCHP-ATR | 0 (CPU) | Solo para el vestidor | **Subutilizado**: no llega a medición (§3.1) |
| SHAPY / SMPL-X | 6.2 GB en disco | **No corre** (`-SkipBody3D`) | **Peso muerto hoy.** Activar como segunda opinión o retirar del entregable |
| Compositor sprites 2D | 0 | Sí, pero calidad con techo bajo | Degradar a diagnóstico; no es el vestidor del demo |
| `geometric_est.py` | 0 | **No corre** (`use_geometry: false`) | Código muerto en la ruta activa |
| FASHN VTON 1.5 | 2.7 GB medido | Sí: es el vestidor real | Mantener. Los defectos son de post-proceso, no del modelo |

---

## 3. Qué sí mueve la aguja, en orden de retorno

### 3.1 Ropa (el que más error elimina en uso real)

En el dataset limpio la cintura da 2.83 cm; en la foto real del audit dio +4.7 cm.
Esa diferencia es ropa, no modelo. Tres capas, de barata a cara:

1. **Protocolo obligatorio, no sugerido.** La UI ya recibe `clothing_fit`, pero el
   parámetro no tiene efecto real en el modelo. Rechazar la captura si la ropa es
   holgada, en vez de medir y avisar después.
2. **SCHP-ATR aplicado a medición.** Ya se descarga, ya corre, ya devuelve
   `upper_clothes`, `pants`, `skirt`, `left-leg`, `right-leg`, `arms`. Con eso:
   medir muslo sobre pierna desnuda visible, calcular fracción de silueta cubierta
   por prenda y rechazar por encima de un umbral. **No requiere entrenar nada.**
3. **Aumentación de ropa en entrenamiento.** Dilatación anisotrópica aleatoria de
   las siluetas (más en torso que en extremidades), condicionada en `clothing_fit`.
   Sin datos nuevos.

### 3.2 Largos desde landmarks — pero en inferencia, no en `geometric_est.py`

El objetivo del roadmap es correcto; la vía no. Con `use_geometry: false`, tocar
`geometric_est.py` no hace nada. La vía que sí funciona:

Calcular en **`predict_four_views.py`** (ruta activa) los largos directamente como
distancias entre landmarks de MediaPipe escaladas por la estatura confirmada, y
compararlos por medida contra la salida del CNN sobre el conjunto local.

- Afecta a: `arm-length`, `leg-length`, `shoulder-breadth`, `shoulder-to-crotch`.
- Ojo metrológico: MediaPipe da centros de articulación; la cinta del sastre pasa
  por encima del hombro y por fuera del brazo. Hay un **offset sistemático** por
  medida. Se calibra como constante sobre el conjunto local y se documenta.
- Criterio de aceptación: quedarse, por medida, con el método de menor MAE en el
  conjunto local. No sustituir a ciegas.

Por qué puede ganar: para el sujeto real, la cinta dio 54 cm y el modelo 47.49 cm.
Una distancia hombro-muñeca medida sobre los landmarks de **esa** persona es
específica por construcción; el CNN, con 18 % de aporte visual, tiende a la media.

### 3.3 Corrección de sesgo post-hoc (barato, medido)

Verificado con validación 2-fold disjunta por sujeto sobre el propio test:

| Medida | MAE actual | MAE tras restar sesgo | Gana |
|---|---|---|---|
| waist | 2.60 | **2.16** | +0.43 |
| arm-length | 1.02 | 0.93 | +0.10 |
| hip | 1.54 | 1.45 | +0.09 |
| agregado | 1.242 | **1.186** | +0.06 |

Restar el sesgo por medida —estimado en el split de calibración, nunca en test—
baja la cintura de 2.60 a 2.16 cm. Es gratis y legítimo. No es un game changer,
pero la cintura es una de las dos que definen el saco.

**Ya está implementado.** Falta un solo paso, que necesita GPU y por eso lo
corre el usuario o el agente:

```bash
conda activate sastre-ia
python -m src.evaluation.evaluate --exp experiments/exp_006_weight_huber --split calibration --config configs/exp_006_weight_huber.yaml
python scripts/fit_bias_correction.py --exp-dir experiments/exp_006_weight_huber
```

Ojo con los flags: `evaluate` usa `--exp` y `fit_bias_correction` usa
`--exp-dir`. No son iguales. Y hay que estar en el entorno `sastre-ia`;
en `base` no hay PyTorch.

El primer comando genera `predictions_calibration.csv`; el segundo mide el
sesgo medio por medida y escribe `experiments/exp_006_weight_huber/bias_correction.json`.
A partir de ahí `src/inference/predict_measurements.py` lo carga solo si el
archivo existe y lo resta antes de armar los intervalos, así que el
comportamiento sin el archivo es exactamente el de antes.

Tres garantías que conviene poder explicar en la defensa:

1. El script **se niega** a estimar el sesgo sobre `predictions_test*.csv`. Si
   se midiera en test, las métricas de test dejarían de ser honestas.
2. Solo corrige medidas cuyo sesgo supera a la vez 0.2 cm y dos errores
   estándar, así no se ajusta ruido. En el test de ejemplo eso acepta cintura,
   cadera, muslo, largo de brazo y ancho de hombros, y rechaza pecho, pantorrilla,
   tobillo, antebrazo, muñeca y bíceps.
3. Los intervalos conformales se calibraron sobre predicciones sin corregir, de
   modo que siguen siendo válidos: quedan más anchos de lo necesario, nunca más
   angostos. La cobertura del 90% no se rompe.

La corrección aplicada viaja en el JSON de salida como `bias_correction_cm`,
para que quede auditable qué se restó a cada medida.

Cubierto por `tests/test_bias_correction.py`.

**Resultado verificado (15/09/2026).** Sesgo ajustado sobre los 303 sujetos del
split de calibración y aplicado a los 87 sujetos de test, que el modelo nunca
vio. Es la comparación honesta: nada del sesgo se estimó con datos de test.

Sesgos aceptados: cintura +1.01, largo de pierna +0.52, cadera +0.34, largo de
brazo +0.29, pantorrilla -0.29 cm. Rechazados por no superar el filtro: pecho,
muslo, tobillo, antebrazo, muñeca, bíceps, ancho de hombros y hombro-entrepierna.

| Medida | MAE antes | MAE después | Dentro de ±2 cm |
|---|---|---|---|
| cintura | 2.60 | **2.16** | 37.9% → **44.8%** |
| cadera | 1.54 | **1.47** | 66.7% → **72.4%** |
| largo de brazo | 1.02 | **0.95** | — |
| agregado | 1.242 | **1.195** | — |

Lo importante para la defensa no es el MAE agregado sino el porcentaje dentro de
tolerancia: casi siete puntos más de cinturas utilizables sin reentrenar nada.
El resultado coincide con la estimación previa por validación 2-fold (2.16 cm
exacto), así que el método es estable y no fue casualidad del split.

La pantorrilla empeoró 0.01 cm. Es ruido, pero conviene poder mencionarlo antes
de que lo pregunten. El pecho no se corrigió porque su sesgo en calibración
(+0.32 cm, error estándar 0.19) no superó los dos errores estándar: corregirlo
habría sido ajustar ruido.

### 3.4 SHAPY como segunda opinión (opcional, alto coste)

6.2 GB ya descargados sin usar. Valor real: definiciones de medida consistentes
sobre malla, mejor tolerancia a ropa, y sobre todo **desacuerdo entre dos métodos
independientes como señal de incertidumbre**, que es más honesto que la `log_var`
de una sola red. Cabe en 6 GB (~2–3 GB). No esperar que baje el MAE por sí solo.

---

## 4. La estatura: la respuesta es no, y con números

Pediste estimarla desde la foto. No lo hagas, por una razón aritmética propia de
tu pipeline, no por pesimismo:

La escala métrica de todo el sistema es `estatura_cm / altura_en_píxeles`. Un error
de 3 cm sobre 165 cm es **1.8 % de error de escala multiplicativo** que se propaga
a **todas** las medidas: ~1.8 cm sobre un pecho de 100 cm, **encima** de los 2.6 cm
que ya tienes. Se come el presupuesto de ±2 cm por sí solo.

La estimación monocular de estatura sin referencia calibrada está en el rango de
varios centímetros; los mejores resultados publicados requieren infraestructura
(líneas de suelo, referencia vertical de altura conocida, puntos de fuga). No es
una limitación de tu máquina: es del problema.

**Dos alternativas honestas, ambas eliminan la cinta métrica:**

1. **Estatura tecleada** (la actual). Se mide contra una pared en 10 segundos con
   ±0.5 cm. Máxima precisión, cero fricción. Es lo que hacen los sistemas
   comerciales de made-to-measure.
2. **Referencia física en la foto**: los marcadores ArUco de suelo de
   `src/calibration/metric.py` —que **ya funcionan**— o una hoja A4 / tarjeta de
   tamaño conocido en el encuadre. Da escala métrica real y además **verifica** la
   estatura declarada y rechaza inconsistencias.

La opción 2 es la más defendible en una tesis: "medición declarada + verificación
métrica independiente" es más riguroso que una estimación, y el código ya existe.

---

## 5. Vestidor: cuatro arreglos, ningún cambio de modelo

Los cinco síntomas (franja negra, fondo alterado, manos imperfectas, pantalón
corto, aspecto de pegatina) vienen de post-proceso. Cambiar FASHN por CatVTON no
arregla ninguno.

1. **Quitar el recorte a la silueta.** `composite_tryon_identity.py:130-131`. La
   máscara de prenda debe ser el cuerpo **dilatado según las holguras** de
   `configs/bespoke_garments.json` para esa prenda, no el cuerpo desnudo + 1 px.
   Esto además conecta el vestidor con el molde: la imagen empieza a reflejar la
   holgura real del patrón.
2. **Matar la franja negra.** Sustituir `draw.rectangle((0,0,size[0],head_bottom))`
   (línea 143) por la máscara semántica de cara/pelo que `human_parsing_schp.py`
   **ya calcula**, con feathering de 2–3 px.
3. **Manos por máscara semántica**, no por las dos elipses dibujadas a mano.
4. **Referencia femenina con blusa visible.** El fallo del audit femenino fue
   `upper_skin_leak_ok=false` (15.06 % de piel en el torso): la referencia tiene
   blazer abierto y el generador rellena el escote con piel. Probar una referencia
   de blazer cerrado antes de tocar nada más.

**Test de aceptación ya existente:** el detector estructural de la Fase 1 reporta
`rectangular_restore_detected: true` sobre el `final.png` actual. Cuando pase a
`false`, la franja está arreglada de verdad.

Sobre cambiar de modelo: CatVTON (basado en SD1.5-inpainting, ~50 M parámetros
entrenables) cabe holgado en 6 GB y es más rápido; puede añadirse como modo rápido.
Licencia **CC BY-NC-SA 4.0**: válida para tesis, **debe declararse**, y bloquea uso
comercial posterior. Pero primero los cuatro arreglos de arriba.

---

## 6. Orden de trabajo para el agente

Una fase a la vez, sin avanzar sin tests.

**FASE 2 — Vestidor (geometría, no modelo).** Puntos 1–4 de §5. Criterio de
aceptación: `rectangular_restore_detected == false`, y el porcentaje de píxeles
alterados fuera de la máscara de prenda (ya medido por la Fase 1) por debajo del
umbral que definas.

**FASE 3 — Medición, en este orden de retorno:**
1. SCHP-ATR conectado a medición + rechazo por cobertura de ropa (§3.1.2).
2. Corrección de sesgo post-hoc por medida, estimada en calibración (§3.3).
3. Largos desde landmarks en `predict_four_views.py`, con offset calibrado y
   comparación por medida contra el CNN (§3.2). **No tocar `geometric_est.py`.**
4. Aumentación de ropa en entrenamiento (§3.1.3).

**FASE 4 — Validación (lo que la tesis necesita de verdad):**
- Protocolo escrito de medición de referencia: punto anatómico y recorrido de cinta
  por cada una de las 13 medidas, alineado con la definición de BodyM. Sin esto no
  puedes distinguir error del modelo de desacuerdo de definición.
- 15–30 personas locales, ropa ceñida, **cinta tomada dos veces por dos personas**
  (eso te da tu piso de error real; no puedes reclamar precisión mejor que tu
  propio ground truth).
- Reportar **por medida**: MAE, % dentro de 1 cm, % dentro de 2 cm, sesgo.
- Recalibrar el conformal sobre ese conjunto local: ese intervalo es el número
  honesto que ve el usuario.

**No hacer:** cambiar el encoder por uno más grande, estimar estatura desde la foto,
tocar `HEIGHT_PRIORS`, cambiar de modelo generativo antes de la Fase 2.

---

## 7. Objetivo realista por familia de medida

Prometer ±1–2 cm en todo no se sostiene. Lo defendible:

| Familia | Medidas | Objetivo realista | Estado hoy |
|---|---|---|---|
| Largos y anchos | arm-length, leg-length, shoulder-breadth | ±1.0–1.5 cm | 0.65–1.55 cm ✔ |
| Contornos finos | wrist, ankle, forearm, bicep, calf | ±1.0 cm | 0.50–1.02 cm ✔ |
| Contornos gruesos | **chest, waist**, hip, thigh | ±2–3 cm con intervalo declarado | 1.43–2.60 cm ✖ para ±2 |
| Medidas de patrón | inclinación de hombro, tiro, cuello | No estimables desde foto | Confirmación humana |

Que pecho y cintura no lleguen a ±2 cm desde una foto de una persona vestida no es
un fallo del proyecto: es el estado del arte. Declararlo por medida, con intervalo
conformal y confirmación humana de las críticas, es más defendible ante un tribunal
que un número único que no se sostiene al primer contraejemplo.

## Nota: pytest y la carpeta temporal de Windows

En esta máquina `%TEMP%` tiene algo (OneDrive, antivirus, o un candado de una
corrida anterior) que bloquea la creación de subcarpetas ahí, y pytest usa esa
carpeta para sus fixtures `tmp_path`. Si `pytest tests -q` falla con
`PermissionError`/`FileNotFoundError` apuntando a `pytest-of-<usuario>`, correr:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\pytest-tmp" | Out-Null
$env:PYTEST_DEBUG_TEMPROOT = "$env:USERPROFILE\pytest-tmp"
pytest tests -q
```

Confirmado el 15/09/2026: 94/94 tests pasan con esto, incluidos los que
ejercitan el pipeline recién traducido al español (identificadores de
`pattern-engine/` y de las funciones/clases públicas de `src/`/`scripts/`).
