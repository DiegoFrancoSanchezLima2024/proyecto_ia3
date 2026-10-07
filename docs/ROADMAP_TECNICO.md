# SASTRE-IA — Auditoría técnica profunda y hoja de ruta

Fecha: 2026-09-15 · Hardware objetivo: RTX 3050 6 GB VRAM · Ejecución 100% local
Basado en lectura directa del código fuente, los resultados de `exp_006_weight_huber`
y el estado del arte publicado.

---

## 0. Veredicto en una página

**La infraestructura está bien construida. El problema no es que falten modelos:
es que los modelos correctos ya están en el repositorio y no están conectados a la
ruta activa.**

Cuatro componentes ya construidos y funcionando están siendo ignorados por el pipeline:

| Componente construido | Qué resuelve | Estado actual |
|---|---|---|
| `src/calibration/metric.py` | Escala métrica real vía ArUco de piso + solvePnP | Pausado ("investigación v2"), `run_assisted_demo` no lo usa |
| `src/geometry/geometric_est.py` | Largos desde geometría | Usado solo como *prior fijo*, no como medición |
| `src/training/conformal.py` | Intervalos con cobertura garantizada | No conectado al bloqueo de la UI |
| `scripts/human_parsing_schp.py` (SCHP-ATR) | Separar piel de prenda | Se ejecuta, pero solo para el vestidor, nunca para medir |

**Sobre el objetivo de ±1–2 cm:** es alcanzable para *largos* (brazo, pierna, hombro)
y no es alcanzable hoy para *contornos* (pecho, cintura) desde fotos de una persona
vestida. La sección 2 explica por qué y qué arquitectura sí lo logra parcialmente.
La sección 3 explica por qué el objetivo debe redefinirse por familia de medida en
lugar de como un número único, y por qué eso hace la tesis **más** defendible, no menos.

---

## 1. Qué está realmente roto (con evidencia)

### 1.1 El error de brazo y pierna no es ruido: el modelo nunca los mide

En `src/geometry/geometric_est.py`:

```python
HEIGHT_PRIORS = {
    "arm-length": 0.295,
    "leg-length": 0.445,
    ...
}
```

Estos son **fracciones fijas de la estatura**. Y en `src/models/regressor.py` la
arquitectura es residual: `m_hat = m_geo + Δm`.

Haz la cuenta para tu sujeto de 165 cm:

| Medida | Prior fijo (`0.295 × 165`) | Predicción real del modelo | Δm aprendido |
|---|---:|---:|---:|
| arm-length | 48.68 cm | **47.49 cm** | −1.19 cm |
| leg-length | 73.43 cm | **73.98 cm** | +0.55 cm |

**El modelo está devolviendo esencialmente el prior antropométrico con un empujón
mínimo.** No está midiendo el brazo: está prediciendo el promedio poblacional.
Tu cinta dice 54 cm (ratio 0.327), o sea que estás por encima de la media — y un
modelo que predice la media nunca te puede alcanzar. De ahí los −6.5 cm.

Esto también explica una trampa estadística importante en tus propias métricas:

```
arm-length en BodyM Test-A:  MAE 1.02 cm, 85.9% dentro de 2 cm
```

Ese 1.02 cm **no es evidencia de que el modelo mida brazos**. El largo de brazo
correlaciona ~0.9 con la estatura en una población; predecir `0.295 × altura`
obtiene buen MAE sin extraer un solo píxel de información de la silueta. El modelo
descubrió ese atajo porque la silueta, en pose A con brazos pegados al cuerpo,
no contiene señal utilizable de largo de brazo.

> **Consecuencia de diseño:** los largos NO deben salir de una CNN de siluetas.
> Son distancias punto a punto entre articulaciones que MediaPipe ya localiza
> (hombro 11/12, codo 13/14, muñeca 15/16, cadera 23/24, rodilla 25/26, tobillo 27/28).
> Con escala métrica, eso es prácticamente una medición directa.

### 1.2 El MAE agregado de 1.24 cm del README es engañoso

Los números reales por medida en `experiments/exp_006_weight_huber/results_test.json`:

| Medida | MAE (cm) | % dentro de 2 cm | Sesgo |
|---|---:|---:|---:|
| **waist** | **2.83** | **38.3%** | **+2.00** |
| **chest** | **2.62** | **44.7%** | −0.65 |
| hip | 1.72 | 64.2% | +1.03 |
| thigh | 1.47 | 72.9% | −0.48 |
| calf | 1.11 | 82.4% | −0.37 |
| arm-length | 1.02 | 85.9% | +0.43 |
| forearm | 0.68 | 95.9% | −0.07 |
| ankle | 0.71 | 96.7% | −0.06 |
| wrist | 0.52 | ~98% | — |

El agregado de 1.24 cm está arrastrado hacia abajo por muñeca, tobillo y antebrazo,
que son medidas pequeñas, fáciles y **antropométricamente irrelevantes para un saco**.
Las dos medidas que definen un saco —pecho y cintura— son las **peores del conjunto**,
con solo 38–45% de sujetos dentro de 2 cm, y eso **en el dataset limpio**.

**Traducción:** el modelo nunca cumplió ±2 cm en pecho y cintura, ni siquiera dentro
de distribución. El fallo en tu foto real no es solo brecha de dominio: es un techo
de precisión que ya existía en el benchmark.

Nota además el **sesgo de cintura de +2.00 cm en BodyM**. De tus +4.7 cm reales,
~2 cm son sesgo del modelo y ~2.7 cm son la ropa. Son dos problemas distintos que
requieren dos arreglos distintos.

### 1.3 El vestidor: un solo error geométrico explica casi todos los síntomas

En `scripts/composite_tryon_identity.py`:

```python
head_bottom = shoulder_y - shoulder_width * 0.07
draw.rectangle((0, 0, size[0], head_bottom), fill=255)
```

**Esa es la franja negra.** Restaura la imagen original en un rectángulo de ancho
completo desde el borde superior hasta justo encima de los hombros. En la línea
`head_bottom` hay una costura horizontal dura que cruza **toda la imagen**, donde
el original se encuentra con lo generado. Si el generador cambió el fondo o el
cuello, ahí aparece la discontinuidad.

En el mismo archivo:

```python
garment_region = silhouette.filter(ImageFilter.MaxFilter(kernel)).filter(GaussianBlur(2.0))
...
print(json.dumps({..., "background_preserved": True, ...}))
```

La silueta se **dilata 13 px y se difumina**, lo que empuja píxeles generados
~8 px hacia el fondo — y acto seguido el script **afirma en su JSON que el fondo
está preservado**. Ese `background_preserved: True` es una constante hardcodeada,
no una verificación. Es el mismo patrón de falso positivo que detectaste en el QA
de sprites, pero incrustado en el compositor.

En `scripts/tryon_geometry.py`, línea 156:

```python
lower = ImageChops.multiply(lower, silhouette)
```

**Este es el techo que hace que todo parezca pegatina.** La máscara de la prenda se
intersecta con la silueta del cuerpo. Resultado: **una prenda generada nunca puede
ser más ancha que el cuerpo desnudo + 13 px de dilatación.** Pero un saco formal
tiene hombro estructurado y caída; un pantalón tiene holgura. Geométricamente les
estás prohibiendo existir. De ahí el aspecto pegado y el pantalón que se ve mal
en las piernas.

**Los cinco síntomas — franja negra, fondo modificado, manos imperfectas, pantalón
corto, aspecto de pegatina — vienen de máscaras mal construidas, no del modelo
generativo.** Cambiar FASHN por CatVTON no arregla ninguno.

### 1.4 El patrón de fondo

SCHP-ATR **ya se está ejecutando** y ya produce máscaras semánticas de cara, pelo,
brazos y piernas (`scripts/human_parsing_schp.py` devuelve `arms`, `hands`, `labels`).
El compositor las ignora y dibuja un rectángulo y dos elipses a mano.

Lo mismo en el lado de medición: SCHP-ATR sabe **dónde termina el short y empieza
la pierna desnuda**, y esa información —que resolvería directamente el sesgo de ropa—
no llega nunca al estimador de medidas.

---

## 2. La arquitectura correcta

### 2.1 Estatura: deja de estimarla

Esta es la recomendación más importante del documento y va en contra de lo que pediste.

La literatura de estimación monocular de estatura reporta:

| Método | Error | Requisito |
|---|---:|---|
| Guan et al. 2009 | 5.75 cm | — |
| Sakina et al. 2024 | 3.10 cm | — |
| Nguyen Trung et al. 2024 | 2.24 cm | — |
| Chen et al. 2025 | 1.92 cm | — |
| SegPose + calibración ortogonal (2026, SOTA) | **1.42 cm** | **Requiere infraestructura calibrada: líneas de suelo, referencia vertical de 2.44 m conocida, puntos de fuga** |

El mejor resultado publicado necesita un **arco de portería de altura conocida y
líneas del campo**. Sin esa calibración, el estado del arte está en 2–6 cm.

Ahora: en tu pipeline la estatura es el **ancla métrica de todo lo demás**
(`front_scale = height_cm / front_height` en `geometric_est.py`). Un error de 3 cm
en estatura es un error de escala del 1.8% que se propaga multiplicativamente a
**todas** las medidas — 1.8 cm sobre un pecho de 100 cm, antes de cualquier otro error.

**Estimar la estatura por foto destruye el presupuesto de error de ±2 cm por sí sola.**

La estatura es el único dato que un usuario puede medir en 10 segundos contra una
pared con precisión de ±0.5 cm. **Pídesela.** Es la decisión de ingeniería correcta,
no una concesión.

Lo que sí puedes hacer, y es defendible en la tesis: usar `calibration/metric.py`
(que ya funciona) con marcadores ArUco de piso para **verificar** la estatura
declarada y rechazar entradas inconsistentes. Medición declarada + verificación
métrica independiente es más riguroso que una estimación, y ya tienes el código.

### 2.2 Tres familias de medidas, tres métodos

El error de diseño actual es tratar 13 medidas como un solo problema de regresión.
Son tres problemas distintos:

**Familia A — Largos y anchos (geométricos).**
`hombro-muñeca`, `largo de pierna`, `ancho de hombros`, `largo de espalda`, `tiro`.
Son distancias entre puntos que MediaPipe localiza. Método: landmarks 3D
(`pose_world_landmarks`) + escala métrica desde la estatura confirmada.
Determinista, explicable, auditable.
**Objetivo realista: ±1.0–1.5 cm.** Alcanzable.

**Familia B — Contornos (aprendidos).**
`pecho`, `cintura`, `cadera`, `muslo`, `bíceps`, `cuello`.
Requieren inferir profundidad y curvatura. Método: CNN multivista + prior elíptico
+ corrección por ropa. Aquí está el techo real.
**Objetivo realista: ±2.0–3.0 cm con intervalo conformal reportado.** No prometas ±1 cm.

**Familia C — Medidas de patrón que ninguna foto da.**
`inclinación de hombro`, `profundidad de tiro`, `contorno de cuello`.
Método: confirmación del usuario, o marcadas explícitamente como derivadas.
Nunca inventadas silenciosamente.

Esta separación es la contribución metodológica más fuerte que puedes defender:
**no todas las medidas corporales son igual de estimables desde una foto, y un
sistema honesto lo declara por medida.**

### 2.3 El arreglo de la ropa (el que más error elimina)

Tu mayor error individual de contorno es +4.84 cm en pecho, causado por la camiseta.
Hay tres capas de arreglo, en orden de costo/beneficio:

**(a) Protocolo de captura — gratis, el más efectivo.**
3DLOOK, que es un sistema comercial en producción, exige explícitamente ropa ceñida
y declara que *"nuestra IA no puede medir de forma fiable cuerpos bajo ropa holgada"*.
Sus requisitos: leggings o ropa ajustada, pose A, cámara a ~2.1 m a la altura de la
cintura, pelo recogido, sin zapatos, sin accesorios, fondo contrastado.
Tu UI debe **imponer y verificar** esto, no sugerirlo. Tu propia captura de auditoría
violó el requisito principal.

**(b) Usar SCHP-ATR para separar piel de prenda — ya está descargado y corriendo.**
ATR etiqueta `upper-clothes`, `pants`, `skirt`, `left-leg`, `right-leg`, `left-arm`,
`right-arm`, `face`, `hair`. Con eso puedes:
- medir el muslo en la **pierna desnuda visible** bajo el short, no en el short;
- detectar cuánta prenda hay y en qué regiones;
- rechazar la captura si `upper-clothes` cubre demasiada área respecto a la silueta.

Esto es la mejora de medición de mayor retorno disponible y no requiere entrenar nada.

**(c) Aumentación de ropa en entrenamiento — barato, sin datos nuevos.**
BodyM tiene sujetos en ropa ceñida gris; tus usuarios no. Dilata sintéticamente las
siluetas de entrenamiento de forma anisotrópica y aleatoria (más en torso, menos en
extremidades) para simular prendas holgadas, condicionando en el parámetro
`ClothingFit` que **ya existe en tu CLI pero no hace nada real**. El modelo aprende
la corrección en vez de sufrirla.

### 2.4 Desbloquear el 3D que ya pagaste

Tienes 6.2 GB de `body3d` y 160 MB de `smplx` descargados, con readiness completo,
y el servidor pasa `-SkipBody3D`. SHAPY/SMPL-X entra en 6 GB.

El valor no es cosmético: de una malla SMPL-X extraes **todas** las medidas con
definiciones geométricas consistentes (`src/body3d/mesh_measurements.py` ya existe),
lo que resuelve de raíz el problema de "definiciones de cinta vs dataset" que
mencionaste. Además SHAPY está entrenado sobre imágenes de personas vestidas en
entornos reales, así que tolera la ropa mejor que una silueta pura.

Úsalo como **segunda opinión**: cuando SHAPY y la CNN discrepen más de X cm en una
medida, esa medida se marca como no confiable. Desacuerdo entre dos métodos
independientes es una señal de incertidumbre mucho más honesta que la `log_var`
de una sola red.

### 2.5 Vestidor: arreglar geometría antes de cambiar modelo

Orden correcto de trabajo:

1. **Máscara de prenda = cuerpo + holgura del patrón, no cuerpo ∩ silueta.**
   Elimina `ImageChops.multiply(lower, silhouette)`. Construye la máscara dilatando
   la región corporal según las holguras que ya tienes en `configs/bespoke_garments.json`.
   Esto conecta el vestidor con el molde: la imagen pasa a reflejar la holgura real
   del patrón, que es exactamente la "relación métrica" que dijiste que falta.
2. **Restauración de identidad por máscara semántica, no por rectángulo.**
   Usa las salidas de SCHP-ATR (`face`, `hair`, `hands`, `arms`) que ya se calculan.
   Borra el `draw.rectangle`. Fin de la franja negra.
3. **Fondo: componer con la silueta sin dilatar.**
   `final = original*(1−M) + generado*M` con `M` = región de prenda, y la dilatación
   aplicada solo hacia dentro de la prenda. Y que `background_preserved` se **verifique**
   comparando píxeles fuera de M, no se declare.
4. **Pantalón hasta el tobillo:** la máscara inferior debe llegar a los landmarks
   27/28 con ancho de prenda, no de pierna desnuda.
5. **Zapatos: desactivados** hasta tener assets RGBA limpios.

Solo después de esto tiene sentido evaluar cambiar de modelo.

---

## 3. Modelos recomendados para RTX 3050 6 GB

### Medición

| Etapa | Recomendación | Por qué | VRAM |
|---|---|---|---|
| Pose | **MediaPipe Pose Heavy** (subir desde Lite) + `pose_world_landmarks` | Los landmarks pasan a ser portadores de medida (Familia A), la precisión ahora importa. Corre en CPU. | 0 (CPU) |
| Segmentación | **SAM 2.1 Hiera Small** — mantener | Funciona bien, no es el cuello de botella | ~1.0 GB |
| Parsing | **SCHP-ATR INT8** — mantener, pero **conectar a medición** | Separa piel de prenda; hoy se desperdicia | 0 (CPU) |
| Contornos | **EfficientNet-B0 multivista** — mantener arquitectura, cambiar *entrada y entrenamiento* | El encoder no es el problema; la señal lo es. Cambiar a ConvNeXt no arregla un atajo estadístico. | ~1.5 GB |
| Largos | **Nuevo: estimador geométrico de landmarks** (reemplaza los priors fijos) | Elimina la regresión a la media | 0 |
| Verificación 3D | **SHAPY / SMPL-X** — activar como segunda opinión | Definiciones consistentes + tolera ropa | ~2–3 GB |
| Incertidumbre | **`CalibradorConforme`** — conectar a la UI | Ya escrito, cobertura garantizada | 0 |

> **No cambies el encoder.** La tentación es pasar a un modelo más grande. Tu problema
> no es capacidad: es que la silueta de un brazo pegado al torso no contiene el largo
> del brazo. Ninguna arquitectura extrae información que no está en el píxel.

### Vestidor

| Modelo | Params | VRAM @576×768 fp16 | Tiempo est. 3050 | Licencia | Veredicto |
|---|---:|---:|---:|---|---|
| **FASHN VTON 1.5** (actual) | — | ~2.7 GB medido | ~180 s | Revisar | **Mantener como calidad máxima** |
| **CatVTON** | 899 M total / 49.6 M entrenables | <8 GB @1024×768, holgado a 576×768 | ~20–40 s | **CC BY-NC-SA 4.0** | **Añadir como modo rápido** (ya tienes `tryon-catvton-direct`) |
| IDM-VTON | SDXL | Alto | Lento | CC BY-NC-SA 4.0 | Descartar: SDXL es pesado para 6 GB |
| OOTDiffusion | — | — | ~46 s | — | **Descartar: no soporta prendas inferiores** |

CatVTON está basado en SD1.5-inpainting, soporta prenda superior e inferior, y no
requiere obligatoriamente DensePose ni parsing (solo persona + prenda + máscara).
Es la opción correcta para iteración rápida en tu GPU.

> **Aviso de licencia relevante para tu tesis:** CatVTON e IDM-VTON son
> **CC BY-NC-SA 4.0 — prohíben el uso comercial** sin licencia aparte. Para un
> proyecto de grado es correcto, pero debe declararse explícitamente en el documento,
> y bloquea una comercialización posterior sin sustituir el modelo.

---

## 4. Plan de validación (lo que la tesis necesita de verdad)

Sin esto, cualquier mejora es una corazonada. Con esto, tienes resultados publicables.

1. **Protocolo de medición de referencia escrito.** Define cada una de las 13 medidas
   con el punto anatómico exacto y el recorrido de la cinta, **alineado con la
   definición de BodyM**. Tu discrepancia de brazo probablemente tiene 2–4 cm de
   desalineación de definición además del error de modelo. Sin esto no puedes
   distinguir error metrológico de desacuerdo de definición.
2. **Conjunto local de 15–30 personas**, mixto hombre/mujer, con:
   - 3 fotos en ropa ceñida siguiendo el protocolo,
   - medidas de cinta tomadas **dos veces por dos personas** (esto te da la
     repetibilidad de la cinta, que es tu piso de error — típicamente ±0.5–1 cm;
     no puedes reclamar precisión mejor que tu propio ground truth),
   - estatura medida contra pared.
3. **Reportar por medida, nunca agregado.** MAE, % dentro de 1 cm, % dentro de 2 cm,
   y sesgo. El agregado oculta que pecho y cintura son las peores.
4. **Calibrar el `CalibradorConforme` sobre ese conjunto local** y usar su `q_hat`
   como el intervalo real mostrado al usuario. Es el número honesto.
5. **Gating duro:** `manual_confirmation` o `ready=false` ⇒ la UI no emite molde de
   corte, solo `BORRADOR NO VALIDADO`.

---

## 5. Brief para tu agente de código

> Copia desde aquí. Está ordenado por dependencia: cada fase desbloquea la siguiente.
> **Pídele que haga una fase a la vez y que no avance sin tests.**

---

**Contexto para el agente:**

Trabajas en `proyecto-sastre-ia`, un sistema local de medición corporal y patronaje.
Una auditoría encontró que varios módulos correctos ya existen en el repositorio pero
no están conectados a la ruta de ejecución activa. Tu trabajo es conectarlos y corregir
errores geométricos concretos. **No entrenes modelos nuevos ni cambies arquitecturas
de red hasta la Fase 4.** No inventes proporciones antropométricas para rellenar
medidas faltantes.

**FASE 1 — Gating de integridad (bloqueante, hacer primero)**

1. En el servidor de `pattern-engine`, bloquear la acción "Generar molde" cuando
   `decision == "manual_confirmation"` o `decision == "repeat_capture"` o el auditor
   estricto devuelve `ready == false`. En esos casos solo se permite exportar con la
   marca `BORRADOR NO VALIDADO` en el SVG y en el PDF.
2. Conectar `src/training/conformal.py` (`CalibradorConforme`) a la ruta de inferencia.
   Cada medida devuelta debe llevar su intervalo `[pred − q_hat, pred + q_hat]` y su
   bandera `is_reliable`. La UI muestra el intervalo junto al valor, siempre.
3. En `scripts/composite_tryon_identity.py`, eliminar los campos hardcodeados
   `"background_preserved": True`, `"head_preserved": True`, `"hands_preserved": True`.
   Sustituirlos por verificaciones reales: comparar píxeles fuera de la máscara de
   prenda contra el original y reportar el porcentaje de píxeles alterados.
4. En el QA de sprites 2D, renombrar el estado `bueno` a `tecnicamente_compuesto`
   hasta que exista una métrica de realismo. Un QA que no mide realismo no puede
   emitir un veredicto de calidad.

**FASE 2 — Arreglo geométrico del vestidor (no cambiar de modelo generativo)**

5. En `scripts/composite_tryon_identity.py`, eliminar el bloque:
   ```python
   head_bottom = shoulder_y - shoulder_width * 0.07
   draw.rectangle((0, 0, size[0], head_bottom), fill=255)
   ```
   Es la causa de la franja negra horizontal: restaura un rectángulo de ancho completo
   y crea una costura dura que cruza toda la imagen. Sustituirlo por una máscara
   construida con las salidas semánticas de `scripts/human_parsing_schp.py`
   (clases de cara, pelo, y las máscaras `hands` y `arms` que esa función **ya devuelve**),
   con feathering de 2–3 px. Lo mismo para las manos: usar la máscara semántica de
   brazos/manos en lugar de las dos elipses dibujadas a mano.
6. En `scripts/tryon_geometry.py` línea ~156, eliminar `lower = ImageChops.multiply(lower, silhouette)`.
   Intersectar la máscara de prenda con la silueta del cuerpo impide que la prenda
   generada sea más ancha que el cuerpo desnudo, que es la causa del aspecto de
   pegatina y del pantalón mal formado. La máscara debe ser la **región esperada de
   la prenda**: cuerpo dilatado según las holguras definidas en
   `configs/bespoke_garments.json` para esa prenda.
7. Asegurar que la máscara inferior para pantalón llegue hasta los landmarks de
   tobillo (27/28) con ancho de prenda, no de pierna desnuda.
8. Componer el resultado final como `final = original*(1−M) + generado*M`, donde `M`
   es la región de prenda **sin dilatar hacia el fondo**. La dilatación actual de
   13 px con blur empuja píxeles generados hacia el fondo y es la causa de
   "fondo modificado".
9. Desactivar el overlay de zapatos por defecto.
10. Preservar `tryon/selected.png` como mejor salida cuando el postprocesado empeore
    las métricas verificadas del punto 3.

**FASE 3 — Medición: conectar lo que ya existe**

11. **Largos desde landmarks, no desde priors.** En `src/geometry/geometric_est.py`,
    `HEIGHT_PRIORS` devuelve fracciones fijas de la estatura para `arm-length`,
    `leg-length`, `shoulder-to-crotch`, etc. Como la arquitectura es residual
    (`m_hat = m_geo + Δm`), el modelo está devolviendo esencialmente el prior:
    para un sujeto de 165 cm el prior de brazo da 48.68 cm y el modelo predijo 47.49 cm,
    contra 54 cm reales de cinta. Sustituir esos priors por distancias reales calculadas
    sobre `pose_world_landmarks` de MediaPipe (hombro 11/12, codo 13/14, muñeca 15/16,
    cadera 23/24, rodilla 25/26, tobillo 27/28), escaladas por la estatura confirmada.
    Mantener el prior únicamente como fallback cuando los landmarks tengan baja visibilidad.
12. **Subir MediaPipe Pose de Lite a Heavy.** Los landmarks pasan a ser portadores de
    medida, la precisión ahora importa. Corre en CPU, no consume VRAM.
13. **Conectar SCHP-ATR a la medición.** Hoy solo se usa para el vestidor. Usar sus
    etiquetas para: (a) medir muslo sobre la pierna desnuda visible en vez de sobre el
    short; (b) calcular la fracción de silueta cubierta por prenda y exponerla como
    señal de calidad de captura; (c) rechazar la captura si la cobertura de prenda
    supera un umbral.
14. **Activar SHAPY/SMPL-X como segunda opinión.** Quitar `-SkipBody3D` de la ruta de
    la interfaz. Usar `src/body3d/mesh_measurements.py` para extraer medidas de la malla.
    Cuando SHAPY y la CNN discrepen más de un umbral por medida, marcar esa medida como
    no confiable. El desacuerdo entre dos métodos independientes es mejor señal de
    incertidumbre que la `log_var` de una sola red.
15. **La estatura es entrada obligatoria, nunca estimada.** Documentarlo en la UI.
    Opcionalmente, usar `src/calibration/metric.py` (que ya funciona, con ArUco de piso)
    para **verificar** la estatura declarada y rechazar inconsistencias.

**FASE 4 — Entrenamiento (solo después de las fases 1–3)**

16. **Aumentación de ropa.** Entrenar con dilatación anisotrópica aleatoria de las
    siluetas (más en torso, menos en extremidades) para simular prendas holgadas,
    condicionando en el parámetro `ClothingFit` que hoy existe en el CLI pero no tiene
    efecto real en el modelo.
17. **Corregir el sesgo de cintura.** `results_test.json` muestra sesgo sistemático de
    +2.00 cm en cintura y +1.03 cm en cadera **en el dataset limpio**. Investigar si es
    desalineación de la definición del nivel de cintura en `CIRCUMFERENCE_LEVELS`
    (actualmente 0.63 de la estatura desde los pies) antes de asumir que es error del modelo.
18. **Reportar métricas por medida, nunca agregadas.** El MAE agregado de 1.24 cm está
    arrastrado por muñeca y tobillo; pecho y cintura tienen MAE 2.62 y 2.83 cm con solo
    38–45% de sujetos dentro de 2 cm. El agregado oculta el problema real.
19. **No cambiar el encoder a una red más grande.** El problema no es capacidad del modelo:
    la silueta de un brazo pegado al torso no contiene información del largo del brazo.

---

## 6. Respuesta directa a tu pregunta

**¿Es posible en tu máquina?** Sí. Nada de lo recomendado excede 6 GB de VRAM.
SHAPY (~2–3 GB) y CatVTON (<8 GB a 1024×768, holgado a 576×768) caben, y la mayor
parte del trabajo pendiente es geometría y conexión de módulos, que no cuesta VRAM.

**¿Es posible ±1–2 cm?** Por familia:
- Largos (brazo, pierna, hombro): **sí**, ±1–1.5 cm, con landmarks en vez de priors.
- Contornos (pecho, cintura): **no de forma fiable con la ropa puesta.** Con ropa ceñida,
  protocolo controlado y las correcciones de la Fase 3–4, apunta a ±2–3 cm con intervalo
  conformal declarado. Prometer ±1 cm en pecho desde una foto no es sostenible y el
  tribunal lo va a notar.
- Estatura: **no la estimes.** Pídela. Es la decisión correcta, no una renuncia.

**¿Hay una ruta mejor?** La ruta que ya elegiste es buena. El cambio de rumbo que
recomiendo no es de modelos sino de **encuadre**: pasar de "sastre automático" a
**"medidor asistido con incertidumbre declarada y confirmación humana de las medidas
críticas"**. Es lo que hacen los sistemas comerciales de made-to-measure en producción,
es honesto, y convierte tus hallazgos negativos —el atajo del prior, el sesgo de cintura,
el techo de precisión— en contribuciones metodológicas en lugar de fallos.

---

## Fuentes

- [Robust Monocular Human Height Estimation via a Temporal SegPose Framework (MDPI Sensors, 2026)](https://www.mdpi.com/1424-8220/26/16/5252)
- [A Focused Human Body Model for Accurate Anthropometric Measurements Extraction (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_A_Focused_Human_Body_Model_for_Accurate_Anthropometric_Measurements_Extraction_CVPR_2025_paper.pdf)
- [Comparing the Top 4 Open Source Virtual Try-On Models (FASHN)](https://fashn.ai/blog/comparing-the-top-4-open-source-virtual-try-on-viton-models)
- [CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models](https://zheng-chong.github.io/CatVTON/)
- [How to take photos with 3DLOOK — requisitos de captura](https://3dlook.ai/content-hub/how-to-take-photos-with-3dlook-the-ultimate-guide/)
- [SHAPY: Accurate 3D Body Shape Regression using Metric and Semantic Attributes (CVPR 2022)](https://github.com/muelea/shapy)
- [Awesome Try-On Models](https://github.com/Zheng-Chong/Awesome-Try-On-Models)
- Evidencia interna: `experiments/exp_006_weight_huber/results_test.json`, `src/geometry/geometric_est.py`, `src/models/regressor.py`, `scripts/composite_tryon_identity.py`, `scripts/tryon_geometry.py`
