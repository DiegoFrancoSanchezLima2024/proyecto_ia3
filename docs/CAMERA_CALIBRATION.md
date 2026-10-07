# Primera etapa métrica: cámara + ChArUco + suelo ArUco

Estado: implementación experimental, **sin validación física todavía**. No se
solicita altura, peso ni sexo en estos comandos. El regresor BodyM anterior sí
necesita metadatos y permanece separado: no se convirtió automáticamente en un
modelo sin datos personales.

## Decisión técnica

Para este prototipo, usar cámara RGB calibrada, referencia física y percepción
preentrenada es una ruta defendible. La referencia resuelve la escala; la IA
ayudará a localizar el cuerpo y estimar su forma. Ninguno de esos componentes por
separado garantiza medidas listas para confección.

- **ChArUco:** tablero que se fotografía en distintas posiciones para estimar
  intrínsecos y distorsión de la lente. Se retira durante la demo.
- **ArUco en el suelo:** cuatro marcadores en posiciones conocidas permiten
  estimar la orientación y distancia de la cámara respecto a un suelo métrico.
  Pueden integrarse en el borde de una base de captura; no se sostienen delante
  del torso. Esta primera versión utiliza tres o cuatro visibles, sin oclusiones.
- **No usar una tarjeta de débito como referencia principal:** es pequeña en una
  foto de cuerpo entero, puede inclinarse y estar a distinta profundidad. Un
  tamaño conocido no elimina esos errores. Tampoco hace falta fotografiar datos
  bancarios. Los marcadores no necesitan una tarjeta real.

OpenCV recomienda esquinas ChArUco para calibración por su precisión de
localización: [calibración oficial](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html).
La pose usa correspondencias 3D–2D y longitud física de los marcadores:
[ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html),
[PnP e IPPE](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
Los tamaños y umbrales de este kit son decisiones iniciales del proyecto, no
recomendaciones de precisión certificadas por OpenCV.

## 1. Lo que debes hacer ahora

Actualización 08/09/2026: las fotos ya se revisaron y hay 30 originales conservados,
28 preferidos y 2 problemáticos, en `dataset/local/calibration/infinix_x6876_20260908`.
El perfil diagnóstico aún tiene una vista sobre el umbral; no está aprobado.
Para esta captura concreta, seguir primero la [guía de selección y siguiente paso](SIGUIENTE_PASO_CALIBRACION_20260908.md)
en vez de repetir toda la adquisición indicada abajo. El nuevo `inspect-floor`
permite preparar/comprobar visibilidad del suelo antes de aprobar la lente,
sin estimar geometría métrica ni omitir sus controles.

El kit ya se generó en `outputs/calibration_kit/imprimir.html`. Abrirlo en un
navegador e imprimir **las cinco páginas A4**, tamaño real **100 %**, sin
«ajustar a página», sin encabezados/pies del navegador. Conservar los SVG junto
al HTML; no imprimir capturas de pantalla.

Verificar físicamente con una regla:

- Línea de control: 100 mm, de extremo exterior a extremo exterior.
- Tablero ChArUco: 150 × 210 mm; cada cuadrado: 30 mm.
- ArUco de suelo: lado exterior negro 160 mm, excluyendo margen blanco.
- Revisar escala tanto horizontal como vertical. Si la impresora altera el
  tamaño, corregir la impresión; no seguir con una escala equivocada.

Pegar el tablero ChArUco sobre soporte rígido, plano y mate. No doblarlo, cubrirlo
con plástico brillante ni cambiar las proporciones.

Para regenerar el kit, si se necesita:

```powershell
Set-Location 'C:\Users\diego\Desktop\proyecto de ia3\proyecto-sastre-ia'
conda activate sastre-ia-perception
python -m src.calibration.cli kit --output outputs/calibration_kit_v2
```

### Fotografías para calibrar la cámara (no son un dataset de personas)

Usar la misma cámara, lente y modo de fotografía que se utilizarán en la demo:

1. Elegir cámara principal sin ultra gran angular, modo foto normal, sin retrato,
   belleza, zoom digital ni filtros. Bloquear zoom y, si es posible, enfoque.
   Mantener orientación vertical, resolución y procesamiento iguales.
2. Tomar unas **20 fotos diferentes del tablero**: centro, bordes, esquinas,
   distintas inclinaciones en ambos ejes y tamaños aparentes. Inclinar el tablero,
   no cambiar entre orientación vertical y horizontal del teléfono. Mantenerlo
   nítido y suficientemente grande para leer sus cuadros. Debe recorrer el
   encuadre; 20 fotos idénticas no sirven.
3. Tomar **otras 6 fotos** con posiciones distintas para validación. No copiar
   imágenes de la primera carpeta. El programa detecta duplicados exactos en
   píxeles, no todas las fotografías casi iguales: tú debes variar las tomas.
4. Transferir los archivos originales JPG/PNG. No comprimir por mensajería,
   recortar o redimensionar. HEIC no está admitido en esta primera versión.

Carpetas ya creadas:

```text
dataset/local/calibration/camera01/
  train/        <- 20 fotos del tablero para estimar la lente
  validation/   <- 6 fotos nuevas para comprobar la lente
```

Son fotografías de calibración del equipo. No se entrena una red con ellas ni
se construye un dataset corporal propio.

## 2. Ejecutar y compartir el resultado

Desde la raíz del proyecto:

```powershell
conda activate sastre-ia-perception
python -m src.calibration.cli calibrate
```

Se guarda `outputs/calibration/camera01.json`: matriz de cámara, distorsión,
resolución, errores por foto, cobertura, diversidad angular, archivos y hashes.
El resultado esperado para continuar es `passed_reprojection_checks`.

`needs_recapture` significa repetir/mejorar fotos y revisar los avisos; el
programa guarda el diagnóstico y termina con código 2. Los errores estructurales
(carpetas vacías, tamaños distintos, insuficientes fotos válidas) también terminan
con código 2, sin producir un perfil nuevo. No usar un perfil antiguo como si
fuera el resultado de una ejecución fallida. Las salidas existentes no se
sobrescriben sin `--force`; preferir `--output` distinto al comparar intentos.

El mínimo implementado es 12 fotos válidas + 4 de validación, con al menos 10
esquinas no colineales cada una. Se recomienda 20 + 6 para disponer de margen.
Las fotos rechazadas se documentan en el perfil si hay suficientes válidas.

**Compartir primero el resumen y el JSON de calibración.** No lanzar otro
entrenamiento BodyM todavía. Calibrar la cámara requiere CPU, no ocupa VRAM de la
RTX 3050. SAM 2.1 seguirá utilizando GPU en su etapa, de forma secuencial.

## 3. Montar el suelo métrico después de aprobar la lente

La configuración `configs/calibration_station.json` define un rectángulo de
**100 × 80 cm**, medido entre bordes negros exteriores. No es necesario imprimir
una alfombra completa: se colocan las cuatro hojas sobre un suelo plano y se
verifican posiciones. El origen está en el centro; +X derecha, +Y fondo, +Z arriba.

Vista desde arriba, mirando en la misma dirección que la cámara:

```text
                      FONDO (+Y)
           ID 30                    ID 31
          (-42,+32)                 (+42,+32) cm

                      persona
                      (0, 0)

           ID 32                    ID 33
          (-42,-32)                 (+42,-32) cm
                       CÁMARA
```

Cada coordenada corresponde al **centro del cuadrado negro de 16 cm**, no al
centro de la hoja. Los centros están separados 84 cm lateralmente y 64 cm en
profundidad. Comprobar paralelismo y escuadra, además de las distancias. En todas
las hojas, la flecha impresa apunta hacia +Y; no rotar cada marcador libremente.
Conservar al menos 10 mm de blanco alrededor del cuadrado negro y no tapar
ninguna esquina. Las hojas deben estar planas, inmóviles y sin reflejos.

Las coordenadas están en Z=0: si se usa una base elevada, la persona y los
marcadores deben apoyarse en el mismo plano superior. No poner los marcadores
sobre una pared y aplicar estas coordenadas de suelo.

Empezar con cámara fija, aproximadamente a 2.5–3.5 m y a media altura corporal,
encuadrando persona completa y los cuatro marcadores. Son valores iniciales,
no distancias que debas introducir en el software. El detector calcula la pose.
Si los marcadores se ven demasiado pequeños/achatados, revisar resolución
original, posición y encuadre; **no bajar el umbral para forzar aprobación**.

Guardar una foto de la estación y ejecutar, cambiando la ruta a una foto real:

```powershell
python -m src.calibration.cli floor --image dataset/local/calibration/camera01/station.jpg --output outputs/calibration/station_pose.json
```

Se necesitan al menos 3 IDs conocidos, lado proyectado mínimo de 20 px, RMS
máximo 1.5 px y error por esquina máximo 3 px. Estos umbrales son provisionales.
Un código detectado no garantiza bordes suficientemente precisos. Rehacer la
pose en cada foto si se mueve la cámara; el JSON se vincula a la foto exacta
mediante hash para evitar reutilizar una geometría incorrecta.

## 4. Prueba métrica antes de usar un cuerpo

El módulo ofrece `vertical-check`: estima la altura de una **varilla rígida
vertical**, cuya base está sobre el suelo de la estación, a partir de sus dos
extremos marcados en píxeles. No recibe la longitud real como entrada. La longitud
medida con cinta se reserva para evaluar el error posteriormente.

Crear un JSON con `base_pixel` y `top_pixel`, ambos `[x, y]` en la foto original
orientada (origen arriba a la izquierda). No usar coordenadas de una previsualización
reducida. Comando para cuando se disponga de esa foto y anotaciones:

```powershell
python -m src.calibration.cli vertical-check --image dataset/local/calibration/camera01/station.jpg --pose outputs/calibration/station_pose.json --points outputs/calibration/vertical_points.json --output outputs/calibration/vertical_result.json
```

La altura se calcula intersectando el rayo de la base con Z=0 y buscando el
punto sobre su vertical que corresponde al rayo del extremo superior. **Es un
diagnóstico manual de geometría, no detección automática de estatura.** Un tobillo
no es el contacto con el suelo; cabello no es cráneo; una cabeza puede estar
adelantada respecto a los pies. La inclinación en profundidad puede sesgar la
altura aunque el residuo geométrico sea pequeño.

Repetir con objeto vertical de longitud comparable a una persona, en el centro
y distintas posiciones/distancias admitidas. Esto permite separar errores de
escala/cámara de errores posteriores de percepción humana. Si la impresión está
un 5 % fuera de escala, la altura también puede desviarse un 5 % aunque la
reproyección parezca perfecta: existe una prueba automatizada de este caso.

## 5. Lo que sigue después de esta etapa

1. Aprobar geometría con referencia rígida y medición independiente. Sin esto,
   cambiar el backbone no resuelve el error de escala.
2. Integrar detector de persona/postura con SAM 2.1; detectar apoyo de pies y
   contorno de cabeza, comprobar postura y estimar estatura con varias vistas.
   No sustituir cabeza/pies por la caja completa de la persona. Exigir ropa
   ajustada, descalzo, cabeza erguida, cabello controlado y cuerpo no recortado.
3. Capturar frente, perfil izquierdo, espalda y perfil derecho con cámara fija;
   gira la persona. No tratar estas fotos como un objeto rígido de SfM: cambia
   la postura. La integración corporal deberá compartir forma y permitir poses
   distintas por vista.
4. Evaluar un modelo de proporciones/forma que no consuma peso, sexo ni altura
   real en inferencia, usando datos públicos. El checkpoint exp_006 no cumple
   este contrato; mantenerlo sólo como comparación asistida. La altura estimada
   por cámara se usa para escala y se propaga su incertidumbre.
5. Comparar SMPL/SMPL-X sólo después de medir el baseline; procesar vistas y
   modelos secuencialmente en 6 GB. No hay necesidad de Colab para la calibración.
6. Validar estatura y cada contorno frente a mediciones independientes en una
   pequeña evaluación consentida, sin usarla para entrenar. Datos públicos no
   sustituyen comprobar la cámara real de la demo. Reportar MAE, P90, porcentaje
   de error ≤2 cm, repetibilidad y tasa de rechazos; separar persona y captura.
7. Sólo entonces convertir medidas a patrones con holguras, costuras y revisión
   de patronista. Un render 2D atractivo no demuestra precisión ni ajuste textil.

**Objetivo, no promesa:** P90 de error absoluto de estatura ≤2 cm en capturas
admitidas, junto con tasa de rechazo. No afirmar ±2 cm garantizados, ni extender
ese objetivo a pecho/cintura/cadera sin resultados separados. Ninguna prueba
sintética de este módulo acredita todavía precisión en personas.

## Verificación de software

```powershell
conda activate sastre-ia-perception
python -m pytest tests/test_calibration.py tests/test_perception.py -q
```

Incluye proyección con distorsión, recuperación de pose y altura de verticales,
rechazo por geometría inválida, marcadores renderizados, detección ChArUco,
duplicados, calibración con validación independiente y fallo por escala impresa
incorrecta. OpenCV local verificado: 4.12.0, con ArUco y ChArUco disponibles.
