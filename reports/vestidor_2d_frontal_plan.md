# SATRE-IA — arquitectura del vestidor 2D frontal

## Decisión

La salida oficial debe ser una **composición 2D determinista y auditable**. La
generación por difusión queda como postproceso experimental. Una sola foto
frontal puede comunicar estilo, color y colocación aproximada, pero no puede
validar por sí sola caída física, presión, holgura posterior ni ajuste 3D.

## Estado implementado — 11 de septiembre de 2026

- Biblioteca inicial de cuatro sprites RGBA 1024×2048 con manifiesto y anclas:
  saco y pantalón masculinos, saco y falda lápiz femeninos.
- Compositor oficial determinista conectado a `/api/tryon/layered` y ejecutado
  automáticamente después de una medición.
- Orden aplicado: foto, sombras restringidas semánticamente, prenda inferior,
  saco, manos/cuello/cara/pelo, grano y etiqueta persistente dentro del PNG.
- Adaptación conservadora del canal L en LAB. Los canales A/B no se igualan al
  cuerpo porque eso destruiría el color elegido de la tela.
- SCHP-ATR 18 se ejecuta localmente en ONNX INT8 sobre CPU. ATR no contiene
  clases independientes de mano o cuello: esas dos máscaras se refinan con
  landmarks MediaPipe y se declaran como refinamientos, no como clases nativas.
- QA no circular: extensión respecto de SAM original, distancia del borde
  visible del cuello, distancia del borde visible de puños, invasión de
  oclusiones y reducción de diferencia de luminancia.
- Validación masculina real: 3.517 % del área de prenda queda fuera de SAM;
  extensión estructural máxima 10.252 px con límite 11.650 px; separación de
  cuello 26.845 px; separación media de puños 16.604 px; solape con oclusiones
  0.961 %. El estado correcto es `ajuste_dudoso` por el fallo de puños.
- La ruta femenina fue comprobada mecánicamente con saco y falda; requiere una
  sesión femenina procesada para evaluación visual válida.

Pendiente prioritario: añadir sprites de saco específicos para brazos abiertos.
SCHP ya resuelve la separación semántica, pero no corrige una manga cuyo sprite
de origen no corresponde a la pose. Si el parser falla, la interfaz y el JSON
lo declaran y regresan al fallback Pose; nunca se presenta ese fallback como SCHP.

## Tres vistas, sin mezclar significados

1. **Ajuste 2D calculado:** foto + prenda preajustada + oclusiones + iluminación.
2. **Diagnóstico:** pose, máscara, anclas, confianza y advertencias.
3. **Experimental:** CatVTON, SD-inpainting o IDM-VTON, con etiqueta naranja.

Las metas visuales externas permanecen separadas y marcadas como no generadas
por el sistema.

## Pipeline oficial propuesto

```text
foto frontal normalizada
  -> pose + silueta SAM
  -> parsing semántico (cara, pelo, cuello, brazos, manos, torso, piernas)
  -> selección del sprite por sexo/prenda/pose
  -> warp por piezas
       saco: cuerpo + 4 segmentos de manga
       pantalón: cintura + 4 segmentos de pierna
       falda: malla cintura/cadera/rodilla
  -> regularización con medidas finales disponibles
  -> composición: foto -> sombra -> prenda -> manos/brazos -> cara/pelo
  -> adaptación de luz y grano
  -> controles automáticos de calidad
  -> PNG + JSON de auditoría
```

## Observación importante sobre VTON

CatVTON e IDM-VTON normalmente esperan como condición una prenda de catálogo.
Entregarles directamente una prenda ya deformada cambia la distribución de su
entrada y no garantiza una mejora. Se deben comparar dos variantes:

- **A:** prenda plana como condición; pre-warp usado como guía/máscara.
- **B:** composición pre-warp como imagen inicial para inpainting de baja fuerza.

Para SATRE-IA, la variante B es conceptualmente más segura: el modelo retoca
bordes, textura y sombra sin volver a decidir toda la geometría.

## Parsing y oclusión

SAM produce una silueta de persona, no clases corporales. El primer MVP puede
combinar pose + SAM, pero para colocar manos, brazos, cabello y rostro delante de
la prenda hace falta parsing semántico. El candidato local práctico es SCHP ATR
en ONNX INT8; debe compararse con SegFormer en un pequeño conjunto propio antes
de elegirlo.

## Biblioteca mínima de prendas

No hacen falta treinta sprites al principio. Para validar la arquitectura:

- saco masculino frontal con brazos abajo y abiertos;
- pantalón masculino frontal;
- saco femenino frontal con brazos abajo y abiertos;
- falda lápiz femenina frontal;
- todos en PNG RGBA, sin fondo, misma convención de anclas.

Después se agregan estilos y poses solamente cuando el control de calidad detecte
que el sprite actual excede su rango de deformación.

## Control de calidad antes de mostrar el resultado

- visibilidad de hombros, codos, muñecas, caderas, rodillas y tobillos;
- persona completa dentro del encuadre;
- intersección prenda/máscara dentro de límites;
- estiramiento local máximo de la textura;
- cuello, cara y manos sin invasión;
- simetría razonable cuando la pose es frontal;
- si falla: mostrar diagnóstico y pedir repetir la foto, no inventar imagen.

## Métricas de aceptación del MVP

- 0 píxeles de prenda sobre cara;
- menos de 1 % de invasión visible sobre manos;
- extensión estructural fuera de SAM no mayor al 8 % del ancho de hombros;
- separación visible cuello/prenda menor al 20 % del ancho de hombros;
- separación visible puño/muñeca menor al 10 % del ancho de hombros;
- resultado determinista en menos de 3 s después de tener pose y parsing;
- prueba ciega de preferencia frente al overlay geométrico actual;
- etiqueta visible e incorporada al PNG exportado.

## Orden recomendado

1. Terminar composición determinista usando el pre-warp ya creado.
2. Añadir parsing semántico real y capas de oclusión.
3. Incorporar sombras, igualación LAB, grano y bordes.
4. Añadir selector de sprite por pose y una compuerta de calidad.
5. Evaluar SD-inpainting de baja fuerza sobre la composición.
6. Solo después comparar CatVTON e IDM-VTON en un lote fijo.

## Referencias técnicas

- CatVTON: https://github.com/Zheng-Chong/CatVTON
- IDM-VTON: https://github.com/yisol/IDM-VTON
- StableVITON: https://github.com/rlawjdghek/StableVITON
- CP-VTON+: https://github.com/minar09/cp-vton-plus
- PF-AFN: https://github.com/geyuying/PF-AFN
- SCHP Human Parsing: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
