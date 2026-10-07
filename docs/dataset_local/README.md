# Protocolo del dataset local SATRE-IA

Versión del protocolo: `SATRE-LOCAL-1.0`

## Propósito y alcance

Este conjunto sirve para **calibrar y evaluar** SATRE-IA en fotografías reales de
teléfono. Con 15–50 personas no se debe entrenar desde cero una red profunda. Sí
permite medir el cambio de dominio respecto de BodyM, calibrar offsets, seleccionar
entre métodos y estimar incertidumbre real.

Las fotografías, las medidas corporales y las medidas de la prenda son datos
distintos. Nunca se debe usar una medida de prenda terminada como si fuera una
medida corporal.

## Tamaño recomendado y fases

1. **Piloto técnico:** 5 adultos. Sirve para encontrar errores del protocolo y no
   debe formar parte de la prueba final.
2. **Calibración:** 15 adultos. Sirve para elegir offsets, umbrales y correcciones.
3. **Prueba bloqueada:** 15–30 adultos diferentes. No se mira para ajustar el
   sistema; se usa una sola vez para reportar resultados finales.
4. **Entrenamiento o adaptación futura:** aspirar a 150–300 adultos como mínimo,
   con división por persona. La cantidad necesaria definitiva se decide mediante
   curvas de aprendizaje, no por una cifra arbitraria.

La división es siempre por `sujeto_id`. Ninguna foto, repetición ni medida de una
misma persona puede aparecer en más de una división.

## Ética y privacidad

- Empezar únicamente con personas adultas. Si posteriormente se incluyen menores,
  se necesita un protocolo institucional y consentimiento del representante.
- Usar identificadores como `SATRE_0001`; no poner nombres, cédulas, teléfonos ni
  direcciones en nombres de archivos o CSV.
- Conservar el documento firmado y la tabla que relaciona identidad con
  `sujeto_id` fuera del repositorio y cifrados.
- Explicar finalidad, uso académico, tiempo de retención, quién tendrá acceso y
  procedimiento de retiro.
- Guardar una copia original protegida y trabajar con copias derivadas. Para
  publicación, usar siluetas o imágenes con rostro anonimizado, según el permiso.
- No subir fotografías personales a GitHub, servicios públicos o almacenamiento
  de terceros sin autorización específica.

La plantilla de consentimiento incluida es una base operativa, no asesoría legal
ni aprobación de un comité de ética.

## Preparación del espacio

### Cámara

- Usar la cámara trasera del mismo teléfono durante una sesión.
- Lente `1x`; desactivar gran angular, modo retrato, belleza, filtros y zoom digital.
- Formato vertical y resolución constante. Conservar EXIF si es posible.
- Colocar el teléfono en trípode, nivelado, sin inclinarlo hacia arriba o abajo.
- Altura inicial del centro de lente: 85–90 cm. Registrar el valor real.
- Distancia inicial: 280–320 cm. Marcar en el piso la posición del trípode y la de
  los pies. Registrar la distancia real.
- No mover la cámara entre las cuatro vistas; la persona es quien gira.

### Escena

- Fondo liso que contraste con la ropa y sin espejos ni otras personas.
- Luz frontal y lateral uniforme. Evitar ventana o foco fuerte detrás de la persona.
- Debe verse el cuerpo completo, incluida la parte superior de la cabeza y ambos
  pies, con aproximadamente 8–12 % de margen arriba y abajo.
- Retirar objetos que crucen visualmente brazos, cintura o piernas.
- Una referencia rígida o regla visible puede guardarse para auditoría, pero no
  sustituye la estatura medida ni debe tocar el contorno corporal.

### Persona

- Descalza, bolsillos vacíos, sin reloj, cinturón, chaqueta o prendas compresoras.
- Cabello largo recogido para dejar visibles cuello, hombros y espalda.
- Ropa válida: camiseta deportiva ceñida y short/legging ceñido, opacos y de color
  contrastante. No es necesario fotografiar en ropa interior.
- Abdomen relajado; no sacar pecho ni contener la respiración.
- Mirada horizontal, hombros relajados, rodillas extendidas sin bloquearlas.
- Pies paralelos, separados 10–15 cm, sobre marcas permanentes.
- Brazos en A, separados aproximadamente 20–30 grados del torso; codos extendidos
  y manos relajadas. La pose debe mantenerse en las cuatro vistas.

## Secuencia de captura por persona

Cada persona produce **ocho fotografías válidas**, no cuatro:

1. Ronda 1: `frente`, `izquierda`, `espalda`, `derecha`.
2. La persona sale de las marcas, descansa 30 segundos y vuelve a colocarse.
3. Ronda 2: `frente`, `izquierda`, `espalda`, `derecha`.

Esto permite evaluar repetibilidad. Una ráfaga de imágenes sin recolocar a la
persona no cuenta como segunda ronda.

Convención de archivos:

```text
dataset/local/raw/SATRE_0001/
  ronda_01/frente.jpg
  ronda_01/izquierda.jpg
  ronda_01/espalda.jpg
  ronda_01/derecha.jpg
  ronda_02/frente.jpg
  ronda_02/izquierda.jpg
  ronda_02/espalda.jpg
  ronda_02/derecha.jpg
```

Opcionalmente se puede capturar una tercera ronda con ropa cotidiana u holgada,
marcada como `desafio_ropa`. Esa ronda sirve para entrenar/evaluar el **rechazo de
captura**, no debe tratarse como una observación métrica equivalente a la ropa
ceñida.

## Criterios para aceptar una fotografía

La captura se rechaza y se repite si ocurre cualquiera de estos casos:

- Falta cabeza, mano, pie o cualquier parte del cuerpo.
- La cámara está inclinada o se cambió de lugar entre vistas.
- Hay desenfoque, movimiento, contraluz o sombra que borra el contorno.
- La persona rota incorrectamente; el perfil no está cerca de 90 grados.
- Los brazos tocan el torso o las piernas se superponen en la vista frontal.
- La ropa es holgada, hay objetos en bolsillos o el cabello cubre hombros/cintura.
- La postura cambia visiblemente entre vistas.
- MediaPipe/RTMPose no encuentra los puntos clave con la confianza mínima.
- SAM pierde partes del cuerpo o incluye objetos grandes del fondo.

No se debe corregir una mala fotografía rellenando centímetros manualmente y
marcándola como captura válida. Se conserva como rechazo y se toma otra.

## Medición de referencia

### Regla general

- Utilizar cinta flexible no elástica y balanza sobre piso firme.
- Marcar cintura natural y puntos anatómicos con cinta adhesiva de papel lavable.
- La cinta debe quedar horizontal en los contornos, en contacto sin comprimir piel.
- Medir altura y peso dos veces.
- Cada una de las 13 medidas debe tomarse dos veces por dos medidores cuando sea
  posible: cuatro lecturas independientes.
- El segundo medidor no debe ver la lectura del primero.
- Si dos lecturas de contorno difieren más de 1,0 cm, o dos largos más de 0,7 cm,
  realizar una tercera lectura y documentar la causa.
- Guardar todas las lecturas; no sustituirlas por el promedio en el archivo crudo.

### Definiciones operativas de las 13 variables

Estas definiciones deben conservar la misma `version_protocolo`. BodyM publica los
nombres, pero no toda la geometría de medición en su página pública. Por ello los
largos ambiguos se registran además con medidas auxiliares antes de decidir la
correspondencia definitiva.

| Clave del modelo | Nombre | Definición operativa SATRE-LOCAL-1.0 |
|---|---|---|
| `chest` | Pecho/busto | Contorno horizontal máximo durante respiración normal, pasando bajo axilas y sobre escápulas y zona más prominente del pecho. |
| `waist` | Cintura natural | Contorno horizontal entre última costilla y cresta ilíaca, abdomen relajado. No usar automáticamente la pretina del pantalón. |
| `hip` | Cadera/asiento | Contorno horizontal máximo de glúteos y cadera. |
| `thigh` | Muslo | Contorno máximo del muslo derecho, inmediatamente debajo del pliegue glúteo. |
| `calf` | Pantorrilla | Contorno máximo de la pantorrilla derecha. |
| `ankle` | Tobillo | Contorno mínimo inmediatamente por encima de los maléolos. |
| `arm-length` | Largo de brazo | Desde acromion, pasando por codo, hasta prominencia cubital de muñeca. Registrar además si se tomó con brazo recto o flexionado; usar siempre el modo `flexionado_90` en esta versión. |
| `forearm` | Antebrazo | Contorno máximo del antebrazo derecho, brazo relajado. |
| `wrist` | Muñeca | Contorno sobre las prominencias óseas de la muñeca derecha. |
| `bicep` | Bíceps/brazo superior | Contorno máximo del brazo superior derecho relajado, sin flexionar músculo. |
| `shoulder-breadth` | Ancho biacromial | Distancia recta entre acromion izquierdo y derecho. No medir de borde de manga a borde de manga. |
| `leg-length` | Largo de pierna BodyM | Variable ambigua. Registrar provisionalmente `crotch_to_floor` y `waist_to_floor`; no elegir una por intuición. |
| `shoulder-to-crotch` | Hombro a entrepierna BodyM | Variable ambigua. Registrar recorrido desde punto alto de hombro a entrepierna por delante y largo vertical directo como auxiliares. |

Para pecho, cintura y cadera, la referencia de confección recomendada es
ISO 8559-1. Si se cambia un punto anatómico, se incrementa la versión del
protocolo; nunca se mezclan definiciones con el mismo nombre.

## Medidas adicionales para los moldes

Las 13 variables de BodyM no satisfacen todos los requisitos de Jaeger, Charlie y
Penelope. Deben medirse también, en un archivo independiente:

- cuello;
- punto alto de hombro a busto;
- punto alto de hombro a cintura por espalda;
- inclinación de hombro;
- cintura a axila, cadera, asiento, rodilla, piso y parte alta del muslo;
- hombro a codo;
- entrepierna completa (`crossSeam`) y delantera (`crossSeamFront`);
- contorno de rodilla;
- ancho de cintura y asiento por espalda si el patrón lo solicita.

Estas medidas no se inventan a partir de porcentajes de estatura cuando el molde se
marca como `apto_para_corte`. Si falta una, el resultado sigue siendo borrador.

## Archivos tabulares

Copiar las plantillas de `docs/dataset_local/plantillas/` al área privada de datos:

```text
dataset/local/registro/
  sujetos.csv
  capturas.csv
  mediciones.csv
  mediciones_patronaje.csv
  asignacion_splits.csv
```

`mediciones.csv` usa formato largo: una fila por lectura, no una columna por medida.
Esto conserva medidor y repetición y permite calcular error entre observadores.

## División y prevención de fuga

- Crear `asignacion_splits.csv` antes de evaluar el modelo final.
- El piloto queda como `pilot` y nunca se presenta como test.
- Las correcciones, offsets y umbrales usan solamente `calibration`.
- El conjunto `test` queda bloqueado hasta congelar código y configuración.
- Agrupar por persona: nunca dividir fotografías individualmente.
- Congelar una copia del manifiesto con hashes SHA-256 antes de la evaluación.

## Métricas mínimas

Reportar por cada medida y por persona:

- MAE en centímetros;
- sesgo medio firmado;
- mediana del error absoluto;
- porcentaje dentro de ±1 cm y ±2 cm;
- repetibilidad entre las dos rondas de fotos;
- diferencia media entre medidores y entre repeticiones de cinta;
- cobertura de los intervalos de incertidumbre;
- métricas separadas por sexo corporal, rango de IMC y condición de ropa, sin
  publicar grupos tan pequeños que puedan identificar personas.

No seleccionar el mejor modelo por MAE global solamente. Pecho, cintura, cadera y
las medidas necesarias para cada patrón deben aprobarse de manera individual.

## Lista rápida del día de captura

1. Confirmar consentimiento y asignar `sujeto_id`.
2. Medir dos veces estatura y peso.
3. Confirmar vestimenta, cabello, accesorios y bolsillos.
4. Registrar teléfono, distancia y altura de cámara.
5. Capturar y revisar ronda 1 antes de continuar.
6. Recolocar a la persona y capturar ronda 2.
7. Tomar las medidas corporales sin mostrar predicciones del sistema.
8. Segundo medidor repite sin ver las primeras lecturas.
9. Completar medidas adicionales de patronaje.
10. Ejecutar el validador y corregir omisiones antes de terminar la sesión.

