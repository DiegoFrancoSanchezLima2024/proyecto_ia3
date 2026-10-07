# Plan de implementacion Sastre-IA para RTX 3050 6 GB

## 1. Objetivo verificable

Construir y evaluar un sistema que, a partir de cuatro fotografias guiadas y una
referencia metrica, estime medidas con incertidumbre, reconstruya una forma
corporal 3D, genere patrones base imprimibles para dos conjuntos formales y muestre
una vista previa frontal.

Alcance cerrado:

- linea masculina: saco y pantalon;
- linea femenina: saco y falda lapiz;
- tejido plano de sastreria, no elastico;
- una silueta regular y opciones limitadas de holgura/largo;
- VTON frontal como visualizacion, no como prueba de ajuste.

## 2. Arquitectura por contratos

Cada modulo debe poder probarse sin cargar los demas:

1. `capture`: cuatro fotos, referencia ArUco/altura y control de calidad.
2. `segmentation`: silueta, keypoints y puntuacion de calidad.
3. `measurement`: regresor directo de dos/cuatro vistas.
4. `body3d`: inicializador monocular y ajuste SMPL-X multivista.
5. `fusion`: combina medidas directas y de malla; calibra incertidumbre.
6. `patterns`: adapta medidas corporales a medidas de patron y holguras.
7. `pattern_service`: genera SVG/PDF mediante FreeSewing.
8. `tryon`: produce solamente la vista previa 2D.
9. `app`: orquesta el flujo y descarga resultados.

Contrato de salida del nucleo antropometrico:

```json
{
  "measurement": "chest",
  "value_cm": 96.4,
  "lower_90_cm": 94.8,
  "upper_90_cm": 98.1,
  "source": "fused",
  "quality": "high"
}
```

## 3. Perfil local optimizado

Configuracion inicial validable:

- 320 x 320, un canal;
- EfficientNet-B0 compartido;
- batch fisico 16 y acumulacion 2;
- FP16, channels-last y TF32;
- encoder congelado 5 epocas, BatchNorm adaptado solo 3 epocas y ultimo bloque con LR `3e-5`;
- cabezas/atencion con LR `2e-4`;
- 2 workers persistentes y prefetch 2;
- un unico modelo grande residente en GPU;
- checkpoint por menor MAE de validacion.

El lote final se decide ejecutando `scripts/benchmark_gpu.py`. Se reserva al menos
0.8 GiB para Windows, pantalla y variaciones del allocator. No se selecciona el
lote que apenas cabe.

## 4. Hitos y puertas de calidad

### H0 - Integridad del baseline, semanas 1-2

Entregables:

- splits train/val/calibration por sujeto;
- Test-A/Test-B intactos;
- smoke train en CPU y CUDA;
- benchmark de VRAM;
- baseline de media de train y red sin geometria.

Puerta: ninguna imagen ausente, ningun NaN, tests verdes y resultados
reproducibles con seed fija.

### H1 - Medicion BodyM de dos vistas, semanas 3-5

Experimentos obligatorios:

- media de train;
- altura + genero mediante MLP;
- frontal solamente;
- lateral solamente;
- frontal + lateral;
- frontal + lateral + geometria.

Metricas por medida: MAE, RMSE, mediana, P90, porcentajes dentro de 1/2/3 cm,
sesgo y Bland-Altman. La geometria se conserva solo si mejora Test-A y no degrada
de forma importante Test-B.

### H2 - Captura local de cuatro vistas, semanas 6-8

- camara fija; persona rota a 0/90/180/270 grados;
- ropa ajustada, pies visibles y brazos separados;
- altura introducida o marcador de medida conocida;
- rechazo automatico de desenfoque, recorte, oclusion y pose incorrecta;
- consentimiento y seudonimizacion.

Primero se implementa la captura con siluetas fiables. No se inicia SMPL-X hasta
que la tasa de captura valida supere 90%.

### H3 - Dataset local, semanas 8-11

- piloto: 40 personas;
- objetivo defendible: 80-120 personas;
- dos lecturas de cinta por un evaluador entrenado;
- tercera lectura si las primeras difieren mas que la tolerancia definida;
- division por persona;
- subconjunto local de test bloqueado desde el inicio.

BodyM preentrena dos vistas. El conjunto local calibra el dominio de telefono y
permite comparar dos contra cuatro vistas.

### H4 - Cuerpo 3D, semanas 11-14

Evaluar SHAPY y Multi-HMR 672-S como inicializadores. Mantener uno. Compartir los
parametros de forma entre vistas y optimizar pose/camara por imagen con perdidas de
keypoints, silueta, altura y prior corporal.

Ruta de ejecucion:

- local: inicializador pequeno, una vista por vez, FP16;
- Colab: modelos pesados y experimentos;
- local: optimizacion final de pocos parametros y extraccion de medidas.

Puerta: la fusion 3D debe mejorar medidas relevantes o reducir P90/incertidumbre.
Si no mejora, queda como visualizacion y no controla el patron.

### H5 - Patronaje, semanas 15-19

Motor primario: FreeSewing.

- hombre: Jaeger y bloque Titan/Charlie;
- mujer: Penelope y adaptacion del saco con patronista;
- salida: SVG, A4, A0, margenes, hilo, piquetes, etiquetas y cuadro de escala;
- separacion entre medida corporal, holgura y opcion de estilo.

Pruebas: compatibilidad de costuras, ausencia de autointersecciones, escala de
impresion y holgura digital. Confeccionar primero en muselina/toile.

GarmentCode se incorpora despues para simulacion 3D; nunca bloquea el PDF.

### H6 - Vista previa y demo, semanas 20-22

CatVTON se ejecuta a baja resolucion local solo si el benchmark deja margen; la
version de demostracion de alta resolucion va a Colab. La interfaz rotula la salida
como aproximacion visual.

Flujo de demo:

1. captura guiada;
2. control de calidad;
3. medidas e intervalos;
4. malla 3D;
5. seleccion de prenda y holgura;
6. descarga de patron;
7. vista previa frontal;
8. pantalla de evaluacion contra cinta para el jurado.

### H7 - Validacion fisica y tesis, semanas 23-24

Minimo: tres hombres con saco/pantalon y tres mujeres con saco/falda. Registrar
correcciones realizadas por un sastre en cuello, hombro, pecho, cintura, manga,
tiro y largos.

## 5. Mejoras despues del prototipo

Prioridad 1 - calidad de datos:

- active learning de capturas con mayor incertidumbre;
- detector de ropa holgada;
- calibracion por tipo corporal y condiciones de captura;
- auditoria de error por sexo, talla, altura y sitio de captura.

Prioridad 2 - mejor modelo:

- preentrenamiento sintetico de cuatro vistas SMPL-X;
- distillation desde SHAPY/SAM 3D Body hacia un encoder pequeno;
- view dropout para tolerar una foto defectuosa;
- fusion con mascara de vistas y embeddings de angulo reales;
- calibracion conformal adaptativa por medida.

Prioridad 3 - inferencia:

- exportar el regresor estable a ONNX;
- TensorRT FP16 despues de verificar paridad numerica;
- cachear mascaras, keypoints y betas;
- descargar cada modelo antes de iniciar el siguiente;
- medir latencia P50/P95 y pico real de VRAM.

Prioridad 4 - patronaje:

- perfiles de holgura aprobados por patronista;
- correcciones automaticas aprendidas de las pruebas fisicas;
- simulacion GarmentCode por material;
- exportacion DXF solo despues de validar escala y convenciones de corte.

## 6. Criterios de exito

- el modelo multivista supera media de train y modelos de una sola vista;
- Test-A controlado: objetivo inicial MAE <= 2 cm en pecho/cintura/cadera;
- Test-B/local: objetivo inicial MAE <= 3 cm en medidas primarias;
- cobertura conformal de 90% cercana a 87-93% por medida;
- captura aceptada en mas de 90% de intentos guiados;
- patron generado en menos de 5 segundos en CPU;
- flujo local de medicion por debajo de 30 segundos;
- cero OOM en la configuracion publicada;
- seis conjuntos fisicos evaluados por un patronista.

Los umbrales son hipotesis de trabajo. Si no se alcanzan, se reportan el error,
las causas y la ruta de mejora; no se ocultan ni se redefine el test.

## 7. Riesgos que no deben ocultarse

- BodyM contiene solo dos vistas y no todas las medidas de un patron formal.
- una foto sin referencia no determina escala absoluta;
- una silueta con ropa holgada representa la ropa, no el cuerpo;
- VTON no demuestra ajuste fisico;
- el saco femenino requiere trabajo real de patronaje;
- BodyM, SMPL-X, SHAPY y varios VTON tienen restricciones no comerciales;
- Colab ayuda a entrenar, pero no reemplaza un entorno reproducible y versionado.
