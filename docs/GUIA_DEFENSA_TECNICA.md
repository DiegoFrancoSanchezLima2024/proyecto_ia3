# Guía técnica para la defensa de SATRE-IA

## 1. Problema y alcance

SATRE-IA es una herramienta local de apoyo al sastre. Recibe una fotografía
frontal, un perfil izquierdo y la estatura; devuelve peso estimado, 13 medidas
corporales con incertidumbre, patrones paramétricos y una visualización virtual
experimental.

El sistema no promete reemplazar la validación del sastre ni garantizar ±2 cm
en todas las medidas. Su aporte es automatizar una primera estimación trazable,
reducir captura manual y producir un borrador reproducible de patronaje.

## 2. Por qué BodyM

BodyM es adecuado para el problema porque vincula:

- vistas frontal y lateral;
- siluetas segmentadas;
- estatura y peso;
- medidas antropométricas objetivo;
- identificadores de persona para separar sujetos entre entrenamiento y prueba.

La elección permite entrenar exactamente con las dos vistas que acepta la
interfaz. No se piden cuatro fotografías porque añadir entradas ausentes en el
dataset crearía una diferencia injustificada entre entrenamiento e inferencia.

Limitaciones que deben declararse:

- el dominio de BodyM se parece más a siluetas limpias que a fotografías con
  ropa holgada;
- la distribución demográfica puede no representar perfectamente la población
  local;
- una silueta 2D no contiene toda la geometría de un torso 3D;
- pecho, cintura y cadera son más difíciles que longitudes y articulaciones.

## 3. Modelos y procedencia

### Preentrenados

1. **MediaPipe Pose Landmarker Lite**: localiza landmarks y controla pose.
2. **SAM 2.1 Hiera Small**: segmenta a la persona en ambas fotografías.
3. **SCHP-ATR ONNX INT8**: separa cara, pelo y manos para el compositor 2D y
   para preservar identidad después de FASHN.
4. **FASHN VTON 1.5**: genera la visualización virtual experimental.

### Entrenado para el proyecto

**`exp_008_perfiles_arboles`** es un ensamble ExtraTrees entrenado con BodyM.
Convierte cada silueta en perfiles de ancho normalizados por estatura y predice
simultáneamente peso y 13 medidas. Los intervalos se calibran separadamente.

### Motor no neuronal

**FreeSewing 4.10.1** genera patrones paramétricos Jaeger, Charlie y Penelope.
No debe llamarse modelo de IA. Recibe medidas; no las predice.

### Modelos excluidos

- `exp_001`–`exp_007`: comparaciones y ablaciones, no runtime.
- SHAPY/SMPL-X: técnicamente instalados, pero desconectados del protocolo
  oficial porque no mejoraron la ruta validada y aumentaban complejidad.
- CatVTON/compositor de sprites: ruta histórica descartada como resultado
  realista. La superposición queda solo como diagnóstico geométrico.

## 4. Resultados que sí se pueden defender

| Evaluación | N | MAE 13 | RMSE 13 | R² 13 | Cobertura 90 % | MAE peso |
|---|---:|---:|---:|---:|---:|---:|
| Test-A | 1684 | 1,92 cm | 2,46 cm | 0,685 | 90,3 % | 3,60 kg |
| Test-B / libre | 1160 | 2,18 cm | 2,92 cm | 0,701 | 86,9 % | 4,86 kg |

En Test-B, aproximadamente:

- pecho: 4,14 cm MAE;
- cintura: 4,84 cm;
- cadera: 4,02 cm;
- largo de brazo: 1,34 cm;
- ancho de hombros: 1,17 cm;
- largo de pierna: 2,18 cm.

Conclusión correcta: el promedio global se acerca a 2 cm, pero los contornos
principales todavía requieren confirmación antes de cortar tela.

## 5. Arquitectura y separación de responsabilidades

```text
Captura
  ├─ MediaPipe: pose y caja
  └─ SAM2.1: silueta
          ↓
Características físicas
  └─ perfiles de ancho frontal/lateral × estatura
          ↓
Regresión exp_008
  ├─ peso
  ├─ 13 medidas
  └─ intervalos conformales
          ↓
Contrato de confección
  ├─ medida corporal
  ├─ holgura de prenda
  └─ estado de validación
          ↓
FreeSewing
  ├─ SVG 1:1
  ├─ PDF A4
  └─ JSON de proyecto

Ruta visual separada
  └─ FASHN VTON → imagen experimental
```

La visualización no retroalimenta medidas ni patrones. Esto evita que una
imagen aparentemente realista sea confundida con una validación métrica.

## 6. Guion de demo en vivo

Antes de entrar al aula:

```powershell
.\scripts\verificar_demo_defensa.ps1 -Completa
```

Para iniciar toda la demo local:

```powershell
.\scripts\iniciar_demo.ps1 -ConVestidor
```

1. Mostrar `configs/modelos_runtime.json` para probar qué modelos están activos.
2. Elegir explícitamente el conjunto de hombre o mujer.
3. Introducir estatura y cargar frontal + perfil con ropa ceñida.
4. Ejecutar la medición.
5. Mostrar `capture_manifest.json`: MediaPipe, SAM2.1, dispositivo y calidad.
6. Mostrar `prediction.json`: nombre de `exp_008`, 13 medidas, peso e intervalos.
7. Generar un molde y explicar por qué aparece como borrador si hay medidas no
   confirmadas.
8. Ejecutar o mostrar el resultado FASHN previamente generado, etiquetándolo
   como experimental.

Se recomienda conservar dos sesiones de respaldo —una masculina y otra
femenina— porque una inferencia generativa en vivo puede tardar más por carga en
frío aunque la medición y el patronaje funcionen correctamente.

## 7. Preguntas previsibles

**¿Por qué no usar todos los modelos descargados?**  Porque cada componente debe
tener una función y evidencia de mejora. Cargar modelos redundantes aumenta
latencia y dificulta atribuir el resultado.

**¿Por qué ExtraTrees y no la CNN?**  La CNN `exp_007` generalizó mal al eliminar
peso como entrada. ExtraTrees sobre perfiles físicos obtuvo menor error y es
determinista, rápido y explicable para el alcance del prototipo.

**¿El vestidor demuestra que el traje ajusta?**  No. Demuestra una posible
apariencia. El ajuste depende de las medidas, holguras, patrón, material y prueba
de confección.

**¿Por qué se bloquea el corte?**  Porque una medida estimada o derivada no debe
tratarse como confirmada. Es una decisión de seguridad y honestidad técnica.

**¿Funciona sin Internet?**  Sí. Pesos, inferencia, interfaz, patronaje y salidas
se ejecutan localmente.

## 8. Evidencia reproducible

- `experiments/exp_008_perfiles_arboles/results.json`
- `docs/SELECCION_MODELO_DOS_VISTAS.md`
- `docs/AUDITORIA_MODELOS_RUNTIME.md`
- `outputs/auditoria_modelos_runtime.json`
- `tests/test_auditoria_modelos_runtime.py`
- `pattern-engine/pruebas-contratos.mjs`
