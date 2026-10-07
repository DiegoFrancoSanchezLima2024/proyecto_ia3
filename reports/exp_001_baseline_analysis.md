# Análisis de `exp_001_baseline`

Fecha de evaluación: 2026-09-03  
GPU: NVIDIA GeForce RTX 3050 Laptop, 6 GB  
Checkpoint seleccionado: época 23 (no la última época)

## Dictamen

El experimento es un **baseline de investigación válido y reproducible**. El
modelo aprende información visual y generaliza a sujetos no vistos; no se limita
a reproducir estatura y sexo. Todavía **no es seguro usar sus salidas para cortar
tela**, porque los contornos principales de sastrería siguen teniendo errores e
intervalos demasiado grandes.

## Resultados principales

| Evaluación | Casos | MAE global | Dentro de ±2 cm | Cobertura conformal |
|---|---:|---:|---:|---:|
| Validación por captura | 892 | 1.778 cm | 71.1 % | — |
| Test-A por captura | 1,684 | 1.858 cm | 68.8 % | 89.1 % |
| Test-A consolidado por persona | 87 | 1.706 cm | — | 91.2 % |
| Test-B por captura | 1,160 | 2.090 cm | 67.4 % | 85.5 % |
| Test-B consolidado por persona | 400 | 1.961 cm | — | 87.4 % |

La consolidación promedia las predicciones de las capturas repetidas de cada
sujeto. Es la métrica más cercana a una demo con varias fotografías, pero se
deben conservar ambos niveles en la memoria de grado.

## Comparación contra referencias simples

| Split | Modelo visual | Ridge (estatura + sexo) | Mejora relativa |
|---|---:|---:|---:|
| Test-A | 1.858 cm | 3.293 cm | 43.6 % |
| Test-B | 2.090 cm | 3.664 cm | 42.9 % |

Esto demuestra que las siluetas aportan señal útil. La diferencia Validación →
Test-A es solo +0.080 cm y Test-A → Test-B es +0.233 cm por captura.

## Medidas críticas para patronaje

MAE consolidado por persona:

| Medida | Test-A | Test-B | Dictamen actual |
|---|---:|---:|---|
| Pecho | 3.36 cm | 4.15 cm | No apta para corte automático |
| Cintura | 4.08 cm | 4.58 cm | No apta; existe sesgo positivo |
| Cadera | 2.91 cm | 3.52 cm | Requiere mejora |
| Muslo | 2.09 cm | 2.37 cm | Útil como estimación asistida |
| Hombros | 0.76 cm | 0.93 cm | Buen resultado |
| Largo de brazo | 1.07 cm | 1.07 cm | Buen resultado |
| Largo de pierna | 1.63 cm | 1.65 cm | Aceptable para prototipo |

Los intervalos conformales de 90 % tienen semianchos de 6.79 cm para pecho,
8.26 cm para cintura y 5.89 cm para cadera. En Test-B la cobertura media cae a
85.5 % por captura (87.4 % por persona), lo que evidencia cambio de distribución.
No debe afirmarse que existe garantía del 90 % sobre fotografías reales externas.

## Lectura correcta del entrenamiento

- El `early stopping` funcionó: detuvo en la época 31 y preservó la época 23.
- Las pérdidas negativas son posibles en la pérdida Huber heterocedástica por el
  término de log-varianza; no significan que el entrenamiento esté corrupto.
- El pico de 2.46 GiB deja margen de VRAM. Aun así, el benchmark previo mostró
  mayor rendimiento con lote 16; subir el lote no acelera esta GPU.
- El aviso de `flash attention` es informativo. Con dos tokens de vista no es un
  cuello de botella relevante.

## Próximo experimento controlado

La pérdida heterocedástica puede restar peso a pecho, cintura y cadera por ser
los objetivos más ruidosos. La siguiente ablación debe cambiar **solo** la pérdida
a Huber, conservando datos, semilla, arquitectura y particiones:

```powershell
python src/training/train.py --exp exp_002_huber --loss huber
python src/evaluation/evaluate.py --exp experiments/exp_002_huber --split test
python src/evaluation/evaluate.py --exp experiments/exp_002_huber --split test_wild
```

No se debe iniciar todavía SMPL-X, moldes ni visualización de traje como si las
medidas fueran definitivas. Primero se compara `exp_002_huber` con estos criterios:

1. reducir pecho, cintura y cadera en Test-B consolidado;
2. no empeorar el MAE global consolidado por encima de 1.961 cm;
3. no degradar largos y hombros más de 0.2 cm;
4. después validar con cinta métrica en personas locales, separadas del ajuste.

## Estado real del proyecto

- Captura y medidas: baseline funcional de **dos vistas**, aún no cuatro vistas reales.
- Reconstrucción SMPL-X: activo planificado, no validado en este experimento.
- Extracción antropométrica: funcional como asistencia; todavía no automática para corte.
- Moldes de saco/pantalón/falda: no deben calificarse como terminados hasta producir
  piezas 2D con márgenes, piquetes, hilo, etiquetas y prueba de ensamblaje.
- Visualización frontal 2D: debe construirse después del módulo paramétrico y mostrarse
  como aproximación visual, no como prueba de ajuste físico.

