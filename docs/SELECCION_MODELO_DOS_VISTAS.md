# Selección del modelo oficial de dos vistas

## Entradas permitidas

La ruta oficial usa únicamente:

1. silueta frontal;
2. silueta de perfil izquierdo;
3. estatura en centímetros.

El peso y el sexo no son entradas del estimador. La opción de conjunto
`saco+pantalón` o `saco+falda` solo selecciona el patronaje posterior.

## Comparación independiente

Se compararon dos modelos entrenados con BodyM:

| Modelo | Test-A MAE 13 medidas | Test-A MAE peso | Test-B MAE 13 medidas | Test-B MAE peso |
|---|---:|---:|---:|---:|
| `exp_007_multitarea_sin_peso` (CNN) | 4.79 cm* | 17.45 kg | 4.48 cm* | 16.31 kg |
| `exp_008_perfiles_arboles` (ExtraTrees) | **1.92 cm** | **3.60 kg** | **2.18 cm** | **4.86 kg** |

\* El reporte global de la CNN mezclaba 13 centímetros con kilogramos. Para
esta tabla se promedian solo las 13 medidas corporales a partir de los archivos
de predicción independientes.

La CNN alcanzó 1.84 cm dentro de validación, pero sufrió cambio de dominio en
Test-A/Test-B y sobrestimó el peso con sesgos de +17.45 y +15.88 kg. Por eso no
se usa en la demo, aunque queda conservada como ablación.

`exp_008` transforma cada silueta en perfiles de ancho físico aproximado:

```text
ancho_cm ≈ ancho_px / alto_corporal_px × estatura_cm
```

Después, un ensamble ExtraTrees aprende conjuntamente las 13 medidas y el peso.
Este diseño fue más estable entre los tres dominios BodyM y puede explicarse y
auditarse con facilidad.

## Incertidumbre y decisión

Los intervalos se calibraron exclusivamente con las 904 muestras del split de
calibración. El radio conformal de peso al 90 % es 6.89 kg. Pecho, cintura,
cadera y muslo conservan intervalos amplios y quedan marcados para revisión.

Una media global cercana a 2 cm no garantiza ±2 cm para toda persona ni para
toda medida. El software debe mostrar los intervalos y bloquear el estado
“listo para corte” cuando la captura o la incertidumbre no sean suficientes.

## Limitación principal

La silueta de ropa holgada describe la ropa, no el cuerpo. En la prueba femenina
de 148 cm y 46 kg con sudadera ancha, el modelo estimó 70.36 kg. Esa captura no
es válida para medición corporal. La interfaz parte de “ajuste no confirmado” y
solo debe aceptar como confiables fotografías con ropa ceñida y protocolo BodyM.

## Prueba real disponible

Para el hombre de 165 cm y 70 kg:

| Variable | Real | Estimación | Error |
|---|---:|---:|---:|
| Peso | 70 kg | 73.04 kg | +3.04 kg |
| Pecho | 98 cm | 101.32 cm | +3.32 cm |
| Cintura | 87 cm | 88.85 cm | +1.85 cm |
| Muslo | 54 cm | 54.12 cm | +0.12 cm |
| Pierna | 78 cm | 76.32 cm | −1.68 cm |
| Brazo | 54 cm | 48.66 cm | −5.34 cm |

El largo de brazo debe permanecer marcado para validación: además del error,
la definición BodyM puede no coincidir exactamente con “hombro hasta muñeca”.
