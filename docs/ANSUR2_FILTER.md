# Control antropométrico ANSUR II

Sastre-IA usa ANSUR II como una segunda opinión estadística después de la
predicción de `exp_006_weight_huber`. Este control **no entrena el modelo, no
reemplaza BodyM y no modifica centímetros**.

## Qué hace

- Selecciona la referencia masculina o femenina indicada por la interfaz.
- Condiciona cada referencia por estatura y peso mediante una regresión lineal.
- Compara pecho, cintura, cadera, muslo, pantorrilla, tobillo, muñeca y anchura
  biacromial con un intervalo residual empírico del 95 %.
- Para una salida automática compara intervalos completos y solo alerta cuando
  no existe solapamiento; para una medida manual compara el valor confirmado.
- Marca valores atípicos para confirmación y reduce la confianza de talla cuando
  una medida empleada por esa prenda es atípica.
- Se desactiva fuera del percentil 1–99 de estatura o peso de la población de
  referencia; no extrapola silenciosamente.

No se mapearon largo de brazo ni largo de pierna porque sus puntos anatómicos no
son idénticos entre BodyM y ANSUR II. Tampoco se usa ANSUR II como tabla universal
de tallas: su población es militar estadounidense y puede introducir sesgos.

## Datos y reproducción

Fuente institucional: [U.S. Army / Defense Centers for Public Health – Aberdeen](https://ph.health.mil/topics/workplacehealth/ergo/Pages/Anthropometric-Database.aspx).
El lanzamiento público contiene 4.082 hombres y 1.986 mujeres. Los CSV locales
se obtuvieron de un espejo público de GitHub, están en `dataset/ansur2/raw/` y se
omiten de Git por tratarse de datos regenerables. El JSON registra tanto la
fuente institucional como el espejo y los SHA-256 para que la procedencia sea
auditable.

Para reconstruir la referencia compacta:

```powershell
C:\Users\diego\miniconda3\Scripts\conda.exe run -n sastre-ia `
  python scripts/build_ansur2_reference.py
```

El artefacto auditable resultante es
`configs/anthropometry/ansur2_reference.json`; incluye conteos, equivalencias,
coeficientes, dominios y SHA-256 de cada CSV.

## Interpretación

`plausible` solo significa compatible con la distribución condicionada de esta
referencia. No certifica exactitud de ±2 cm. `outlier` tampoco significa que el
cuerpo esté mal: indica que conviene revisar la predicción, la ropa o el punto
anatómico antes de fabricar un molde.
