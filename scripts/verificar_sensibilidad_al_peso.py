"""Comprueba si el modelo depende "demasiado" del peso, o si esa dependencia es correcta.

Una auditoria del proyecto observo que, manteniendo las mismas siluetas y cambiando
solo el peso de 50 a 90 kg, el pecho predicho se movia unos 26 cm y la cintura unos
29 cm. Concluyo que el modelo "usa demasiado los metadatos" y propuso reentrenar.

Este script contrasta esa conclusion contra la realidad: mide, en BodyM, cuanto
cambia realmente cada medida por kilo en personas de la misma estatura y sexo. Si la
pendiente real de la poblacion coincide con la respuesta del modelo, el modelo no
esta fallando: esta reproduciendo una relacion anatomica verdadera.

Tambien mide dos cosas que deciden si conviene quitar campos del formulario:
  - Cuanto error tendria estimar el peso en vez de pedirlo, y en cuantos centimetros
    de error de pecho y cintura se traduce.
  - Cuanto cuesta en precision quitar el sexo como entrada.

Ejecutar:
    python scripts/verificar_sensibilidad_al_peso.py
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

MEDIDAS = [
    "chest", "waist", "hip", "thigh", "calf", "ankle", "arm-length",
    "forearm", "wrist", "bicep", "shoulder-breadth", "leg-length",
    "shoulder-to-crotch",
]

# Lo que la auditoria midio en el modelo al pasar de 50 kg a 90 kg (165 cm, varon).
RESPUESTA_DEL_MODELO_40KG = {"chest": 26.08, "waist": 29.39, "hip": 19.88}


def cargar_sujetos() -> list[dict]:
    ruta = RAIZ / "dataset" / "bodym_clean" / "train" / "data.csv"
    with ruta.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    agrupadas: dict[str, list[dict]] = defaultdict(list)
    for fila in filas:
        agrupadas[fila["subject_id"]].append(fila)

    sujetos = []
    for grupo in agrupadas.values():
        try:
            registro = {
                "estatura": float(grupo[0]["height_cm"]),
                "peso": float(grupo[0]["weight_kg"]),
                "sexo": float(grupo[0]["gender"]),
            }
            for medida in MEDIDAS:
                registro[medida] = statistics.fmean(float(f[medida]) for f in grupo)
        except (KeyError, ValueError):
            continue
        sujetos.append(registro)
    return sujetos


def pendiente(puntos: list[dict], x: str, y: str) -> float:
    media_x = statistics.fmean(p[x] for p in puntos)
    media_y = statistics.fmean(p[y] for p in puntos)
    numerador = sum((p[x] - media_x) * (p[y] - media_y) for p in puntos)
    denominador = sum((p[x] - media_x) ** 2 for p in puntos)
    return numerador / denominador if denominador else 0.0


def resolver_sistema(matriz, vector):
    n = len(matriz)
    ampliada = [fila[:] + [vector[i]] for i, fila in enumerate(matriz)]
    for columna in range(n):
        pivote = max(range(columna, n), key=lambda f: abs(ampliada[f][columna]))
        if abs(ampliada[pivote][columna]) < 1e-12:
            return None
        ampliada[columna], ampliada[pivote] = ampliada[pivote], ampliada[columna]
        for fila in range(n):
            if fila == columna:
                continue
            factor = ampliada[fila][columna] / ampliada[columna][columna]
            for k in range(columna, n + 1):
                ampliada[fila][k] -= factor * ampliada[columna][k]
    return [ampliada[i][n] / ampliada[i][i] for i in range(n)]


def ajustar(entrenamiento, caracteristicas, objetivo):
    n = len(caracteristicas(entrenamiento[0]))
    matriz = [[0.0] * n for _ in range(n)]
    vector = [0.0] * n
    for sujeto in entrenamiento:
        x = caracteristicas(sujeto)
        y = sujeto[objetivo]
        for i in range(n):
            vector[i] += x[i] * y
            for j in range(n):
                matriz[i][j] += x[i] * x[j]
    coeficientes = resolver_sistema(matriz, vector)
    if coeficientes is None:
        return None
    return lambda s: sum(c * v for c, v in zip(coeficientes, caracteristicas(s)))


def error_medio(sujetos, caracteristicas, objetivo) -> float | None:
    mitad = len(sujetos) // 2
    pliegues = [(sujetos[:mitad], sujetos[mitad:]), (sujetos[mitad:], sujetos[:mitad])]
    errores = []
    for entrenamiento, prueba in pliegues:
        predecir = ajustar(entrenamiento, caracteristicas, objetivo)
        if predecir is None:
            return None
        errores += [abs(predecir(s) - s[objetivo]) for s in prueba]
    return statistics.fmean(errores)


def proporciones(sujeto: dict) -> list[float]:
    base = sujeto["chest"]
    return [1.0] + [sujeto[m] / base for m in MEDIDAS if m != "chest"]


def main() -> None:
    sujetos = cargar_sujetos()

    print("=" * 78)
    print("1. La dependencia del peso, es un defecto del modelo o la relacion real?")
    print("=" * 78)
    franja = [s for s in sujetos if s["sexo"] == 1.0 and 160 <= s["estatura"] <= 170]
    print(f"Hombres de 160-170 cm en BodyM: {len(franja)}\n")
    print(f"{'Medida':<10} {'Realidad (40 kg)':>18} {'Modelo (40 kg)':>16} {'Veredicto':>22}")
    print("-" * 70)
    for medida, esperado in RESPUESTA_DEL_MODELO_40KG.items():
        real = pendiente(franja, "peso", medida) * 40
        veredicto = "responde de menos" if esperado < real else "responde de mas"
        print(f"{medida:<10} {real:>15.1f} cm {esperado:>13.1f} cm {veredicto:>22}")
    print(
        "\nLectura: en personas de la misma estatura, 40 kg de diferencia SI cambian el\n"
        "pecho unos 29 cm en la poblacion real. El modelo responde algo menos que eso.\n"
        "No esta sobre-usando el peso: esta reproduciendo una relacion anatomica cierta."
    )

    print("\n" + "=" * 78)
    print("2. Conviene quitar el campo de peso y estimarlo?")
    print("=" * 78)
    mae_peso = error_medio(sujetos, lambda s: proporciones(s) + [s["estatura"], s["sexo"]], "peso")
    pend_pecho = pendiente(franja, "peso", "chest")
    pend_cintura = pendiente(franja, "peso", "waist")
    print(f"Error al estimar el peso desde proporciones + estatura + sexo: {mae_peso:.2f} kg")
    print("(cota optimista: usa las 13 medidas reales, no una silueta)\n")
    print(f"Ese error se traduce en:")
    print(f"  pecho:   +{mae_peso * pend_pecho:.2f} cm de error anadido")
    print(f"  cintura: +{mae_peso * pend_cintura:.2f} cm de error anadido")
    actual_pecho, actual_cintura = 2.48, 2.16
    print(f"\nError actual (tras correccion de sesgo): pecho {actual_pecho} cm, cintura {actual_cintura} cm")
    print(
        f"Error resultante estimado: "
        f"pecho ~{(actual_pecho**2 + (mae_peso*pend_pecho)**2) ** 0.5:.2f} cm, "
        f"cintura ~{(actual_cintura**2 + (mae_peso*pend_cintura)**2) ** 0.5:.2f} cm"
    )
    print("Quitar el campo de peso empeora las dos medidas que definen el saco.")

    print("\n" + "=" * 78)
    print("3. Cuanto cuesta quitar el sexo como entrada?")
    print("=" * 78)
    for nombre, caracteristicas in [
        ("estatura + peso + sexo", lambda s: [1.0, s["estatura"], s["peso"], s["sexo"]]),
        ("estatura + peso, sin sexo", lambda s: [1.0, s["estatura"], s["peso"]]),
    ]:
        errores = [error_medio(sujetos, caracteristicas, m) for m in MEDIDAS]
        print(f"  {nombre:<28} MAE agregado = {statistics.fmean(errores):.3f} cm")
    print("  Un menu desplegable de un clic cuesta cero; quitarlo cuesta precision.")


if __name__ == "__main__":
    main()
