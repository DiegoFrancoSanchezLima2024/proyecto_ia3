"""Mide si la estatura se puede estimar desde la forma del cuerpo, con datos propios.

La pregunta que esto responde: si la interfaz dejara de pedir la estatura escrita
y la estimara desde las fotos, cuanto error tendria esa estimacion?

El experimento calcula una COTA SUPERIOR OPTIMISTA. En vez de estimar la estatura
desde una silueta (que es lo que haria el sistema real), la estima desde las 13
medidas REALES tomadas con cinta metrica en BodyM. Esas medidas son mucho mas
informativas y mas precisas que cualquier cosa que un modelo pueda extraer de una
silueta de 320x320. Por lo tanto: si la estatura no se puede estimar bien ni con
esta informacion privilegiada, desde una silueta es imposible.

El caso decisivo es el 2 y el 3: solo PROPORCIONES entre medidas, sin escala
absoluta. Eso es exactamente lo que le queda a una silueta, porque el
preprocesamiento (src/data/bodym_dataset.py) redimensiona toda silueta a 320x320
y con eso borra el tamano real de la persona.

Ejecutar:
    python scripts/experimento_estimabilidad_estatura.py
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


def cargar_sujetos(ruta_csv: Path) -> list[dict]:
    """Promedia las fotos de cada sujeto: evita contar dos veces a la misma persona."""
    with ruta_csv.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))

    por_sujeto: dict[str, list[dict]] = defaultdict(list)
    for fila in filas:
        por_sujeto[fila["subject_id"]].append(fila)

    sujetos = []
    for grupo in por_sujeto.values():
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


def resolver_sistema(matriz: list[list[float]], vector: list[float]) -> list[float] | None:
    """Eliminacion gaussiana con pivoteo parcial."""
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


def ajustar_minimos_cuadrados(entrenamiento: list[dict], caracteristicas):
    n = len(caracteristicas(entrenamiento[0]))
    matriz = [[0.0] * n for _ in range(n)]
    vector = [0.0] * n
    for sujeto in entrenamiento:
        x = caracteristicas(sujeto)
        y = sujeto["estatura"]
        for i in range(n):
            vector[i] += x[i] * y
            for j in range(n):
                matriz[i][j] += x[i] * x[j]
    coeficientes = resolver_sistema(matriz, vector)
    if coeficientes is None:
        return None
    return lambda s: sum(c * v for c, v in zip(coeficientes, caracteristicas(s)))


def evaluar(nombre: str, sujetos: list[dict], caracteristicas) -> None:
    """Validacion cruzada 2-fold disjunta por sujeto."""
    mitad = len(sujetos) // 2
    pliegues = [(sujetos[:mitad], sujetos[mitad:]), (sujetos[mitad:], sujetos[:mitad])]
    errores = []
    for entrenamiento, prueba in pliegues:
        predecir = ajustar_minimos_cuadrados(entrenamiento, caracteristicas)
        if predecir is None:
            print(f"{nombre}: sistema singular, no evaluable")
            return
        errores += [abs(predecir(s) - s["estatura"]) for s in prueba]
    mae = statistics.fmean(errores)
    dentro_2 = statistics.fmean(e <= 2 for e in errores) * 100
    dentro_5 = statistics.fmean(e <= 5 for e in errores) * 100
    print(f"{nombre:<54} {mae:>6.2f} cm {dentro_2:>7.1f}% {dentro_5:>7.1f}%")


def solo_proporciones(sujeto: dict) -> list[float]:
    """Divide cada medida por el pecho: queda la forma, se pierde la escala."""
    base = sujeto["chest"]
    return [1.0] + [sujeto[m] / base for m in MEDIDAS if m != "chest"]


def main() -> None:
    sujetos = cargar_sujetos(RAIZ / "dataset" / "bodym_clean" / "train" / "data.csv")
    print(
        f"Sujetos: {len(sujetos)} (BodyM train, agregado por sujeto, "
        "validacion 2-fold disjunta)\n"
    )
    print(f"{'Informacion disponible para estimar la estatura':<54} {'MAE':>9} {'<=2cm':>7} {'<=5cm':>7}")
    print("-" * 82)

    evaluar("0. Nada: solo la media de la poblacion", sujetos, lambda s: [1.0])
    evaluar(
        "1. Las 13 medidas reales en cm (irreal: ya traen escala)",
        sujetos,
        lambda s: [1.0] + [s[m] for m in MEDIDAS],
    )
    evaluar(
        "2. Solo proporciones, sin escala (lo que ve la silueta)",
        sujetos,
        solo_proporciones,
    )
    evaluar(
        "3. Proporciones + sexo",
        sujetos,
        lambda s: solo_proporciones(s) + [s["sexo"]],
    )
    evaluar(
        "4. Proporciones + sexo + peso escrito por el usuario",
        sujetos,
        lambda s: solo_proporciones(s) + [s["sexo"], s["peso"]],
    )

    print(
        "\nLectura: el caso 2 y el 3 son la cota superior de cualquier estimador\n"
        "visual de estatura. Un error de 5 cm sobre 165 cm es 3% de error de escala,\n"
        "y la escala multiplica TODAS las medidas: sobre un pecho de 100 cm son 3 cm\n"
        "de error anadido, mas que el error actual del modelo en esa medida.\n"
        "El caso 4 muestra que el peso escrito aporta la escala que falta: si hay que\n"
        "quitar un campo del formulario, conviene quitar la estatura y conservar el\n"
        "peso, no lo contrario."
    )


if __name__ == "__main__":
    main()
