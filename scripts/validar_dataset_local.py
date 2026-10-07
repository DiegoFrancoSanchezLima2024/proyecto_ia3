"""Valida la estructura tabular del dataset local SATRE-IA sin leer imágenes.

Uso:
    python scripts/validar_dataset_local.py --registro dataset/local/registro

El validador detecta omisiones, duplicados y fuga de una persona entre splits.
No modifica los archivos ni decide si una medición anatómica fue bien tomada.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


VISTAS = {"frente", "izquierda", "espalda", "derecha"}
MEDIDAS = {
    "chest",
    "waist",
    "hip",
    "thigh",
    "calf",
    "ankle",
    "arm-length",
    "forearm",
    "wrist",
    "bicep",
    "shoulder-breadth",
    "leg-length",
    "shoulder-to-crotch",
}
ARCHIVOS = {
    "sujetos": "sujetos.csv",
    "capturas": "capturas.csv",
    "mediciones": "mediciones.csv",
    "splits": "asignacion_splits.csv",
}


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta {ruta}")
    with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


def es_verdadero(valor: str) -> bool:
    return valor.strip().lower() in {"1", "true", "si", "sí", "yes"}


def validar(registro: Path, repeticiones_medida: int) -> list[str]:
    errores: list[str] = []
    tablas = {nombre: leer_csv(registro / archivo) for nombre, archivo in ARCHIVOS.items()}

    ids = [fila.get("sujeto_id", "").strip() for fila in tablas["sujetos"]]
    ids_validos = {valor for valor in ids if valor}
    duplicados = sorted(valor for valor, total in Counter(ids).items() if valor and total > 1)
    if duplicados:
        errores.append(f"Sujetos duplicados: {', '.join(duplicados)}")

    splits_por_sujeto: dict[str, set[str]] = defaultdict(set)
    for fila in tablas["splits"]:
        sujeto = fila.get("sujeto_id", "").strip()
        split = fila.get("split", "").strip()
        if sujeto not in ids_validos:
            errores.append(f"Split referencia sujeto inexistente: {sujeto}")
        if split not in {"pilot", "calibration", "test", "train", "val"}:
            errores.append(f"Split inválido para {sujeto}: {split}")
        splits_por_sujeto[sujeto].add(split)
    for sujeto, splits in splits_por_sujeto.items():
        if len(splits) > 1:
            errores.append(f"Fuga de sujeto {sujeto} entre splits: {sorted(splits)}")
    for sujeto in sorted(ids_validos - set(splits_por_sujeto)):
        errores.append(f"Sujeto sin split: {sujeto}")

    capturas: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fila in tablas["capturas"]:
        sujeto = fila.get("sujeto_id", "").strip()
        ronda = fila.get("ronda", "").strip()
        vista = fila.get("vista", "").strip().lower()
        if sujeto not in ids_validos:
            errores.append(f"Captura referencia sujeto inexistente: {sujeto}")
        if vista not in VISTAS:
            errores.append(f"Vista inválida para {sujeto}, ronda {ronda}: {vista}")
        if es_verdadero(fila.get("aceptada", "")):
            capturas[(sujeto, ronda)].add(vista)
    for sujeto in sorted(ids_validos):
        for ronda in ("1", "2"):
            faltantes = VISTAS - capturas[(sujeto, ronda)]
            if faltantes:
                errores.append(
                    f"{sujeto}, ronda {ronda}: faltan capturas aceptadas {sorted(faltantes)}"
                )

    lecturas: Counter[tuple[str, str]] = Counter()
    combinaciones: Counter[tuple[str, str, str, str]] = Counter()
    for fila in tablas["mediciones"]:
        sujeto = fila.get("sujeto_id", "").strip()
        clave = fila.get("clave_medida", "").strip()
        medidor = fila.get("medidor_id", "").strip()
        repeticion = fila.get("repeticion", "").strip()
        if sujeto not in ids_validos:
            errores.append(f"Medición referencia sujeto inexistente: {sujeto}")
        if clave not in MEDIDAS:
            errores.append(f"Clave de medida inválida para {sujeto}: {clave}")
        try:
            valor = float(fila.get("valor_cm", ""))
            if not 0 < valor < 300:
                errores.append(f"Valor fuera de rango para {sujeto}/{clave}: {valor}")
        except ValueError:
            errores.append(f"Valor no numérico para {sujeto}/{clave}")
        if es_verdadero(fila.get("valida", "")):
            lecturas[(sujeto, clave)] += 1
        combinaciones[(sujeto, clave, medidor, repeticion)] += 1

    for combinacion, total in combinaciones.items():
        if total > 1:
            errores.append(f"Lectura duplicada {combinacion}")
    for sujeto in sorted(ids_validos):
        for clave in sorted(MEDIDAS):
            total = lecturas[(sujeto, clave)]
            if total < repeticiones_medida:
                errores.append(
                    f"{sujeto}/{clave}: {total} lecturas válidas; se requieren "
                    f"{repeticiones_medida}"
                )
    return errores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registro", type=Path, required=True)
    parser.add_argument(
        "--repeticiones-medida",
        type=int,
        default=4,
        help="Lecturas válidas requeridas por persona y medida (por defecto: 4)",
    )
    args = parser.parse_args()

    try:
        errores = validar(args.registro, args.repeticiones_medida)
    except (FileNotFoundError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if errores:
        print(f"Dataset inválido: {len(errores)} problema(s)")
        for error in errores:
            print(f"- {error}")
        return 1
    print("Dataset local válido: estructura, vistas, medidas y splits completos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
