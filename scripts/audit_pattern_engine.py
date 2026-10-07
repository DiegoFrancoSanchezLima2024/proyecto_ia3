"""Informa qué falta para generar patrones reales con FreeSewing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.patterns.freesewing_bridge import auditar_entradas_freesewing, cargar_mapa_medidas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument(
        "--mapping", default="configs/freesewing_measurement_map.json"
    )
    parser.add_argument(
        "--output", default="outputs/pattern_engine_readiness.json"
    )
    args = parser.parse_args()
    prediction = json.loads((ROOT / args.prediction).read_text(encoding="utf-8"))
    mapping = cargar_mapa_medidas(ROOT / args.mapping)
    report = auditar_entradas_freesewing(prediction, mapping)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

