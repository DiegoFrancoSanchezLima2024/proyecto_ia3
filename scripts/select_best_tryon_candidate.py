#!/usr/bin/env python3
"""Score every FASHN VTON candidate for a session and promote the best one.

Keeps generation and quality judgement separate and auditable:
`run_fashn_vton_local.py` never decides what "looks good", it only generates
candidates; this script is the only place that picks a winner, and it always
records why (`selection_report.json`) instead of silently shipping whichever
candidate happened to be last.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tryon_quality_checks import score_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--candidates-manifest", type=Path, required=True)
    parser.add_argument("--sex", choices=["male", "female"], required=True)
    parser.add_argument("--garment-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates_manifest.read_text(encoding="utf-8"))["candidates"]
    if not candidates:
        raise RuntimeError("No se genero ningun candidato utilizable: revisar candidates.json.")

    scored = []
    for entry in candidates:
        result = score_candidate(Path(entry["path"]), args.session_dir, args.sex, args.garment_image)
        result["seed"] = entry.get("seed")
        result["steps"] = entry.get("steps")
        result["inference_seconds"] = entry.get("inference_seconds")
        result["output_resolution"] = entry.get("output_resolution")
        scored.append(result)

    scored.sort(key=lambda item: item["score"], reverse=True)
    winner = scored[0]
    all_suspicious = all(item["verdict"] == "sospechoso" for item in scored)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(winner["candidate"], args.output)

    report = {
        "candidates_evaluated": len(scored),
        "winner": winner,
        "all_candidates": scored,
        "all_candidates_suspicious": all_suspicious,
        "notice": (
            "Seleccion automatica por heuristica de color/piel entre varios candidatos "
            "generativos. Revisar visualmente antes de usar en la defensa."
        ),
    }
    args.output.with_name("selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
