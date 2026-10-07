from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.body3d.readiness import preparacion_cuerpo3d


if __name__ == "__main__":
    status = preparacion_cuerpo3d(ROOT)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    raise SystemExit(0 if status["ready"] else 2)

