"""
tests/test_bias_correction.py
==============================
Verifica que el ajuste de sesgo por medida solo corrija corrimientos reales
y que se niegue a estimarlos sobre el split de test.
Ejecutar: pytest tests/ -v
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "fit_bias_correction", ROOT / "scripts" / "fit_bias_correction.py"
)
fit_bias_correction = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_bias_correction)


def _write_predictions(path: Path, waist_offset: float, n_subjects: int = 40) -> None:
    """Cintura con un corrimiento fijo; pecho con ruido simetrico sin sesgo."""
    fields = [
        "subject_id",
        "true_waist",
        "pred_waist",
        "true_chest",
        "pred_chest",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(n_subjects):
            noise = 0.5 if index % 2 == 0 else -0.5
            writer.writerow(
                {
                    "subject_id": f"s{index}",
                    "true_waist": 80.0,
                    "pred_waist": 80.0 + waist_offset + noise,
                    "true_chest": 95.0,
                    "pred_chest": 95.0 + noise,
                }
            )


def test_detecta_sesgo_real_e_ignora_ruido(tmp_path):
    predictions = tmp_path / "predictions_calibration.csv"
    _write_predictions(predictions, waist_offset=2.0)

    _, errors = fit_bias_correction.load_subject_errors(predictions)
    payload = fit_bias_correction.fit(errors, min_cm=0.2, z=2.0)

    assert payload["bias_cm"]["waist"] == pytest.approx(2.0, abs=0.01)
    assert "chest" not in payload["bias_cm"]
    assert payload["detail"]["chest"]["applied"] is False


def test_no_corrige_cuando_el_sesgo_es_despreciable(tmp_path):
    predictions = tmp_path / "predictions_calibration.csv"
    _write_predictions(predictions, waist_offset=0.05)

    _, errors = fit_bias_correction.load_subject_errors(predictions)
    payload = fit_bias_correction.fit(errors, min_cm=0.2, z=2.0)

    assert payload["bias_cm"] == {}


def test_rechaza_el_split_de_test(tmp_path, monkeypatch):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    predictions = exp_dir / "predictions_test.csv"
    _write_predictions(predictions, waist_offset=2.0)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit_bias_correction.py",
            "--exp-dir",
            str(exp_dir),
            "--predictions",
            str(predictions),
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        fit_bias_correction.main()
    assert "test" in str(excinfo.value)


def test_el_archivo_generado_es_legible_por_el_predictor(tmp_path):
    """El predictor lee bias_correction.json con la clave 'bias_cm'."""
    predictions = tmp_path / "predictions_calibration.csv"
    _write_predictions(predictions, waist_offset=2.0)
    _, errors = fit_bias_correction.load_subject_errors(predictions)
    payload = fit_bias_correction.fit(errors, min_cm=0.2, z=2.0)

    destination = tmp_path / "bias_correction.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["bias_cm"]["waist"] == pytest.approx(2.0, abs=0.01)
