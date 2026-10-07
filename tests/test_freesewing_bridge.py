from src.patterns.freesewing_bridge import auditar_entradas_freesewing


def test_bridge_reports_exact_missing_measurements_without_guessing():
    prediction = {
        "measurements": {
            "waist": {"estimate_cm": 87},
            "hip": {"estimate_cm": 98},
        },
        "bespoke": {"garments": {"skirt": {}}},
    }
    mapping = {
        "engine": "FreeSewing",
        "designs": {
            "skirt": {
                "design": "penelope",
                "package": "@freesewing/penelope",
                "required": {
                    "waist": "waist",
                    "seat": "hip",
                    "waistToKnee": None,
                },
            }
        },
    }
    report = auditar_entradas_freesewing(prediction, mapping)
    skirt = report["garments"]["skirt"]
    assert skirt["measurements_mm"]["waist"] == 870
    assert skirt["measurements_mm"]["seat"] == 980
    assert skirt["missing_measurements"] == ["waistToKnee"]
    assert not report["ready"]
