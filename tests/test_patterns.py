from src.patterns.draft_export import trazar_piezas, renderizar_svg


def _jacket(status="draft_ready"):
    return {
        "garment": "jacket",
        "pattern_status": status,
        "finished_garment_measurements_cm": {
            "chest": 108, "waist": 95, "hip": 104,
            "shoulder-to-crotch": 64, "arm-length": 54,
            "bicep": 36, "wrist": 21,
        },
        "seam_allowance_cm": 1.5,
        "hem_allowance_cm": 4,
    }


def test_draft_geometry_has_physical_dimensions_and_assumptions():
    pieces, assumptions = trazar_piezas(_jacket(), 164)
    assert {piece.name for piece in pieces} == {"frente", "espalda", "manga base"}
    assert all(piece.width_cm > 0 and piece.height_cm > 0 for piece in pieces)
    assert assumptions


def test_svg_is_one_to_one_and_watermarks_unvalidated_draft():
    svg = renderizar_svg(_jacket(), 164)
    assert 'width="' in svg and "mm" in svg
    assert "Línea de control: 10 cm" in svg
    assert "BORRADOR - NO CORTAR" in svg
    assert "S/M/L" not in svg


def test_svg_removes_watermark_only_for_cut_ready_contract():
    svg = renderizar_svg(_jacket("cut_ready"), 164)
    assert "BORRADOR - NO CORTAR" not in svg
