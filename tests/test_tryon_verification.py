"""Verificacion medida del compositor de identidad del vestidor.

Estas comprobaciones existen porque `composite_tryon_identity.py` afirmaba
`background_preserved: True`, `head_preserved: True` y `hands_preserved: True`
como constantes escritas a mano, sin comprobar un solo pixel. Un compositor que
declara su propia correccion no es auditable.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts" / "composite_tryon_identity.py").read_text(encoding="utf-8")


def _load(name: str):
    """Extrae una funcion pura del script sin importar sus dependencias de GPU."""
    block = re.search(rf"def {name}\(.*?\n\n\n", SOURCE, re.S)
    assert block, f"No se encontro la funcion {name}"
    namespace: dict = {"np": np, "Image": Image}
    exec(block.group(0), namespace)
    return namespace[name]


restore_mask_report = _load("restore_mask_report")
boundary_step_report = _load("boundary_step_report")
preservation_report = _load("preservation_report")

SIZE = (240, 320)


def _rectangular_restore() -> Image.Image:
    """La mascara que produce la franja: rectangulo de ancho completo."""
    mask = Image.new("L", SIZE, 0)
    ImageDraw.Draw(mask).rectangle((0, 0, SIZE[0], 120), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(2.2))


def _semantic_restore() -> Image.Image:
    """Una mascara de cara/pelo plausible: nunca cubre el ancho completo."""
    mask = Image.new("L", SIZE, 0)
    ImageDraw.Draw(mask).ellipse((90, 30, 150, 110), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(2.2))


def test_detecta_mascara_rectangular_de_ancho_completo():
    report = restore_mask_report(_rectangular_restore())
    assert report["rectangular_restore_detected"] is True
    assert report["max_row_coverage"] > 0.9
    assert report["full_width_rows"] > 50


def test_mascara_semantica_no_se_marca_como_rectangular():
    report = restore_mask_report(_semantic_restore())
    assert report["rectangular_restore_detected"] is False
    assert report["max_row_coverage"] < 0.9


def test_detecta_salto_tonal_en_el_borde_de_restauracion():
    """Original claro arriba, generado oscuro abajo: la costura debe verse."""
    array = np.full((SIZE[1], SIZE[0], 3), 200, dtype=np.uint8)
    array[125:, :, :] = 120
    report = boundary_step_report(Image.fromarray(array), _rectangular_restore())
    assert report["visible_band_detected"] is True
    assert report["max_row_step"] > 8.0


def test_imagen_continua_no_reporta_franja():
    gradient = np.linspace(80, 200, SIZE[1], dtype=np.uint8)
    array = np.repeat(gradient[:, None], SIZE[0], axis=1)
    image = Image.fromarray(np.stack([array] * 3, axis=2))
    assert boundary_step_report(image, _rectangular_restore())["visible_band_detected"] is False


def test_fondo_alterado_se_reporta_como_no_preservado():
    original = Image.new("RGB", SIZE, (30, 60, 90))
    changed = original.copy()
    ImageDraw.Draw(changed).rectangle((0, 0, SIZE[0], 200), fill=(200, 40, 40))
    region = Image.new("L", SIZE, 255)
    report = preservation_report(original, changed, region)
    assert report["preserved"] is False
    assert report["changed_percent"] > 1.0


def test_fondo_intacto_se_reporta_como_preservado():
    original = Image.new("RGB", SIZE, (30, 60, 90))
    region = Image.new("L", SIZE, 255)
    report = preservation_report(original, original.copy(), region)
    assert report["preserved"] is True
    assert report["changed_percent"] == 0.0


def test_el_compositor_no_vuelve_a_afirmar_preservacion_sin_medirla():
    """Guarda contra la regresion concreta que motivo este archivo."""
    for claim in ('"background_preserved": True', '"head_preserved": True', '"hands_preserved": True'):
        assert claim not in SOURCE, f"Volvio la afirmacion hardcodeada {claim}"


def test_runtime_usa_identidad_semantica_y_no_franja_de_cabeza():
    assert "infer_atr(original_full" in SOURCE
    assert 'regiones["face"]' in SOURCE
    assert 'regiones["hair"]' in SOURCE
    assert 'regiones["hands"]' in SOURCE
    assert "draw.rectangle((0, 0, size[0]" not in SOURCE
