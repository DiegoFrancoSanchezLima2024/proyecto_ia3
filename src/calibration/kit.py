"""Patrones oficiales OpenCV en SVG métrico (no son imágenes generadas por IA)."""
from __future__ import annotations

from html import escape
from pathlib import Path

import cv2
import numpy as np

from .common import tablero_charuco, dictionary, write_json


def binary_svg(pixels, width_mm, height_mm):
    """Agrupa filas idénticas para preservar módulos rectos sin interpolación."""
    height, width = pixels.shape
    paths = []
    start = 0
    for end in range(1, height + 1):
        if end < height and np.array_equal(pixels[start], pixels[end]):
            continue
        black = pixels[start] < 128
        transitions = np.diff(np.r_[False, black, False].astype(int))
        for left, right in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            paths.append(f"M{left} {start}h{right-left}v{end-start}h{left-right}z")
        start = end
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" '
            f'viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="white"/>'
            f'<path d="{"".join(paths)}" fill="black"/></svg>')


def generar_kit(config, output, force=False):
    output = Path(output)
    names = ["imprimir.html", "charuco.svg", "station.json"] + [f"aruco_{m['id']}.svg" for m in config["floor"]["markers"]]
    if not force and any((output / name).exists() for name in names):
        raise ValueError("El kit ya existe. Use otra carpeta o --force para regenerar estos archivos.")
    output.mkdir(parents=True, exist_ok=True)
    board = tablero_charuco(config)
    spec = config["charuco"]
    width_mm = spec["squares_x"] * spec["square_m"] * 1000
    height_mm = spec["squares_y"] * spec["square_m"] * 1000
    if width_mm > 180 or height_mm > 210 or config["floor"]["marker_m"] * 1000 > 170:
        raise ValueError("El patrón no cabe en el diseño A4 actual; usar la configuración suministrada")
    # 10 px/mm; cuadrado de 30 mm ocupa exactamente 300 px.
    raster = board.generateImage((round(width_mm * 10), round(height_mm * 10)), marginSize=0, borderBits=1)
    (output / "charuco.svg").write_text(binary_svg(raster, width_mm, height_mm), encoding="utf-8")
    pages = [f'''<section><h1>Sastre-IA · Calibración de lente</h1>
<p>ChArUco {spec['squares_x']} × {spec['squares_y']} · cuadrados {spec['square_m']*1000:g} mm · marcadores {spec['marker_m']*1000:g} mm<br>Imprimir A4 al 100 %. Sin ajustar a página.</p>
<img class="board" src="charuco.svg" style="width:{width_mm}mm;height:{height_mm}mm"/>
<div class="ruler"></div><p class="small">Verificar con regla: línea = 100 mm; cuadrado = {spec['square_m']*1000:g} mm.<br>Pegar plano sobre soporte rígido, sin deformar el papel.</p></section>''']
    for marker in config["floor"]["markers"]:
        marker_id = marker["id"]
        side_mm = config["floor"]["marker_m"] * 1000
        raster = cv2.aruco.generateImageMarker(dictionary(config), marker_id, 600, borderBits=1)
        (output / f"aruco_{marker_id}.svg").write_text(binary_svg(raster, side_mm, side_mm), encoding="utf-8")
        x, y = marker["center_m"]
        pages.append(f'''<section><h1>Sastre-IA · ArUco {marker_id}</h1>
<p>{escape(marker['label'])} · centro X={x*100:.0f} cm, Y={y*100:.0f} cm<br>↑ ARRIBA DE ESTA HOJA HACIA EL FONDO (+Y) ↑</p>
<img class="marker" src="aruco_{marker_id}.svg" style="width:{side_mm}mm;height:{side_mm}mm"/>
<p>Lado negro exterior: {side_mm:.0f} mm. Conservar al menos 10 mm<br>de margen blanco. No girar al montar.</p>
<div class="ruler"></div><p class="small">Línea = 100 mm. A4 al 100 %, sin ajuste automático.<br>El margen blanco NO forma parte de los {side_mm:.0f} mm.</p></section>''')
    html = '''<!doctype html><html lang="es"><meta charset="utf-8"><title>Sastre-IA — Kit métrico A4</title>
<style>@page{size:A4;margin:0}*{box-sizing:border-box}body{margin:0;background:#ddd;font-family:Arial,sans-serif;color:#000}
section{width:210mm;height:297mm;padding:10mm 12mm;background:white;margin:8mm auto;break-after:page;text-align:center;overflow:hidden}
section:last-child{break-after:auto}h1{font-size:17px;margin:0 0 3mm}p{font-size:12px;line-height:1.4;margin:0 0 4mm}
img{display:block;margin:0 auto 4mm}.marker{margin-top:12mm;margin-bottom:12mm}.ruler{width:100mm;height:3mm;border:0.3mm solid black;border-top:0;margin:3mm auto}
.small{font-size:11px}@media print{body{background:white}section{margin:0}}</style>''' + "\n".join(pages) + "</html>"
    (output / "imprimir.html").write_text(html, encoding="utf-8")
    write_json(output / "station.json", config)
    return {"print_file": str((output / "imprimir.html").resolve()), "pages": len(pages), "physical_scale_verified": False}
