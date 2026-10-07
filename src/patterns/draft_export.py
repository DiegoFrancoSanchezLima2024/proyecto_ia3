"""Exporta bloques 2D de verificación; no sustituye patronaje profesional."""

from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class Piece:
    name: str
    cut: str
    width_cm: float
    height_cm: float
    points: tuple[tuple[float, float], ...]


def _trapezoid(name: str, cut: str, top: float, bottom: float, height: float) -> Piece:
    width = max(top, bottom)
    offset_top = (width - top) / 2
    offset_bottom = (width - bottom) / 2
    return Piece(
        name,
        cut,
        width,
        height,
        (
            (offset_top, 0),
            (offset_top + top, 0),
            (offset_bottom + bottom, height),
            (offset_bottom, height),
        ),
    )


def trazar_piezas(garment: dict, height_cm: float) -> tuple[list[Piece], list[str]]:
    """Crea bloques esquemáticos y devuelve sus supuestos explícitos."""
    name = garment["garment"]
    m = garment["finished_garment_measurements_cm"]
    sa = float(garment["seam_allowance_cm"])
    hem = float(garment["hem_allowance_cm"])
    pieces: list[Piece]
    assumptions: list[str]

    if name == "jacket":
        quarter_chest = m["chest"] / 4
        quarter_waist = m["waist"] / 4
        length = m["shoulder-to-crotch"] + 8 + hem
        sleeve = m["arm-length"] + hem
        sleeve_top = m["bicep"] / 2 + 2 * sa
        sleeve_wrist = m["wrist"] / 2 + 2 * sa
        pieces = [
            _trapezoid("frente", "cortar 2 espejo", quarter_chest + 2*sa, quarter_waist + 2*sa, length),
            _trapezoid("espalda", "cortar 1 al doblez", quarter_chest + 2*sa, quarter_waist + 2*sa, length),
            _trapezoid("manga base", "cortar 2 espejo", sleeve_top, sleeve_wrist, sleeve),
        ]
        assumptions = [
            "Largo base de saco = hombro-entrepierna + 8 cm.",
            "La manga mostrada es un bloque de control; falta desarrollar copa y dos piezas.",
            "Faltan pinzas, solapa, cuello, sisas, bolsillos y balance delantero/espalda.",
        ]
    elif name == "trousers":
        rise = round(max(22.0, 0.15 * height_cm), 2)
        length = m["leg-length"] + rise + hem
        hip_front = m["hip"] / 4 + 2 * sa
        hem_front = m["ankle"] / 4 + 2 * sa
        pieces = [
            _trapezoid("pantalón delantero", "cortar 2 espejo", hip_front, hem_front, length),
            _trapezoid("pantalón espalda", "cortar 2 espejo", hip_front + 2, hem_front + 1, length),
        ]
        assumptions = [
            f"Tiro preliminar derivado de estatura: {rise:.1f} cm.",
            "El largo usa pierna + tiro + dobladillo; confirmar definición de leg-length.",
            "Faltan curvas de tiro, pinzas, pretina, bolsillos y aplomos de producción.",
        ]
    elif name == "skirt":
        length = round(0.38 * height_cm + hem, 2)
        waist = m["waist"] / 4 + 2 * sa
        hip = m["hip"] / 4 + 2 * sa
        pieces = [
            _trapezoid("falda delantera", "cortar 1 al doblez", waist, hip, length),
            _trapezoid("falda espalda", "cortar 2 espejo", waist, hip, length),
        ]
        assumptions = [
            "Largo preliminar = 38% de la estatura; debe elegirse en la interfaz.",
            "Faltan pinzas, abertura, pretina y ajuste de cadera para producción.",
        ]
    else:
        raise ValueError(f"Prenda no soportada: {name}")
    return pieces, assumptions


def renderizar_svg(garment: dict, height_cm: float) -> str:
    pieces, assumptions = trazar_piezas(garment, height_cm)
    gap = 8.0
    margin = 8.0
    max_height = max(piece.height_cm for piece in pieces)
    total_width = sum(piece.width_cm for piece in pieces) + gap * (len(pieces) - 1)
    width_cm = total_width + 2 * margin
    height_total_cm = max_height + 2 * margin + 18
    scale = 10.0  # 1 cm = 10 SVG units = 10 mm físico.
    watermark = garment["pattern_status"] != "cut_ready"

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_cm*10:.1f}mm" '
        f'height="{height_total_cm*10:.1f}mm" viewBox="0 0 {width_cm*scale:.1f} {height_total_cm*scale:.1f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">'
        '<path d="M 50 0 L 0 0 0 50" fill="none" stroke="#d7dde5" stroke-width="0.8"/>'
        '</pattern></defs>',
        '<rect width="100%" height="100%" fill="url(#grid)"/>',
        f'<text x="{margin*scale}" y="45" font-family="Arial" font-size="22" font-weight="bold">'
        f'SATRE-IA - {html.escape(garment["garment"])}</text>',
        f'<text x="{margin*scale}" y="70" font-family="Arial" font-size="14">'
        f'Estado: {html.escape(garment["pattern_status"])} | escala geométrica 1:1</text>',
    ]
    x = margin
    y = margin + 4
    for piece in pieces:
        points = " ".join(f"{(x+px)*scale:.2f},{(y+py)*scale:.2f}" for px, py in piece.points)
        chunks.append(f'<polygon points="{points}" fill="none" stroke="#18212f" stroke-width="2"/>')
        chunks.append(
            f'<text x="{(x+piece.width_cm/2)*scale:.2f}" y="{(y+piece.height_cm/2)*scale:.2f}" '
            'text-anchor="middle" font-family="Arial" font-size="13">'
            f'{html.escape(piece.name)} - {html.escape(piece.cut)}</text>'
        )
        x += piece.width_cm + gap
    note_y = (margin + max_height + 7) * scale
    for index, assumption in enumerate(assumptions):
        chunks.append(
            f'<text x="{margin*scale}" y="{note_y + index*18}" font-family="Arial" font-size="12">'
            f'- {html.escape(assumption)}</text>'
        )
    chunks.append(
        f'<line x1="{margin*scale}" y1="{height_total_cm*scale-35}" '
        f'x2="{(margin+10)*scale}" y2="{height_total_cm*scale-35}" stroke="#000" stroke-width="3"/>'
    )
    chunks.append(
        f'<text x="{margin*scale}" y="{height_total_cm*scale-15}" font-family="Arial" font-size="12">Línea de control: 10 cm</text>'
    )
    if watermark:
        chunks.append(
            f'<text x="{width_cm*scale/2}" y="{height_total_cm*scale/2}" '
            'text-anchor="middle" transform="rotate(-25)" font-family="Arial" '
            'font-size="52" font-weight="bold" fill="#d22630" opacity="0.28">BORRADOR - NO CORTAR</text>'
        )
    chunks.append("</svg>")
    return "\n".join(chunks)

