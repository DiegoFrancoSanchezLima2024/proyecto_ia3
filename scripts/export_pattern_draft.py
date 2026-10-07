"""Genera SVG 1:1 y un PDF A4 de revisión para el contrato a medida."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.patterns.draft_export import trazar_piezas, renderizar_svg


def _draw_pdf(input_path: Path, payload: dict, output_path: Path) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    page_w, page_h = A4
    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle("SATRE-IA - borrador técnico de moldes")

    c.setFillColor(HexColor("#18212f"))
    c.setFont("Helvetica-Bold", 22)
    c.drawString(45, page_h - 65, "SATRE-IA")
    c.setFont("Helvetica-Bold", 15)
    c.drawString(45, page_h - 92, "Borrador técnico de moldes personalizados")
    c.setFont("Helvetica", 10)
    c.drawString(45, page_h - 115, f"Entrada: {input_path.name}")
    sex_label = {"male": "hombre", "female": "mujer"}.get(
        payload["bespoke"]["sex_profile"], payload["bespoke"]["sex_profile"]
    )
    c.drawString(45, page_h - 132, f"Perfil: {sex_label}")
    c.drawString(45, page_h - 149, f"Estatura: {payload['bespoke']['height_cm']:.1f} cm")
    c.setFillColor(HexColor("#b4232d"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(page_w / 2, page_h / 2 + 30, "BORRADOR - NO CORTAR")
    c.setFont("Helvetica", 11)
    c.drawCentredString(page_w / 2, page_h / 2, "Bloques geométricos para verificar medidas y flujo del prototipo.")
    c.drawCentredString(page_w / 2, page_h / 2 - 18, "No contienen todavía todas las curvas y reglas de sastrería.")
    c.setFillColor(HexColor("#18212f"))
    c.setFont("Helvetica", 9)
    c.drawString(45, 50, "La barra de 10 cm de cada página permite comprobar la impresión al 100%.")
    c.showPage()

    for garment_name, garment in payload["bespoke"]["garments"].items():
        pieces, assumptions = trazar_piezas(garment, payload["bespoke"]["height_cm"])
        garment_label = {
            "jacket": "saco",
            "trousers": "pantalón",
            "skirt": "falda",
        }.get(garment_name, garment_name)
        c.setFillColor(HexColor("#18212f"))
        c.setFont("Helvetica-Bold", 17)
        c.drawString(40, page_h - 50, f"Prenda: {garment_label}")
        c.setFont("Helvetica", 9)
        c.drawString(40, page_h - 68, f"Estado: {garment['pattern_status']} | vista reducida de revisión")

        available_w = page_w - 80
        available_h = page_h - 230
        total_width = sum(p.width_cm for p in pieces) + 5 * (len(pieces) - 1)
        max_height = max(p.height_cm for p in pieces)
        scale = min(available_w / total_width, available_h / max_height)
        x_cm = 0.0
        for piece in pieces:
            pts = [(40 + (x_cm + px) * scale, 145 + py * scale) for px, py in piece.points]
            path = c.beginPath()
            path.moveTo(*pts[0])
            for point in pts[1:]:
                path.lineTo(*point)
            path.close()
            c.setStrokeColor(HexColor("#18212f"))
            c.setLineWidth(1.2)
            c.drawPath(path, stroke=1, fill=0)
            c.setFont("Helvetica", 7)
            c.drawCentredString(40 + (x_cm + piece.width_cm / 2) * scale, 135, piece.name)
            x_cm += piece.width_cm + 5

        c.setStrokeColor(HexColor("#000000"))
        c.setLineWidth(2)
        c.line(40, 100, 40 + 10 * 28.346457, 100)
        c.setFont("Helvetica", 8)
        c.drawString(40, 87, "Control físico 10 cm (imprimir al 100%)")
        c.setFillColor(HexColor("#b4232d"))
        c.saveState()
        c.translate(page_w / 2, page_h / 2)
        c.rotate(25)
        c.setFont("Helvetica-Bold", 30)
        c.drawCentredString(0, 0, "BORRADOR - NO CORTAR")
        c.restoreState()
        c.setFillColor(HexColor("#18212f"))
        c.setFont("Helvetica", 7.5)
        note_y = 70
        for assumption in assumptions:
            c.drawString(40, note_y, f"- {assumption}")
            note_y -= 10
        c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--svg-dir", default="outputs/patterns")
    parser.add_argument("--pdf", default="output/pdf/bespoke_patterns_draft.pdf")
    parser.add_argument(
        "--allow-schematic",
        action="store_true",
        help="Habilita explícitamente los bloques geométricos no aptos para confección",
    )
    args = parser.parse_args()

    if not args.allow_schematic:
        raise SystemExit(
            "Exportador esquemático desactivado: no genera patrones reales. "
            "Use scripts/audit_pattern_engine.py para preparar FreeSewing."
        )

    input_path = ROOT / args.prediction
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if "bespoke" not in payload:
        raise ValueError("La predicción no contiene un contrato bespoke")
    if payload["bespoke"]["overall_pattern_status"] == "blocked":
        raise ValueError("La captura está bloqueada y no permite generar ni borrador")

    svg_dir = ROOT / args.svg_dir
    svg_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, garment in payload["bespoke"]["garments"].items():
        svg_path = svg_dir / f"{name}_draft_1to1.svg"
        svg_path.write_text(
            renderizar_svg(garment, payload["bespoke"]["height_cm"]), encoding="utf-8"
        )
        generated.append(str(svg_path))

    pdf_path = ROOT / args.pdf
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _draw_pdf(input_path, payload, pdf_path)
    print(json.dumps({"pdf": str(pdf_path), "svg": generated}, indent=2))


if __name__ == "__main__":
    main()
