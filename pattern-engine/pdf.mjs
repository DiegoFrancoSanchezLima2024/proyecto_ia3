import PDFDocument from 'pdfkit';
import SVGtoPDF from 'svg-to-pdfkit';

const mm = 72 / 25.4;
// Las hojas incluyen una franja repetida de 5 mm a la derecha y abajo.
export async function pdfFalda(pattern) {
  return pdfPatron(pattern);
}

export async function pdfPatron(pattern) {
  const anchoHoja = 190, altoHoja = 267, solape = 5;
  const pasoX = anchoHoja - solape, pasoY = altoHoja - solape;
  const columnas = Math.max(1, Math.ceil((pattern.width_mm - solape) / pasoX));
  const filas = Math.max(1, Math.ceil((pattern.height_mm - solape) / pasoY));
  if (columnas * filas > 100) throw new Error('El patrón supera 100 hojas A4.');
  const documento = new PDFDocument({ size: 'A4', margin: 0, autoFirstPage: false,
    info: { Title: `SATRE-IA - ${pattern.garment_label || 'Penelope'} - borrador A4`, Author: 'SATRE-IA local' } });
  const fragmentos = [];
  const completado = new Promise((resolve, reject) => {
    documento.on('data', chunk => fragmentos.push(chunk));
    documento.on('end', () => resolve(Buffer.concat(fragmentos)));
    documento.on('error', reject);
  });
  documento.addPage();
  documento.font('Helvetica-Bold').fontSize(24).text('SATRE-IA', 40, 40);
  documento.fontSize(16).text(`${pattern.garment_label || 'Falda Penelope'} / Impresion A4`, 40, 80);
  documento.font('Helvetica').fontSize(11).text(
    `${pattern.source === 'reference' ? 'EJEMPLO DE REFERENCIA' : 'BORRADOR PERSONALIZADO'}\n\n` +
    `Patron para prueba de confeccion. Ajuste corporal sin validar.\n` +
    `Margen de costura: ${pattern.seam_allowance_mm} mm.\n\n` +
    'Imprime a TAMANO REAL / 100%. Desactiva Ajustar a pagina.\n' +
    'Comprueba ambos lados del cuadrado: deben medir 10 cm.\n' +
    'Cada hoja incluye otro control de 5 x 5 mm en el pie.\n' +
    'Ordena las hojas por fila y columna (F1-C1, F1-C2...).\n' +
    'Recorta el margen exterior y superpone las bandas repetidas\n' +
    'de 5 mm. Alinea las lineas del patron y los marcos.\n\n' +
    `Mosaico: ${filas} filas x ${columnas} columnas (${filas * columnas} hojas).\n` +
    'Las paginas contienen fragmentos del patron, no piezas enteras.', 40, 115, { width: 510, lineGap: 4 });
  documento.rect(40, 410, 100 * mm, 100 * mm).lineWidth(0.7).stroke();
  documento.fontSize(12).text('10 x 10 cm', 60, 440);
  documento.fontSize(10).text('CONTROL DE ESCALA', 60, 463);
  for (let row = 0; row < filas; row++) for (let col = 0; col < columnas; col++) {
    documento.addPage();
    documento.fontSize(8).text(`SATRE-IA / ${pattern.garment_label || 'Penelope'} / F${row + 1}-C${col + 1} / 100% / BORRADOR`, 10 * mm, 5 * mm);
    documento.save().rect(10 * mm, 15 * mm, anchoHoja * mm, altoHoja * mm).clip();
    SVGtoPDF(documento, pattern.svg, (10 - col * pasoX) * mm, (15 - row * pasoY) * mm,
      { width: pattern.width_mm * mm, height: pattern.height_mm * mm, preserveAspectRatio: 'none' });
    documento.restore();
    documento.lineWidth(0.4).strokeColor('#777777').rect(10 * mm, 15 * mm, anchoHoja * mm, altoHoja * mm).stroke();
    documento.dash(2).moveTo((10 + pasoX) * mm, 15 * mm).lineTo((10 + pasoX) * mm, 282 * mm).stroke();
    documento.moveTo(10 * mm, (15 + pasoY) * mm).lineTo(200 * mm, (15 + pasoY) * mm).stroke().undash();
    documento.fillColor('black').fontSize(8).text('Banda punteada: solape 5 mm. No ajustar al papel.', 10 * mm, 286 * mm);
    // Fuera de la region recortada del patron: nunca oculta geometria de corte.
    documento.save().strokeColor('black').lineWidth(0.3).undash();
    documento.rect(195 * mm, 285 * mm, 5 * mm, 5 * mm).stroke();
    documento.fontSize(7).text('Control 5 x 5 mm', 168 * mm, 286 * mm, { width: 26 * mm });
    documento.restore();
  }
  documento.end();
  return completado;
}
