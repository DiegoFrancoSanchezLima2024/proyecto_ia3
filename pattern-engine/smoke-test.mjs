import { strict as assert } from 'node:assert';
import { Charlie } from '@freesewing/charlie';
import { Jaeger } from '@freesewing/jaeger';
import { Penelope } from '@freesewing/penelope';
import { cisFemaleAdult36, cisMaleAdult40 } from '@freesewing/models';
import { generarPrenda } from './garments.mjs';
import { evaluarListoParaCorte, MARCA_NO_VALIDADO } from './readiness.mjs';

const casos = [
  ['Jaeger', Jaeger, cisMaleAdult40],
  ['Charlie', Charlie, cisMaleAdult40],
  ['Penelope', Penelope, cisFemaleAdult36],
];

for (const [name, Design, medidas] of casos) {
  const patron = new Design({ measurements: medidas, sa: 10 }).draft();
  const errores = patron.getLogs().sets.flatMap((set) => set.error);
  assert.deepEqual(errores, [], `${name} devolvió errores de trazado`);
  const svg = patron.render();
  assert.match(svg, /^<\?xml/);
  assert.ok(patron.width > 0 && patron.height > 0, `${name} no creó un lienzo válido`);
  console.log(`${name}: SVG válido (${Math.round(patron.width)} x ${Math.round(patron.height)} mm)`);
}

for (const [garment, sex, Design, model] of [
  ['jacket', 'male', Jaeger, cisMaleAdult40],
  ['trousers', 'male', Charlie, cisMaleAdult40],
  ['skirt', 'female', Penelope, cisFemaleAdult36],
]) {
  const requeridas = new Design().getConfig().measurements;
  const medidas = Object.fromEntries(requeridas.map(name => [
    name,
    name === 'shoulderSlope' ? model[name] : model[name] / 10,
  ]));
  // Sin procedencia, el contrato falla cerrado: un trazado coherente no basta
  // para declarar un molde cortable.
  const sinProcedencia = generarPrenda({ garment, sex, measurements: medidas });
  assert.equal(sinProcedencia.status, 'draft_unvalidated');
  assert.equal(sinProcedencia.cut_ready, false);
  assert.ok(sinProcedencia.svg.includes(MARCA_NO_VALIDADO), `${garment}: falta la marca en el SVG`);

  // Con medidas confirmadas y una decision aceptada, si se declara cortable.
  const fuentes = Object.fromEntries(Object.keys(medidas).map(name => [name, 'guardada']));
  const confirmado = generarPrenda({
    garment, sex, measurements: medidas,
    decision: 'accepted_assisted',
    measurement_sources: fuentes,
  });
  assert.equal(confirmado.status, 'draft_ready');
  assert.equal(confirmado.cut_ready, true);
  assert.ok(!confirmado.svg.includes(MARCA_NO_VALIDADO), `${garment}: marca indebida en molde validado`);
  assert.ok(confirmado.svg.includes('<svg'));
  console.log(`${garment}: contrato personalizado valido (bloquea sin procedencia, permite con medidas confirmadas)`);
}

// El gating por si mismo, sin depender del motor de patronaje.
{
  const conDerivadas = evaluarListoParaCorte({
    decision: 'accepted_assisted', required: ['chest', 'waist'],
    sources: { chest: 'guardada', waist: 'derivada' },
  });
  assert.equal(conDerivadas.cut_ready, false, 'una proporcion generica debe bloquear el corte');
  assert.deepEqual(conDerivadas.derived_measurements, ['waist']);

  const conAdvertencia = evaluarListoParaCorte({
    decision: 'accepted_assisted', required: ['chest'],
    sources: { chest: 'medida IA' },
    intervals: { chest: { requires_manual_confirmation: true } },
  });
  assert.equal(conAdvertencia.cut_ready, false, 'un intervalo conformal ancho debe bloquear el corte');
  assert.deepEqual(conAdvertencia.unreliable_measurements, ['chest']);

  for (const decision of ['manual_confirmation', 'repeat_capture']) {
    const blocked = evaluarListoParaCorte({ decision, required: [], sources: {} });
    assert.equal(blocked.cut_ready, false, `${decision} debe bloquear el corte`);
  }

  const sinDecision = evaluarListoParaCorte({ required: ['chest'], sources: { chest: 'guardada' } });
  assert.equal(sinDecision.cut_ready, false, 'sin decision debe fallar cerrado');
  console.log('gating de corte: bloquea derivadas, intervalos anchos, decisiones marcadas y falta de procedencia');
}

assert.throws(
  () => generarPrenda({ garment: 'trousers', sex: 'female', measurements: {} }),
  /no corresponde/,
);
