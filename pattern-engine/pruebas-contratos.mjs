import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';

const servidor = await readFile(new URL('./server.mjs', import.meta.url), 'utf8');
const interfaz = await readFile(new URL('./ui.html', import.meta.url), 'utf8');
const registro = JSON.parse(await readFile(new URL('../configs/modelos_runtime.json', import.meta.url), 'utf8'));
const scripts = Object.fromEntries(await Promise.all([
  'run_fashn_vton_local.py',
  'select_best_tryon_candidate.py',
  'composite_tryon_identity.py',
  'build_layered_tryon_preview.py',
].map(async nombre => [
  nombre,
  await readFile(new URL(`../scripts/${nombre}`, import.meta.url), 'utf8'),
])));

// Los nombres propios del proyecto pueden estar en español, pero estos valores
// pertenecen a estándares del navegador o a interfaces CLI de otros procesos.
for (const mime of ['image/jpeg', 'image/png', 'image/webp']) {
  assert.ok(servidor.includes(`'${mime}'`), `Falta el MIME estándar ${mime}`);
}
assert.doesNotMatch(servidor, /imagen\/(?:jpeg|png|webp)/);
assert.doesNotMatch(servidor, /data:imagen\//);

for (const opcion of ['--garment-image', '--output', '--height']) {
  assert.ok(servidor.includes(`'${opcion}'`), `Falta la opción CLI ${opcion}`);
}
assert.doesNotMatch(servidor, /--(?:garment-imagen|salida|altura)\b/);

const contratosCli = {
  'run_fashn_vton_local.py': ['--session-dir', '--garment-image', '--output'],
  'select_best_tryon_candidate.py': ['--session-dir', '--garment-image', '--output'],
  'composite_tryon_identity.py': ['--session-dir', '--generated', '--output'],
  'build_layered_tryon_preview.py': ['--session-dir', '--sex', '--output', '--width', '--height'],
};
assert.doesNotMatch(servidor, /add_shoes_overlay|shoes_added|result-with-shoes/);
assert.doesNotMatch(interfaz, /shoes_added|Zapatos añadidos/);
assert.match(servidor, /VERSION_CACHE_VESTIDOR = 2/);
for (const [nombre, opciones] of Object.entries(contratosCli)) {
  for (const opcion of opciones) {
    assert.ok(scripts[nombre].includes(`"${opcion}"`), `${nombre} no declara ${opcion}`);
    assert.ok(servidor.includes(`'${opcion}'`), `server.mjs no usa ${opcion} requerido por ${nombre}`);
  }
}

// El contrato JSON que comparten navegador, servidor y FreeSewing permanece
// estable aunque las variables locales de la interfaz estén en español.
assert.match(interfaz, /ultimaCargaUtil=\{garment:prenda,/);
assert.match(interfaz, /person:persona\(\),measurements/);
assert.doesNotMatch(interfaz, /ultimaCargaUtil=\{prenda,/);

// Todo selector JavaScript por id debe apuntar a un elemento que exista.
const ids = new Set([...interfaz.matchAll(/\bid=["']([^"']+)/g)].map(match => match[1]));
const selectores = new Set([...interfaz.matchAll(/\$\(["']#([A-Za-z_][\w-]*)/g)].map(match => match[1]));
for (const selector of selectores) {
  assert.ok(ids.has(selector), `El selector #${selector} no tiene elemento HTML`);
}

for (const idAnterior of ['status', 'measureStatus', 'measurements', 'preview', 'predictionSummary']) {
  assert.doesNotMatch(interfaz, new RegExp(`#${idAnterior}\\b`), `Quedó el selector anterior #${idAnterior}`);
}
assert.match(interfaz, /cursor:not-allowed/);

// La ruta oficial BodyM acepta exactamente dos fotos y estatura. El peso es
// una salida del modelo multitarea, nunca un campo ni un argumento del usuario.
assert.match(interfaz, /id="fotoFrontal"/);
assert.match(interfaz, /id="fotoIzquierda"/);
assert.doesNotMatch(interfaz, /id="fotoDerecha"|id="fotoEspalda"|id="peso"/);
assert.doesNotMatch(servidor, /'-WeightKg'|'-Gender'|'-Right'/);
assert.match(servidor, /'-GarmentRoute'/);
assert.match(interfaz, /estimated_weight/);
assert.equal(registro.medicion_oficial.regresion.nombre, 'exp_008_perfiles_arboles');
assert.equal(registro.medicion_oficial.regresion.backend, 'profiles');
assert.equal(registro.vestidor.perfil_demo_en_vivo.pasos, 30);
assert.equal(registro.vestidor.perfil_demo_en_vivo.candidatos, 1);
assert.match(servidor, /perfilVestidor\.pasos/);
assert.match(servidor, /perfilVestidor\.candidatos/);
assert.match(servidor, /identity_qa/);
assert.match(servidor, /rectangular_restore_detected/);
assert.match(servidor, /runtime_report\.json/);
assert.match(interfaz, /id="generarVestidorReal"[^>]*>Vestidor</);
assert.match(interfaz, /Visualización generada localmente para esta persona/);
assert.doesNotMatch(interfaz, /Referencia externa|Mockup preparado|Modo experimental/);

console.log('Contratos navegador/servidor/Python: válidos');
