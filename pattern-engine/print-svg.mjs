// Los atributos de presentacion funcionan tanto en el navegador como en el renderizador PDF sin conexion.
export function imprimirSvg(svg) {
  const etiquetas = {
    'plugin-annotations:cut,2,plugin-annotations:mirrored,plugin-annotations:from,plugin-annotations:fabric,': 'Cortar 2 en espejo / tela',
    'plugin-annotations:cut,1,plugin-annotations:from,plugin-annotations:fabric,': 'Cortar 1 / tela',
    'plugin-annotations:cutOnFoldAndGrainline': 'Doblez / hilo',
    'plugin-annotations:cutOnFold': 'Doblez de tela',
    'plugin-annotations:grainline': 'Hilo',
    'plugin-annotations:supportFreeSewingBecomeAPatron': 'Diseno: FreeSewing',
    'plugin-annotations:theWhiteInsideOfThisBoxShouldMeasure': 'Interior blanco:',
    'plugin-annotations:theBlackOutsideOfThisBoxShouldMeasure': 'Exterior negro:',
  };
  for (const [key,value] of Object.entries(etiquetas)) svg=svg.replaceAll(key,value);
  svg=svg.replace(/>\s*back\s*</g,'>Espalda<').replace(/>\s*front\s*</g,'>Delantero<').replace(/>\s*waistband\s*</g,'>Pretina<');
  // FreeSewing stamps its own "<Design> v<version> ( ephemeral )" caption plus a
  // skull-and-needle logo mark on every piece (free-tier branding). Both are
  // FreeSewing's own attribution, not this project's, and read as an unexplained
  // watermark on a formal pattern printout, so they are stripped here. The
  // short translated "Diseno: FreeSewing" credit line stays.
  svg=svg.replace(/<text\b[^>]*>\s*<tspan>[^<]*FreeSewing[^<]*\(\s*ephemeral\s*\)[^<]*<\/tspan>\s*<\/text>/g,'');
  svg=svg.replace(/<use\b[^>]*xlink:href="#logo"[^>]*>\s*(?:<\/use>)?/g,'');
  // FreeSewing's raw pattern.render() also emits a malformed <defs> block: its
  // marker/icon library is serialized as bare `key="<xml.../>"` text pairs
  // instead of real child elements (a bug in @freesewing/core Defs.render(),
  // src/defs.mjs). Browsers and svg-to-pdfkit currently tolerate it, but a
  // strict XML parser (e.g. Illustrator/Inkscape opening the exported SVG for
  // real cutting) would not, so it is rebuilt into valid <defs> hijos here.
  svg=svg.replace(/<defs>([\s\S]*?)<\/defs>/,(all,body)=>{
    const hijos=[];
    let i=0;
    while (i<body.length) {
      const lt=body.indexOf('<',i);
      if (lt===-1) break;
      const coincidenciaEtiqueta=/^<([a-zA-Z][\w-]*)/.exec(body.slice(lt,lt+40));
      if (!coincidenciaEtiqueta) { i=lt+1; continue; }
      const nombreEtiqueta=coincidenciaEtiqueta[1];
      let j=lt,inQuote=null;
      while (j<body.length) {
        const c=body[j];
        if (inQuote) { if (c===inQuote) inQuote=null; }
        else if (c==='"'||c==="'") inQuote=c;
        else if (c==='>') break;
        j++;
      }
      if (body[j-1]==='/') { hijos.push(body.slice(lt,j+1)); i=j+1; continue; }
      const aguaApertura=`<${nombreEtiqueta}`,aguaCierre=`</${nombreEtiqueta}>`;
      let depth=1,k=j+1;
      while (depth>0) {
        const nextOpen=body.indexOf(aguaApertura,k),nextClose=body.indexOf(aguaCierre,k);
        if (nextClose===-1) { k=body.length; break; }
        if (nextOpen!==-1 && nextOpen<nextClose) { depth++; k=nextOpen+aguaApertura.length; }
        else { depth--; k=nextClose+aguaCierre.length; }
      }
      hijos.push(body.slice(lt,k));
      i=k;
    }
    return hijos.length ? `<defs>${hijos.join('\n')}</defs>` : all;
  });
  svg=svg.replace(/<(path|circle|rect|polygon)\b([^>]*?)(\/?)>/g,(all,tag,attrs,close)=>{
    if (/\b(fill|stroke)=/.test(attrs)) return all;
    const isSa=/class="[^"]*\bsa\b/.test(attrs);
    const fill=/fill-bg/.test(attrs)?'white':/fill-current|fill-mark/.test(attrs)?'black':'none';
    return `<${tag} fill="${fill}" stroke="black" stroke-width="${isSa?0.4:0.22}" ${isSa?'stroke-dasharray="2 1" ':''}${attrs}${close}>`;
  });
  svg=svg.replace(/<text\b([^>]*)>/g,(all,attrs)=>`<text font-family="Helvetica" font-size="${attrs.includes('text-4xl')?8:attrs.includes('font-bold')?4:2.5}" fill="black" ${attrs}>`);
  return svg;
}
