/* ═══════════════════════════════════════════════════════════════════════════
   app.js — lógica de la SPA Rough Paths Lab
   Carga datos.json (AMI + simulados + neuralde), construye índices y
   renderiza las 13 secciones interactivas.
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

let D = null;                 // payload global
const IDX = { pares: {}, detCasa: {}, simDet: {} };
const ANIMADORES = {};        // canvas activos
const PCFG = { responsive: true, displayModeBar: false };

const LAYOUT_BASE = {
  margin: { l: 54, r: 18, t: 34, b: 44 },
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Segoe UI, sans-serif", size: 12, color: "#33415c" },
};
const lay = (extra) => Object.assign({}, JSON.parse(JSON.stringify(LAYOUT_BASE)), extra);

const $ = (id) => document.getElementById(id);
const on = (id, fn) => { const el = $(id); if (el) el.addEventListener("input", () => { syncRange(el); fn(); }); };

function syncRange(el) {
  if (el.type !== "range") return;
  const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty("--pct", pct + "%");
  const v = $(el.id + "-v");
  if (v) v.textContent = el.id.includes("casa") ? (parseInt(el.value) + 1) : el.value;
}

function mt(el) {  // render KaTeX dentro de un elemento
  if (window.renderMathInElement)
    renderMathInElement(el, {
      delimiters: [{ left: "$$", right: "$$", display: true },
                   { left: "\\(", right: "\\)", display: false }],
      throwOnError: false,
    });
}

function setAnim(key, canvasId, fnName, st = {}) {
  if (ANIMADORES[key]) ANIMADORES[key].pause();
  const cv = $(canvasId);
  if (!cv || !window.ANIMS[fnName]) return null;
  const a = new Animador(cv, ANIMS[fnName], st);
  ANIMADORES[key] = a; a.play();
  return a;
}

const CTXS = () => Object.keys(D.simulados || {});
const SIM = (c) => D.simulados[c];
const DETS_TODOS = () => {
  const extras = [];
  for (const c of CTXS())
    for (const r of SIM(c).metricas)
      if (!D.DETECTORES.includes(r.Detector) && !extras.includes(r.Detector))
        extras.push(r.Detector);
  return D.DETECTORES.concat(extras);
};
const famDe = (det) => D.DET_FAM[det] || (det.includes("Sig") || det.includes("sig") ? "Signatures" : "C-ML");
const colorFam = (det) => D.FAM_COLOR[famDe(det)] || "#888";
const tipoLabel = (ds, t) => {
  if (t === "general" || t === "Todos") return t;
  if (ds.startsWith("ami")) return D.TIPO_LABEL[t] || t;
  const m = SIM(ds.split(":")[1]).meta.descrip_tipos[t];
  return m ? m.titulo : t;
};

/* ═══ arranque ═══════════════════════════════════════════════════════════ */
async function init() {
  // El servidor tarda ~10-15 s en generar datos.json la primera vez (lee todo
  // outputs/). Reintentamos hasta ~45 s con margen para no rendirnos antes.
  const MAX_INTENTOS = 45;
  for (let i = 1; i <= MAX_INTENTOS; i++) {
    try {
      const ctl = new AbortController();                 // timeout por intento:
      const tmr = setTimeout(() => ctl.abort(), 20000);  // un fetch colgado no bloquea
      const res = await fetch("datos.json?_=" + Date.now(),
                              { cache: "no-store", signal: ctl.signal });
      if (!res.ok) throw new Error("HTTP " + res.status);
      D = await res.json(); clearTimeout(tmr); break;
    } catch (e) {
      const lt = $("loader").querySelector(".ltxt");
      if (lt) lt.textContent = `preparando datos… (${i}/${MAX_INTENTOS})`;
      await new Promise(r => setTimeout(r, 1000));
    }
  }
  if (!D) { $("loader").querySelector(".ltxt").textContent = "error cargando datos.json"; return; }

  buildIndices();
  renderInicio(); renderFundamentos(); renderDatos();
  populateControles();
  renderSeries(); renderMultivar(); renderDetectores();
  renderDR(); renderJaccard(); renderInspector(); renderSig();
  renderNDETeoria(); renderNDELab(); renderConclusiones();
  bindEventos(); setupNav(); setupReveal();
  mt(document.body);
  $("loader").classList.add("hide");
}

function buildIndices() {
  for (const p of (D.pares || [])) {
    (IDX.pares[p.casa_idx] = IDX.pares[p.casa_idx] || {})[p.tipo] = p;
  }
  for (const r of (D.detecciones || []))
    IDX.detCasa[`${r.detector}|${r.casa_idx}|${r.tipo}`] = r;
  for (const c of CTXS())
    for (const r of SIM(c).detecciones)
      IDX.simDet[`${c}|${r.detector}|${r.par_idx}`] = r;
}

/* ═══ INICIO ═════════════════════════════════════════════════════════════ */
function renderInicio() {
  const nDet = DETS_TODOS().length;
  const nCtx = CTXS().length;
  let nVent = 0, nAnom = 7;
  for (const c of CTXS()) { nVent += SIM(c).meta.n_ventanas; nAnom += SIM(c).meta.tipos.length; }
  const runs = D.neuralde?.resultados?.runs || [];
  const cards = [
    [nDet, "detectores en 5 familias"],
    [(1 + nCtx) + " / 4", "datasets · regímenes de muestreo"],
    [nAnom, "tipos de anomalía etiquetados"],
    [nVent.toLocaleString("es"), "ventanas multivariadas simuladas"],
    [new Set(runs.map(r => r.modelo)).size || 5, "modelos: RNN → GRU → ODE → CDE → RDE"],
  ];
  $("hero-cards").innerHTML = cards.map(c =>
    `<div class="rcard"><div class="num">${c[0]}</div><div class="lbl">${c[1]}</div></div>`).join("");

  $("mapa-sistema").innerHTML = `
  <div class="grid-2">
    <div class="info-card naranja"><h4>⚡ Detección de anomalías por signaturas</h4>
      La firma del camino (con aumento temporal) describe la geometría de cada serie. Sobre ella, una
      familia de detectores localiza anomalías que las medidas estáticas no ven —incluso anomalías del
      propio muestreo. Se compara con 13 detectores clásicos bajo el mismo presupuesto de alertas.</div>
    <div class="info-card purpura"><h4>🧭 Ecuaciones diferenciales neuronales</h4>
      De las redes recurrentes a las Neural RDE: discreto → continuo → controlado por los datos →
      resumido por la geometría de cada ventana (log-signatura). Cada etapa corrige un límite demostrable
      de la anterior, con experimentos reproducibles.</div>
  </div>`;
}

/* ═══ FUNDAMENTOS ════════════════════════════════════════════════════════ */
function renderFundamentos() {
  const niveles = NIVELES_SIG.map(nv => `
    <div class="info-card" style="border-left-color:${nv.color}">
      <h4 style="color:${nv.color}">Nivel ${nv.n} · ${nv.titulo}</h4>
      <div class="formula-box" style="background:#101e38">$$${nv.formula}$$</div>
      <p>${nv.texto}</p>
      <p style="margin-top:6px;font-size:12.5px;color:var(--tinta-2)"><b>Ejemplo:</b> ${nv.ami}</p>
    </div>`).join("");

  $("fund-contenido").innerHTML = `
  <div class="plot-card teoria-panel">
    <h3>Definición</h3>
    <p>Para un camino \\(X:[0,T]\\to\\mathbb{R}^d\\) de variación acotada, su <b>signatura</b> es la
    colección de todas las integrales iteradas:</p>
    <div class="formula-box">$$S(X)^{(i_1,\\dots,i_k)} \\;=\\; \\int\\limits_{0\\lt t_1\\lt \\cdots\\lt t_k\\lt T} dX^{i_1}_{t_1}\\cdots\\, dX^{i_k}_{t_k},\\qquad k = 1, 2, \\dots$$</div>
    <p>Es un elemento del álgebra tensorial \\(T(\\mathbb{R}^d)=\\bigoplus_k (\\mathbb{R}^d)^{\\otimes k}\\):
    el «desarrollo de Taylor» no conmutativo del camino. Tres propiedades la hacen el descriptor canónico:</p>
    <div class="callout azul"><b>1 · Identidad de Chen.</b> La signatura convierte concatenación en producto:
    $$S(X * Y) = S(X)\\otimes S(Y)$$
    y para un segmento lineal de incremento \\(\\delta\\): \\(S^{(k)} = \\delta^{\\otimes k}/k!\\) — la exponencial
    tensorial. Así se calcula exactamente para caminos lineales a trozos (es lo que ejecuta este navegador en el
    explorador y el backend vectorizado por lotes).</div>
    <div class="callout naranja"><b>2 · Producto shuffle.</b> El producto de dos coordenadas es combinación de
    coordenadas de orden superior:
    $$S^{(i)}S^{(j)} = S^{(i,j)} + S^{(j,i)}$$
    Consecuencia: los funcionales <i>lineales</i> sobre la signatura forman un álgebra que separa puntos — con
    Stone–Weierstrass se obtiene la <b>universalidad</b>: todo funcional continuo del camino se aproxima
    linealmente sobre signaturas. (Por eso detectores lineales en \\(S(X)\\) son tan expresivos.)</div>
    <div class="callout morado"><b>3 · Unicidad (Hambly–Lyons 2010).</b> \\(S(X)\\) determina a \\(X\\) módulo
    reparametrización y tramos «tree-like». Añadiendo el tiempo como canal (aumento temporal) se elimina la
    invariancia y el muestreo — regular, irregular o por eventos — queda <b>codificado</b> en la firma.</div>
    <div class="callout verde"><b>Factorial decay.</b> \\(\\lVert S^{(k)}\\rVert \\le \\lVert X\\rVert_{1\\text{-var}}^k / k!\\)
    — los niveles altos pesan factorialmente menos: truncar en \\(m=2..4\\) retiene casi toda la información útil,
    con dimensión \\(\\sum_{k\\le m} d^k\\).</div>
  </div>
  <h3 style="margin:18px 0 12px;color:var(--azul-dk)">Interpretación geométrica de cada nivel</h3>
  <div class="grid-2">${niveles}</div>`;
}

/* ═══ DATOS & MUESTREO ═══════════════════════════════════════════════════ */
function renderDatos() {
  const cards = [`
    <div class="info-card"><h4>⚡ AMI Colombia (real)</h4>
      Lecturas horarias (Wh) de medidores residenciales y comerciales, ventanas semanales de 168 h.
      <div style="margin-top:8px"><span class="chip chip-azul">univariada</span>
      <span class="chip">muestreo regular 1 h</span> <span class="chip">7 anomalías inyectadas</span></div></div>`];
  for (const c of CTXS()) {
    const m = SIM(c).meta;
    const tipoM = { regular: "chip-verde", irregular: "chip-ambar", alta_frecuencia: "chip-morado" }[m.muestreo.tipo] || "";
    cards.push(`
    <div class="info-card naranja"><h4>${m.icono} ${m.titulo}</h4>
      ${m.descripcion}
      <div style="margin-top:8px"><span class="chip chip-azul">${m.canales.length} canales: ${m.canales.join(", ")}</span>
      <span class="chip ${tipoM}">muestreo ${m.muestreo.tipo.replace("_", " ")}</span>
      <span class="chip">${m.n_ventanas.toLocaleString("es")} ventanas + ${m.n_anomalias} anómalas</span></div>
      <p style="margin-top:6px;font-size:12.5px;color:var(--tinta-2)">${m.muestreo.descripcion}</p></div>`);
  }
  $("datos-cards").innerHTML = `<div class="grid-2">${cards.join("")}</div>`;

  // demo de muestreo (espiral exportada por el pipeline)
  const md = D.neuralde?.trayectorias?.muestreo_demo;
  if (md) {
    const cont = md.continua;
    const tr = [{ x: cont.x, y: cont.y, mode: "lines", name: "trayectoria continua",
                  line: { color: "#c3cfe0", width: 2 } }];
    const estilos = { regular: ["#1565c0", "circle"], irregular: ["#e65100", "diamond"], eventos: ["#6a1b9a", "triangle-up"] };
    for (const k of ["regular", "irregular", "eventos"]) {
      tr.push({ x: md[k].map(i => cont.x[i]), y: md[k].map(i => cont.y[i]),
        mode: "markers", name: "muestreo " + k,
        marker: { color: estilos[k][0], symbol: estilos[k][1], size: 9, line: { color: "#fff", width: 1 } } });
    }
    Plotly.newPlot("plot-muestreo", tr, lay({
      xaxis: { title: "x", zeroline: false }, yaxis: { title: "y", zeroline: false, scaleanchor: "x" },
      legend: { orientation: "h", y: -0.15 },
    }), PCFG);
  }

  $("modos-ami").innerHTML = `
    ${["single_house|1 casa × todas las semanas del año — dinámica individual",
       "weekly_month|todas las casas × 1 semana representativa por mes — corte transversal",
       "monthly|todas las casas × todas las semanas del mes — densidad media",
       "annual|todas las casas × todo el año — máxima evidencia"].map(s => {
      const [k, txt] = s.split("|");
      return `<div style="display:flex;gap:10px;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--linea)">
        <span class="chip chip-azul" style="min-width:120px;justify-content:center">${D.MODOS_ES[k]}</span>
        <span style="font-size:13px">${txt}</span></div>`;
    }).join("")}
    <p style="font-size:12.5px;color:var(--tinta-2);margin-top:10px">El mismo detector cambia de
    comportamiento según la muestra: con pocas series (Casa Individual) los métodos de densidad pierden
    poder; con el año completo los vecindarios se vuelven informativos — compáralo en Resultados.</p>`;
}

/* ═══ SERIES AMI ═════════════════════════════════════════════════════════ */
function renderSeries() {
  const tipo = $("series-tipo").value;
  const casa = +$("series-casa").value;
  let h0 = +$("series-h0").value, h1 = +$("series-h1").value;
  if (h1 <= h0) h1 = Math.min(167, h0 + 5);
  const par = IDX.pares[casa] || {};
  const sN = par["normal"], sA = par[tipo];
  if (!sN || !sA) return;
  const horas = [...Array(168).keys()];
  const colorA = D.ANOM_COLOR[tipo] || "#e65100";
  const tr = [
    { x: horas, y: sN.h, name: "normal", mode: "lines", line: { color: "#1565c0", width: 2 } },
    { x: horas, y: sA.h, name: D.TIPO_LABEL[tipo], mode: "lines", line: { color: colorA, width: 2 } },
  ];
  Plotly.react("plot-series", tr, lay({
    title: { text: `Casa ${casa + 1} (${sN.ID}, ${sN.mes}) — semana de 168 h`, font: { size: 14 } },
    xaxis: { title: "hora de la semana" }, yaxis: { title: "consumo (Wh)" },
    legend: { orientation: "h", y: 1.12 },
    shapes: [{ type: "rect", x0: h0, x1: h1, y0: 0, y1: 1, yref: "paper",
               fillcolor: "rgba(255,183,77,.13)", line: { width: 0 } }],
  }), PCFG);

  // Segundo gráfico: ZOOM al tramo [h0, h1] seleccionado con los deslizadores.
  const horasZ = horas.slice(h0, h1 + 1);
  Plotly.react("plot-series-zoom", [
    { x: horasZ, y: sN.h.slice(h0, h1 + 1), name: "normal", mode: "lines+markers",
      line: { color: "#1565c0", width: 2 }, marker: { size: 4 } },
    { x: horasZ, y: sA.h.slice(h0, h1 + 1), name: D.TIPO_LABEL[tipo], mode: "lines+markers",
      line: { color: colorA, width: 2 }, marker: { size: 4 } },
  ], lay({
    title: { text: `Ventana ampliada: horas ${h0}–${h1} (${h1 - h0 + 1} h)`, font: { size: 13 } },
    xaxis: { title: "hora de la semana", range: [h0, h1] },
    yaxis: { title: "consumo (Wh)" },
    legend: { orientation: "h", y: 1.16 }, margin: { t: 40, b: 40 },
    plot_bgcolor: "rgba(255,183,77,.06)",
  }), PCFG);

  const de = D.DESCRIP[tipo] || {};
  $("series-info").innerHTML = `
    <div class="info-card" style="border-left-color:${colorA}">
      <h4>${de.emoji || ""} ${de.titulo || tipo}</h4><p>${de.texto || ""}</p>
      <div style="margin-top:10px">
        <span class="chip">ID ${sN.ID}</span> <span class="chip">${sN.mes}</span>
        <span class="chip chip-ambar">ventana ${h0}–${h1} h</span></div>
    </div>
    <div class="callout azul" style="font-size:13px">La franja ámbar es la ventana activa del
    <a href="#sec-signaturas">explorador de signaturas</a>: muévela con los deslizadores y compara
    la firma del tramo normal y anómalo.</div>`;
}

/* ═══ MULTIVARIADOS ══════════════════════════════════════════════════════ */
function poblarMultivar() {
  const sel = $("mv-ctx");
  sel.innerHTML = CTXS().map(c => `<option value="${c}">${SIM(c).meta.icono} ${SIM(c).meta.titulo}</option>`).join("");
  poblarMvTipo();
}
function poblarMvTipo() {
  const c = $("mv-ctx").value;
  $("mv-tipo").innerHTML = SIM(c).meta.tipos.map(t =>
    `<option value="${t}">${SIM(c).meta.descrip_tipos[t].emoji} ${SIM(c).meta.descrip_tipos[t].titulo}</option>`).join("");
  poblarMvPar();
}
function poblarMvPar() {
  const c = $("mv-ctx").value, t = $("mv-tipo").value;
  const pares = SIM(c).pares.filter(p => p.tipo === t);
  $("mv-par").innerHTML = pares.map((p, i) => `<option value="${p.par_idx}">ejemplar ${i + 1}</option>`).join("");
}
function renderMultivar() {
  const c = $("mv-ctx").value, t = $("mv-tipo").value, pi = +$("mv-par").value;
  const meta = SIM(c).meta;
  const par = SIM(c).pares.find(p => p.par_idx === pi);
  if (!par) return;
  const canales = meta.canales;
  const nC = canales.length;
  const tr = [], anots = [];
  canales.forEach((ch, i) => {
    const ax = i === 0 ? "" : (i + 1);
    tr.push({ x: par.t_base, y: par.base[ch], mode: "lines", name: "normal",
      legendgroup: "n", showlegend: i === 0,
      line: { color: "#9fb2c8", width: 1.6 }, xaxis: "x" + ax, yaxis: "y" + ax });
    tr.push({ x: par.t, y: par.canales[ch], mode: "lines", name: "anómala",
      legendgroup: "a", showlegend: i === 0,
      line: { color: "#e65100", width: 1.8 }, xaxis: "x" + ax, yaxis: "y" + ax });
    anots.push({ text: `<b>${ch}</b> [${meta.unidades[i]}]`, xref: "paper", yref: "y" + ax + " domain",
      x: 0.005, y: 1.12, showarrow: false, font: { size: 11, color: "#4a5a72" } });
  });
  const layout = lay({
    grid: { rows: nC, columns: 1, pattern: "independent", roworder: "top to bottom" },
    margin: { l: 54, r: 14, t: 28, b: 36 }, legend: { orientation: "h", y: 1.06 },
    annotations: anots, showlegend: true,
  });
  for (let i = 0; i < nC; i++) {
    const ax = i === 0 ? "" : (i + 1);
    layout["xaxis" + ax] = { showticklabels: i === nC - 1, title: i === nC - 1 ? "tiempo normalizado" : "" };
    layout["yaxis" + ax] = {};
  }
  Plotly.react("plot-multivar", tr, layout, PCFG);

  const de = meta.descrip_tipos[t];
  $("mv-info").innerHTML = `
    <div class="info-card naranja"><h4>${de.emoji} ${de.titulo}</h4><p>${de.texto}</p>
      <div style="margin-top:8px"><span class="chip chip-azul">${meta.titulo}</span>
      <span class="chip">${meta.muestreo.tipo.replace("_", " ")}</span></div></div>
    <div class="callout azul" style="font-size:13px">¿Quién la detecta? Ve al
    <a href="#sec-inspector">Inspector</a> y selecciona este contexto y tipo: verás el score de los
    23 detectores sobre este mismo ejemplar.</div>`;
}

/* ═══ CATÁLOGO DE DETECTORES ═════════════════════════════════════════════ */
function renderDetectores(detSel) {
  const dets = DETS_TODOS();
  detSel = detSel || "SigMaHaKNN_d2_k10";
  $("algo-galeria").innerHTML = dets.map(d => {
    const col = colorFam(d);
    const act = d === detSel;
    return `<button class="algo-pill ${act ? "activa" : ""}" data-det="${d}"
      style="color:${col};border-color:${act ? col : ""};background:${act ? col : ""}">${d}</button>`;
  }).join("");
  $("algo-galeria").querySelectorAll(".algo-pill").forEach(b =>
    b.addEventListener("click", () => renderDetectores(b.dataset.det)));

  const info = teoriaDetector(detSel);
  if (!info) return;
  const fam = famDe(detSel), col = colorFam(detSel);
  const bloques = info.bloques.map(b => `
    <div style="margin-bottom:10px">
      <p>${b.t || ""}</p>${b.f ? `<div class="formula-box">$$${b.f}$$</div>` : ""}
    </div>`).join("");
  const props = Object.entries(info.props).map(([k, v]) =>
    `<div class="prop-mini"><b>${k}</b>${v}</div>`).join("");

  $("algo-panel").innerHTML = `
  <div class="teoria-panel">
    <div class="grid-31">
      <div class="plot-card">
        <h3 style="color:${col}">${detSel} <span class="chip" style="background:${col};color:#fff;margin-left:8px">${D.FAM_LABEL[fam] || fam}</span></h3>
        <canvas id="cv-algo" class="canvas-anim" style="height:320px"></canvas>
        <div class="anim-controles">
          <button class="btn fantasma" id="btn-algo-replay">↻ Reanimar</button>
          <span class="desc">${info.resumen}</span>
        </div>
      </div>
      <div>
        <div class="info-card" style="border-left-color:${col}"><h4>Vista de features</h4>
          <p style="font-size:13px">${info.vista}</p>
          ${info.nota ? `<p style="font-size:12.5px;color:var(--tinta-2);margin-top:6px">${info.nota}</p>` : ""}
        </div>
        <div class="plot-card"><h3>Propiedades</h3><div class="props-grid">${props}</div></div>
      </div>
    </div>
    <div class="plot-card"><h3>Construcción matemática</h3>${bloques}</div>
  </div>`;
  setAnim("algo", "cv-algo", info.anim);
  $("btn-algo-replay").addEventListener("click", () => ANIMADORES["algo"] && ANIMADORES["algo"].reset());
  mt($("algo-panel"));
}

/* ═══ RESULTADOS DR/AR ═══════════════════════════════════════════════════ */
function dsOptions() {
  let h = D.MODOS.map(m => `<option value="ami:${m}">AMI · ${D.MODOS_ES[m]}</option>`).join("");
  h += CTXS().map(c => `<option value="sim:${c}">${SIM(c).meta.icono} Simulado · ${SIM(c).meta.titulo}</option>`).join("");
  return h;
}
function getMetricas(ds) {
  const [k, v] = ds.split(":");
  return k === "ami" ? D.metricas[v] : SIM(v).metricas;
}
function getTipos(ds) {
  const [k, v] = ds.split(":");
  return k === "ami" ? D.TIPOS_ANOM : SIM(v).meta.tipos;
}
function renderDR() {
  const ds = $("dr-dataset").value, tipo = $("dr-tipo").value;
  const rows = getMetricas(ds);
  const dedup = {};
  for (const r of rows) {
    if (!dedup[r.Detector]) dedup[r.Detector] = { ...r };
    if (r.TipoAnomalia === tipo) dedup[r.Detector].DR_sel = r.DR_tipo;
  }
  const lista = Object.values(dedup).map(r => ({
    det: r.Detector, fam: r.Familia, ar: r.AR_global,
    dr: tipo === "Todos" ? r.DR_global : (r.DR_sel ?? 0),
  })).sort((a, b) => (a.fam + "Z").localeCompare(b.fam + "Z") || b.dr - a.dr);

  Plotly.react("plot-dr", [{
    x: lista.map(r => r.det), y: lista.map(r => r.dr), type: "bar",
    marker: { color: lista.map(r => D.FAM_COLOR[r.fam] || "#888") },
    text: lista.map(r => (r.dr * 100).toFixed(0) + "%"), textposition: "outside",
    textfont: { size: 9.5 }, hovertemplate: "%{x}<br>DR = %{y:.3f}<extra></extra>",
  }], lay({
    title: { text: `Tasa de detección — ${tipo === "Todos" ? "global" : tipoLabel(ds, tipo)} (AR ≈ 10 %)`, font: { size: 14 } },
    xaxis: { tickangle: -42, tickfont: { size: 9.5 } },
    yaxis: { title: "DR", range: [0, 1.12], tickformat: ".0%" },
    margin: { b: 110 },
  }), PCFG);

  // heatmap detector × tipo
  const tipos = getTipos(ds);
  const detsOrd = Object.values(dedup).sort((a, b) => b.DR_global - a.DR_global).map(r => r.Detector);
  const z = detsOrd.map(d => tipos.map(t => {
    const r = rows.find(x => x.Detector === d && x.TipoAnomalia === t);
    return r ? r.DR_tipo : null;
  }));
  Plotly.react("plot-dr-heat", [{
    z, x: tipos.map(t => tipoLabel(ds, t)), y: detsOrd, type: "heatmap",
    colorscale: "RdYlGn", zmin: 0, zmax: 1,
    colorbar: { title: "DR", thickness: 12 },
    hovertemplate: "%{y} · %{x}<br>DR = %{z:.2f}<extra></extra>",
  }], lay({ margin: { l: 150, r: 10, t: 10, b: 90 }, xaxis: { tickangle: -38, tickfont: { size: 10 } },
            yaxis: { tickfont: { size: 9.5 }, autorange: "reversed" } }), PCFG);

  // top-5
  const top = [...lista].sort((a, b) => b.dr - a.dr).slice(0, 5);
  $("dr-top5").innerHTML = `<table class="tabla"><tr><th>#</th><th>Detector</th><th>Familia</th><th>DR</th></tr>
    ${top.map((r, i) => `<tr><td>${i + 1}</td>
      <td><b style="color:${D.FAM_COLOR[r.fam]}">${r.det}</b></td>
      <td><span class="chip" style="font-size:10.5px">${D.FAM_LABEL[r.fam] || r.fam}</span></td>
      <td><span class="barra-mini" style="width:${r.dr * 70}px"></span> ${(r.dr * 100).toFixed(1)}%</td></tr>`).join("")}
  </table>`;

  const sigs = lista.filter(r => r.fam === "Signatures"), clas = lista.filter(r => r.fam !== "Signatures");
  const mean = a => a.reduce((s, r) => s + r.dr, 0) / Math.max(a.length, 1);
  $("dr-lectura").innerHTML = `<div class="callout naranja" style="font-size:13.5px">
    <b>Lectura.</b> DR medio familia Signatures: <b>${(mean(sigs) * 100).toFixed(1)}%</b> ·
    resto: <b>${(mean(clas) * 100).toFixed(1)}%</b>. Con el mismo presupuesto de alertas (τ = p90),
    ${mean(sigs) > mean(clas) ? "la geometría del camino captura información que las features estáticas pierden."
      : "en este corte las features de magnitud dominan — revisa los tipos de FORMA donde las signaturas remontan."}
  </div>`;

  // ── Resumen del caso de estudio: cuántos datos se toman y cuántos se ──────
  //    detectan como anomalía, según el tipo de muestreo seleccionado.
  const [k, v] = ds.split(":");
  const r0 = rows[0];
  const nOrig = r0.n_orig;                                  // series normales analizadas
  const filasUnDet = rows.filter(r => r.Detector === r0.Detector);
  const nSintTotal = filasUnDet.reduce((s, r) => s + (r.n_tipo || 0), 0);
  const nSel = (tipo === "Todos")
      ? nSintTotal
      : (filasUnDet.find(r => r.TipoAnomalia === tipo)?.n_tipo || 0);
  const mejor = [...lista].sort((a, b) => b.dr - a.dr)[0];   // mejor detector para el corte
  const nDetectadas = Math.round(mejor.dr * nSel);           // anomalías detectadas
  const nFalsas = Math.round(mejor.ar * nOrig);              // falsas alarmas (AR)
  // Etiqueta del muestreo según el dataset (AMI = modo; simulado = régimen).
  let casoTxt, muestreoTxt;
  if (k === "ami") {
    casoTxt = "AMI · " + D.MODOS_ES[v];
    muestreoTxt = "Caso de muestreo AMI: <b>" + D.MODOS_ES[v] + "</b> (ventanas semanales de 168 h).";
  } else {
    const m = SIM(v).meta;
    casoTxt = m.icono + " " + m.titulo;
    muestreoTxt = "Muestreo <b>" + m.muestreo.tipo.replace("_", " ") + "</b>: " + m.muestreo.descripcion;
  }
  const tipoTxt = tipo === "Todos" ? "anomalías inyectadas (todos los tipos)" : "anomalías de tipo " + tipoLabel(ds, tipo);
  const card = (num, lbl, col) =>
    `<div style="flex:1;min-width:120px;background:#fff;border:1px solid var(--linea);border-radius:12px;padding:12px 14px">
       <div style="font-size:26px;font-weight:800;color:${col}">${num}</div>
       <div style="font-size:11.5px;color:var(--tinta-2)">${lbl}</div></div>`;
  $("dr-resumen").innerHTML = `
    <div class="info-card naranja">
      <h4>Caso de estudio: ${casoTxt}</h4>
      <p style="font-size:13px;margin-bottom:10px">${muestreoTxt}</p>
      <div style="display:flex;flex-wrap:wrap;gap:12px">
        ${card(nOrig.toLocaleString("es"), "datos normales analizados", "#1565c0")}
        ${card(nSel.toLocaleString("es"), tipoTxt, "#e65100")}
        ${card(nDetectadas + " / " + nSel, "detectadas por " + mejor.det + " (DR " + (mejor.dr * 100).toFixed(0) + "%)", "#2e7d32")}
        ${card(nFalsas, "falsas alarmas sobre normales (AR " + (mejor.ar * 100).toFixed(0) + "%)", "#c62828")}
      </div>
      <p style="font-size:12px;color:var(--tinta-2);margin-top:10px">
        Total de ventanas en el experimento: <b>${(nOrig + nSintTotal).toLocaleString("es")}</b>
        (${nOrig.toLocaleString("es")} normales + ${nSintTotal.toLocaleString("es")} anomalías inyectadas de 7 tipos).
        El umbral τ se fija en el percentil 90 de los scores sobre las series normales.</p>
    </div>`;
}

/* ═══ JACCARD ════════════════════════════════════════════════════════════ */
function getJaccard(ds, tipo) {
  const [k, v] = ds.split(":");
  return k === "ami" ? D.jaccard[`${tipo}__${v}`] : SIM(v).jaccard[tipo];
}
function renderJaccard() {
  const ds = $("jac-dataset").value, tipo = $("jac-tipo").value;
  const blk = getJaccard(ds, tipo);
  if (!blk) return;
  Plotly.react("plot-jaccard", [{
    z: blk.mat, x: blk.dets, y: blk.dets, type: "heatmap",
    // Escala explícita: J=0 -> blanco puro, J=1 -> azul intenso.
    colorscale: [[0, "#ffffff"], [0.5, "#90caf9"], [1, "#0d47a1"]],
    zmin: 0, zmax: 1, colorbar: { title: "J", thickness: 12 },
    hovertemplate: "%{y} ∩ %{x}<br>J = %{z:.2f}<extra></extra>",
  }], lay({
    margin: { l: 150, r: 10, t: 10, b: 130 },
    xaxis: { tickangle: -55, tickfont: { size: 8.5 } },
    yaxis: { tickfont: { size: 8.5 }, autorange: "reversed" },
  }), PCFG);

  // estadística intra/inter familia
  let intra = [], inter = [];
  blk.dets.forEach((a, i) => blk.dets.forEach((b, j) => {
    if (i < j) (famDe(a) === famDe(b) ? intra : inter).push(blk.mat[i][j] ?? 0);
  }));
  const mean = a => a.length ? a.reduce((x, y) => x + y) / a.length : 0;
  $("jac-info").innerHTML = `
    <div class="info-card"><h4>Cómo leer esta matriz</h4>
      <p style="font-size:13.5px">Cada celda: fracción de anomalías detectadas en común.
      J medio <b>dentro</b> de la misma familia: <b>${mean(intra).toFixed(2)}</b> ·
      <b>entre</b> familias: <b>${mean(inter).toFixed(2)}</b>.</p>
      <p style="font-size:13px;margin-top:8px">Detectores con J alto son intercambiables (elige el más barato);
      con J bajo y DR alto son <b>complementarios</b>: un ensamble OR amplía la cobertura sin duplicar alertas.</p></div>
    <div class="callout azul" style="font-size:13px">Los bloques diagonales de la familia Signatures
    (mismo nivel, distinto k) confirman la estabilidad del espacio de firmas; el bloque cruzado con
    los detectores de magnitud baja en anomalías de FORMA pura (FlipSchedule, DesacopleCanales).</div>`;
}

/* ═══ INSPECTOR ══════════════════════════════════════════════════════════ */
function poblarInspector() {
  const ds = $("insp-dataset").value;
  const [k, v] = ds.split(":");
  let dets, tipos;
  if (k === "ami") {
    dets = [...new Set(D.detecciones.map(r => r.detector))];
    tipos = D.TIPOS_ANOM;
  } else {
    dets = [...new Set(SIM(v).detecciones.map(r => r.detector))];
    tipos = SIM(v).meta.tipos;
  }
  const detPrev = $("insp-det").value;
  $("insp-det").innerHTML = dets.map(d => `<option ${d === detPrev ? "selected" : ""}>${d}</option>`).join("");
  $("insp-tipo").innerHTML = tipos.map(t => `<option value="${t}">${tipoLabel(ds, t)}</option>`).join("");
  poblarInspCaso();
}
function poblarInspCaso() {
  const ds = $("insp-dataset").value, tipo = $("insp-tipo").value;
  const [k, v] = ds.split(":");
  if (k === "ami") {
    $("insp-caso").innerHTML = [...Array(D.N_CASAS_REF).keys()].map(i =>
      `<option value="${i}">casa ${i + 1}</option>`).join("");
  } else {
    const pares = SIM(v).pares.filter(p => p.tipo === tipo);
    $("insp-caso").innerHTML = pares.map((p, i) =>
      `<option value="${p.par_idx}">ejemplar ${i + 1}</option>`).join("");
  }
}
function renderInspector() {
  const ds = $("insp-dataset").value, det = $("insp-det").value,
        tipo = $("insp-tipo").value, caso = $("insp-caso").value;
  const [k, v] = ds.split(":");
  let reg, serieTr = [], titulo = "", todosScores = [];

  if (k === "ami") {
    reg = IDX.detCasa[`${det}|${caso}|${tipo}`];
    const par = IDX.pares[caso] || {};
    if (par.normal && par[tipo]) {
      const horas = [...Array(168).keys()];
      serieTr = [
        { x: horas, y: par.normal.h, name: "normal", line: { color: "#1565c0", width: 1.8 } },
        { x: horas, y: par[tipo].h, name: tipoLabel(ds, tipo), line: { color: D.ANOM_COLOR[tipo] || "#e65100", width: 1.8 } }];
      titulo = `Casa ${+caso + 1} · ${tipoLabel(ds, tipo)}`;
    }
    for (const d of new Set(D.detecciones.map(r => r.detector))) {
      const r = IDX.detCasa[`${d}|${caso}|${tipo}`];
      if (r) todosScores.push({ det: d, margen: r.score_norm - r.tau_norm, detectado: r.detectado });
    }
  } else {
    reg = IDX.simDet[`${v}|${det}|${caso}`];
    const par = SIM(v).pares.find(p => p.par_idx === +caso);
    if (par) {
      const meta = SIM(v).meta;
      meta.canales.forEach((ch, i) => {
        const norm = arr => { const lo = Math.min(...arr), hi = Math.max(...arr); return arr.map(x => (x - lo) / ((hi - lo) || 1) + i * 1.15); };
        serieTr.push({ x: par.t_base, y: norm(par.base[ch]), name: ch + " normal", legendgroup: "n", showlegend: i === 0,
          line: { color: "#9fb2c8", width: 1.2 }, hoverinfo: "name" });
        serieTr.push({ x: par.t, y: norm(par.canales[ch]), name: ch + " anómala", legendgroup: "a", showlegend: i === 0,
          line: { color: "#e65100", width: 1.4 }, hoverinfo: "name" });
      });
      titulo = `${meta.titulo} · ${tipoLabel(ds, tipo)} (canales apilados)`;
    }
    for (const d of new Set(SIM(v).detecciones.map(r => r.detector))) {
      const r = IDX.simDet[`${v}|${d}|${caso}`];
      if (r) todosScores.push({ det: d, margen: r.score_norm - r.tau_norm, detectado: r.detectado });
    }
  }

  Plotly.react("plot-insp-serie", serieTr, lay({
    title: { text: titulo, font: { size: 13.5 } }, legend: { orientation: "h", y: 1.18 },
    yaxis: { showticklabels: k === "ami" }, margin: { t: 40, b: 34 },
  }), PCFG);

  todosScores.sort((a, b) => b.margen - a.margen);
  Plotly.react("plot-insp-scores", [{
    x: todosScores.map(r => r.margen), y: todosScores.map(r => r.det),
    type: "bar", orientation: "h",
    marker: { color: todosScores.map(r => r.det === det ? "#ffb300" : (r.detectado ? colorFam(r.det) : "#c9d4e3")) },
    hovertemplate: "%{y}<br>score − τ = %{x:.3f}<extra></extra>",
  }], lay({
    title: { text: "distancia al umbral de cada detector  ·  (+) detectado / (−) no detectado", font: { size: 12.5 } },
    margin: { l: 150, t: 36, b: 44 },
    xaxis: { title: "score − umbral", zeroline: true, zerolinecolor: "#15233a", zerolinewidth: 2 },
    yaxis: { tickfont: { size: 8.5 }, autorange: "reversed" },
  }), PCFG);

  // ── Conteos REALES (no solo la razón): sobre todos los casos de este tipo,
  //    cuántas anomalías marcó ESTE detector, y cuántas series se analizaron.
  let nTot = 0, nDet = 0;
  if (k === "ami") {
    for (let c = 0; c < D.N_CASAS_REF; c++) {
      const rr = IDX.detCasa[`${det}|${c}|${tipo}`];
      if (rr) { nTot++; if (rr.detectado) nDet++; }
    }
  } else {
    SIM(v).pares.filter(p => p.tipo === tipo).forEach(p => {
      const rr = IDX.simDet[`${v}|${det}|${p.par_idx}`];
      if (rr) { nTot++; if (rr.detectado) nDet++; }
    });
  }
  const dr = nTot ? (nDet / nTot) : 0;
  // Conteo del experimento COMPLETO (de las métricas) cuando está disponible:
  // da la "cantidad de datos" grande además del conjunto del inspector.
  let expTxt = "";
  const metr = (k === "ami") ? (D.metricas.annual || []) : SIM(v).metricas;
  const fila = metr.find(r => r.Detector === det && r.TipoAnomalia === tipo);
  if (fila && fila.n_tipo) {
    const detExp = Math.round((fila.DR_tipo || 0) * fila.n_tipo);
    expTxt = `<p style="font-size:12px;color:var(--tinta-2);margin-top:8px">
      En el experimento completo${k === "ami" ? " (modo Anual)" : ""}: <b>${fila.n_orig}</b> series normales,
      <b>${fila.n_tipo}</b> anomalías de este tipo, de las cuales <b>${det}</b> detectó
      <b>${detExp} / ${fila.n_tipo}</b> (DR ${((fila.DR_tipo || 0) * 100).toFixed(0)}%).</p>`;
  }

  if (reg) {
    const det_ok = reg.detectado;
    $("insp-info").innerHTML = `
      <div class="plot-card" style="text-align:center">
        <h3>${det}</h3>
        <div style="font-size:40px;font-weight:800;color:${det_ok ? "var(--verde)" : "var(--rojo)"}">
          ${(reg.tau_norm > 0 ? reg.score_norm / reg.tau_norm : 0).toFixed(2)}×</div>
        <div style="font-size:12.5px;color:var(--tinta-2)">el score del caso vale ${(reg.tau_norm > 0 ? reg.score_norm / reg.tau_norm : 0).toFixed(2)} veces el umbral
          (${det_ok ? "≥ 1 ⇒ alerta" : "< 1 ⇒ sin alerta"})</div>
        <div style="margin-top:8px;font-size:13px;color:var(--tinta-2)">
          índice de anomalía <b>${reg.score_norm.toFixed(2)}</b> · umbral <b>${reg.tau_norm.toFixed(2)}</b>
          <span style="font-size:11px">(escala normalizada)</span></div>
        <div style="margin-top:10px"><span class="${det_ok ? "badge-ok" : "badge-no"}">
          ${det_ok ? "✓ DETECTADA" : "✗ no detectada"}</span></div>
      </div>
      <div class="plot-card">
        <h3>Conteo real de este detector</h3>
        <div style="display:flex;gap:12px;flex-wrap:wrap">
          <div style="flex:1;min-width:90px;text-align:center">
            <div style="font-size:28px;font-weight:800;color:#1565c0">${nTot}</div>
            <div style="font-size:11.5px;color:var(--tinta-2)">casos de este tipo analizados</div></div>
          <div style="flex:1;min-width:90px;text-align:center">
            <div style="font-size:28px;font-weight:800;color:#2e7d32">${nDet} / ${nTot}</div>
            <div style="font-size:11.5px;color:var(--tinta-2)">anomalías detectadas (DR ${(dr * 100).toFixed(0)}%)</div></div>
        </div>
        ${expTxt}
      </div>
      <div class="callout azul" style="font-size:13px">Las barras doradas/grises de abajo muestran qué
      detectores cruzan su umbral en ESTE caso concreto.</div>`;
  }
}

/* ═══ EXPLORADOR DE SIGNATURAS ═══════════════════════════════════════════ */
function poblarSig() {
  let h = `<option value="ami">⚡ AMI (univariada)</option>`;
  h += CTXS().map(c => `<option value="sim:${c}">${SIM(c).meta.icono} ${SIM(c).meta.titulo}</option>`).join("");
  $("sig-fuente").innerHTML = h;
  poblarSigTipo();
}
function poblarSigTipo() {
  const f = $("sig-fuente").value;
  if (f === "ami") {
    $("sig-tipo").innerHTML = D.TIPOS_ANOM.map(t => `<option value="${t}">${D.TIPO_LABEL[t]}</option>`).join("");
    $("sig-caso").innerHTML = [...Array(D.N_CASAS_REF).keys()].map(i => `<option value="${i}">casa ${i + 1}</option>`).join("");
    $("sig-rango-wrap").style.display = "";
  } else {
    const c = f.split(":")[1];
    $("sig-tipo").innerHTML = SIM(c).meta.tipos.map(t =>
      `<option value="${t}">${SIM(c).meta.descrip_tipos[t].titulo}</option>`).join("");
    poblarSigCaso();
    $("sig-rango-wrap").style.display = "none";
  }
}
function poblarSigCaso() {
  const f = $("sig-fuente").value;
  if (f === "ami") return;
  const c = f.split(":")[1], t = $("sig-tipo").value;
  const pares = SIM(c).pares.filter(p => p.tipo === t);
  $("sig-caso").innerHTML = pares.map((p, i) => `<option value="${p.par_idx}">ejemplar ${i + 1}</option>`).join("");
}

function renderSig() {
  const fuente = $("sig-fuente").value;
  const grado = +$("sig-grado").value;
  $("sig-grado-v").textContent = grado;
  let sigN, sigA, etiq, serieAnim, canales = null, nombres;

  if (fuente === "ami") {
    const tipo = $("sig-tipo").value, casa = +$("sig-caso").value;
    let h0 = +$("sig-h0").value, h1 = +$("sig-h1").value;
    if (h1 <= h0 + 3) h1 = Math.min(167, h0 + 4);
    $("sig-rango-v").textContent = `${h0}–${h1}`;
    const par = IDX.pares[casa] || {};
    if (!par.normal || !par[tipo]) return;
    const yN = par.normal.h.slice(h0, h1 + 1), yA = par[tipo].h.slice(h0, h1 + 1);
    sigN = Sig.calcular(Sig.caminoDeSerie(yN), grado);
    sigA = Sig.calcular(Sig.caminoDeSerie(yA), grado);
    nombres = ["t", "x"];
    etiq = Sig.etiquetas(2, grado, nombres);
    serieAnim = yA;
  } else {
    const c = fuente.split(":")[1], pi = +$("sig-caso").value;
    const par = SIM(c).pares.find(p => p.par_idx === pi);
    if (!par) return;
    const gradoM = Math.min(grado, 3);
    const cn = Sig.caminoMultivariado(par.base, par.t_base);
    const ca = Sig.caminoMultivariado(par.canales, par.t);
    sigN = Sig.calcular(cn.path, gradoM);
    sigA = Sig.calcular(ca.path, gradoM);
    nombres = ca.nombres;
    etiq = Sig.etiquetas(nombres.length, gradoM, nombres);
    const ch0 = SIM(c).meta.canales[0];
    serieAnim = par.canales[ch0];
  }

  // animación de desarrollo
  const anim = ANIMADORES["firma"];
  if (anim) { anim.st.serie = serieAnim; }
  else setAnim("firma", "cv-firma", "firma_dev", { serie: serieAnim });

  // barras por coordenada (≤ 30) o por nivel (multivariado grande)
  const coordsN = [], coordsA = [];
  sigN.niveles.forEach(lv => coordsN.push(...lv));
  sigA.niveles.forEach(lv => coordsA.push(...lv));
  if (coordsN.length <= 32) {
    Plotly.react("plot-sig-bars", [
      { x: etiq, y: coordsN, type: "bar", name: "normal", marker: { color: "#1565c0" } },
      { x: etiq, y: coordsA, type: "bar", name: "anómala", marker: { color: "#e65100" } },
    ], lay({
      title: { text: "Coordenadas de la signatura (Chen, en vivo)", font: { size: 13.5 } },
      barmode: "group", xaxis: { tickangle: -50, tickfont: { size: 9 } },
      legend: { orientation: "h", y: 1.12 }, margin: { b: 90 },
    }), PCFG);
  } else {
    const nN = Sig.normasPorNivel(sigN), nA = Sig.normasPorNivel(sigA);
    const lv = nN.map((_, i) => "nivel " + (i + 1));
    Plotly.react("plot-sig-bars", [
      { x: lv, y: nN, type: "bar", name: "normal", marker: { color: "#1565c0" } },
      { x: lv, y: nA, type: "bar", name: "anómala", marker: { color: "#e65100" } },
    ], lay({
      title: { text: `Norma por nivel (${coordsN.length} coordenadas — demasiadas para barras)`, font: { size: 13 } },
      barmode: "group", legend: { orientation: "h", y: 1.12 },
    }), PCFG);
  }

  // áreas de Lévy
  const Ln = Sig.levy(sigN), La = Sig.levy(sigA);
  const dif = Ln.map((fila, i) => fila.map((v, j) => La[i][j] - v));
  Plotly.react("plot-levy", [{
    z: dif, x: nombres, y: nombres, type: "heatmap",
    colorscale: "RdBu", zmid: 0, colorbar: { title: "ΔA", thickness: 12 },
    hovertemplate: "A(%{y},%{x})<br>anómala − normal = %{z:.4f}<extra></extra>",
  }], lay({
    title: { text: "ΔA<sup>ij</sup> = áreas de Lévy (anómala − normal)", font: { size: 13 } },
    yaxis: { autorange: "reversed" }, margin: { l: 60, t: 40 },
  }), PCFG);

  // panel de niveles
  const dist = Sig.distancia(sigN, sigA);
  $("sig-niveles-panel").innerHTML = `
    <div class="plot-card"><h3>Distancia entre firmas</h3>
      <div style="font-size:34px;font-weight:800;color:var(--naranja)">‖S(X) − S(X̃)‖ = ${dist.toFixed(3)}</div>
      <p style="font-size:13px;color:var(--tinta-2)">La métrica que usan SigMaHaKNN y compañía —
      aquí calculada en tu navegador con la identidad de Chen.</p></div>
    ${NIVELES_SIG.slice(0, Math.min(grado, 4)).map(nv => `
      <div class="info-card" style="border-left-color:${nv.color};padding:10px 16px">
        <h4 style="color:${nv.color};font-size:13.5px">Nivel ${nv.n}: ${nv.titulo}</h4>
        <p style="font-size:12.5px">${nv.texto.split(".")[0]}.</p></div>`).join("")}`;
  mt($("sig-niveles-panel"));   // renderiza las ecuaciones \(...\) del panel
}

/* ═══ NDE TEORÍA ═════════════════════════════════════════════════════════ */
function renderNDETeoria(idSel) {
  idSel = idSel || "ncde";
  $("nde-tabs").innerHTML = NDE_ETAPAS.map((e, i) => `
    <button class="nde-tab ${e.id === idSel ? "activa" : ""}" data-id="${e.id}">
      <span class="paso">${e.paso}</span><span class="nom" style="color:${e.color}">${e.nombre}</span>
      <span class="eq">\\(${e.eq}\\)</span>
    </button>${i < NDE_ETAPAS.length - 1 ? '<span class="flecha-evo">→</span>' : ""}`).join("");
  $("nde-tabs").querySelectorAll(".nde-tab").forEach(b =>
    b.addEventListener("click", () => renderNDETeoria(b.dataset.id)));

  const e = NDE_ETAPAS.find(x => x.id === idSel);
  $("nde-panel").innerHTML = `
  <div class="teoria-panel">
    <div class="grid-2">
      <div class="plot-card">
        <h3 style="color:${e.color}">${e.nombre} — intuición animada</h3>
        <canvas id="cv-nde" class="canvas-anim" style="height:330px"></canvas>
        <div class="anim-controles"><button class="btn fantasma" id="btn-nde-replay">↻ Reanimar</button>
          <span class="desc">${e.motiv}</span></div>
      </div>
      <div>
        <div class="formula-box" style="font-size:17px;text-align:center;padding:26px">$$${e.eq}$$</div>
        <div class="props-grid">${e.props.map(p => `<div class="prop-mini"><b>propiedad</b>${p}</div>`).join("")}</div>
      </div>
    </div>
    <div class="plot-card"><h3>Construcción</h3>
      ${e.bloques.map(b => `
        <h4 style="color:${e.color};margin:14px 0 4px;font-size:14px">${b.h}</h4>
        <p>${b.t || ""}</p>${b.f ? `<div class="formula-box">$$${b.f}$$</div>` : ""}`).join("")}
    </div>
  </div>`;
  setAnim("nde", "cv-nde", e.anim);
  $("btn-nde-replay").addEventListener("click", () => ANIMADORES["nde"] && ANIMADORES["nde"].reset());
  mt($("nde-tabs")); mt($("nde-panel"));
}

/* ═══ NDE LAB ════════════════════════════════════════════════════════════ */
const NDE_COLOR = { RNN: "#546e7a", GRU: "#90a4ae", NeuralODE: "#00838f",
                    NeuralCDE: "#6a1b9a", NeuralRDE: "#e65100" };
function renderNDELab() {
  const nde = D.neuralde || {};
  const res = nde.resultados;
  if (res) {
    $("lab-dataset").innerHTML = Object.entries(res.datasets).map(([k, v]) =>
      `<option value="${k}">${v.titulo}</option>`).join("");
    renderLabComparativa();
  }

  // Van der Pol
  const v = nde.vanderpol;
  if (v) {
    const seg = (rej, campo, color, nombre) => {
      const xs = [], ys = [];
      const esc = 0.16;
      rej.forEach((p, i) => {
        const n = Math.hypot(campo[i][0], campo[i][1]) + 1e-9;
        xs.push(p[0], p[0] + campo[i][0] / n * esc * Math.min(n, 3), null);
        ys.push(p[1], p[1] + campo[i][1] / n * esc * Math.min(n, 3), null);
      });
      return { x: xs, y: ys, mode: "lines", name: nombre, line: { color, width: 1.1 }, opacity: .75 };
    };
    Plotly.newPlot("plot-vdp-campo", [
      seg(v.rejilla, v.campo_real, "#90a4ae", "campo real f"),
      seg(v.rejilla, v.campo_aprendido, "#e65100", "campo aprendido f_θ"),
    ], lay({
      title: { text: `Van der Pol (μ=${v.mu}): f vs f_θ aprendido de trayectorias`, font: { size: 13.5 } },
      xaxis: { title: "x", zeroline: false }, yaxis: { title: "ẋ", zeroline: false, scaleanchor: "x" },
      legend: { orientation: "h", y: -0.14 },
    }), PCFG);
    Plotly.newPlot("plot-vdp-tray", [
      { x: v.tray_real.map(p => p[0]), y: v.tray_real.map(p => p[1]), mode: "lines",
        name: "trayectoria real", line: { color: "#1565c0", width: 2.4 } },
      { x: v.tray_aprendida.map(p => p[0]), y: v.tray_aprendida.map(p => p[1]), mode: "lines",
        name: "integrada con f_θ", line: { color: "#e65100", width: 2, dash: "dash" } },
      { x: [v.tray_real[0][0]], y: [v.tray_real[0][1]], mode: "markers", name: "x(0)",
        marker: { color: "#2e7d32", size: 11, symbol: "star" } },
    ], lay({
      title: { text: `pérdida final RK4: ${v.perdidas[v.perdidas.length - 1].toExponential(2)} — el ciclo límite se reproduce`, font: { size: 13 } },
      xaxis: { title: "x" }, yaxis: { title: "ẋ", scaleanchor: "x" }, legend: { orientation: "h", y: -0.14 },
    }), PCFG);
  }

  // NCDE z(t)
  const tz = nde.trayectorias?.ncde_trayectorias;
  if (tz) {
    const tr = tz.map((s, i) => ({
      x: s.z.map(p => p[0]), y: s.z.map(p => p[1]), z: s.z.map(p => p[2]),
      type: "scatter3d", mode: "lines+markers",
      name: `espiral ${s.clase ? "antihoraria" : "horaria"} #${i % 2 + 1}`,
      line: { color: s.clase ? "#e65100" : "#1565c0", width: 5 },
      marker: { size: 2.5, color: s.clase ? "#e65100" : "#1565c0" },
    }));
    Plotly.newPlot("plot-ncde-z", tr, lay({
      margin: { l: 0, r: 0, t: 10, b: 0 },
      scene: { xaxis: { title: "PC1" }, yaxis: { title: "PC2" }, zaxis: { title: "PC3" } },
      legend: { orientation: "h", y: -0.06 },
    }), PCFG);
  }

  // NRDE logsig por ventana
  const lg = nde.trayectorias?.nrde_logode_demo;
  if (lg) {
    const z = lg.ventanas.map(w => w.logsig);
    Plotly.newPlot("plot-nrde-logsig", [{
      z: z, x: lg.etiquetas_logsig, y: lg.ventanas.map((w, i) => `ventana ${i + 1} [${w.rango[0]}–${w.rango[1]}]`),
      type: "heatmap", colorscale: "RdBu", zmid: 0, colorbar: { thickness: 12 },
      hovertemplate: "%{y} · %{x} = %{z:.3f}<extra></extra>",
    }], lay({
      title: { text: "log-signaturas nivel 2 por ventana (espiral): incrementos + áreas de Lévy", font: { size: 12.5 } },
      margin: { l: 130, t: 36 }, yaxis: { autorange: "reversed" },
    }), PCFG);
  }
}
function renderLabComparativa() {
  const res = D.neuralde?.resultados;
  if (!res) return;
  const dsk = $("lab-dataset").value;
  const runs = res.runs.filter(r => r.dataset === dsk);
  const modelos = [...new Set(runs.map(r => r.modelo))];

  Plotly.react("plot-lab-acc", ["regular", "irregular"].map(reg => ({
    x: modelos, y: modelos.map(m => runs.find(r => r.modelo === m && r.regimen === reg)?.acc_final ?? 0),
    type: "bar", name: "muestreo " + reg,
    marker: { color: modelos.map(m => NDE_COLOR[m]), opacity: reg === "regular" ? 0.95 : 0.55,
              line: { color: "#fff", width: 1 } },
  })), lay({
    barmode: "group", yaxis: { title: "accuracy validación", range: [0.4, 1.05], tickformat: ".0%" },
    legend: { orientation: "h", y: 1.14 }, title: { text: "accuracy final (media últimas 5 épocas)", font: { size: 13 } },
    shapes: [{ type: "line", x0: -0.5, x1: modelos.length - 0.5, y0: 0.5, y1: 0.5,
               line: { color: "#c62828", dash: "dot", width: 1.5 } }],
    annotations: [{ x: modelos.length - 1, y: 0.5, text: "azar", showarrow: false, yshift: 10, font: { color: "#c62828", size: 11 } }],
  }), PCFG);

  const curvas = [];
  for (const reg of ["regular", "irregular"])
    for (const m of modelos) {
      const r = runs.find(x => x.modelo === m && x.regimen === reg);
      if (r) curvas.push({
        y: r.acc_val, mode: "lines", name: `${m} (${reg})`,
        line: { color: NDE_COLOR[m], width: reg === "regular" ? 2.2 : 1.4, dash: reg === "regular" ? "solid" : "dot" },
        showlegend: reg === "regular",
      });
    }
  Plotly.react("plot-lab-curvas", curvas, lay({
    title: { text: "convergencia: accuracy por época (línea punteada = irregular)", font: { size: 13 } },
    xaxis: { title: "época" }, yaxis: { range: [0.35, 1.04], tickformat: ".0%" },
    legend: { orientation: "h", y: -0.2 },
  }), PCFG);

  const node = runs.find(r => r.modelo === "NeuralODE" && r.regimen === "regular");
  const ncde = runs.find(r => r.modelo === "NeuralCDE" && r.regimen === "irregular");
  const nrde = runs.find(r => r.modelo === "NeuralRDE" && r.regimen === "irregular");
  $("lab-lectura").innerHTML = `<div class="callout morado" style="font-size:13.5px">
    <b>Lo que muestra el experimento.</b> El Neural ODE queda en ${(100 * (node?.acc_final ?? 0.5)).toFixed(0)}%:
    solo ve la condición inicial \\(h(0)=\\mathrm{enc}(x_0)\\) (fase aleatoria ⇒ no informativa) — la
    limitación de Picard–Lindelöf en acción. Las ecuaciones CONTROLADAS la resuelven:
    NCDE ${(100 * (ncde?.acc_final ?? 0)).toFixed(0)}% y NRDE ${(100 * (nrde?.acc_final ?? 0)).toFixed(0)}%
    incluso con muestreo irregular, porque Δt y la geometría intra-ventana entran en la estructura
    (no como parche). El NRDE entrena ~${node ? Math.round((runs.find(r => r.modelo === "NeuralCDE" && r.regimen === "regular")?.segundos ?? 3) / Math.max(runs.find(r => r.modelo === "NeuralRDE" && r.regimen === "regular")?.segundos ?? 1, 0.1)) : 3}× más rápido
    que el NCDE con igual accuracy: paga n/s pasos gracias al método log-ODE.</div>`;
  mt($("lab-lectura"));
}

/* ═══ CONCLUSIONES ═══════════════════════════════════════════════════════ */
function renderConclusiones() {
  $("conclusiones").innerHTML = `
  <div class="grid-2">
    <div class="info-card naranja"><h4>1 · La geometría detecta lo que la magnitud no ve</h4>
      Los detectores de signaturas (Mahalanobis local sobre S(X) con aumento temporal) lideran el DR en los
      tres contextos multivariados con AR fijo del 10%, y son los únicos que capturan anomalías de ORDEN
      (FlipSchedule, DesacopleCanales, RafagaMuestreo) — invisibles para features estáticas.</div>
    <div class="info-card"><h4>2 · El muestreo es información, no un obstáculo</h4>
      Con el tiempo como canal del camino, el muestreo irregular queda codificado en la signatura
      (la anomalía RafagaMuestreo se detecta con valores PERFECTAMENTE normales). El mismo principio hace
      que NCDE/NRDE manejen datos irregulares de forma estructural.</div>
    <div class="info-card purpura"><h4>3 · RNN → NODE → NCDE → NRDE es una sola idea madurando</h4>
      Discretización → límite continuo → control por datos → resumen geométrico por ventanas. Cada etapa
      corrige un defecto demostrable de la anterior; el experimento de espirales lo exhibe con precisión
      cuantitativa (NODE ≈ azar, CDE/RDE ≈ 100%).</div>
    <div class="info-card verde"><h4>4 · Un solo lenguaje matemático</h4>
      La log-signatura nivel 2 (incrementos + áreas de Lévy) es a la vez el mejor descriptor para detección
      (LogSigMaHa_d2) y el motor del método log-ODE de las Neural RDE: detección de anomalías y modelado
      dinámico comparten el álgebra de los rough paths de Lyons.</div>
  </div>`;
}

/* ═══ controles & navegación ═════════════════════════════════════════════ */
function populateControles() {
  $("series-tipo").innerHTML = D.TIPOS_ANOM.map(t => `<option value="${t}">${D.TIPO_LABEL[t]}</option>`).join("");
  poblarMultivar();
  $("dr-dataset").innerHTML = dsOptions();
  $("dr-dataset").value = "ami:annual";
  poblarDrTipo();
  $("jac-dataset").innerHTML = dsOptions();
  $("jac-dataset").value = "ami:annual";
  poblarJacTipo();
  $("insp-dataset").innerHTML = dsOptions();
  $("insp-dataset").value = "ami:annual";
  poblarInspector();
  poblarSig();
  document.querySelectorAll("input[type=range]").forEach(syncRange);
}
function poblarDrTipo() {
  const ds = $("dr-dataset").value;
  $("dr-tipo").innerHTML = `<option value="Todos">Todos (global)</option>` +
    getTipos(ds).map(t => `<option value="${t}">${tipoLabel(ds, t)}</option>`).join("");
}
function poblarJacTipo() {
  const ds = $("jac-dataset").value;
  $("jac-tipo").innerHTML = `<option value="general">General</option>` +
    getTipos(ds).map(t => `<option value="${t}">${tipoLabel(ds, t)}</option>`).join("");
}

function bindEventos() {
  ["series-tipo", "series-casa", "series-h0", "series-h1"].forEach(id => on(id, renderSeries));
  on("mv-ctx", () => { poblarMvTipo(); renderMultivar(); });
  on("mv-tipo", () => { poblarMvPar(); renderMultivar(); });
  on("mv-par", renderMultivar);
  on("dr-dataset", () => { poblarDrTipo(); renderDR(); });
  on("dr-tipo", renderDR);
  on("jac-dataset", () => { poblarJacTipo(); renderJaccard(); });
  on("jac-tipo", renderJaccard);
  on("insp-dataset", () => { poblarInspector(); renderInspector(); });
  on("insp-det", renderInspector);
  on("insp-tipo", () => { poblarInspCaso(); renderInspector(); });
  on("insp-caso", renderInspector);
  on("sig-fuente", () => { poblarSigTipo(); renderSig(); });
  on("sig-tipo", () => { poblarSigCaso(); renderSig(); });
  ["sig-caso", "sig-grado", "sig-h0", "sig-h1"].forEach(id => on(id, renderSig));
  on("lab-dataset", renderLabComparativa);
  const btn = $("btn-firma-replay");
  if (btn) btn.addEventListener("click", () => ANIMADORES["firma"] && ANIMADORES["firma"].reset());
}

function setupNav() {
  const links = [...document.querySelectorAll(".side-nav a")];
  const secs = links.map(a => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const obs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        const id = "#" + en.target.id;
        links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === id));
      }
    });
  }, { rootMargin: "-35% 0px -55% 0px" });
  secs.forEach(s => obs.observe(s));
}

function setupReveal() {
  document.querySelectorAll(".plot-card, .info-card, .callout").forEach(el => el.classList.add("reveal"));
  const obs = new IntersectionObserver(entries =>
    entries.forEach(en => { if (en.isIntersecting) { en.target.classList.add("vis"); obs.unobserve(en.target); } }),
    { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".reveal").forEach(el => obs.observe(el));
}

window.addEventListener("DOMContentLoaded", init);
