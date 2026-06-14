/* ═══════════════════════════════════════════════════════════════════════════
   anims.js — animaciones canvas explicativas

   * Un Animador por canvas (requestAnimationFrame, escala DPR, play/pausa).
   * ANIMS: catálogo de funciones de dibujo — cada detector y cada etapa de
     las ecuaciones diferenciales neuronales tiene su explicación animada.
   Convención: fn(ctx, W, H, t, st) con t en segundos y st = estado persistente.
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

/* ── utilidades ─────────────────────────────────────────────────────────── */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss2(rnd) {
  const u = Math.max(rnd(), 1e-9), v = rnd();
  return [Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v),
          Math.sqrt(-2 * Math.log(u)) * Math.sin(2 * Math.PI * v)];
}
const C = {
  punto: "#8fc1ff", punto2: "#5e93d8", anom: "#ff9e4d", rojo: "#ff6b6b",
  linea: "rgba(143,193,255,.35)", texto: "#cfe2ff", acento: "#ffd166",
  verde: "#6fe3a5", morado: "#c792ea", fondo: "rgba(10,28,61,0)",
};
function txt(ctx, s, x, y, color = C.texto, size = 12, align = "left", bold = false) {
  ctx.fillStyle = color;
  ctx.font = `${bold ? "700 " : ""}${size}px Segoe UI, sans-serif`;
  ctx.textAlign = align; ctx.fillText(s, x, y);
}
function punto(ctx, x, y, r, color, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832);
  ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
}
function aro(ctx, x, y, r, color, lw = 1.5, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); ctx.globalAlpha = 1;
}
function linea(ctx, x1, y1, x2, y2, color, lw = 1.5, alpha = 1, dash = null) {
  ctx.globalAlpha = alpha;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
}
function elipse(ctx, x, y, rx, ry, rot, color, lw = 1.5, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.beginPath(); ctx.ellipse(x, y, rx, ry, rot, 0, 6.2832);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); ctx.globalAlpha = 1;
}
function pulso(t, per = 1.6) { return 0.5 + 0.5 * Math.sin((t / per) * 6.2832); }

/* nube de puntos en 2 clusters + 1 outlier — base de muchas anims */
function nubeBase(st, seed = 7, n = 46) {
  if (st.pts) return st;
  const rnd = mulberry32(seed);
  st.pts = [];
  for (let i = 0; i < n; i++) {
    const [gx, gy] = gauss2(rnd);
    if (i % 2) st.pts.push([0.33 + gx * 0.055, 0.42 + gy * 0.085]);
    else       st.pts.push([0.62 + gx * 0.075, 0.62 + gy * 0.060]);
  }
  st.out = [0.84, 0.22];
  return st;
}
const PX = (W, p) => 40 + p[0] * (W - 80);
const PY = (H, p) => 24 + p[1] * (H - 64);

/* ── clase Animador ─────────────────────────────────────────────────────── */
class Animador {
  constructor(canvas, fnDibujo, st = {}) {
    this.cv = canvas; this.fn = fnDibujo; this.st = st;
    this.activo = false; this.t0 = performance.now(); this._raf = null;
    this._resize();
  }
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.cv.clientWidth || 720;
    const h = this.cv.clientHeight || (this.cv.height / (window.devicePixelRatio || 1)) || 320;
    this.cv.width = w * dpr; this.cv.height = h * dpr;
    this.W = w; this.H = h; this.dpr = dpr;
  }
  frame() {
    const ctx = this.cv.getContext("2d");
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.W, this.H);
    const t = (performance.now() - this.t0) / 1000;
    try { this.fn(ctx, this.W, this.H, t, this.st); } catch (e) { /* anim segura */ }
    if (this.activo) this._raf = requestAnimationFrame(() => this.frame());
  }
  play() { if (this.activo) return; this.activo = true; this.t0 = performance.now() - (this.st._t || 0) * 1000; this.frame(); }
  pause() { this.activo = false; if (this._raf) cancelAnimationFrame(this._raf); }
  reset() { this.st = { serie: this.st.serie, extra: this.st.extra }; this.t0 = performance.now(); if (!this.activo) { this.activo = true; this.frame(); } }
}

/* ═══════════════════════════════════════════════════════════════════════════
   CATÁLOGO DE ANIMACIONES — DETECTORES
   ═══════════════════════════════════════════════════════════════════════════ */
const ANIMS = {};

/* 1 ─ RobustZMAD: mediana, banda MAD y outlier */
ANIMS.zscore = (ctx, W, H, t, st) => {
  if (!st.v) {
    const rnd = mulberry32(11);
    st.v = Array.from({ length: 60 }, (_, i) => 0.5 + 0.16 * (gauss2(rnd)[0]));
    st.v[44] = 0.06;
  }
  const n = st.v.length, x0 = 50, x1 = W - 30, ym = H * 0.52;
  const X = i => x0 + (i / (n - 1)) * (x1 - x0);
  const Y = v => H - 30 - v * (H - 75);
  const med = 0.5, mad = 0.16, k = 3.0;
  // banda mediana ± k·MAD
  ctx.fillStyle = "rgba(111,227,165,.10)";
  ctx.fillRect(x0, Y(med + k * mad / 2.2), x1 - x0, Y(med - k * mad / 2.2) - Y(med + k * mad / 2.2));
  linea(ctx, x0, Y(med), x1, Y(med), C.verde, 1.6, .9, [6, 4]);
  txt(ctx, "mediana", x0 + 4, Y(med) - 6, C.verde, 11);
  txt(ctx, "± 3·MAD / 1.4826", x0 + 4, Y(med + k * mad / 2.2) - 5, "rgba(111,227,165,.8)", 10.5);
  const vis = Math.min(n, Math.floor(t * 14) + 1);
  for (let i = 0; i < vis; i++) {
    const fuera = Math.abs(st.v[i] - med) > k * mad / 2.2;
    punto(ctx, X(i), Y(st.v[i]), fuera ? 6 + 2 * pulso(t) : 3.4,
          fuera ? C.rojo : C.punto, fuera ? .95 : .8);
    if (fuera && vis > 46) {
      linea(ctx, X(i), Y(st.v[i]), X(i), Y(med), C.rojo, 1.4, .8, [3, 3]);
      txt(ctx, "|z| = |x−med|/MAD ≫ 3", X(i) - 8, Y(st.v[i]) + 22, C.rojo, 11.5, "right", true);
    }
  }
  txt(ctx, "Cada feature se estandariza con mediana y MAD (robustos); alerta por el máximo |z|.", 14, 18, C.texto, 12);
};

/* 2 ─ Elipse de Mahalanobis (PCAT2Q) */
ANIMS.ellipse = (ctx, W, H, t, st) => {
  nubeBase(st, 5);
  const cx = W * 0.45, cy = H * 0.55, rot = -0.5;
  for (let k = 1; k <= 3; k++)
    elipse(ctx, cx, cy, 52 * k, 26 * k, rot, k === 3 ? C.acento : C.linea, k === 3 ? 2 : 1.2, k === 3 ? .9 : .6);
  txt(ctx, "T² = cte", cx + 158 * Math.cos(rot) + 8, cy + 158 * Math.sin(rot), C.acento, 11);
  const rnd = mulberry32(9);
  for (let i = 0; i < 60; i++) {
    const [gx, gy] = gauss2(rnd);
    const ex = gx * 48, ey = gy * 22;
    const x = cx + ex * Math.cos(rot) - ey * Math.sin(rot);
    const y = cy + ex * Math.sin(rot) + ey * Math.cos(rot);
    punto(ctx, x, y, 3.2, C.punto, .75);
  }
  const ox = W * 0.82, oy = H * 0.25;
  punto(ctx, ox, oy, 6 + 2 * pulso(t), C.rojo);
  const f = Math.min(1, t / 2.5);
  linea(ctx, cx, cy, cx + (ox - cx) * f, cy + (oy - cy) * f, C.rojo, 2, .85, [5, 4]);
  punto(ctx, cx, cy, 4, C.acento);
  txt(ctx, "μ", cx - 14, cy + 4, C.acento, 13, "left", true);
  txt(ctx, "T² = (x−μ)ᵀ Σ⁻¹ (x−μ)", ox - 6, oy - 14, C.rojo, 12.5, "right", true);
  txt(ctx, "La covarianza Σ define la métrica: la distancia se mide en 'desviaciones elípticas'.", 14, 18, C.texto, 12);
};

/* 3 ─ KDE: suma de núcleos gaussianos */
ANIMS.kde = (ctx, W, H, t, st) => {
  if (!st.c) {
    const rnd = mulberry32(21);
    st.c = Array.from({ length: 13 }, () => 0.18 + 0.5 * rnd());
    st.c.push(0.88);
  }
  const x0 = 46, x1 = W - 30, yb = H - 38, bw = 0.045;
  const X = u => x0 + u * (x1 - x0);
  const dens = u => st.c.reduce((s, c) => s + Math.exp(-((u - c) ** 2) / (2 * bw * bw)), 0);
  const vis = Math.min(st.c.length, Math.floor(t * 2.2) + 1);
  for (let i = 0; i < vis; i++) {
    ctx.beginPath();
    for (let px = 0; px <= 200; px++) {
      const u = px / 200;
      const y = yb - 52 * Math.exp(-((u - st.c[i]) ** 2) / (2 * bw * bw));
      px ? ctx.lineTo(X(u), y) : ctx.moveTo(X(u), y);
    }
    ctx.strokeStyle = i === st.c.length - 1 ? C.rojo : C.linea; ctx.lineWidth = 1.1; ctx.stroke();
    punto(ctx, X(st.c[i]), yb, 3.4, i === st.c.length - 1 ? C.rojo : C.punto);
  }
  if (vis >= st.c.length) {
    let max = 0; for (let px = 0; px <= 280; px++) max = Math.max(max, dens(px / 280));
    ctx.beginPath();
    for (let px = 0; px <= 280; px++) {
      const u = px / 280, y = yb - (dens(u) / max) * (H - 95);
      px ? ctx.lineTo(X(u), y) : ctx.moveTo(X(u), y);
    }
    ctx.strokeStyle = C.acento; ctx.lineWidth = 2.4; ctx.stroke();
    txt(ctx, "f̂(x) = (1/Nh) Σ K((x−xᵢ)/h)", X(0.07), yb - (dens(0.3) / max) * (H - 95) - 22, C.acento, 12.5, "left", true);
    const u = 0.88;
    txt(ctx, "score = −log f̂", X(u), yb - 60 - 6 * pulso(t), C.rojo, 12, "center", true);
    linea(ctx, X(u), yb - 54, X(u), yb - (dens(u) / max) * (H - 95) - 4, C.rojo, 1.4, .8, [4, 3]);
  }
  txt(ctx, "Cada observación aporta un núcleo de ancho h (regla de Scott). Densidad baja ⇒ anómalo.", 14, 18, C.texto, 12);
};

/* 4 ─ GMM: mezcla de gaussianas */
ANIMS.gmm = (ctx, W, H, t, st) => {
  const comps = [
    { x: 0.30, y: 0.42, rx: 60, ry: 30, rot: 0.4, c: C.punto },
    { x: 0.62, y: 0.64, rx: 48, ry: 26, rot: -0.5, c: C.verde },
    { x: 0.72, y: 0.30, rx: 36, ry: 22, rot: 0.2, c: C.morado },
  ];
  comps.forEach((g, gi) => {
    const cx = 40 + g.x * (W - 80), cy = 24 + g.y * (H - 64);
    for (let k = 1; k <= 2; k++)
      elipse(ctx, cx, cy, g.rx * k * (0.9 + 0.08 * pulso(t + gi)), g.ry * k * (0.9 + 0.08 * pulso(t + gi)), g.rot, g.c, k === 1 ? 2 : 1, k === 1 ? .85 : .4);
    punto(ctx, cx, cy, 4, g.c);
    txt(ctx, "π" + (gi + 1) + " 𝒩(μ" + (gi + 1) + ",Σ" + (gi + 1) + ")", cx, cy - g.ry * 2 - 8, g.c, 11.5, "center");
    const rnd = mulberry32(31 + gi);
    for (let i = 0; i < 22; i++) {
      const [a, b] = gauss2(rnd);
      const ex = a * g.rx * 0.55, ey = b * g.ry * 0.55;
      punto(ctx, cx + ex * Math.cos(g.rot) - ey * Math.sin(g.rot),
                 cy + ex * Math.sin(g.rot) + ey * Math.cos(g.rot), 2.8, g.c, .65);
    }
  });
  const ox = W * 0.16, oy = H * 0.82;
  punto(ctx, ox, oy, 6 + 2 * pulso(t), C.rojo);
  txt(ctx, "log p(x) muy bajo", ox + 12, oy + 4, C.rojo, 12, "left", true);
  txt(ctx, "k elegido por BIC = −2·logL + p·log N (penaliza complejidad).", 14, 18, C.texto, 12);
};

/* 5 ─ KMeans */
ANIMS.kmeans = (ctx, W, H, t, st) => {
  nubeBase(st, 41);
  const fase = (t % 6) / 6;
  const cents = [[0.33 + 0.05 * Math.sin(t * 0.9), 0.42], [0.62, 0.62 + 0.04 * Math.cos(t * 0.8)]];
  st.pts.forEach((p, i) => {
    const d0 = (p[0] - cents[0][0]) ** 2 + (p[1] - cents[0][1]) ** 2;
    const d1 = (p[0] - cents[1][0]) ** 2 + (p[1] - cents[1][1]) ** 2;
    const c = d0 < d1 ? 0 : 1;
    if (fase > 0.3 && i % 3 === 0)
      linea(ctx, PX(W, p), PY(H, p), PX(W, cents[c]), PY(H, cents[c]), C.linea, 0.8, .4);
    punto(ctx, PX(W, p), PY(H, p), 3.2, c ? C.verde : C.punto, .8);
  });
  cents.forEach((c, i) => {
    punto(ctx, PX(W, c), PY(H, c), 7, i ? C.verde : C.punto);
    aro(ctx, PX(W, c), PY(H, c), 11, "#fff", 1.5, .8);
  });
  punto(ctx, PX(W, st.out), PY(H, st.out), 6 + 2 * pulso(t), C.rojo);
  linea(ctx, PX(W, st.out), PY(H, st.out), PX(W, cents[1]), PY(H, cents[1]), C.rojo, 1.6, .8, [5, 4]);
  txt(ctx, "score = min‖x − c‖", PX(W, st.out) - 10, PY(H, st.out) + 22, C.rojo, 12, "right", true);
  txt(ctx, "k por codo-kneedle sobre WCSS(k); lejos de TODO centroide ⇒ anómalo.", 14, 18, C.texto, 12);
};

/* 6 ─ HDBSCAN: densidad jerárquica */
ANIMS.densidad = (ctx, W, H, t, st) => {
  nubeBase(st, 55, 52);
  const umbral = 0.16 - 0.07 * pulso(t, 5);
  for (let i = 0; i < st.pts.length; i++)
    for (let j = i + 1; j < st.pts.length; j++) {
      const d = Math.hypot(st.pts[i][0] - st.pts[j][0], st.pts[i][1] - st.pts[j][1]);
      if (d < umbral)
        linea(ctx, PX(W, st.pts[i]), PY(H, st.pts[i]), PX(W, st.pts[j]), PY(H, st.pts[j]), C.linea, 0.7, .5);
    }
  st.pts.forEach((p, i) => punto(ctx, PX(W, p), PY(H, p), 3.2, i % 2 ? C.punto : C.verde, .85));
  punto(ctx, PX(W, st.out), PY(H, st.out), 5.6 + 1.8 * pulso(t), C.rojo);
  txt(ctx, "ruido (sin clúster)", PX(W, st.out) - 10, PY(H, st.out) - 12, C.rojo, 11.5, "right", true);
  txt(ctx, "λ = 1/dist", W - 30, H - 14, C.acento, 11, "right");
  txt(ctx, "Jerarquía de densidad: al variar λ los clústeres emergen; lo que nunca se une es ruido.", 14, 18, C.texto, 12);
};

/* 7 ─ OPTICS: perfil de alcanzabilidad */
ANIMS.reach = (ctx, W, H, t, st) => {
  if (!st.r) {
    const rnd = mulberry32(61);
    st.r = [];
    for (let i = 0; i < 44; i++) {
      let v = 0.18 + 0.1 * rnd();
      if (i > 13 && i < 19) v = 0.65 + 0.2 * rnd();
      if (i === 33) v = 0.92;
      st.r.push(v);
    }
  }
  const x0 = 50, x1 = W - 30, yb = H - 36;
  const n = st.r.length, vis = Math.min(n, Math.floor(t * 9) + 1);
  for (let i = 0; i < vis; i++) {
    const x = x0 + (i / (n - 1)) * (x1 - x0);
    const h = st.r[i] * (H - 90);
    const alto = st.r[i] > 0.6;
    ctx.fillStyle = alto ? (i === 33 ? C.rojo : C.acento) : C.punto2;
    ctx.globalAlpha = .9;
    ctx.fillRect(x - 3.5, yb - h, 7, h);
    ctx.globalAlpha = 1;
  }
  linea(ctx, x0, yb, x1, yb, C.linea, 1.2);
  txt(ctx, "orden de visita →", x1, yb + 16, C.texto, 11, "right");
  txt(ctx, "distancia de alcanzabilidad", x0, 40, C.acento, 11.5);
  if (vis > 33) txt(ctx, "pico aislado = outlier", x0 + (33 / (n - 1)) * (x1 - x0), yb - st.r[33] * (H - 90) - 10, C.rojo, 11.5, "center", true);
  txt(ctx, "OPTICS ordena los puntos por densidad; los valles son clústeres, los picos anomalías.", 14, 18, C.texto, 12);
};

/* 8 ─ LOF: densidad local relativa */
ANIMS.lof = (ctx, W, H, t, st) => {
  nubeBase(st, 71, 40);
  st.pts.forEach(p => punto(ctx, PX(W, p), PY(H, p), 3.2, C.punto, .8));
  const denso = [0.36, 0.45], ralo = st.out;
  aro(ctx, PX(W, denso), PY(H, denso), 26 + 3 * pulso(t), C.verde, 2, .9);
  punto(ctx, PX(W, denso), PY(H, denso), 4.4, C.verde);
  txt(ctx, "lrd alta", PX(W, denso), PY(H, denso) - 34, C.verde, 11.5, "center", true);
  aro(ctx, PX(W, ralo), PY(H, ralo), 64 + 6 * pulso(t), C.rojo, 2, .9);
  punto(ctx, PX(W, ralo), PY(H, ralo), 5.4, C.rojo);
  txt(ctx, "lrd baja", PX(W, ralo), PY(H, ralo) - 72, C.rojo, 11.5, "center", true);
  txt(ctx, "LOF(x) = media( lrd(vecinos) / lrd(x) ) ≫ 1 ⇒ outlier local", W / 2, H - 16, C.acento, 12.5, "center", true);
  txt(ctx, "Compara la densidad local propia con la de los k vecinos (k = √N).", 14, 18, C.texto, 12);
};

/* 9 ─ Isolation Forest: particiones aleatorias */
ANIMS.iforest = (ctx, W, H, t, st) => {
  nubeBase(st, 81, 44);
  if (!st.cortes) {
    const rnd = mulberry32(99);
    st.cortes = Array.from({ length: 9 }, () => ({
      vert: rnd() > 0.5, pos: 0.15 + 0.7 * rnd(),
      a: rnd() * 0.6, b: 0.4 + rnd() * 0.6,
    }));
  }
  st.pts.forEach(p => punto(ctx, PX(W, p), PY(H, p), 3.0, C.punto, .75));
  punto(ctx, PX(W, st.out), PY(H, st.out), 5.6 + 1.6 * pulso(t), C.rojo);
  const vis = Math.min(st.cortes.length, Math.floor(t * 1.6) + 1);
  for (let i = 0; i < vis; i++) {
    const c = st.cortes[i];
    if (c.vert) linea(ctx, PX(W, [c.pos, 0]), PY(H, [0, c.a]), PX(W, [c.pos, 0]), PY(H, [0, c.b]), C.acento, 1.1, .6);
    else        linea(ctx, PX(W, [c.a, 0]), PY(H, [0, c.pos]), PX(W, [c.b, 0]), PY(H, [0, c.pos]), C.acento, 1.1, .6);
  }
  // caja rápida alrededor del outlier
  if (vis >= 3) {
    const o = st.out;
    ctx.strokeStyle = C.rojo; ctx.lineWidth = 1.8; ctx.globalAlpha = .9;
    ctx.strokeRect(PX(W, [o[0] - 0.07, 0]), PY(H, [0, o[1] - 0.10]),
                   0.14 * (W - 80), 0.20 * (H - 64));
    ctx.globalAlpha = 1;
    txt(ctx, "aislado en ~3 cortes", PX(W, o) - 12, PY(H, o) + 30, C.rojo, 11.5, "right", true);
  }
  txt(ctx, "Cortes aleatorios recursivos: profundidad de aislamiento corta ⇒ score alto. E[h(x)] ~ c(N).", 14, 18, C.texto, 12);
};

/* 10 ─ One-Class SVM */
ANIMS.ocsvm = (ctx, W, H, t, st) => {
  nubeBase(st, 91, 48);
  // frontera suave (blob) alrededor de los datos
  ctx.beginPath();
  for (let a = 0; a <= 64; a++) {
    const ang = (a / 64) * 6.2832;
    const r = 0.30 + 0.07 * Math.sin(3 * ang + t * 0.7) + 0.03 * Math.sin(5 * ang - t);
    const x = PX(W, [0.48 + r * Math.cos(ang) * 1.15, 0]);
    const y = PY(H, [0, 0.52 + r * Math.sin(ang)]);
    a ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(143,193,255,.07)"; ctx.fill();
  ctx.strokeStyle = C.acento; ctx.lineWidth = 2.2; ctx.stroke();
  st.pts.forEach(p => punto(ctx, PX(W, p), PY(H, p), 3.1, C.punto, .8));
  punto(ctx, PX(W, st.out), PY(H, st.out), 5.6 + 1.8 * pulso(t), C.rojo);
  txt(ctx, "f(x) < 0", PX(W, st.out) + 12, PY(H, st.out) + 4, C.rojo, 12.5, "left", true);
  txt(ctx, "f(x) ≥ 0", PX(W, [0.48, 0]), PY(H, [0, 0.52]), C.acento, 12.5, "center", true);
  txt(ctx, "En el RKHS del kernel RBF se separa la masa de datos del origen; ν acota la fracción fuera.", 14, 18, C.texto, 12);
};

/* 11 ─ Autoencoder */
ANIMS.autoencoder = (ctx, W, H, t, st) => {
  const capas = [7, 4, 2, 4, 7];
  const xs = capas.map((_, i) => 70 + (i / (capas.length - 1)) * (W - 150));
  const nodos = capas.map((n, i) => Array.from({ length: n }, (_, j) =>
    [xs[i], H / 2 + (j - (n - 1) / 2) * Math.min(34, (H - 90) / n)]));
  // conexiones con pulso de flujo
  const flujo = (t * 0.9) % 1;
  for (let i = 0; i < capas.length - 1; i++)
    for (const a of nodos[i]) for (const b of nodos[i + 1])
      linea(ctx, a[0], a[1], b[0], b[1], C.linea, 0.6, .35);
  // pulso viajero
  const seg = Math.floor(flujo * (capas.length - 1));
  const fr = flujo * (capas.length - 1) - seg;
  for (let j = 0; j < Math.min(capas[seg], capas[seg + 1]); j++) {
    const a = nodos[seg][j % capas[seg]], b = nodos[seg + 1][j % capas[seg + 1]];
    punto(ctx, a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr, 3, C.acento, .9);
  }
  nodos.forEach((capa, i) => capa.forEach(p =>
    punto(ctx, p[0], p[1], i === 2 ? 7 : 5.4, i === 2 ? C.morado : C.punto, .95)));
  txt(ctx, "x ∈ ℝᶠ", xs[0], H - 18, C.texto, 11.5, "center");
  txt(ctx, "cuello z ∈ ℝ⁸", xs[2], H - 18, C.morado, 11.5, "center", true);
  txt(ctx, "x̂ = D(E(x))", xs[4], H - 18, C.texto, 11.5, "center");
  const err = 0.25 + 0.6 * pulso(t, 3.4);
  ctx.fillStyle = C.rojo; ctx.globalAlpha = .85;
  ctx.fillRect(W - 56, H / 2 + 40 - err * 80, 14, err * 80);
  ctx.globalAlpha = 1;
  txt(ctx, "‖x − x̂‖²", W - 49, H / 2 + 56, C.rojo, 11, "center", true);
  txt(ctx, "Entrenado en lo normal: lo anómalo no cabe por el cuello de botella ⇒ error alto.", 14, 18, C.texto, 12);
};

/* 12 ─ Robust PCA */
ANIMS.rpca = (ctx, W, H, t, st) => {
  if (!st.pp) {
    const rnd = mulberry32(13);
    st.pp = Array.from({ length: 40 }, () => {
      const u = rnd(), [g] = gauss2(rnd);
      return [0.12 + u * 0.72, 0.78 - u * 0.5 + g * 0.035];
    });
    st.o2 = [0.42, 0.20];
  }
  const a = [PX(W, [0.08, 0]), PY(H, [0, 0.83])], b = [PX(W, [0.9, 0]), PY(H, [0, 0.25])];
  linea(ctx, a[0], a[1], b[0], b[1], C.acento, 2.4, .95);
  txt(ctx, "subespacio principal (IRLS-Huber)", b[0] - 6, b[1] - 10, C.acento, 11.5, "right");
  const proy = (p) => {
    const px = PX(W, p), py = PY(H, p);
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const tt = ((px - a[0]) * dx + (py - a[1]) * dy) / (dx * dx + dy * dy);
    return [a[0] + tt * dx, a[1] + tt * dy, px, py];
  };
  st.pp.forEach(p => {
    const [qx, qy, px, py] = proy(p);
    linea(ctx, px, py, qx, qy, C.linea, 0.8, .5);
    punto(ctx, px, py, 3, C.punto, .8);
  });
  const [qx, qy, px, py] = proy(st.o2);
  const f = Math.min(1, (t % 4) / 1.4);
  linea(ctx, px, py, px + (qx - px) * f, py + (qy - py) * f, C.rojo, 2.2, .95, [4, 3]);
  punto(ctx, px, py, 5.6 + 1.6 * pulso(t), C.rojo);
  txt(ctx, "residuo ‖x − P x‖", px + 12, py - 8, C.rojo, 12, "left", true);
  txt(ctx, "Los pesos de Huber quitan influencia a los outliers al ajustar el subespacio.", 14, 18, C.texto, 12);
};

/* 13 ─ Conformal */
ANIMS.conformal = (ctx, W, H, t, st) => {
  if (!st.hist) {
    const rnd = mulberry32(17);
    st.hist = Array.from({ length: 120 }, () => Math.abs(gauss2(rnd)[0]) * 0.32 + 0.08);
  }
  const x0 = 50, x1 = W - 36, yb = H - 40;
  const bins = 22, cnt = new Array(bins).fill(0);
  st.hist.forEach(v => cnt[Math.min(bins - 1, Math.floor(v * bins))]++);
  const cmax = Math.max(...cnt);
  const visBins = Math.min(bins, Math.floor(t * 5) + 1);
  for (let i = 0; i < visBins; i++) {
    const x = x0 + (i / bins) * (x1 - x0);
    const h = (cnt[i] / cmax) * (H - 105);
    ctx.fillStyle = C.punto2; ctx.globalAlpha = .8;
    ctx.fillRect(x, yb - h, (x1 - x0) / bins - 2, h);
    ctx.globalAlpha = 1;
  }
  txt(ctx, "α de calibración (dist. k-ésimo vecino)", x0, 40, C.texto, 11.5);
  const am = 0.55 + 0.28 * pulso(t, 5);
  const xa = x0 + am * (x1 - x0);
  linea(ctx, xa, yb, xa, 52, C.rojo, 2, .9);
  punto(ctx, xa, 52, 5, C.rojo);
  const p = st.hist.filter(v => v >= am).length / (st.hist.length + 1);
  txt(ctx, `α(x) → p-valor = ${(p).toFixed(3)}`, xa + 8, 64, C.rojo, 12.5, "left", true);
  txt(ctx, "p = (1 + #{αᵢ ≥ α}) / (n+1): válido sin supuestos distribucionales (intercambiabilidad).", 14, 18, C.texto, 12);
};

/* 14 ─ SigKernel: dos caminos y su producto interno */
ANIMS.sigkernel = (ctx, W, H, t, st) => {
  const f = Math.min(1, (t % 7) / 3.2);
  const dib = (cx, cy, esc, fase, color, prog) => {
    ctx.beginPath();
    const n = Math.max(2, Math.floor(prog * 90));
    for (let i = 0; i < n; i++) {
      const u = i / 89;
      const x = cx + esc * u * 1.6;
      const y = cy - esc * (0.5 * Math.sin(6.28 * u * 2 + fase) * (1 - u * 0.5) + u * 0.4);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2.4; ctx.stroke();
  };
  dib(60, H * 0.52, 100, 0, C.punto, f);
  dib(60, H * 0.85, 100, 0.9, C.verde, f);
  txt(ctx, "X", 56, H * 0.32, C.punto, 13, "center", true);
  txt(ctx, "Y", 56, H * 0.66, C.verde, 13, "center", true);
  // vectores de signatura como barras
  const bx = W * 0.48, bw = 9, n = 12;
  const rnd = mulberry32(3);
  if (!st.s1) { st.s1 = Array.from({ length: n }, () => rnd() * 2 - 1); st.s2 = st.s1.map(v => v * 0.85 + (rnd() - 0.5) * 0.5); }
  for (let i = 0; i < Math.floor(f * n); i++) {
    ctx.fillStyle = C.punto; ctx.globalAlpha = .85;
    ctx.fillRect(bx + i * (bw + 3), H * 0.42 - st.s1[i] * 32, bw, Math.abs(st.s1[i]) * 32);
    ctx.fillStyle = C.verde;
    ctx.fillRect(bx + i * (bw + 3), H * 0.80 - st.s2[i] * 32, bw, Math.abs(st.s2[i]) * 32);
    ctx.globalAlpha = 1;
  }
  txt(ctx, "S(X)", bx - 8, H * 0.40, C.punto, 11.5, "right", true);
  txt(ctx, "S(Y)", bx - 8, H * 0.78, C.verde, 11.5, "right", true);
  // medidor de coseno
  if (f >= 1) {
    const cos = 0.86, gx = W - 86, gy = H * 0.52, R = 44;
    ctx.beginPath(); ctx.arc(gx, gy, R, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = C.linea; ctx.lineWidth = 7; ctx.stroke();
    ctx.beginPath(); ctx.arc(gx, gy, R, Math.PI, Math.PI * (1 + cos * pulso(t, 8) * 0.2 + cos * 0.8));
    ctx.strokeStyle = C.acento; ctx.lineWidth = 7; ctx.stroke();
    txt(ctx, "K(X,Y)", gx, gy + 18, C.acento, 13, "center", true);
    txt(ctx, "⟨S(X),S(Y)⟩ / ‖S(X)‖‖S(Y)‖", gx, gy + 34, C.texto, 10.5, "center");
  }
  txt(ctx, "El kernel lineal entre signaturas normalizadas compara la GEOMETRÍA de los caminos; alimenta un One-Class SVM.", 14, 18, C.texto, 12);
};

/* 15 ─ SigMaHaKNN / espacio de signaturas */
ANIMS.sigmaha = (ctx, W, H, t, st) => {
  nubeBase(st, 23, 50);
  txt(ctx, "espacio de signaturas Φ = S(X) ∈ ℝᴰ", W / 2, 38, C.morado, 12.5, "center", true);
  st.pts.forEach(p => punto(ctx, PX(W, p), PY(H, p), 3.1, C.morado, .7));
  // vecindario local del query
  const q = st.out;
  const vecinos = st.pts.filter(p => Math.hypot(p[0] - q[0], p[1] - q[1]) < 0.33)
    .sort((a, b) => Math.hypot(a[0]-q[0],a[1]-q[1]) - Math.hypot(b[0]-q[0],b[1]-q[1])).slice(0, 8);
  let mx = 0, my = 0;
  vecinos.forEach(p => { mx += p[0] / vecinos.length; my += p[1] / vecinos.length; });
  const f = Math.min(1, (t % 6) / 2);
  vecinos.forEach((p, i) => {
    if (i / vecinos.length <= f)
      linea(ctx, PX(W, q), PY(H, q), PX(W, p), PY(H, p), C.linea, 1, .6, [3, 3]);
  });
  if (f > 0.6) {
    elipse(ctx, PX(W, [mx, 0]), PY(H, [0, my]), 52, 26, 0.5, C.verde, 2, .9);
    punto(ctx, PX(W, [mx, 0]), PY(H, [0, my]), 4, C.verde);
    txt(ctx, "μ_loc, Σ_loc (k vecinos)", PX(W, [mx, 0]), PY(H, [0, my]) + 42, C.verde, 11.5, "center");
  }
  punto(ctx, PX(W, q), PY(H, q), 6 + 1.8 * pulso(t), C.rojo);
  if (f >= 1) {
    linea(ctx, PX(W, q), PY(H, q), PX(W, [mx, 0]), PY(H, [0, my]), C.rojo, 2, .9);
    txt(ctx, "s = √((φ−μ)ᵀ(Σ+λI)⁻¹(φ−μ))", PX(W, q) - 10, PY(H, q) - 14, C.rojo, 12, "right", true);
  }
  txt(ctx, "Mahalanobis LOCAL: cada signatura se compara con la geometría de su propio vecindario.", 14, 18, C.texto, 12);
};

/* 16 ─ Área de Lévy (log-signatura) */
ANIMS.levy = (ctx, W, H, t, st) => {
  const cx = W * 0.40, cy = H * 0.56, esc = Math.min(W * 0.3, H * 0.36);
  const per = 6, f = Math.min(1, (t % per) / 4.2);
  const N = 140, pts = [];
  for (let i = 0; i <= Math.floor(f * (N - 1)); i++) {
    const u = i / (N - 1);
    const ang = u * 4.6 - 0.6;
    const r = 0.25 + 0.75 * u;
    pts.push([cx + esc * r * Math.cos(ang) * 0.9, cy - esc * r * Math.sin(ang) * 0.65]);
  }
  if (pts.length > 2) {
    // área barrida respecto a la cuerda
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.forEach(p => ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = "rgba(255,209,102,.16)"; ctx.fill();
  }
  ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.strokeStyle = C.punto; ctx.lineWidth = 2.6; ctx.stroke();
  if (pts.length > 1) {
    const a = pts[0], b = pts[pts.length - 1];
    linea(ctx, a[0], a[1], b[0], b[1], C.acento, 1.8, .9, [6, 4]);
    punto(ctx, a[0], a[1], 4.5, C.verde); punto(ctx, b[0], b[1], 4.5, C.rojo);
    txt(ctx, "cuerda (nivel 1: ΔX)", (a[0] + b[0]) / 2 + 10, (a[1] + b[1]) / 2 + 16, C.acento, 11.5);
    // área acumulada aproximada
    let area = 0;
    for (let i = 1; i < pts.length; i++)
      area += 0.5 * ((pts[i - 1][0] - a[0]) * (pts[i][1] - a[1]) - (pts[i][0] - a[0]) * (pts[i - 1][1] - a[1]));
    txt(ctx, "A = " + Math.abs(area / (esc * esc)).toFixed(3), cx + esc * 0.55, cy - esc * 0.45, C.acento, 15, "center", true);
  }
  txt(ctx, "A^{ij} = ½(S^{ij} − S^{ji}): área firmada entre el camino y su cuerda", W / 2, H - 16, C.morado, 12.5, "center", true);
  txt(ctx, "El nivel 2 've' el ORDEN de los movimientos: misma cuerda, distinta área ⇒ distinta firma.", 14, 18, C.texto, 12);
};

/* ═══════════════════════════════════════════════════════════════════════════
   ANIMACIONES — ECUACIONES DIFERENCIALES NEURONALES
   ═══════════════════════════════════════════════════════════════════════════ */

/* RNN: recurrencia desplegada */
ANIMS.nde_rnn = (ctx, W, H, t, st) => {
  const n = 6, y = H * 0.55, dx = (W - 140) / (n - 1);
  const fase = (t * 0.8) % n;
  for (let i = 0; i < n; i++) {
    const x = 70 + i * dx;
    // entrada x_k
    const act = fase >= i && fase < i + 1;
    punto(ctx, x, H * 0.86, act ? 6 : 4.4, C.verde, act ? 1 : .7);
    linea(ctx, x, H * 0.82, x, y + 22, C.verde, act ? 2.2 : 1.2, act ? .95 : .45);
    txt(ctx, "x" + (i + 1), x, H * 0.86 + 18, C.verde, 11, "center");
    // celda h_k
    ctx.fillStyle = act ? "rgba(143,193,255,.30)" : "rgba(143,193,255,.12)";
    ctx.strokeStyle = act ? C.acento : C.linea; ctx.lineWidth = act ? 2.2 : 1.2;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x - 24, y - 20, 48, 40, 9); else ctx.rect(x - 24, y - 20, 48, 40);
    ctx.fill(); ctx.stroke();
    txt(ctx, "h" + (i + 1), x, y + 5, "#fff", 13, "center", true);
    if (i < n - 1) {
      const fr = Math.max(0, Math.min(1, fase - i));
      linea(ctx, x + 24, y, x + 24 + (dx - 48) * fr, y, C.acento, 2.4, .9);
      if (fr > 0 && fr < 1) punto(ctx, x + 24 + (dx - 48) * fr, y, 4, C.acento);
      linea(ctx, x + 24, y, x + dx - 24, y, C.linea, 1.1, .4);
    }
  }
  txt(ctx, "h_{k+1} = φ(W_h h_k + W_x x_{k+1} + b)   — paso DISCRETO, ciego al tiempo entre muestras", W / 2, 26, C.texto, 12.5, "center");
  txt(ctx, "→ salida y = W_o h_n", W - 60, H * 0.55 + 5, C.morado, 12, "right", true);
};

/* Neural ODE: campo vectorial con partículas fluyendo */
ANIMS.nde_ode = (ctx, W, H, t, st) => {
  const cx = W / 2, cy = H * 0.55, esc = Math.min(W / 7.2, H / 3.4);
  const f = (p) => {  // espiral estable hacia ciclo
    const x = p[0], y = p[1];
    const r2 = x * x + y * y;
    return [y + x * (1 - r2) * 0.6, -x + y * (1 - r2) * 0.6];
  };
  // rejilla de flechas
  for (let i = -3; i <= 3; i++) for (let j = -2; j <= 2; j++) {
    const p = [i * 0.55, j * 0.62];
    const v = f(p);
    const nv = Math.hypot(v[0], v[1]) + 1e-9;
    const x = cx + p[0] * esc, y = cy - p[1] * esc;
    const L = 13;
    linea(ctx, x, y, x + (v[0] / nv) * L, y - (v[1] / nv) * L, C.linea, 1.3, .7);
    punto(ctx, x + (v[0] / nv) * L, y - (v[1] / nv) * L, 1.6, C.punto2, .8);
  }
  // partículas integrando dh/dt = f(h)
  if (!st.parts) {
    const rnd = mulberry32(2);
    st.parts = Array.from({ length: 26 }, () => ({ p: [(rnd() - 0.5) * 3.4, (rnd() - 0.5) * 2.6], tr: [] }));
  }
  st.parts.forEach(pt => {
    for (let s = 0; s < 2; s++) {
      const v = f(pt.p);
      pt.p = [pt.p[0] + v[0] * 0.012, pt.p[1] + v[1] * 0.012];
    }
    pt.tr.push([...pt.p]); if (pt.tr.length > 36) pt.tr.shift();
    ctx.beginPath();
    pt.tr.forEach((q, i) => {
      const x = cx + q[0] * esc, y = cy - q[1] * esc;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = "rgba(255,209,102,.55)"; ctx.lineWidth = 1.6; ctx.stroke();
    punto(ctx, cx + pt.p[0] * esc, cy - pt.p[1] * esc, 2.8, C.acento, .95);
  });
  txt(ctx, "dh/dt = f_θ(h) — la 'profundidad' es tiempo continuo; el estado fluye por el campo aprendido", W / 2, 26, C.texto, 12.5, "center");
  txt(ctx, "h(0) fija TODA la trayectoria (Picard–Lindelöf): los datos posteriores no entran", W / 2, H - 14, C.naranja || "#ffb74d", 12, "center", true);
};

/* Neural CDE: el camino de datos controla la dinámica */
ANIMS.nde_cde = (ctx, W, H, t, st) => {
  const per = 9, f = ((t % per) / per);
  const n = 160;
  const X = (u) => 0.35 * Math.sin(6.28 * u * 1.4) + 0.18 * Math.sin(6.28 * u * 3.1 + 1);
  // panel superior: camino de datos X(t)
  const y1 = H * 0.30, esc1 = H * 0.14;
  ctx.beginPath();
  for (let i = 0; i <= Math.floor(f * (n - 1)); i++) {
    const u = i / (n - 1);
    const x = 60 + u * (W - 120), y = y1 - X(u) * esc1 / 0.4;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = C.verde; ctx.lineWidth = 2.4; ctx.stroke();
  txt(ctx, "control X(t) = interpolación de las observaciones (t, x)", 60, y1 - esc1 / 0.4 - 12, C.verde, 11.5);
  // observaciones marcadas
  for (let k = 0; k <= 8; k++) {
    const u = k / 8;
    if (u <= f) punto(ctx, 60 + u * (W - 120), y1 - X(u) * esc1 / 0.4, 4, C.verde);
  }
  // panel inferior: estado z(t) respondiendo
  const y2 = H * 0.72, esc2 = H * 0.13;
  ctx.beginPath();
  let z = 0, zPrev = 0;
  for (let i = 0; i <= Math.floor(f * (n - 1)); i++) {
    const u = i / (n - 1);
    const dX = X(u) - X(Math.max(0, u - 1 / (n - 1)));
    z = z + Math.tanh(1.6 * z + 1.0) * dX * 3.2;   // dz = f(z)·dX
    const x = 60 + u * (W - 120), y = y2 - z * esc2;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    zPrev = z;
  }
  ctx.strokeStyle = C.morado; ctx.lineWidth = 2.6; ctx.stroke();
  txt(ctx, "estado z(t):  dz = f_θ(z) dX", 60, y2 + esc2 + 22, C.morado, 11.5);
  // línea de acople vertical
  const xu = 60 + f * (W - 120);
  linea(ctx, xu, y1 - X(f) * esc1 / 0.4, xu, y2 - zPrev * esc2, C.acento, 1.4, .8, [4, 4]);
  punto(ctx, xu, y1 - X(f) * esc1 / 0.4, 4.4, C.acento);
  punto(ctx, xu, y2 - zPrev * esc2, 4.4, C.acento);
  txt(ctx, "dX", xu + 8, (y1 + y2) / 2, C.acento, 12, "left", true);
  txt(ctx, "El dato deja de ser condición inicial y pasa a ser CONTROL: cada incremento dX tuerce la dinámica (integral de Riemann–Stieltjes).", W / 2, 22, C.texto, 12, "center");
};

/* Neural RDE: ventanas + log-signatura */
ANIMS.nde_rde = (ctx, W, H, t, st) => {
  const per = 10, f = (t % per) / per;
  const n = 200, x0 = 60, x1 = W - 60, ymid = H * 0.34, esc = H * 0.15;
  const Xf = (u) => 0.5 * Math.sin(6.28 * u * 2.2) * Math.exp(-u * 0.6) + 0.3 * Math.sin(6.28 * u * 5.1);
  // camino fino (rugoso)
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const x = x0 + u * (x1 - x0), y = ymid - Xf(u) * esc / 0.5;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = "rgba(143,193,255,.65)"; ctx.lineWidth = 1.6; ctx.stroke();
  // ventanas
  const Wn = 5;
  const winF = Math.min(Wn, Math.floor(f * (Wn + 2)));
  for (let w = 0; w < Wn; w++) {
    const ua = w / Wn, ub = (w + 1) / Wn;
    const xa = x0 + ua * (x1 - x0), xb = x0 + ub * (x1 - x0);
    if (w < winF) {
      ctx.fillStyle = w % 2 ? "rgba(199,146,234,.10)" : "rgba(255,209,102,.10)";
      ctx.fillRect(xa, ymid - esc / 0.5 - 14, xb - xa, esc / 0.5 * 2 + 18);
      // glifo logsig: flecha (incremento) + disco (área)
      const dx = (Xf(ub) - Xf(ua)) * esc / 0.5;
      const gx = (xa + xb) / 2, gy = H * 0.62;
      linea(ctx, gx - 16, gy + dx / 2, gx + 16, gy - dx / 2, C.acento, 2.6, .95);
      punto(ctx, gx + 16, gy - dx / 2, 3, C.acento);
      aro(ctx, gx, gy + 26, 7 + 5 * Math.abs(Math.sin(w * 2.3 + 1)), C.morado, 2.2, .9);
      txt(ctx, "logsig²", gx, gy + 52, C.texto, 10, "center");
      // paso del estado z
      const zy = H * 0.87;
      punto(ctx, gx, zy, 6, C.verde, .95);
      if (w > 0) {
        const gxp = (x0 + (w - 0.5) / Wn * (x1 - x0));
        linea(ctx, gxp + 6, zy, gx - 6, zy, C.verde, 2, .8);
      }
      txt(ctx, "z" + w, gx, zy + 18, C.verde, 10.5, "center");
    } else {
      linea(ctx, xa, ymid - esc / 0.5 - 14, xa, ymid + esc / 0.5 + 4, C.linea, 1, .4, [3, 4]);
    }
  }
  txt(ctx, "Método log-ODE: la geometría de cada ventana se comprime en su log-signatura (ΔX + áreas de Lévy)", W / 2, 22, C.texto, 12, "center");
  txt(ctx, "z_{j+1} = z_j + g_θ(z_j) · logsig_j   —  n/s pasos en lugar de n", W / 2, H - 12, C.verde, 12, "center", true);
};

/* ── Desarrollo de la signatura (fundamentos): camino + niveles en vivo ──── */
ANIMS.firma_dev = (ctx, W, H, t, st) => {
  const serie = st.serie || [];
  if (!serie.length) return;
  const per = 12, f = (t % per) / (per * 0.8);
  const prog = Math.min(1, f);
  const n = serie.length;
  const k = Math.max(2, Math.floor(prog * n));
  const sub = serie.slice(0, k);
  let lo = Math.min(...serie), hi = Math.max(...serie);
  const x0 = 56, x1 = W * 0.56, yb = H - 40, yt = 44;
  const X = i => x0 + (i / (n - 1)) * (x1 - x0);
  const Y = v => yb - ((v - lo) / (hi - lo || 1)) * (yb - yt);
  // área bajo la curva (S^(t,x))
  ctx.beginPath(); ctx.moveTo(X(0), yb);
  sub.forEach((v, i) => ctx.lineTo(X(i), Y(v)));
  ctx.lineTo(X(k - 1), yb); ctx.closePath();
  ctx.fillStyle = "rgba(255,209,102,.13)"; ctx.fill();
  // camino
  ctx.beginPath();
  sub.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
  ctx.strokeStyle = C.punto; ctx.lineWidth = 2.2; ctx.stroke();
  // cuerda nivel 1
  linea(ctx, X(0), Y(serie[0]), X(k - 1), Y(sub[k - 1]), C.acento, 1.8, .9, [6, 4]);
  punto(ctx, X(k - 1), Y(sub[k - 1]), 5, C.rojo);
  txt(ctx, "X_t  (camino tiempo–valor)", x0, 30, C.texto, 12);
  // niveles en vivo
  const path = Sig.caminoDeSerie(sub);
  const sig = Sig.calcular(path, 4);
  const normas = Sig.normasPorNivel(sig);
  const bx = W * 0.64, bw = (W - bx - 50) / 4;
  const maxN = Math.max(...normas, 1e-9);
  const colores = [C.verde, C.acento, C.morado, C.rojo];
  normas.forEach((nv, i) => {
    const h = (nv / maxN) * (H - 130);
    const x = bx + i * bw;
    ctx.fillStyle = colores[i]; ctx.globalAlpha = .85;
    ctx.fillRect(x, yb - h, bw - 14, h);
    ctx.globalAlpha = 1;
    txt(ctx, "‖S⁽" + (i + 1) + "⁾‖", x + (bw - 14) / 2, yb + 18, colores[i], 12, "center", true);
    txt(ctx, nv.toFixed(2), x + (bw - 14) / 2, yb - h - 8, colores[i], 11.5, "center");
  });
  txt(ctx, "normas por nivel (en vivo)", bx, 30, C.texto, 12);
  txt(ctx, "S(t)=1 ✓   S^(t,x) ∝ área   niveles ↑ = geometría fina", bx, H - 8, "rgba(207,226,255,.7)", 10.5);
};

window.Animador = Animador;
window.ANIMS = ANIMS;
