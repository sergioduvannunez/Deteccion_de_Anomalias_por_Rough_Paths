/* ═══════════════════════════════════════════════════════════════════════════
   signatures.js — Rough Path Signatures en JavaScript (cálculo en vivo)

   Réplica exacta de backend/signaturas.py:
   identidad de Chen nivel a nivel con productos tensoriales aplanados.
   Soporta caminos multivariados en R^d (d arbitrario) hasta nivel 4.

   Verificación: con aumento temporal, S(t) = 1.000000 exacto.
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

const Sig = (() => {

  // producto tensorial aplanado: (p) ⊗ (q) → (p·q)
  function kron(a, b) {
    const out = new Float64Array(a.length * b.length);
    let k = 0;
    for (let i = 0; i < a.length; i++) {
      const ai = a[i];
      for (let j = 0; j < b.length; j++) out[k++] = ai * b[j];
    }
    return out;
  }

  function suma(dst, src, esc = 1.0) {
    for (let i = 0; i < dst.length; i++) dst[i] += esc * src[i];
  }

  /**
   * Signatura truncada de un camino lineal a trozos.
   * @param {Array<Array<number>>} path - n puntos en R^d
   * @param {number} depth - nivel de truncación (1..4)
   * @returns {{niveles: Float64Array[], d: number, depth: number}}
   */
  function calcular(path, depth) {
    const n = path.length, d = path[0].length;
    const S = [new Float64Array(d)];
    for (let k = 2; k <= depth; k++) S.push(new Float64Array(Math.pow(d, k)));

    for (let t = 0; t < n - 1; t++) {
      const e1 = new Float64Array(d);
      for (let j = 0; j < d; j++) e1[j] = path[t + 1][j] - path[t][j];
      const e2 = depth >= 2 ? kron(e1, e1).map(v => v / 2) : null;
      const e3 = depth >= 3 ? kron(e2, e1).map(v => v / 3) : null;
      const e4 = depth >= 4 ? kron(e3, e1).map(v => v / 4) : null;

      if (depth >= 4) {
        suma(S[3], kron(S[2], e1)); suma(S[3], kron(S[1], e2));
        suma(S[3], kron(S[0], e3)); suma(S[3], e4);
      }
      if (depth >= 3) {
        suma(S[2], kron(S[1], e1)); suma(S[2], kron(S[0], e2)); suma(S[2], e3);
      }
      if (depth >= 2) {
        suma(S[1], kron(S[0], e1)); suma(S[1], e2);
      }
      suma(S[0], e1);
    }
    return { niveles: S, d, depth };
  }

  /** Serie univariada → camino (t, x) con aumento temporal y normalización. */
  function caminoDeSerie(y, t = null) {
    const n = y.length;
    let lo = Infinity, hi = -Infinity;
    for (const v of y) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const rango = hi > lo ? hi - lo : 1;
    const path = new Array(n);
    if (t) {
      const t0 = t[0], t1 = t[t.length - 1], rt = (t1 - t0) || 1;
      for (let i = 0; i < n; i++) path[i] = [(t[i] - t0) / rt, (y[i] - lo) / rango];
    } else {
      for (let i = 0; i < n; i++) path[i] = [i / Math.max(n - 1, 1), (y[i] - lo) / rango];
    }
    return path;
  }

  /** Canales multivariados {nombre: array} → camino (t, c1, c2, ...). */
  function caminoMultivariado(canales, t = null) {
    const noms = Object.keys(canales);
    const n = canales[noms[0]].length;
    const normas = noms.map(nm => {
      let lo = Infinity, hi = -Infinity;
      for (const v of canales[nm]) { if (v < lo) lo = v; if (v > hi) hi = v; }
      return [lo, hi > lo ? hi - lo : 1];
    });
    const path = new Array(n);
    for (let i = 0; i < n; i++) {
      const p = [t ? (t[i] - t[0]) / ((t[t.length - 1] - t[0]) || 1) : i / Math.max(n - 1, 1)];
      noms.forEach((nm, j) => p.push((canales[nm][i] - normas[j][0]) / normas[j][1]));
      path[i] = p;
    }
    return { path, nombres: ["t", ...noms] };
  }

  /** Palabras (multi-índices) del nivel k en orden canónico. */
  function palabras(d, k) {
    const out = [];
    const rec = (w) => {
      if (w.length === k) { out.push(w.slice()); return; }
      for (let i = 0; i < d; i++) { w.push(i); rec(w); w.pop(); }
    };
    rec([]);
    return out;
  }

  /** Etiquetas S(a,b,...) usando nombres de canal. */
  function etiquetas(d, depth, nombres) {
    const noms = nombres || Array.from({ length: d }, (_, i) => "x" + (i + 1));
    const out = [];
    for (let k = 1; k <= depth; k++)
      for (const w of palabras(d, k))
        out.push("S(" + w.map(i => noms[i]).join(",") + ")");
    return out;
  }

  /** Áreas de Lévy A^{ij} = (S^{ij} − S^{ji})/2 a partir del nivel 2. */
  function levy(sig) {
    const d = sig.d, S2 = sig.niveles[1];
    const A = [];
    for (let i = 0; i < d; i++) {
      A.push(new Float64Array(d));
      for (let j = 0; j < d; j++)
        A[i][j] = 0.5 * (S2[i * d + j] - S2[j * d + i]);
    }
    return A;
  }

  /** Norma euclídea por nivel. */
  function normasPorNivel(sig) {
    return sig.niveles.map(lv => {
      let s = 0; for (const v of lv) s += v * v; return Math.sqrt(s);
    });
  }

  /** Distancia L2 entre dos signaturas (mismos d/depth). */
  function distancia(a, b) {
    let s = 0;
    for (let k = 0; k < a.niveles.length; k++)
      for (let i = 0; i < a.niveles[k].length; i++) {
        const dd = a.niveles[k][i] - b.niveles[k][i]; s += dd * dd;
      }
    return Math.sqrt(s);
  }

  return { calcular, caminoDeSerie, caminoMultivariado, palabras, etiquetas,
           levy, normasPorNivel, distancia, kron };
})();
