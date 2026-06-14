/* ═══════════════════════════════════════════════════════════════════════════
   teoria.js — contenido matemático riguroso

   * TEORIA_DETECTORES : construcción formal de cada algoritmo de detección
     (definición, score, hiperparámetros automáticos y propiedades) + clave
     de su animación explicativa.
   * NDE_ETAPAS : la progresión RNN → Neural ODE → Neural CDE → Neural RDE
     con la construcción matemática completa de cada salto.
   * NIVELES_SIG : interpretación geométrica de cada nivel de la signatura.

   Las fórmulas usan KaTeX (delimitadores \( \) y $$ $$).
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";
const R = String.raw;

/* ── plantillas por tipo de algoritmo ───────────────────────────────────── */
const _KINDS = {

  zmad: {
    anim: "zscore", vista: "aug (forma + magnitud)",
    resumen: "Estandarización robusta por mediana y MAD; alerta por la desviación máxima entre features.",
    bloques: [
      { t: "Sea \\(x \\in \\mathbb{R}^F\\) el vector de features. La media y la desviación estándar son sensibles a outliers (punto de ruptura 0); se sustituyen por estimadores con punto de ruptura 1/2:",
        f: R`\mathrm{med}_j=\operatorname{mediana}_i(x_{ij}),\qquad \mathrm{MAD}_j=\operatorname{mediana}_i\,|x_{ij}-\mathrm{med}_j|` },
      { t: "El z-score robusto por feature y el score final (norma del máximo) son:",
        f: R`z_{ij}=\frac{|x_{ij}-\mathrm{med}_j|}{\mathrm{MAD}_j/0.6745},\qquad s(x_i)=\max_j z_{ij}` },
      { t: "La constante 0.6745 calibra el MAD para que sea consistente con σ bajo normalidad: \\(\\mathbb{E}[\\mathrm{MAD}] = 0.6745\\,\\sigma\\). Detecta anomalías univariadas extremas en cualquier coordenada, pero ignora la estructura de correlación." },
    ],
    props: { "Tipo": "Estadístico robusto", "Supuestos": "Unimodalidad por feature", "Complejidad": "O(NF)", "HP automático": "umbral implícito vía τ = p90" },
  },

  t2: {
    anim: "ellipse", vista: "pca (subespacio k=10)",
    resumen: "Estadístico T² de Hotelling: distancia de Mahalanobis al centroide en el subespacio PCA.",
    bloques: [
      { t: "Sobre los scores PCA \\(z\\in\\mathbb{R}^k\\) se estima \\(\\mu\\) y \\(\\Sigma\\), y la distancia de Mahalanobis define elipsoides de equiprobabilidad:",
        f: R`T^2(z)=(z-\mu)^\top \Sigma^{-1} (z-\mu)` },
      { t: "Bajo normalidad multivariada, \\(\\tfrac{N-k}{k(N-1)}\\,T^2 \\sim F_{k,\\,N-k}\\), lo que da el control de calidad clásico (Hotelling, 1931). La métrica \\(\\Sigma^{-1}\\) blanquea el espacio: una desviación a lo largo de una dirección de alta varianza cuenta menos que la misma desviación en una dirección de baja varianza." },
      { t: "Limitación estructural: es global y unimodal — con clústeres múltiples el centroide cae en medio y la elipse se infla. (Compárese con el Mahalanobis LOCAL de SigMaHaKNN.)" },
    ],
    props: { "Tipo": "Estadístico paramétrico", "Supuestos": "Gaussianidad aprox.", "Complejidad": "O(Nk²+k³)", "HP automático": "k PCA = 10" },
  },

  kde: {
    anim: "kde", vista: "pca",
    resumen: "Verosimilitud negativa bajo un estimador de densidad por núcleos gaussianos.",
    bloques: [
      { t: "El estimador de Parzen–Rosenblatt con núcleo gaussiano y ancho de banda \\(h\\):",
        f: R`\hat f_h(x)=\frac{1}{N h^d}\sum_{i=1}^N K\!\Big(\frac{x-x_i}{h}\Big),\qquad s(x) = -\log \hat f_h(x)` },
      { t: "El ancho óptimo minimiza el error cuadrático integrado medio asintótico (AMISE). Para datos aproximadamente gaussianos en dimensión \\(d\\), la regla de Scott da:",
        f: R`h^\* = N^{-1/(d+4)}` },
      { t: "Es el balance sesgo–varianza canónico: \\(h\\) grande sobre-suaviza (sesgo), \\(h\\) pequeño produce densidades espinosas (varianza). KDE es no paramétrico y multimodal, pero sufre la maldición de la dimensión — por eso se aplica en el subespacio PCA." },
    ],
    props: { "Tipo": "Densidad no paramétrica", "Supuestos": "Suavidad de f", "Complejidad": "O(N² d)", "HP automático": "h por regla de Scott" },
  },

  gmm: {
    anim: "gmm", vista: "pca",
    resumen: "Verosimilitud negativa bajo mezcla finita de gaussianas ajustada por EM; k elegido por BIC.",
    bloques: [
      { t: "Modelo generativo de mezcla con pesos \\(\\pi_m\\):",
        f: R`p(x)=\sum_{m=1}^{k}\pi_m\,\mathcal N(x\mid \mu_m,\Sigma_m),\qquad s(x)=-\log p(x)` },
      { t: "Los parámetros se estiman por Expectation–Maximization, que asciende monótonamente la verosimilitud. El número de componentes se selecciona minimizando el criterio de información bayesiano:",
        f: R`\mathrm{BIC}(k) = -2\log \mathcal{L}_k + p_k \log N` },
      { t: "El BIC es una aproximación de Laplace a la evidencia marginal: penaliza los \\(p_k\\) parámetros con \\(\\log N\\), favoreciendo el modelo más simple compatible con los datos (consistente para mezclas identificables). Captura multimodalidad que T² no puede." },
    ],
    props: { "Tipo": "Mezcla paramétrica (EM)", "Supuestos": "Componentes gaussianas", "Complejidad": "O(TNk d²)", "HP automático": "k = argmin BIC, k∈[2,10]" },
  },

  kmeans: {
    anim: "kmeans", vista: "pca",
    resumen: "Distancia al centroide más cercano; k elegido por el codo de la curva WCSS (kneedle).",
    bloques: [
      { t: "k-means minimiza la suma de cuadrados intra-clúster (WCSS) mediante el algoritmo de Lloyd:",
        f: R`\min_{C_1..C_k}\;\sum_{m=1}^{k}\sum_{x\in C_m}\lVert x-c_m\rVert^2,\qquad s(x)=\min_m \lVert x-c_m\rVert` },
      { t: "WCSS(k) decrece monótonamente: añadir clústeres siempre ayuda. El 'codo' es el punto de retornos decrecientes; el método kneedle lo formaliza como el k que maximiza la distancia perpendicular entre la curva normalizada y la cuerda que une sus extremos:",
        f: R`k^\* = \arg\max_k\; \frac{|y(k) - (1-x(k))|}{\sqrt 2}` },
      { t: "Score geométrico simple y rápido; asume clústeres convexos e isótropos (Voronoi). Anomalía = lejos de todo prototipo." },
    ],
    props: { "Tipo": "Clustering particional", "Supuestos": "Clústeres esféricos", "Complejidad": "O(TNkd)", "HP automático": "k por codo-kneedle" },
  },

  hdbscan: {
    anim: "densidad", vista: "pca",
    resumen: "Clustering jerárquico por densidad; el score es la distancia media a los k vecinos del núcleo no-ruido.",
    bloques: [
      { t: "HDBSCAN define la distancia de alcanzabilidad mutua, que infla las distancias en zonas ralas:",
        f: R`d_{\mathrm{mreach}}(a,b)=\max\{\mathrm{core}_k(a),\,\mathrm{core}_k(b),\,d(a,b)\}` },
      { t: "Sobre el árbol de expansión mínima de esa métrica se construye la jerarquía de clústeres al variar el umbral de densidad \\(\\lambda = 1/d\\); los clústeres que maximizan la estabilidad \\(\\sum_{x}(\\lambda_x - \\lambda_{\\text{nac}})\\) sobreviven; lo demás es ruido." },
      { t: "Para puntuar puntos nuevos usamos la distancia media a los 5 vecinos más próximos del NÚCLEO (puntos no-ruido del entrenamiento): un punto lejos de toda región densa recibe score alto. min_cluster_size ≈ 2% de N evita la fragmentación en datasets pequeños." },
    ],
    props: { "Tipo": "Densidad jerárquica", "Supuestos": "Ninguno sobre forma", "Complejidad": "O(N log N)", "HP automático": "mcs = max(5, N/50)" },
  },

  lof: {
    anim: "lof", vista: "aug",
    resumen: "Local Outlier Factor: cociente entre la densidad local de los vecinos y la propia.",
    bloques: [
      { t: "Con la distancia de alcanzabilidad \\(\\mathrm{rd}_k(a,b)=\\max\\{d_k(b), d(a,b)\\}\\), la densidad local de alcanzabilidad es:",
        f: R`\mathrm{lrd}_k(a)=\Big(\tfrac{1}{k}\textstyle\sum_{b\in N_k(a)} \mathrm{rd}_k(a,b)\Big)^{-1}` },
      { t: "El factor de outlier local compara cada punto con su vecindario:",
        f: R`\mathrm{LOF}_k(a)=\frac{1}{k}\sum_{b\in N_k(a)}\frac{\mathrm{lrd}_k(b)}{\mathrm{lrd}_k(a)}` },
      { t: "LOF ≈ 1 ⇒ densidad comparable a la de los vecinos; LOF ≫ 1 ⇒ outlier. La clave es la RELATIVIDAD: detecta puntos anómalos respecto a su clúster aunque globalmente no sean extremos. k = √N equilibra sesgo (k grande suaviza) y varianza (k pequeño es ruidoso)." },
    ],
    props: { "Tipo": "Densidad local (kNN)", "Supuestos": "Métrica significativa", "Complejidad": "O(N² d) / O(N log N) con árboles", "HP automático": "k = max(5, √N)" },
  },

  optics: {
    anim: "reach", vista: "pca",
    resumen: "Ordenamiento por densidad: el perfil de alcanzabilidad revela clústeres (valles) y outliers (picos).",
    bloques: [
      { t: "OPTICS generaliza DBSCAN a todos los radios simultáneamente: produce un ORDEN de visita y, para cada punto, su distancia de alcanzabilidad",
        f: R`\mathrm{reach}(p) = \max\{\mathrm{core}_{k}(q),\, d(q,p)\}` },
      { t: "respecto al punto previo \\(q\\) del recorrido de expansión. El diagrama de alcanzabilidad es una 'silueta de montañas': los valles son clústeres densos, los picos son fronteras o anomalías." },
      { t: "Score de evaluación: distancia mínima al subconjunto de entrenamiento normalizada por la alcanzabilidad máxima finita — puntos nuevos lejos de cualquier valle puntúan alto." },
    ],
    props: { "Tipo": "Densidad (ordenamiento)", "Supuestos": "Ninguno sobre k clústeres", "Complejidad": "O(N²)", "HP automático": "min_samples = mcs/3" },
  },

  iforest: {
    anim: "iforest", vista: "aug",
    resumen: "Bosque de aislamiento: los outliers se aíslan con pocos cortes aleatorios.",
    bloques: [
      { t: "Cada árbol particiona recursivamente con cortes (feature, umbral) uniformes. La profundidad esperada de un punto en un árbol binario aleatorio de \\(n\\) hojas es \\(c(n)=2H_{n-1}-2(n-1)/n \\approx 2\\ln n\\). El score normaliza la profundidad media de aislamiento:",
        f: R`s(x)=2^{-\,\mathbb{E}[h(x)]/c(n)}\in(0,1)` },
      { t: "Intuición probabilística: un punto en una región rala queda solo en su celda tras pocos cortes (\\(h\\) pequeño ⇒ \\(s\\to 1\\)); un punto inmerso en masa densa requiere muchos cortes. No usa distancias: solo separabilidad axis-aligned." },
      { t: "max_samples crece con N (256/512/1024): submuestrear descorrelaciona los árboles y mitiga el enmascaramiento (swamping/masking) de outliers múltiples." },
    ],
    props: { "Tipo": "Ensamble de aislamiento", "Supuestos": "Ninguno", "Complejidad": "O(T·ψ log ψ)", "HP automático": "max_samples por tamaño de N" },
  },

  ocsvm: {
    anim: "ocsvm", vista: "aug",
    resumen: "One-Class SVM: separa la masa de datos del origen en el espacio de características del kernel RBF.",
    bloques: [
      { t: "Problema primal de Schölkopf et al. (2001):",
        f: R`\min_{w,\rho,\xi}\;\tfrac12\lVert w\rVert^2 + \tfrac{1}{\nu N}\sum_i \xi_i - \rho \quad \text{s.a.}\;\; \langle w,\phi(x_i)\rangle \ge \rho-\xi_i` },
      { t: "El score es la distancia (con signo) a la frontera aprendida: \\(s(x) = \\rho - \\langle w, \\phi(x)\\rangle\\). El parámetro \\(\\nu\\) tiene una interpretación exacta (propiedad ν): es cota superior de la fracción de outliers de entrenamiento y cota inferior de la fracción de vectores soporte." },
      { t: "Aquí \\(\\nu\\) se estima de los datos: fracción de puntos con |z-MAD| > 3.5, recortada a [0.02, 0.15] — el detector se autocalibra a la contaminación aparente del dataset. Kernel RBF con γ = 'scale'." },
    ],
    props: { "Tipo": "Frontera en RKHS", "Supuestos": "Kernel adecuado", "Complejidad": "O(N²)–O(N³)", "HP automático": "ν por MAD-Z; γ scale" },
  },

  autoencoder: {
    anim: "autoencoder", vista: "aug",
    resumen: "Red neuronal con cuello de botella entrenada con el motor autodiff propio; score = error de reconstrucción.",
    bloques: [
      { t: "Arquitectura \\(F\\to 64\\to 8\\to 64\\to F\\) con activaciones tanh, entrenada minimizando el error de reconstrucción con Adam:",
        f: R`\min_\theta\; \frac{1}{N}\sum_i \lVert x_i - D_\theta(E_\theta(x_i))\rVert^2,\qquad s(x)=\lVert x-\hat x\rVert^2` },
      { t: "El cuello de 8 dimensiones obliga a la red a aprender la variedad de baja dimensión donde vive lo NORMAL (un PCA no lineal: con activaciones lineales el óptimo coincide exactamente con PCA, Baldi–Hornik 1989). Lo anómalo, fuera de la variedad, no se puede reconstruir." },
      { t: "Entrenado desde cero con nuestro motor de diferenciación automática en modo reverso (backend/autodiff.py) — el mismo que integra las Neural ODE/CDE/RDE: gradientes verificados numéricamente." },
    ],
    props: { "Tipo": "Variedad no lineal", "Supuestos": "Manifold de lo normal", "Complejidad": "O(E·N·F·64)", "HP automático": "150 épocas, lr 5e-3, Adam" },
  },

  rpca: {
    anim: "rpca", vista: "aug",
    resumen: "PCA robustecido con una pasada IRLS de pesos de Huber; score = norma del residuo ortogonal.",
    bloques: [
      { t: "El PCA clásico minimiza \\(\\sum_i \\lVert x_i - P x_i \\rVert^2\\) — una pérdida cuadrática que los outliers dominan. Se re-pondera con pesos de Huber:",
        f: R`w_i=\min\{1,\, c/r_i\},\quad r_i=\lVert x_i - P x_i\rVert,\quad c = 2.5\cdot\mathrm{med}(r)` },
      { t: "y se reajusta el subespacio con los datos ponderados (un paso de mínimos cuadrados re-iterados, IRLS). El estimador resultante interpola entre L2 (puntos normales) y L1 (outliers), acotando la función de influencia." },
      { t: "Score: \\(s(x) = \\lVert x - P_{rob} x\\rVert\\) — la energía fuera del subespacio principal robusto.",
        f: R`s(x)=\lVert (I - V_k V_k^\top)(x-\bar x)\rVert_2` },
    ],
    props: { "Tipo": "Subespacio robusto", "Supuestos": "Estructura lineal dominante", "Complejidad": "O(NFk)", "HP automático": "k=5; c de Huber por mediana" },
  },

  conformal: {
    anim: "conformal", vista: "aug",
    resumen: "p-valores conformales con no-conformidad kNN: validez distribucional garantizada por intercambiabilidad.",
    bloques: [
      { t: "Con medida de no-conformidad \\(\\alpha(x)\\) = distancia al k-ésimo vecino, el p-valor conformal es:",
        f: R`p(x)=\frac{1+\#\{i:\ \alpha_i \ge \alpha(x)\}}{n+1},\qquad s(x) = 1 - p(x)` },
      { t: "Teorema (validez conformal): si los datos son intercambiables, \\(\\mathbb{P}(p(X) \\le \\epsilon) \\le \\epsilon\\) para todo \\(\\epsilon\\) — SIN supuestos sobre la distribución. Es la única familia aquí con garantía finita exacta de tasa de falsas alarmas." },
      { t: "k = 2·log₂N crece logarítmicamente: en alta dimensión el volumen de las bolas crece exponencialmente y vecindarios logarítmicos mantienen localidad. Implementación vectorizada con búsqueda binaria sobre los α de calibración ordenados." },
    ],
    props: { "Tipo": "Inferencia conformal", "Supuestos": "Intercambiabilidad", "Complejidad": "O(N log N)", "HP automático": "k = 2·log₂N" },
  },

  sigkernel: {
    anim: "sigkernel", vista: "sig{D} (signaturas nivel {D})",
    resumen: "One-Class SVM con el kernel de signaturas normalizado: compara la geometría de los caminos.",
    bloques: [
      { t: "La signatura es una aplicación inyectiva (módulo reparametrización y 'tree-like equivalence', Hambly–Lyons 2010) del camino al álgebra tensorial. El producto interno de signaturas define un kernel sobre caminos (truncamiento del signature kernel de Király–Oberhauser 2019):",
        f: R`K(X,Y)=\frac{\langle S^{\le m}(X),\,S^{\le m}(Y)\rangle}{\lVert S^{\le m}(X)\rVert\,\lVert S^{\le m}(Y)\rVert}` },
      { t: "La normalización elimina el efecto de la escala global: dos caminos con la misma forma pero distinta amplitud quedan próximos — el kernel discrimina GEOMETRÍA (orden de los movimientos, áreas, asimetrías), complementario a los detectores de magnitud." },
      { t: "K se entrega como kernel precomputado a un One-Class SVM con ν = 0.05. Por el teorema de aproximación universal de signaturas, los funcionales continuos del camino son aproximables linealmente en S(X): la frontera del SVM en este espacio es muy expresiva." },
    ],
    props: { "Tipo": "Kernel sobre caminos", "Supuestos": "Variación acotada", "Complejidad": "O(N² D + n d^m)", "HP automático": "ν = 0.05" },
  },

  sigmaha: {
    anim: "sigmaha", vista: "sig{D}",
    resumen: "Distancia de Mahalanobis LOCAL en el espacio de signaturas: geometría del camino + densidad local.",
    bloques: [
      { t: "Cada serie se transforma en su signatura truncada \\(\\Phi = S^{\\le m}(X) \\in \\mathbb{R}^D\\) (con aumento temporal: el muestreo queda codificado). Para una signatura \\(\\varphi\\), sus k vecinos en la base definen la geometría local:",
        f: R`\mu_k(\varphi),\ \Sigma_k(\varphi)\ \text{(media y covarianza de los k vecinos)}` },
      { t: "El score es la distancia de Mahalanobis a ese vecindario, regularizada (λI evita singularidad cuando k < D):",
        f: R`s(\varphi)=\sqrt{(\varphi-\mu_k)^\top(\Sigma_k+\lambda I)^{-1}(\varphi-\mu_k)}` },
      { t: "k controla la escala del contraste: k=3 detecta desviaciones finas (sensible a ruido), k=10 equilibra, k=20 solo marca desviaciones groseras. A diferencia de T² (global), aquí cada punto se mide contra SU región — válido en espacios de signaturas multimodales. Si D > 60 se reduce con PCA (las coordenadas de la signatura son fuertemente colineales por la identidad de shuffle). Implementación vectorizada por lotes de sistemas lineales." },
    ],
    props: { "Tipo": "Mahalanobis local kNN", "Supuestos": "Localidad significativa", "Complejidad": "O(N k D² + N D³ lote)", "HP automático": "λ = 1e-4; k ∈ {3,10,20}" },
  },

  sigconf: {
    anim: "sigmaha", vista: "sig2",
    resumen: "Distancia de conformancia: media a los k vecinos en el espacio de signaturas estandarizado. (Detector añadido vía registro extensible.)",
    bloques: [
      { t: "Variante de la 'conformance distance' (Cochrane et al. 2021): tras estandarizar robustamente las signaturas,",
        f: R`s(\varphi)=\frac{1}{k}\sum_{b \in N_k(\varphi)} \lVert \varphi - \varphi_b \rVert` },
      { t: "Es el detector más simple posible sobre signaturas (sin covarianza local), útil como línea base y demostración del REGISTRO EXTENSIBLE: se añadió con un decorador de 6 líneas en backend/detectores.py, y aparece automáticamente en pipeline, métricas y este frontend." },
    ],
    props: { "Tipo": "kNN sobre signaturas", "Supuestos": "—", "Complejidad": "O(N log N · D)", "HP automático": "k = √N/2" },
  },

  logsigmaha: {
    anim: "levy", vista: "logsig2 (incrementos + áreas de Lévy)",
    resumen: "Mahalanobis local sobre la LOG-signatura nivel 2: la parametrización mínima sin redundancia. (Detector añadido vía registro.)",
    bloques: [
      { t: "La signatura es redundante: la identidad de shuffle implica p. ej. \\(S^{(i,j)} + S^{(j,i)} = S^{(i)}S^{(j)}\\). El logaritmo tensorial elimina esa redundancia; a nivel 2 las coordenadas libres son exactamente:",
        f: R`\log S = \underbrace{\Delta X^i}_{d}\;\oplus\;\underbrace{A^{ij}=\tfrac12(S^{ij}-S^{ji})}_{d(d-1)/2\ \text{áreas de Lévy}}` },
      { t: "Las áreas de Lévy capturan el ORDEN de los movimientos entre pares de canales — exactamente la información que pierde cualquier detector de features estáticas. Es la misma representación que alimenta el método log-ODE de las Neural RDE: detección y modelado comparten lenguaje geométrico." },
    ],
    props: { "Tipo": "Mahalanobis local (Lie)", "Supuestos": "—", "Complejidad": "O(N D²)", "HP automático": "k = 10; λ = 1e-4" },
  },
};

/* resolución nombre de detector → contenido */
function teoriaDetector(nombre) {
  let kind = null, extra = {};
  if (nombre === "RobustZMAD") kind = "zmad";
  else if (nombre === "PCAT2Q") kind = "t2";
  else if (nombre === "KDE") kind = "kde";
  else if (nombre === "GMM") kind = "gmm";
  else if (nombre === "KMeans") kind = "kmeans";
  else if (nombre === "HDBSCAN") kind = "hdbscan";
  else if (nombre === "LOF") kind = "lof";
  else if (nombre === "OPTICS") kind = "optics";
  else if (nombre === "IForest") kind = "iforest";
  else if (nombre === "OCSVM") kind = "ocsvm";
  else if (nombre === "Autoencoder") kind = "autoencoder";
  else if (nombre === "RobustPCA") kind = "rpca";
  else if (nombre === "Conformal") kind = "conformal";
  else if (nombre === "SigConformancia_d2") kind = "sigconf";
  else if (nombre === "LogSigMaHa_d2") kind = "logsigmaha";
  else if (nombre.startsWith("SigKernel")) {
    kind = "sigkernel";
    extra.depth = nombre.match(/_d(\d)/)?.[1];
  } else if (nombre.startsWith("SigMaHaKNN")) {
    kind = "sigmaha";
    extra.depth = nombre.match(/_d(\d)/)?.[1];
    extra.k = nombre.match(/_k(\d+)/)?.[1];
  }
  if (!kind) return null;
  const base = _KINDS[kind];
  const out = JSON.parse(JSON.stringify(base));
  if (extra.depth) {
    out.vista = out.vista.replace(/\{D\}/g, extra.depth);
    const dimd2 = { 2: "6", 3: "14", 4: "30" }[extra.depth];
    out.nota = `Nivel de truncación m = ${extra.depth} ⇒ D = ${dimd2} coordenadas para camino 2D (t, x); ` +
      (extra.depth === "2" ? "captura incrementos y áreas — lo esencial."
        : extra.depth === "3" ? "añade asimetrías temporales de tercer orden."
        : "máxima resolución geométrica antes de la explosión d^k.");
  }
  if (extra.k) out.nota = (out.nota || "") +
    ` Aquí k = ${extra.k}: ${extra.k === "3" ? "vecindario fino — máxima sensibilidad" : extra.k === "10" ? "equilibrio sensibilidad/robustez" : "vecindario amplio — solo desviaciones groseras"}.`;
  return out;
}

/* ── interpretación de niveles de la signatura ──────────────────────────── */
const NIVELES_SIG = [
  {
    n: 1, color: "#2e7d32", titulo: "Incremento total — desplazamiento",
    formula: R`S^{(i)}=\int_0^T dX^i_t = X^i_T - X^i_0`,
    texto: "Las \\(d\\) coordenadas de nivel 1 son los incrementos netos por canal: cuánto subió o bajó la serie en la ventana, ignorando todo lo intermedio. Es la información LINEAL del camino. Verificación: con aumento temporal, \\(S^{(t)}=1\\) exactamente (longitud del intervalo normalizado).",
    ami: "En AMI: \\(S^{(x)}\\) = consumo final − inicial de la semana. Por sí solo no distingue un perfil normal de uno invertido.",
  },
  {
    n: 2, color: "#e65100", titulo: "Áreas de Lévy — orden y co-movimiento",
    formula: R`S^{(i,j)}=\!\!\int\limits_{0\lt t_1\lt t_2\lt T}\!\! dX^i_{t_1} dX^j_{t_2},\qquad A^{ij}=\tfrac12\big(S^{(i,j)}-S^{(j,i)}\big)`,
    texto: "El nivel 2 ve el ORDEN de los movimientos. Su parte simétrica es redundante (shuffle: \\(S^{ij}+S^{ji}=\\Delta X^i\\,\\Delta X^j\\)); la información nueva es el área de Lévy: el área firmada entre el camino proyectado al plano \\((i,j)\\) y su cuerda. \\(A^{ij}>0\\) significa que el canal \\(i\\) se mueve sistemáticamente ANTES que el \\(j\\) (adelanto de fase).",
    ami: "En AMI: \\(S^{(t,x)}\\propto\\) área bajo la curva \\(\\approx\\) energía total de la semana. Para el EEG multivariado: \\(A^{(C3,O1)}\\) mide qué canal lidera la oscilación.",
  },
  {
    n: 3, color: "#6a1b9a", titulo: "Asimetrías de tercer orden",
    formula: R`S^{(i,j,k)}=\!\!\int\limits_{0\lt t_1\lt t_2\lt t_3\lt T}\!\! dX^i\, dX^j\, dX^k`,
    texto: "Integrales triples: asimetría temporal de las fluctuaciones (¿las subidas preceden a las bajadas?), curvatura del área acumulada, co-momentos de tres canales. \\(S^{(t,t,x)}\\) pondera el valor por el tiempo transcurrido al cuadrado — localiza CUÁNDO ocurre la masa del cambio.",
    ami: "Distingue una semana con consumo concentrado el lunes de otra con el mismo total concentrado el domingo: el nivel \\(\\le 2\\) no puede, el 3 sí.",
  },
  {
    n: 4, color: "#c62828", titulo: "Estructura fina — oscilación y rugosidad",
    formula: R`S^{(i_1,\dots,i_4)}=\!\!\int\limits_{0\lt t_1\lt \cdots\lt t_4\lt T}\!\! dX^{i_1}\cdots dX^{i_4}`,
    texto: "Momentos de cuarto orden: kurtosis direccional, textura oscilatoria, combinaciones tiempo-valor finas. El factor \\(1/k!\\) hace decaer la norma factorialmente — el teorema de aproximación universal garantiza que los funcionales continuos se aproximan linealmente sobre los primeros niveles, y en la práctica \\(m=4\\) satura el beneficio frente al costo \\(d^m\\).",
    ami: "El detector SigMaHaKNN_d4 explota esta resolución: detecta SyntheticNoise (textura) mejor que niveles bajos.",
  },
];

/* ── etapas Neural DE ───────────────────────────────────────────────────── */
const NDE_ETAPAS = [
  {
    id: "rnn", paso: "Etapa 1 · punto de partida", nombre: "RNN / GRU",
    color: "#546e7a", anim: "nde_rnn",
    eq: R`h_{k+1}=\varphi(W_h h_k + W_x x_{k+1} + b)`,
    motiv: "Una red recurrente es un SISTEMA DINÁMICO DISCRETO controlado por datos: el estado oculto h se actualiza cada vez que llega una observación.",
    bloques: [
      { h: "Definición", t: "Estado oculto \\(h_k \\in \\mathbb{R}^H\\); en cada paso:",
        f: R`h_{k+1}=\tanh(W_h h_k + W_x x_{k+1} + b),\qquad y = W_o h_n + b_o` },
      { h: "GRU: compuertas", t: "La GRU interpola convexamente entre conservar y renovar el estado — precursora discreta del flujo continuo:",
        f: R`h_{k+1} = (1-z_k)\odot h_k + z_k \odot \tilde h_k,\qquad z_k = \sigma(\cdot)` },
      { h: "Entrenamiento", t: "Backpropagation Through Time: la regla de la cadena a través de la recurrencia. El jacobiano del paso se multiplica n veces ⇒ gradientes que explotan o se desvanecen como \\(\\lambda_{\\max}(J)^n\\) — la motivación histórica de las compuertas." },
      { h: "Muestreo irregular", t: "La estructura es CIEGA al tiempo físico: el paso k→k+1 es idéntico si pasaron 2 s o 2 h. El remiendo estándar — concatenar Δt como entrada — obliga a la red a APRENDER el efecto del tiempo en vez de tenerlo en su estructura. Este defecto motiva todo lo que sigue." },
    ],
    props: ["Discreto, síncrono", "Δt como parche", "BPTT: gradientes frágiles", "Base histórica (Elman 1990)"],
  },
  {
    id: "node", paso: "Etapa 2 · límite continuo", nombre: "Neural ODE",
    color: "#00838f", anim: "nde_ode",
    eq: R`\frac{dh}{dt}=f_\theta(h(t)),\quad h(0)=\mathrm{enc}(x_0)`,
    motiv: "Una ResNet h_{k+1} = h_k + f(h_k) es el método de Euler con paso 1. Al llevar el paso a 0, la profundidad se vuelve TIEMPO CONTINUO (Chen et al., NeurIPS 2018).",
    bloques: [
      { h: "Construcción como límite", t: "De la conexión residual al flujo:",
        f: R`h_{k+1}=h_k+\epsilon f_\theta(h_k)\;\xrightarrow[\epsilon\to 0]{}\;\frac{dh}{dt}=f_\theta(h(t))` },
      { h: "Existencia y unicidad", t: "Si \\(f_\\theta\\) es Lipschitz (lo es: MLP con tanh), Picard–Lindelöf garantiza solución única para cada h(0). Corolario geométrico: las trayectorias NO se cruzan — un Neural ODE es un difeomorfismo del espacio de estados, isotópico a la identidad." },
      { h: "Gradientes: método adjunto", t: "En lugar de guardar el grafo del solver, se integra hacia atrás la ecuación adjunta (memoria O(1)):",
        f: R`\frac{da}{dt}=-a^\top \frac{\partial f_\theta}{\partial h},\qquad \frac{dL}{d\theta}=-\int_T^0 a^\top \frac{\partial f_\theta}{\partial \theta}\,dt,\qquad a(t)=\frac{\partial L}{\partial h(t)}` },
      { h: "Nuestra implementación", t: "Integrador Runge–Kutta 4 con M=12 pasos y gradiente A TRAVÉS del solver (discretizar-luego-optimizar), construido sobre el motor autodiff propio. RK4 tiene error global O(Δt⁴)." },
      { h: "La limitación que motiva la CDE", t: "h(t) queda determinado por h(0): las observaciones x₁,…,xₙ que llegan DESPUÉS no pueden modificar la trayectoria. En el experimento de espirales (fase inicial aleatoria) el Neural ODE queda en ~50% — exactamente lo que la teoría predice." },
    ],
    props: ["Profundidad continua", "Flujo difeomórfico", "Adjunto: memoria O(1)", "No procesa datos en t>0"],
  },
  {
    id: "ncde", paso: "Etapa 3 · el dato como control", nombre: "Neural CDE",
    color: "#6a1b9a", anim: "nde_cde",
    eq: R`dz_t = f_\theta(z_t)\, dX_t`,
    motiv: "La solución (Kidger et al., NeurIPS 2020): el camino de datos X deja de ser condición inicial y pasa a ser el CONTROL de una ecuación diferencial controlada — el análogo continuo exacto de la RNN.",
    bloques: [
      { h: "El control", t: "De las observaciones \\(\\{(t_i, x_i)\\}\\) se construye un camino continuo \\(X:[t_0,t_n]\\to\\mathbb{R}^{d}\\) (interpolación, con el tiempo como canal: \\(X_t=(t, x_t)\\)). La ecuación controlada, como integral de Riemann–Stieltjes:",
        f: R`z_t = z_{t_0} + \int_{t_0}^{t} f_\theta(z_s)\, dX_s,\qquad f_\theta:\mathbb{R}^H\to\mathbb{R}^{H\times d}` },
      { h: "Por qué es la RNN continua", t: "Discretizando con Euler: \\(z_{i+1} = z_i + f_\\theta(z_i)\\,\\Delta X_i\\) — una 'celda recurrente' cuyo efecto de entrada es PROPORCIONAL al incremento real del dato y del tiempo. El muestreo irregular entra de forma exacta: Δt vive dentro de ΔX. Nuestra integración usa punto medio (RK2) por intervalo de observación." },
      { h: "Universalidad", t: "Teorema (Kidger et al.): las Neural CDE son aproximadores universales de funcionales continuos de caminos. La demostración pasa por las signaturas: toda CDE lineal en el control se resuelve como serie de integrales iteradas — la signatura ES la base de la solución.",
        f: R`z_T \approx \sum_{|w|\le m} c_w \, S^{w}(X)_{0,T}` },
      { h: "Robustez", t: "Invariante por reparametrización del tiempo (si se reparametriza X y su canal temporal): la CDE responde a la GEOMETRÍA del camino, no a su velocidad de muestreo — de ahí su solidez con datos faltantes e irregulares." },
    ],
    props: ["Datos en todo t (control)", "Muestreo irregular exacto", "Universal sobre caminos", "Memoria continua"],
  },
  {
    id: "nrde", paso: "Etapa 4 · caminos rugosos", nombre: "Neural RDE",
    color: "#e65100", anim: "nde_rde",
    eq: R`z_{j+1} = z_j + g_\theta(z_j)\cdot \mathrm{logsig}^{(2)}_{[t_j,t_{j+1}]}(X)`,
    motiv: "Para series LARGAS o de alta frecuencia, integrar paso a paso es caro e inestable. La teoría de caminos rugosos de Lyons da la respuesta: el método log-ODE (Morrill et al., 2021).",
    bloques: [
      { h: "De Riemann–Stieltjes a rough paths", t: "Si X es muy rugoso (p.ej. trayectorias brownianas, EEG), la integral \\(\\int f(z)\\,dX\\) deja de estar bien definida en sentido clásico (Young requiere variación p < 2). Lyons (1998): la solución existe y es ÚNICA si se enriquece X con sus integrales iteradas — precisamente su signatura. El mapa solución (Itô–Lyons) es entonces continuo." },
      { h: "El método log-ODE", t: "Se parte el intervalo en ventanas. En cada una, el control se resume por su log-signatura truncada, y se resuelve la EDO autónoma:",
        f: R`\frac{dz}{du} = g_\theta(z)\,\frac{\mathrm{logsig}^{(m)}_{[t_j,t_{j+1}]}(X)}{t_{j+1}-t_j}` },
      { h: "Qué contiene logsig nivel 2", t: "La parametrización mínima de la geometría de la ventana: incrementos \\(\\Delta X\\) (d coordenadas) y áreas de Lévy \\(A^{uv}\\) (d(d−1)/2). El orden de convergencia del método log-ODE con nivel m es O(δ^m) en la malla δ — el nivel 2 ya garantiza orden cuadrático.",
        f: R`\mathrm{logsig}^{(2)} = \big(\Delta X^1,\dots,\Delta X^d,\ A^{12}, A^{13},\dots\big) \in \mathbb{R}^{d + \binom{d}{2}}` },
      { h: "Ventajas prácticas", t: "n/s pasos en lugar de n (aquí s=6–8): menos profundidad efectiva ⇒ gradientes más estables y entrenamiento más rápido (en nuestra comparativa el NRDE entrena ~3× más rápido que el NCDE con igual accuracy). La geometría intra-ventana NO se pierde: viaja comprimida en las áreas de Lévy. Es el mismo objeto matemático que usa el detector LogSigMaHa_d2 — detección y modelado, unificados." },
    ],
    props: ["Series largas/alta frecuencia", "Pasos = n/s (memoria ↓)", "Orden O(δ²) con nivel 2", "Geometría comprimida sin pérdida"],
  },
];

window.teoriaDetector = teoriaDetector;
window.NIVELES_SIG = NIVELES_SIG;
window.NDE_ETAPAS = NDE_ETAPAS;
