"use strict";

// ============================================================
// Réseau de neurones pédagogique – port JS/Canvas de la V1.4
// ============================================================

// ── Résolution de référence (identique au programme Python) ──
const REF_W = 1500, REF_H = 930;

// ── Palette ──────────────────────────────────────────────────
const C = {
  BG:           "#f6f8fc",
  PANEL:        "#ffffff",
  BORDER:       "#dae2eb",
  TEXT:         "#19263b",
  MUTED:        "#64748b",
  GREEN:        "#22c55e",
  GREEN_LIGHT:  "#ebfcf0",
  GREEN_DARK:   "#166534",
  BLUE:         "#2563eb",
  BLUE_LIGHT:   "#eff6ff",
  PURPLE:       "#935cf6",
  PURPLE_LIGHT: "#ede4ff",
  ORANGE_UI:    "#f97316",
  ORANGE_LINE:  "#ff9a43",
  DARK:         "#0f172a",
  LIGHT_GRAY:   "#e2e8f0",
  PINK_LIGHT:   "#fdf2f8",
  YELLOW:       "#ffd400",
  RED:          "#ff3333",
  COLOR_BLUE:   "#2577e5",
  COLOR_ORANGE: "#ff8c00",
  COLOR_GREEN:  "#14aa50",
  COLOR_VIOLET: "#9646dc",
};

// ── Données couleurs ──────────────────────────────────────────
const BASE_COLORS = [
  { name:"JAUNE",  label:"Jaune",  code:[1,0,0], color:C.YELLOW,       target:[1,0,0] },
  { name:"ROUGE",  label:"Rouge",  code:[0,1,0], color:C.RED,          target:[0,1,0] },
  { name:"BLEU",   label:"Bleu",   code:[0,0,1], color:C.COLOR_BLUE,   target:[0,0,1] },
];
const EXPERT_TEST_COLORS = [
  { name:"ORANGE", label:"Orange", code:[1,1,0], color:C.COLOR_ORANGE, comment:"mélange de JAUNE et ROUGE" },
  { name:"VERT",   label:"Vert",   code:[1,0,1], color:C.COLOR_GREEN,  comment:"mélange de JAUNE et BLEU" },
  { name:"VIOLET", label:"Violet", code:[0,1,1], color:C.COLOR_VIOLET, comment:"mélange de ROUGE et BLEU" },
];
const ADVANCED_COLORS = [
  { name:"JAUNE",  label:"Jaune",  code:[1,0,0], color:C.YELLOW,       target:[1,0,0,0,0] },
  { name:"ORANGE", label:"Orange", code:[1,1,0], color:C.COLOR_ORANGE, target:[0,1,0,0,0] },
  { name:"ROUGE",  label:"Rouge",  code:[0,1,0], color:C.RED,          target:[0,0,1,0,0] },
  { name:"VIOLET", label:"Violet", code:[0,1,1], color:C.COLOR_VIOLET, target:[0,0,0,1,0] },
  { name:"BLEU",   label:"Bleu",   code:[0,0,1], color:C.COLOR_BLUE,   target:[0,0,0,0,1] },
];

// ── Math réseau ───────────────────────────────────────────────
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function softmax(vals) {
  const m = Math.max(...vals);
  const e = vals.map(v => Math.exp(v - m));
  const s = e.reduce((a,b) => a+b, 0);
  return e.map(v => v / s);
}
function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }

class SimpleNetwork {
  constructor() {
    this.w = Array.from({length:3}, () => [0.5, 0.5, 0.5]);
    this.b = [0, 0, 0];
  }
  forward(x) {
    const raw = this.w.map((row, o) => dot(row, x) + this.b[o]);
    return { raw, probs: softmax(raw) };
  }
  trainOne(x, target, lr = 0.35) {
    const { probs } = this.forward(x);
    const delta = probs.map((p, i) => p - target[i]);
    for (let o = 0; o < 3; o++) {
      for (let i = 0; i < 3; i++) this.w[o][i] -= lr * delta[o] * x[i];
      this.b[o] -= lr * delta[o];
    }
    return { probs, delta };
  }
}

class ExpertNetwork {
  constructor() {
    // Seed PRNG simple pour reproduire la même initialisation que Python (seed=12)
    let s = 12;
    function rand() {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    }
    function uniform(a, b) { return a + (b - a) * rand(); }

    this.w1 = Array.from({length:5}, () => Array.from({length:3}, () => uniform(-0.8, 0.8)));
    this.b1 = Array.from({length:5}, () => uniform(-0.1, 0.1));
    this.w2 = Array.from({length:5}, () => Array.from({length:5}, () => uniform(-0.8, 0.8)));
    this.b2 = Array.from({length:5}, () => uniform(-0.1, 0.1));
  }
  forward(x) {
    const hidden = this.w1.map((row, h) => sigmoid(dot(row, x) + this.b1[h]));
    const raw    = this.w2.map((row, o) => dot(row, hidden) + this.b2[o]);
    return { hidden, raw, probs: softmax(raw) };
  }
  trainOne(x, target, lr = 0.18) {
    const { hidden, probs } = this.forward(x);
    const delta2 = probs.map((p, i) => p - target[i]);
    const dHid   = hidden.map((h, hi) => {
      const sig = delta2.reduce((s, d, o) => s + d * this.w2[o][hi], 0);
      return sig * h * (1 - h);
    });
    for (let o = 0; o < 5; o++) {
      for (let h = 0; h < 5; h++) this.w2[o][h] -= lr * delta2[o] * hidden[h];
      this.b2[o] -= lr * delta2[o];
    }
    for (let h = 0; h < 5; h++) {
      for (let i = 0; i < 3; i++) this.w1[h][i] -= lr * dHid[h] * x[i];
      this.b1[h] -= lr * dHid[h];
    }
    return { probs, delta: delta2 };
  }
  trainMicroBatch(examples, repeats = 8, lr = 0.18) {
    for (let r = 0; r < repeats; r++) {
      const ex = examples[Math.floor(Math.random() * examples.length)];
      this.trainOne(ex.code, ex.target, lr);
    }
  }
}

// ── Application ───────────────────────────────────────────────
class NeuralDemo {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext("2d");
    this.scale  = 1;
    this.buttons = {};   // { key: {x,y,w,h,label,kind} }
    this._resize();
    window.addEventListener("resize", () => this._resize());
    canvas.addEventListener("click", e => this._onClick(e));
    document.addEventListener("keydown", e => this._onKey(e));
    this._resetBasic();
    this._loop();
  }

  // ── Resize / scaling ──────────────────────────────────────
  _resize() {
    const vw = window.innerWidth, vh = window.innerHeight;
    this.scale = Math.min(vw / REF_W, vh / REF_H);
    this.canvas.width  = Math.round(REF_W * this.scale);
    this.canvas.height = Math.round(REF_H * this.scale);
  }

  // ── State ─────────────────────────────────────────────────
  _resetBasic() {
    this.mode     = "BASIC";
    this.colors   = BASE_COLORS;
    this.outputs  = ["JAUNE","ROUGE","BLEU"];
    this.trainingSet = Array.from({length:21}, (_,i) => BASE_COLORS[i%3]);
    this.net      = new SimpleNetwork();
    this.phase    = "INITIALISATION";
    this.step     = 0;
    this.current  = this.colors[0];
    this.showMatrix = false;
    this.expertTestCount = 0;
    this.lastMessage = "Clique sur « Apprendre l\'exemple suivant » pour observer les poids étape par étape.";
  }

  _switchAdvanced() {
    this.mode     = "ADVANCED";
    this.colors   = ADVANCED_COLORS;
    this.outputs  = ["JAUNE","ORANGE","ROUGE","VIOLET","BLEU"];
    this.trainingSet = Array.from({length:30}, (_,i) => ADVANCED_COLORS[i%5]);
    this.net      = new ExpertNetwork();
    this.phase    = "INITIALISATION";
    this.step     = 0;
    this.current  = this.colors[0];
    this.showMatrix = false;
    this.lastMessage = "Mode expert activé : le réseau va apprendre cinq mots de couleur, dont ORANGE et VIOLET.";
  }

  _learnNext() {
    if (this.phase === "TEST") {
      this.lastMessage = "L\'apprentissage est terminé. Réinitialise ou passe au mode expert.";
      return;
    }
    if (this.step >= this.trainingSet.length) {
      this.phase = "TEST";
      this.current = this.colors[Math.floor(Math.random()*this.colors.length)];
      this.lastMessage = "Apprentissage terminé. Tu peux lancer les tests.";
      return;
    }
    this.phase = "APPRENTISSAGE";
    const ex = this.trainingSet[this.step];
    this.current = ex;
    if (this.mode === "ADVANCED") {
      this.net.trainOne(ex.code, ex.target, 0.18);
      this.net.trainMicroBatch(this.trainingSet, 12, 0.18);
    } else {
      this.net.trainOne(ex.code, ex.target);
    }
    this.step++;
    this.lastMessage = `Exemple ${this.step} appris : ${ex.name}. Observe les poids qui viennent de changer.`;
    if (this.step >= this.trainingSet.length) {
      this.phase = "TEST";
      this.current = this.colors[Math.floor(Math.random()*this.colors.length)];
      this.lastMessage = "Apprentissage terminé. Le réseau peut maintenant être testé.";
    }
  }

  _learnAll() {
    while (this.step < this.trainingSet.length) this._learnNext();
    if (this.mode === "ADVANCED") {
      for (let i = 0; i < 250; i++) {
        const ex = this.trainingSet[Math.floor(Math.random()*this.trainingSet.length)];
        this.net.trainOne(ex.code, ex.target, 0.12);
      }
    }
    this.phase = "TEST";
    this.current = this.colors[Math.floor(Math.random()*this.colors.length)];
    this.lastMessage = "Apprentissage complet effectué. Tu peux tester le réseau.";
  }

  _newTestColor() {
    this.phase = "TEST";
    this.current = this.colors[Math.floor(Math.random()*this.colors.length)];
    this.lastMessage = `Test classique : ${this.current.name}.`;
  }

  _expertTest() {
    if (this.mode === "BASIC") {
      this.phase = "TEST_EXPERT";
      this.current = EXPERT_TEST_COLORS[Math.floor(Math.random()*EXPERT_TEST_COLORS.length)];
      this.expertTestCount++;
      this.lastMessage = `Test expert ${this.expertTestCount}/10 : ${this.current.name} (${this.current.comment}). Le réseau n\'a pas appris cette classe.`;
    } else {
      this.phase = "TEST";
      this.current = this.colors[Math.floor(Math.random()*this.colors.length)];
      this.lastMessage = `En mode expert, ${this.current.name} est maintenant une vraie classe apprise.`;
    }
  }

  _currentProbs() {
    if (this.mode === "BASIC") return this.net.forward(this.current.code).probs;
    return this.net.forward(this.current.code).probs;
  }

  _currentGap() {
    if (!this.current.target) return null;
    const probs = this._currentProbs();
    return probs.reduce((s, p, i) => s + Math.abs((this.current.target[i]||0) - p), 0) / probs.length;
  }

  // ── Events ────────────────────────────────────────────────
  _onClick(e) {
    const rect = this.canvas.getBoundingClientRect();
    const rx = (e.clientX - rect.left) / this.scale;
    const ry = (e.clientY - rect.top)  / this.scale;
    for (const [key, btn] of Object.entries(this.buttons)) {
      if (rx >= btn.x && rx <= btn.x+btn.w && ry >= btn.y && ry <= btn.y+btn.h) {
        if      (key === "reset")       this._resetBasic();
        else if (key === "learn_next")  this._learnNext();
        else if (key === "learn_all")   this._learnAll();
        else if (key === "matrix")      this.showMatrix = !this.showMatrix;
        else if (key === "test")        this._newTestColor();
        else if (key === "expert_test") this._expertTest();
        else if (key === "advanced")    this._switchAdvanced();
        else if (key === "new_color")   this._newTestColor();
        return;
      }
    }
  }

  _onKey(e) {
    if (e.key === " ")    this._learnNext();
    if (e.key === "a")    this._learnAll();
    if (e.key === "r")    this._resetBasic();
    if (e.key === "t")    this._newTestColor();
    if (e.key === "e")    this._expertTest();
    if (e.key === "m")    this.showMatrix = !this.showMatrix;
  }

  // ── Main loop ─────────────────────────────────────────────
  _loop() {
    this._draw();
    requestAnimationFrame(() => this._loop());
  }

  // ══════════════════════════════════════════════════════════
  // DESSIN – helpers canvas
  // ══════════════════════════════════════════════════════════
  get c() { return this.ctx; }

  _px(x) { return x * this.scale; } // unused – on dessine en coordonnées REF puis on scale le ctx

  _setupCtx() {
    // On applique le scale une fois au début du draw
    this.ctx.setTransform(this.scale, 0, 0, this.scale, 0, 0);
  }

  _roundRect(x, y, w, h, r, fill, stroke = null, sw = 1) {
    const ctx = this.c;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    if (fill)   { ctx.fillStyle = fill;   ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke(); }
  }

  _text(txt, x, y, font, color, align = "left", baseline = "top") {
    const ctx = this.c;
    ctx.font = font;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(txt, x, y);
  }

  _textWidth(txt, font) {
    this.c.font = font;
    return this.c.measureText(txt).width;
  }

  _multiline(txt, x, y, maxW, font, color, lineH = 16) {
    const words = txt.split(" ");
    let line = "";
    let yy = y;
    this.c.font = font;
    for (const word of words) {
      const test = line + word + " ";
      if (this.c.measureText(test).width > maxW && line) {
        this._text(line.trim(), x, yy, font, color);
        yy += lineH;
        line = word + " ";
      } else {
        line = test;
      }
    }
    if (line) this._text(line.trim(), x, yy, font, color);
    return yy;
  }

  _progress(x, y, w, h, value, color) {
    const v = Math.max(0, Math.min(1, value));
    this._roundRect(x, y, w, h, 999, C.LIGHT_GRAY);
    if (v > 0) this._roundRect(x, y, Math.round(w * v), h, 999, color);
  }

  _arrowLine(x1, y1, x2, y2, color, lw = 1) {
    const ctx = this.c;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const len = 8, sp = 0.45;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - len*Math.cos(angle-sp), y2 - len*Math.sin(angle-sp));
    ctx.lineTo(x2 - len*Math.cos(angle+sp), y2 - len*Math.sin(angle+sp));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  _circle(x, y, r, fill, stroke = null, sw = 2) {
    const ctx = this.c;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI*2);
    if (fill)   { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke(); }
  }

  _button(key, x, y, w, h, label, kind = "light") {
    this.buttons[key] = { x, y, w, h, label, kind };
    const ctx = this.c;
    const mouse = this._mouse || {x:-1,y:-1};
    const hover = mouse.x>=x && mouse.x<=x+w && mouse.y>=y && mouse.y<=y+h;

    let bg, fg, border;
    if      (kind === "dark")   { bg=C.DARK;     fg="#fff"; border=C.DARK;     }
    else if (kind === "blue")   { bg=C.BLUE;     fg="#fff"; border=C.BLUE;     }
    else if (kind === "orange") { bg=C.ORANGE_UI;fg="#fff"; border=C.ORANGE_UI;}
    else if (kind === "purple") { bg=C.PURPLE;   fg="#fff"; border=C.PURPLE;   }
    else                        { bg="#fff";     fg=C.TEXT; border=C.BORDER;   }

    if (hover) {
      // Darken slightly
      bg = bg.startsWith("#") ? this._darken(bg, 15) : bg;
    }

    this._roundRect(x, y, w, h, 10, bg, border, 1);
    this._text(label, x+w/2, y+h/2, "bold 13px Arial", fg, "center", "middle");
  }

  _darken(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const r = Math.max(0, (n>>16)-amt);
    const g = Math.max(0, ((n>>8)&0xff)-amt);
    const b = Math.max(0, (n&0xff)-amt);
    return `rgb(${r},${g},${b})`;
  }

  // ══════════════════════════════════════════════════════════
  // DESSIN – sections
  // ══════════════════════════════════════════════════════════
  _draw() {
    this.buttons = {};
    this._setupCtx();
    const ctx = this.c;
    ctx.clearRect(0, 0, REF_W, REF_H);
    ctx.fillStyle = C.BG;
    ctx.fillRect(0, 0, REF_W, REF_H);

    this._drawHeader();
    this._drawStepper();
    this._drawTrainingPanel();
    this._drawNetworkPanel();
    this._drawTestPanel();
  }

  // ── Header ────────────────────────────────────────────────
  _drawHeader() {
    this._text("RÉSEAU DE NEURONES – DÉMONSTRATION PÉDAGOGIQUE", 24, 20, "bold 24px Arial", C.TEXT);
    const sub = this.mode === "BASIC"
      ? "Mode simple : 3 entrées codées → 3 sorties. Les couleurs composées ne sont pas encore des classes."
      : "Mode expert : 3 entrées codées → 5 neurones cachés → 5 sorties, avec couleurs composées apprises.";
    this._text(sub, 24, 52, "13px Arial", C.MUTED);

    let x = 750;
    const gap = this._currentGap();
    const gapTxt = gap === null ? "—" : gap.toFixed(2);
    const labels = [
      ["Mode", this.mode],
      ["Phase", this.phase],
      ["Étape", `${this.step} / ${this.trainingSet.length}`],
      ["Écart", gapTxt],
    ];
    for (const [title, value] of labels) {
      this._text(title, x+55, 20, "13px Arial", C.TEXT, "center", "top");
      this._roundRect(x+5, 42, 100, 28, 7, C.GREEN_LIGHT, "#bcebcd", 1);
      this._text(value, x+55, 56, "bold 13px Arial", C.GREEN_DARK, "center", "middle");
      x += 112;
    }

    this._button("reset", 1285, 24, 190, 40, "↻ Réinitialiser simple", "dark");
  }

  // ── Stepper ───────────────────────────────────────────────
  _drawStepper() {
    this._roundRect(36, 88, 1428, 60, 10, C.PANEL, C.BORDER);
    const boxes = [
      ["1", "INITIALISATION", "Poids de départ",    55],
      ["2", "APPRENTISSAGE",  "Un clic = un exemple", 390],
      ["3", "TEST",           "Couleurs apprises",  735],
      ["4", "TEST_EXPERT",    "Couleurs composées", 1045],
    ];
    for (const [num, title, sub, x] of boxes) {
      if (title === this.phase)
        this._roundRect(x-12, 89, 300, 58, 10, C.GREEN_LIGHT);
      this._circle(x+35, 118, 16, C.GREEN);
      this._text(num, x+35, 118, "bold 13px Arial", "#fff", "center", "middle");
      this._text(title, x+60, 104, "bold 14px Arial", C.TEXT);
      this._text(sub,   x+60, 126, "11px Arial", C.TEXT);
    }
  }

  // ── Training panel ────────────────────────────────────────
  _drawTrainingPanel() {
    this._roundRect(36, 170, 300, 570, 12, C.PANEL, C.BORDER);
    this._text("JEU D'APPRENTISSAGE", 50, 186, "bold 16px Arial", C.TEXT);
    this._text(`(${this.trainingSet.length} EXEMPLES)`, 210, 188, "13px Arial", C.TEXT);

    let y = 218;
    const lineH = this.trainingSet.length > 25 ? 18 : 25;
    for (let i = 0; i < this.trainingSet.length; i++) {
      if (y > 715) break;
      const ex = this.trainingSet[i];
      if (this.phase === "APPRENTISSAGE" && i === this.step-1)
        this._roundRect(48, y-2, 260, lineH, 6, C.BLUE_LIGHT);
      this._text(String(i+1), 56, y, "11px Arial", C.MUTED);
      this._roundRect(88, y-1, 22, 15, 3, ex.color);
      this._text("→", 125, y, "11px Arial", C.MUTED);
      this._text(ex.name, 150, y, "11px Arial", C.TEXT);
      y += lineH;
    }

    // Progress box
    this._roundRect(36, 755, 300, 105, 12, C.PANEL, C.BORDER);
    this._text("Progression apprentissage", 54, 776, "11px Arial", C.MUTED);
    this._text(`${this.step} / ${this.trainingSet.length}`, 315, 775, "bold 13px Arial", C.GREEN_DARK, "right");
    this._progress(54, 810, 265, 13, this.step/this.trainingSet.length, C.GREEN);

    if (this.phase === "TEST" || this.phase === "TEST_EXPERT")
      this._text("✓ Apprentissage terminé", 185, 848, "13px Arial", C.GREEN, "center", "top");
    else
      this._text("Clique pour apprendre pas à pas", 185, 848, "11px Arial", C.MUTED, "center", "top");
  }

  // ── Network panel ─────────────────────────────────────────
  _drawNetworkPanel() {
    this._roundRect(350, 170, 790, 705, 12, C.PANEL, C.BORDER);

    if (this.mode === "BASIC") {
      this._text("RÉSEAU SIMPLE : 3 ENTRÉES → 3 SORTIES", 365, 186, "bold 16px Arial", C.TEXT);
      this._text("Un clic apprend un exemple : observe les poids juste après chaque correction.", 365, 210, "13px Arial", C.MUTED);
      this._drawBasicNetwork();
    } else {
      this._text("RÉSEAU EXPERT : 3 ENTRÉES → 5 NEURONES CACHÉS → 5 SORTIES", 365, 186, "bold 16px Arial", C.TEXT);
      this._text("Les couleurs composées deviennent des classes apprises. Le réseau a besoin de plus d'entraînement interne.", 365, 210, "13px Arial", C.MUTED);
      this._drawAdvancedNetwork();
    }

    // Message pédagogique
    const bg  = this.phase === "TEST_EXPERT" ? C.PINK_LIGHT : C.BLUE_LIGHT;
    const bdr = this.phase === "TEST_EXPERT" ? "#f5bed4" : "#bfdbfe";
    this._roundRect(455, 760, 580, 80, 10, bg, bdr);
    const lc = this.phase === "TEST_EXPERT" ? C.PURPLE : C.BLUE;
    this._text("À observer", 475, 775, "bold 14px Arial", lc);
    this._multiline(this.lastMessage, 475, 800, 535, "11px Arial", C.TEXT, 15);

    // Boutons bas
    this._button("learn_next", 350, 880, 230, 38, "Apprendre l'exemple suivant", "blue");
    this._button("learn_all",  590, 880, 150, 38, "Tout apprendre", "light");
    this._button("matrix",     750, 880, 160, 38, "Voir poids", "light");
    this._button("test",       920, 880, 105, 38, "Test", "light");

    if (this.mode === "BASIC" && (this.phase === "TEST" || this.phase === "TEST_EXPERT"))
      this._button("expert_test", 1035, 880, 105, 38, "Test Expert", "orange");

    if (this.mode === "BASIC" && this.expertTestCount >= 10)
      this._button("advanced", 795, 832, 300, 34, "Basculer vers réseau expert 3→5→5", "purple");

    if (this.showMatrix) {
      if (this.mode === "BASIC") this._drawBasicMatrix();
      else                       this._drawAdvancedMatrix();
    }
  }

  _drawInputNodes(nodes) {
    const labels = [
      { label:"Jaune", color:C.YELLOW     },
      { label:"Rouge", color:C.RED        },
      { label:"Bleu",  color:C.COLOR_BLUE },
    ];
    for (let i = 0; i < nodes.length; i++) {
      const [x, y] = nodes[i];
      const active = this.current.code[i] === 1;
      const r = active ? 30 : 22;
      const bw = active ? 4 : 1;
      this._text(labels[i].label,            x-45, y-18, "13px Arial", C.TEXT, "right");
      this._text(`entrée ${this.current.code[i]}`, x-45, y+5,  "11px Arial", C.MUTED, "right");
      this._circle(x, y, r, labels[i].color, active ? C.DARK : "#5a6473", bw);
      this._text(String(this.current.code[i]), x, y+44, "bold 13px Arial", active ? C.DARK : C.MUTED, "center", "top");
    }
  }

  _drawOutputNodes(nodes, probs) {
    const predI = probs.indexOf(Math.max(...probs));
    for (let i = 0; i < nodes.length; i++) {
      const [x, y] = nodes[i];
      const color = i < this.colors.length ? this.colors[i].color : C.LIGHT_GRAY;
      const isPred = i === predI;
      const r  = isPred ? 36 : 30;
      const bw = isPred ? 4  : 2;
      this._circle(x, y, r, "#fff", color, bw);
      this._text(probs[i].toFixed(2), x, y, "bold 13px Arial", C.TEXT, "center", "middle");
      this._text(this.outputs[i], x+52, y-8, "bold 16px Arial", C.TEXT);
      if (isPred) this._text("← plus forte", x+52, y+14, "11px Arial", C.GREEN_DARK);
    }
  }

  _drawBasicNetwork() {
    const { raw, probs } = this.net.forward(this.current.code);
    const inNodes  = [[480,340],[480,500],[480,660]];
    const outNodes = [[990,340],[990,500],[990,660]];

    this._text("ENTRÉES",        480, 260, "bold 16px Arial", C.BLUE,      "center");
    this._text("SORTIES APPRISES", 990, 260, "bold 16px Arial", C.ORANGE_UI, "center");

    for (let i = 0; i < inNodes.length; i++) {
      const [ax, ay] = inNodes[i];
      for (let o = 0; o < outNodes.length; o++) {
        const [bx, by] = outNodes[o];
        const w = this.net.w[o][i];
        const lw = Math.max(1, Math.min(8, Math.abs(w)*3));
        const lc = w >= 0.5 ? C.ORANGE_LINE : "#b4bece";
        this._arrowLine(ax+30, ay, bx-38, by, lc, lw);
        const mx = (ax+bx)>>1, my = (ay+by)>>1;
        this._text(w.toFixed(2), mx, my-8, "11px Arial", C.MUTED, "center");
      }
    }
    this._drawInputNodes(inNodes);
    this._drawOutputNodes(outNodes, probs);
  }

  _drawAdvancedNetwork() {
    const { hidden, raw, probs } = this.net.forward(this.current.code);
    const inNodes  = [[455,310],[455,460],[455,610]];
    const hidNodes = [[735,275],[735,375],[735,475],[735,575],[735,675]];
    const outNodes = [[1030,275],[1030,375],[1030,475],[1030,575],[1030,675]];

    this._text("ENTRÉES",           455,  240, "bold 16px Arial", C.BLUE,      "center");
    this._text("5 NEURONES CACHÉS", 735,  240, "bold 16px Arial", C.PURPLE,    "center");
    this._text("5 SORTIES",         1030, 240, "bold 16px Arial", C.ORANGE_UI, "center");

    // Couche entrée → cachée
    for (let i = 0; i < inNodes.length; i++) {
      const [ax, ay] = inNodes[i];
      for (let h = 0; h < hidNodes.length; h++) {
        const [bx, by] = hidNodes[h];
        const w = this.net.w1[h][i];
        const lw = Math.max(1, Math.min(5, Math.abs(w)*2.5));
        this._arrowLine(ax+25, ay, bx-35, by, "#9baad2", lw);
      }
    }
    // Couche cachée → sortie
    for (let h = 0; h < hidNodes.length; h++) {
      const [ax, ay] = hidNodes[h];
      for (let o = 0; o < outNodes.length; o++) {
        const [bx, by] = outNodes[o];
        const w = this.net.w2[o][h];
        const lw = Math.max(1, Math.min(5, Math.abs(w)*2.5));
        this._arrowLine(ax+35, ay, bx-35, by, C.ORANGE_LINE, lw);
      }
    }

    this._drawInputNodes(inNodes);

    // Neurones cachés
    for (let i = 0; i < hidNodes.length; i++) {
      const [x, y] = hidNodes[i];
      this._circle(x, y, 30, C.PURPLE_LIGHT, C.PURPLE, 2);
      this._text(hidden[i].toFixed(2), x, y,    "bold 13px Arial", C.TEXT,   "center", "middle");
      this._text(`h${i+1}`,            x, y+42,  "11px Arial",      C.TEXT,   "center", "top");
    }

    this._drawOutputNodes(outNodes, probs);
  }

  // ── Matrices ──────────────────────────────────────────────
  _drawBasicMatrix() {
    this._roundRect(430, 555, 620, 270, 12, C.DARK);
    this._text("Matrice des poids", 455, 577, "bold 16px Arial", "#fff");
    this._text("Ligne = sortie ; colonne = entrée", 455, 603, "11px Arial", "#dbeafe");

    const headers = ["entrée JAUNE","entrée ROUGE","entrée BLEU"];
    const x0 = 570;
    for (let c = 0; c < headers.length; c++)
      this._text(headers[c], x0+c*125, 635, "11px Arial", "#dbeafe", "center");

    let y = 675;
    for (let o = 0; o < 3; o++) {
      this._text(this.outputs[o], 455, y, "11px Arial", "#dbeafe");
      for (let i = 0; i < 3; i++)
        this._text(this.net.w[o][i].toFixed(2), x0+i*125, y+6, "bold 13px Arial", "#fff", "center");
      y += 38;
    }
    this._text("Biais :", 455, y+14, "11px Arial", "#dbeafe");
    for (let o = 0; o < 3; o++)
      this._text(this.net.b[o].toFixed(2), x0+o*125, y+20, "11px Arial", "#fff", "center");
  }

  _drawAdvancedMatrix() {
    this._roundRect(395, 540, 690, 300, 12, C.DARK);
    this._text("Aperçu des matrices du réseau expert", 420, 562, "bold 16px Arial", "#fff");
    this._text("w1 : entrées → cachée ; w2 : cachée → sorties", 420, 588, "11px Arial", "#dbeafe");

    let y = 620;
    this._text("w1", 420, y, "bold 14px Arial", "#fff");
    y += 25;
    for (let h = 0; h < 5; h++) {
      const vals = this.net.w1[h].map(v => v.toFixed(2)).join("  ");
      this._text(`h${h+1}: [${vals}]`, 420, y, "11px Arial", "#dbeafe");
      y += 18;
    }
    y += 8;
    this._text("w2", 420, y, "bold 14px Arial", "#fff");
    y += 25;
    for (let o = 0; o < 5; o++) {
      const vals = this.net.w2[o].map(v => v.toFixed(2)).join("  ");
      this._text(`${this.outputs[o].padEnd(7)}: [${vals}]`, 420, y, "11px Arial", "#dbeafe");
      y += 18;
    }
  }

  // ── Test panel ────────────────────────────────────────────
  _drawTestPanel() {
    const nOut = this.outputs.length;
    this._roundRect(1155, 170, 310, 620, 12, C.PANEL, C.BORDER);
    this._text("TEST : PRÉDICTION", 1168, 186, "bold 16px Arial", C.TEXT);

    // Bloc couleur
    this._text("Entrée test", 1310, 228, "bold 14px Arial", C.TEXT, "center");
    this._roundRect(1252, 248, 116, 116, 16, this.current.color);
    this._text(this.current.name, 1310, 382, "bold 14px Arial", C.TEXT, "center");
    this._text(`Code : [${this.current.code.join(",")}]`, 1310, 405, "13px Arial", C.TEXT, "center");
    this._button("new_color", 1195, 428, 240, 36, "↻ Nouvelle couleur", "light");

    // Bloc probabilités
    const probBoxH = nOut <= 3 ? 190 : 205;
    this._roundRect(1170, 485, 280, probBoxH, 10, "#fff", C.BORDER);
    this._text("PROBABILITÉS DE SORTIE", 1185, 502, "bold 16px Arial", C.TEXT);

    const probs = this._currentProbs();
    const predI = probs.indexOf(Math.max(...probs));

    let y      = nOut <= 3 ? 545 : 540;
    const rowG = nOut <= 3 ? 43  : 31;
    const barX = nOut <= 3 ? 1245 : 1242;
    const barW = nOut <= 3 ? 130  : 118;
    const valX = nOut <= 3 ? 1400 : 1390;

    for (let i = 0; i < this.outputs.length; i++) {
      const color = i < this.colors.length ? this.colors[i].color : C.LIGHT_GRAY;
      this._text(this.outputs[i], 1185, y-2, "bold 14px Arial", color);
      this._progress(barX, y, barW, 12, probs[i], color);
      this._text(probs[i].toFixed(2), valX, y-3, "13px Arial", C.TEXT);
      y += rowG;
    }

    // Bloc décision
    const predY = nOut <= 3 ? 700 : 704;
    this._roundRect(1175, predY, 270, 72, 10, C.GREEN_LIGHT, "#bcebcd");
    this._text("Prédiction :", 1260, predY+18, "bold 16px Arial", "#16a34a", "center");
    this._text(this.outputs[predI], 1330, predY+6, "bold 22px Arial", this.colors[predI]?.color ?? C.TEXT);
    this._text("classe avec la probabilité la plus élevée", 1310, predY+50, "11px Arial", C.MUTED, "center");

    // Aide pédagogique
    this._roundRect(1155, 805, 310, 105, 10, C.BLUE_LIGHT, "#bfdbfe");
    this._text("💡 Message pédagogique", 1175, 820, "bold 14px Arial", C.BLUE);

    let msg;
    if (this.mode === "BASIC" && this.phase === "TEST_EXPERT")
      msg = "Cette couleur n'a jamais été apprise comme un mot séparé. Le réseau répartit donc sa confiance entre les sorties connues.";
    else if (this.mode === "ADVANCED")
      msg = "En mode expert, les couleurs composées font partie du vocabulaire appris. Les 5 probabilités sont affichées séparément pour éviter toute ambiguïté.";
    else
      msg = "Le réseau choisit la sortie dont la probabilité est la plus élevée.";

    this._multiline(msg, 1175, 846, 270, "11px Arial", C.TEXT, 15);
  }
}

// ── Suivi souris (pour hover boutons) ──────────────────────
const canvas = document.getElementById("canvas");
let demo;

canvas.addEventListener("mousemove", e => {
  if (!demo) return;
  const rect = canvas.getBoundingClientRect();
  demo._mouse = {
    x: (e.clientX - rect.left) / demo.scale,
    y: (e.clientY - rect.top)  / demo.scale,
  };
});

window.addEventListener("load", () => {
  demo = new NeuralDemo(canvas);
});
