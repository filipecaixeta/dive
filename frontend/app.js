"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const PRESETS = [
  { name: "Natural", p: {} },
  { name: "Vibrante", p: { dehaze: 0.85, contrast: 0.4, vibrance: 0.55, local_contrast: 0.55, sharpen: 0.4 } },
  { name: "Suave", p: { dehaze: 0.5, contrast: 0.12, vibrance: 0.15, local_contrast: 0.2, sharpen: 0.2 } },
  { name: "Água turquesa", p: { water_hue: 180, water_protect: 0.6 } },
  { name: "Azul profundo", p: { water_hue: 215, water_protect: 0.75, contrast: 0.35 } },
  { name: "Água verde", p: { blue_comp: 0.8, red_comp: 1.3, dehaze: 0.8 } },
];
// kept when applying a preset: they describe the photo/method, not the look
const WB_KEYS = ["wb_mode", "wb_x", "wb_y", "method", "z_far"];

const state = {
  id: null,
  info: null,
  defaults: null,
  params: null,
  view: "split",
  split: 0.5,
  scale: 1, tx: 0, ty: 0,
  picking: false,
  stats: null,
  holdBefore: false,
  video: null,        // set when editing a video: {id, name, duration, fps, ...}
  videoMode: "adaptive",
};

/* ------------------------------------------------------------------ utils */
function toast(msg, isErr = false, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}
function busy(on, text = "Processando…") {
  $("#busy").hidden = !on;
  $("#busyText").textContent = text;
}
function store(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch (_) {} }
function load(key) { try { return JSON.parse(localStorage.getItem(key)); } catch (_) { return null; } }

const FMT = {
  pct: (v) => Math.round(v * 100) + "%",
  signed: (v) => (v > 0 ? "+" : "") + Math.round(v * 100),
  x: (v) => v.toFixed(2) + "×",
  ev: (v) => (v > 0 ? "+" : "") + v.toFixed(2) + " EV",
  deg: (v) => Math.round(v) + "°",
  m: (v) => v.toFixed(1) + " m",
};

/* ---------------------------------------------------------------- sliders */
const sliders = {};
function buildSliders() {
  for (const el of $$(".slider")) {
    const key = el.dataset.key;
    el.innerHTML = `<div class="row"><span class="name">${el.dataset.label}</span><span class="val"></span></div>
      <input type="range" min="${el.dataset.min}" max="${el.dataset.max}" step="${el.dataset.step}">`;
    const input = el.querySelector("input");
    const val = el.querySelector(".val");
    const fmt = FMT[el.dataset.fmt] || ((v) => v);
    const s = { el, input, val, fmt, min: +el.dataset.min, max: +el.dataset.max };
    sliders[key] = s;

    input.addEventListener("input", () => {
      state.params[key] = +input.value;
      syncSlider(key);
      clearPresetHighlight();
      liveRender();
    });
    input.addEventListener("change", () => finalRender());
    el.addEventListener("dblclick", () => {
      state.params[key] = state.defaults[key];
      syncSlider(key);
      finalRender();
    });
  }
}
function syncSlider(key) {
  const s = sliders[key];
  const v = state.params[key];
  s.input.value = v;
  s.val.textContent = s.fmt(v);
  s.input.style.setProperty("--fill", ((v - s.min) / (s.max - s.min)) * 100 + "%");
  s.el.classList.toggle("changed", state.defaults && Math.abs(v - state.defaults[key]) > 1e-6);
}
function syncAll() {
  for (const k in sliders) syncSlider(k);
  for (const b of $$("#wbSeg button")) b.classList.toggle("on", b.dataset.wb === state.params.wb_mode);
  for (const b of $$("#methodSeg button")) b.classList.toggle("on", b.dataset.method === (state.params.method || "classic"));
  document.body.classList.toggle("method-depth", state.params.method === "depth");
  $("#wbHint").hidden = state.params.wb_mode !== "pick" || state.params._picked;
  updateMarker();
  store("dive.params", stripPrivate(state.params));
}
function stripPrivate(p) {
  const o = {};
  for (const k in p) if (!k.startsWith("_")) o[k] = p[k];
  return o;
}

/* ---------------------------------------------------------------- presets */
function buildPresets() {
  const box = $("#presets");
  PRESETS.forEach((pr, i) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = pr.name;
    b.onclick = () => {
      const wb = {};
      for (const k of WB_KEYS) wb[k] = state.params[k];
      state.params = { ...state.defaults, ...pr.p, ...wb, _picked: state.params._picked };
      syncAll();
      $$(".chip").forEach((c, j) => c.classList.toggle("on", i === j));
      finalRender();
    };
    box.appendChild(b);
  });
}
function clearPresetHighlight() { $$(".chip").forEach((c) => c.classList.remove("on")); }

/* ------------------------------------------------------------- rendering */
// One request in flight at a time; while dragging we always send the newest
// params at low resolution, and refine at preview resolution once the user stops.
let seq = 0, inFlight = false, queued = null, refineTimer = null;

function liveRender() {
  clearTimeout(refineTimer);
  refineTimer = setTimeout(() => queue("preview"), 280);
  queue("fast");
}
function finalRender() {
  clearTimeout(refineTimer);
  queue("preview");
}
function queue(quality) {
  if (!state.id) return;
  if (!(queued === "preview" && quality === "fast")) queued = quality;
  pump();
}
async function pump() {
  if (inFlight || !queued || !state.id) return;
  const quality = queued;
  queued = null;
  inFlight = true;
  $("#liveDot").classList.add("on");
  const mySeq = ++seq;
  const view = state.view === "haze" || state.view === "skinmask" ? state.view : "result";
  try {
    const res = await fetch(`/api/images/${state.id}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: stripPrivate(state.params), view, quality, seq: mySeq }),
    });
    if (res.status === 404) { toast("O servidor reiniciou. Abra a foto de novo.", true); return; }
    if (!res.ok || res.status === 204) return;
    const stats = JSON.parse(res.headers.get("X-Stats") || "{}");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const img = $("#imgAfter");
    const tmp = new Image();
    tmp.src = url;
    try { await tmp.decode(); } catch (_) {}
    const old = img.src;
    img.src = url;
    if (old.startsWith("blob:")) setTimeout(() => URL.revokeObjectURL(old), 1000);
    state.stats = stats;
    updateStats(stats);
    if (view === "result") drawHistogram(tmp);
  } catch (e) {
    toast("Erro ao processar: " + e.message, true);
  } finally {
    inFlight = false;
    if (!queued) $("#liveDot").classList.remove("on");
    pump();
  }
}

function updateStats(s) {
  const label = s.quality === "fast" ? "rascunho" : "preview";
  $("#statusText").textContent = `${label} · ${s.ms} ms`;
  $("#veilSwatch").style.background = s.veiling_light || "#333";
  if (s.wb_ref) {
    $("#wbRefRow").hidden = false;
    $("#wbSwatch").style.background = s.wb_ref;
  } else {
    $("#wbRefRow").hidden = true;
  }
  if (!$("#veilBox").hidden) placeVeil();
  const sk = (s.skin || [])[0];
  if (sk) {
    $("#skinBefore").style.background = sk.before;
    $("#skinAfter").style.background = sk.after;
  }
}

function drawHistogram(img) {
  const W = 256, H = Math.max(1, Math.round((img.naturalHeight / img.naturalWidth) * 256));
  const c = drawHistogram.c || (drawHistogram.c = document.createElement("canvas"));
  c.width = W; c.height = H;
  const cx = c.getContext("2d", { willReadFrequently: true });
  cx.drawImage(img, 0, 0, W, H);
  const d = cx.getImageData(0, 0, W, H).data;
  const r = new Float32Array(256), g = new Float32Array(256), b = new Float32Array(256);
  for (let i = 0; i < d.length; i += 4) { r[d[i]]++; g[d[i + 1]]++; b[d[i + 2]]++; }
  const hist = $("#hist");
  const hx = hist.getContext("2d");
  const w = hist.width, h = hist.height;
  hx.clearRect(0, 0, w, h);
  let max = 1;
  for (let i = 2; i < 254; i++) max = Math.max(max, r[i], g[i], b[i]);
  hx.globalCompositeOperation = "lighter";
  for (const [arr, col] of [[r, "rgba(230,70,70,.75)"], [g, "rgba(70,200,90,.75)"], [b, "rgba(70,120,240,.8)"]]) {
    hx.fillStyle = col;
    hx.beginPath();
    hx.moveTo(0, h);
    for (let i = 0; i < 256; i++) {
      const v = Math.sqrt(Math.min(1, arr[i] / max));
      hx.lineTo((i / 255) * w, h - v * (h - 4));
    }
    hx.lineTo(w, h);
    hx.closePath();
    hx.fill();
  }
  hx.globalCompositeOperation = "source-over";
}

/* --------------------------------------------------------------- viewing */
const vp = $("#viewport"), canvas = $("#canvas");

function applyTransform() {
  canvas.style.transform = `translate(${state.tx}px, ${state.ty}px) scale(${state.scale})`;
  canvas.style.setProperty("--inv", 1 / state.scale);
}
function fit() {
  if (!state.info) return;
  const r = vp.getBoundingClientRect();
  const { preview_width: pw, preview_height: ph } = state.info;
  state.scale = Math.min(r.width / pw, r.height / ph) * 0.97;
  state.tx = (r.width - pw * state.scale) / 2;
  state.ty = (r.height - ph * state.scale) / 2;
  applyTransform();
}
function zoomTo(scale, cx, cy) {
  const r = vp.getBoundingClientRect();
  if (cx === undefined) { cx = r.width / 2; cy = r.height / 2; }
  scale = Math.max(0.05, Math.min(scale, 12));
  const k = scale / state.scale;
  state.tx = cx - (cx - state.tx) * k;
  state.ty = cy - (cy - state.ty) * k;
  state.scale = scale;
  applyTransform();
}
function applyView() {
  const v = state.holdBefore ? "before" : state.view;
  const after = $("#imgAfter");
  after.style.visibility = v === "before" ? "hidden" : "visible";
  after.style.clipPath = v === "split" ? `inset(0 0 0 ${state.split * 100}%)` : "none";
  $("#splitLine").hidden = v !== "split";
  $("#splitLine").style.left = state.split * 100 + "%";
  for (const b of $$("#viewSeg button")) b.classList.toggle("on", b.dataset.view === state.view);
}
function updateMarker() {
  const m = $("#wbMarker");
  const p = state.params;
  if (!p || p.wb_mode !== "pick" || !p._picked) { m.hidden = true; return; }
  m.hidden = false;
  m.style.left = p.wb_x * 100 + "%";
  m.style.top = p.wb_y * 100 + "%";
}
function placeVeil() {
  const b = state.stats && state.stats.veiling_bbox;
  if (!b) return;
  const box = $("#veilBox");
  box.style.top = b[0] * 100 + "%";
  box.style.left = b[1] * 100 + "%";
  box.style.height = (b[2] - b[0]) * 100 + "%";
  box.style.width = (b[3] - b[1]) * 100 + "%";
}

/* pointer interaction: split handle, white picker, pan */
let drag = null;
vp.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  if (e.target.classList.contains("handle")) {
    drag = { kind: "split" };
  } else if (state.skinPicking) {
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) addSkin(x, y);
    return;
  } else if (state.picking) {
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
      Object.assign(state.params, { wb_mode: "pick", wb_x: x, wb_y: y, _picked: true });
      syncAll();
      finalRender();
    }
    return;
  } else {
    drag = { kind: "pan", x: e.clientX, y: e.clientY, tx: state.tx, ty: state.ty };
    vp.classList.add("panning");
  }
  vp.setPointerCapture(e.pointerId);
});
vp.addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (drag.kind === "split") {
    const rect = canvas.getBoundingClientRect();
    state.split = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    applyView();
  } else {
    state.tx = drag.tx + (e.clientX - drag.x);
    state.ty = drag.ty + (e.clientY - drag.y);
    applyTransform();
  }
});
const endDrag = () => { drag = null; vp.classList.remove("panning"); };
vp.addEventListener("pointerup", endDrag);
vp.addEventListener("pointercancel", endDrag);
vp.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = vp.getBoundingClientRect();
  zoomTo(state.scale * Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
window.addEventListener("resize", () => state.id && fit());

$("#zoomFit").onclick = fit;
$("#zoom100").onclick = () => zoomTo(1 / (window.devicePixelRatio || 1));

for (const b of $$("#viewSeg button")) {
  b.onclick = () => {
    const prev = state.view;
    state.view = b.dataset.view;
    applyView();
    const special = (v) => v === "haze" || v === "skinmask";
    if ((special(prev) || special(state.view)) && prev !== state.view) finalRender();
  };
}

for (const b of $$("#wbSeg button")) {
  b.onclick = () => {
    const mode = b.dataset.wb;
    if (mode === "pick") setSkinPicking(false);
    state.picking = mode === "pick";
    vp.classList.toggle("picking", state.picking);
    if (mode === "pick") {
      // stay on the old WB until the user actually clicks a reference
      if (state.params.wb_mode !== "pick") state.params._picked = false;
      for (const x of $$("#wbSeg button")) x.classList.toggle("on", x === b);
      $("#wbHint").hidden = !!state.params._picked;
      if (state.view === "before") { state.view = "split"; applyView(); }
      return;
    }
    state.params.wb_mode = mode;
    state.params._picked = false;
    syncAll();
    finalRender();
  };
}

let depthLoaded = false;
for (const b of $$("#methodSeg button")) {
  b.onclick = () => {
    state.params.method = b.dataset.method;
    if (state.params.method === "depth" && !depthLoaded) {
      depthLoaded = true;
      toast("Calculando a profundidade com IA. Na primeira vez carrega o modelo e pode levar alguns segundos.", false, 6000);
    }
    syncAll();
    clearPresetHighlight();
    finalRender();
  };
}

/* ------------------------------------------------------------------ skin */
function setSkinPicking(on) {
  state.skinPicking = on;
  if (on) { state.picking = false; vp.classList.remove("picking"); }
  vp.classList.toggle("skin-picking", on);
  $("#skinBtn").classList.toggle("on", on);
  $("#skinHint").hidden = !on;
  if (on && state.view === "before") { state.view = "split"; applyView(); }
}
function showSkin(divers) {
  state.skinDivers = divers || [];
  document.body.classList.toggle("has-skin", state.skinDivers.length > 0);
  if (!state.skinDivers.length && state.view === "skinmask") { state.view = "split"; applyView(); }
  $("#skinClear").hidden = state.skinDivers.length === 0;
  $("#skinBtn").textContent = state.skinDivers.length ? "Pele: adicionar mergulhador" : "Pele: clicar no mergulhador";
  $("#skinCount").textContent = state.skinDivers.length === 1 ? "Pele (1 mergulhador)" : `Pele (${state.skinDivers.length} mergulhadores)`;
  const box = $("#skinBoxes");
  box.innerHTML = "";
  state.skinDivers.forEach((d, i) => {
    const el = document.createElement("div");
    el.className = "skin-box";
    const [x0, y0, x1, y1] = d.bbox;
    Object.assign(el.style, { left: x0 * 100 + "%", top: y0 * 100 + "%", width: (x1 - x0) * 100 + "%", height: (y1 - y0) * 100 + "%" });
    el.innerHTML = `<span>mergulhador ${i + 1}</span>`;
    box.appendChild(el);
  });
}
async function addSkin(x, y) {
  setSkinPicking(false);
  busy(true, state.skinLoaded ? "Encontrando o mergulhador…" : "Encontrando o mergulhador… (na primeira vez carrega os modelos de IA, ~15 s)");
  try {
    const res = await fetch(`/api/images/${state.id}/skin`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, params: stripPrivate(state.params) }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || res.statusText);
    state.skinLoaded = true;
    showSkin(body.divers);
    if (!body.detected) toast("Não reconheci um mergulhador ali; usei o contorno do que foi clicado. Confira o resultado.", false, 5000);
    else if (body.skin_px < 200) toast("Achei o mergulhador, mas quase sem pele visível: a correção vai ser sutil.", false, 5000);
    finalRender();
  } catch (e) {
    toast(e.message, true, 5000);
  } finally {
    busy(false);
  }
}
$("#skinBtn").onclick = () => setSkinPicking(!state.skinPicking);
$("#skinClear").onclick = async () => {
  await fetch(`/api/images/${state.id}/skin`, { method: "DELETE" });
  showSkin([]);
  finalRender();
};

$("#showVeil").onclick = () => {
  const box = $("#veilBox");
  box.hidden = !box.hidden;
  $("#showVeil").textContent = box.hidden ? "mostrar" : "ocultar";
  placeVeil();
};

$("#resetAll").onclick = () => {
  state.params = { ...state.defaults, _picked: false };
  state.picking = false;
  vp.classList.remove("picking");
  syncAll();
  clearPresetHighlight();
  finalRender();
};

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
  if ((e.key === "b" || e.key === "B") && !state.holdBefore) { state.holdBefore = true; applyView(); }
  if (e.key === "Escape" && state.skinPicking) setSkinPicking(false);
  if (e.key === "Escape" && state.picking) {
    state.picking = false;
    vp.classList.remove("picking");
    syncAll();
  }
  if (e.key === "f") fit();
});
window.addEventListener("keyup", (e) => {
  if (e.key === "b" || e.key === "B") { state.holdBefore = false; applyView(); }
});

/* ---------------------------------------------------------------- upload */
const isVideoFile = (f) => f.type.startsWith("video/") || /\.(mp4|mov|m4v|mkv|avi|mts)$/i.test(f.name);

async function openFile(file) {
  if (!file) return;
  if (isVideoFile(file)) return openVideo(file);
  busy(true, "Analisando a foto…");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/images", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const info = await res.json();
    state.video = null;
    document.body.classList.remove("is-video");
    $("#timeline").hidden = true;
    loadImage(info, { newFile: true });
  } catch (e) {
    toast("Não consegui abrir a foto: " + e.message, true);
  } finally {
    busy(false);
  }
}

// Shows an image session in the editor. For a new file the look carries over
// from the last one; for another frame of the same video everything is kept.
function loadImage(info, { newFile = false } = {}) {
  const sameSize = state.info && state.info.preview_width === info.preview_width &&
                   state.info.preview_height === info.preview_height;
  state.id = info.id;
  state.info = info;
  state.defaults = info.defaults;
  if (newFile || !state.params) {
    const saved = load("dive.params") || {};
    // a picked white point belongs to one photo
    state.params = { ...info.defaults, ...saved, wb_mode: saved.wb_mode === "pick" ? "auto" : (saved.wb_mode || "auto"), _picked: false };
    state.picking = false;
    vp.classList.remove("picking");
  }
  setSkinPicking(false);
  showSkin([]);
  const label = state.video ? `${state.video.name} · ${state.video.width}×${state.video.height} · ${fmtTime(state.video.duration)}`
                            : `${info.name} · ${info.width}×${info.height}`;
  $("#filename").textContent = label;
  canvas.style.width = info.preview_width + "px";
  canvas.style.height = info.preview_height + "px";
  $("#imgBefore").src = `/api/images/${info.id}/original`;
  if (newFile) $("#imgAfter").removeAttribute("src");
  $("#empty").hidden = true;
  vp.hidden = false;
  $("#stageToolbar").hidden = false;
  $("#exportBtn").disabled = false;
  $("#panel").classList.remove("disabled");
  if (newFile) { $("#veilBox").hidden = true; $("#showVeil").textContent = "mostrar"; }
  syncAll();
  applyView();
  if (newFile || !sameSize) fit();
  queue("fast");
  queue("preview");
}

/* ----------------------------------------------------------------- video */
function fmtTime(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function uploadWithProgress(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      xhr.status < 300 ? resolve(body) : reject(new Error(body.detail || xhr.statusText));
    };
    xhr.onerror = () => reject(new Error("falha de rede"));
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

async function openVideo(file) {
  busy(true, "Enviando vídeo… 0%");
  try {
    const res = await uploadWithProgress("/api/videos", file, (f) => {
      $("#busyText").textContent = f < 1 ? `Enviando vídeo… ${Math.round(f * 100)}%` : "Lendo o vídeo…";
    });
    state.video = res.video;
    state.videoMode = "adaptive";
    $("#lutBtn").firstChild.textContent = "LUT de entrada";
    $("#lutBtn").classList.remove("on-lut");
    $("#lutClear").hidden = true;
    document.body.classList.add("is-video");
    const scrub = $("#scrub");
    scrub.max = res.video.duration;
    scrub.step = 1 / res.video.fps;
    scrub.value = res.image.t;
    $("#timeLabel").textContent = `${fmtTime(res.image.t)} / ${fmtTime(res.video.duration)}`;
    $("#timeline").hidden = false;
    $("#thumbs").style.backgroundImage = `url(/api/videos/${res.video.id}/thumbs)`;
    for (const b of $$("#colorMode button")) b.classList.toggle("on", b.dataset.mode === "adaptive");
    loadImage(res.image, { newFile: true });
  } catch (e) {
    toast("Não consegui abrir o vídeo: " + e.message, true);
  } finally {
    busy(false);
  }
}

let frameReq = 0;
async function gotoFrame(t) {
  if (!state.video) return;
  const my = ++frameReq;
  $("#statusText").textContent = "lendo frame…";
  try {
    const res = await fetch(`/api/videos/${state.video.id}/frame`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ t }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const info = await res.json();
    if (my !== frameReq) return; // user already moved on
    // a picked white point refers to the old frame's pixels
    if (state.params.wb_mode === "pick") {
      Object.assign(state.params, { wb_mode: "auto", _picked: false });
      state.picking = false;
      vp.classList.remove("picking");
      toast("O conta-gotas vale para um frame. Neste frame voltei para o automático; escolha a referência de novo se quiser.");
    }
    loadImage(info);
  } catch (e) {
    toast("Não consegui ler o frame: " + e.message, true);
  }
}

$("#scrub").addEventListener("input", (e) => {
  $("#timeLabel").textContent = `${fmtTime(+e.target.value)} / ${fmtTime(state.video.duration)}`;
});
$("#scrub").addEventListener("change", (e) => gotoFrame(+e.target.value));
for (const b of $$("#colorMode button")) {
  b.onclick = () => {
    state.videoMode = b.dataset.mode;
    for (const x of $$("#colorMode button")) x.classList.toggle("on", x === b);
  };
}

async function setInputLut(file) {
  const fd = new FormData();
  if (file) fd.append("file", file);
  busy(true, file ? "Aplicando LUT de entrada…" : "Removendo LUT…");
  try {
    const t = +$("#scrub").value;
    const res = await fetch(`/api/videos/${state.video.id}/input_lut?t=${t}`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const info = await res.json();
    state.video.input_lut = info.input_lut;
    $("#lutBtn").firstChild.textContent = info.input_lut ? `LUT: ${info.input_lut}` : "LUT de entrada";
    $("#lutBtn").classList.toggle("on-lut", !!info.input_lut);
    $("#lutClear").hidden = !info.input_lut;
    loadImage(info);
  } catch (e) {
    toast("Não consegui aplicar o LUT: " + e.message, true);
  } finally {
    busy(false);
  }
}
$("#lutInput").addEventListener("change", (e) => { if (e.target.files[0]) setInputLut(e.target.files[0]); e.target.value = ""; });
$("#lutClear").onclick = () => setInputLut(null);

let jobTimer = null;
$("#jobPreviewBox").onclick = () => $("#job").classList.toggle("big");

async function startVideoExport(scale, codec = "hevc") {
  showJob.lastSeq = 0;
  $("#jobPreviewBox").hidden = true;
  $("#job").classList.remove("big");
  try {
    const res = await fetch(`/api/videos/${state.video.id}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: stripPrivate(state.params), image_id: state.id, scale, mode: state.videoMode, codec }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    showJob(await res.json());
  } catch (e) {
    toast("Falha ao iniciar a exportação: " + e.message, true);
  }
}

function showJob(job) {
  const box = $("#job");
  box.hidden = false;
  box.classList.toggle("done", job.status === "done");
  box.classList.toggle("err", job.status === "error" || job.status === "cancelled");
  $("#jobBar").style.width = (job.progress * 100).toFixed(1) + "%";
  const eta = job.eta == null ? "" : ` · faltam ${job.eta >= 60 ? Math.round(job.eta / 60) + " min" : job.eta + " s"}`;
  const text = {
    queued: "Na fila…",
    analyzing: "Analisando as cenas para manter a cor estável…",
    tracking: "Seguindo o mergulhador pelo vídeo para corrigir a pele…",
    rendering: `Frame ${job.frame} de ${job.frames} · ${job.fps.toFixed(1)} fps${eta}`,
    done: `Pronto em ${Math.round(job.elapsed / 60)} min. Salvo em ${job.path}`,
    error: "Erro: " + job.error,
    cancelled: "Exportação cancelada.",
  }[job.status];
  $("#jobTitle").textContent = job.status === "done" ? "Vídeo exportado" : "Exportando vídeo";
  // live preview: refresh only when the server has a new frame
  if (job.preview_seq && job.preview_seq !== showJob.lastSeq) {
    showJob.lastSeq = job.preview_seq;
    const img = $("#jobPreview");
    const next = new Image();
    next.onload = () => { img.src = next.src; $("#jobPreviewBox").hidden = false; };
    next.src = `/api/jobs/${job.id}/preview?seq=${job.preview_seq}`;
    const what = job.preview_kind === "tracking" ? "rastreando" : "frame final";
    $("#jobPreviewCap").textContent = `${what} · ${fmtTime(job.preview_t)}`;
  }
  $("#jobText").textContent = text;
  const finished = ["done", "error", "cancelled"].includes(job.status);
  $("#jobCancel").hidden = finished;
  $("#jobClose").hidden = !finished;
  const dl = $("#jobDownload");
  dl.hidden = job.status !== "done";
  dl.textContent = job.file && job.file.endsWith(".mov") ? "Baixar MOV" : "Baixar MP4";
  dl.href = `/api/jobs/${job.id}/download`;
  dl.download = job.file;
  clearTimeout(jobTimer);
  if (!finished) {
    jobTimer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/jobs/${job.id}`);
        if (r.ok) return showJob(await r.json());
      } catch (_) {}
      showJob(job);
    }, 1000);
  }
  $("#jobCancel").onclick = async () => {
    await fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" });
  };
}
$("#jobClose").onclick = () => { $("#job").hidden = true; };

$("#fileInput").addEventListener("change", (e) => { openFile(e.target.files[0]); e.target.value = ""; });

// The overlay ignores pointer events, so drag events always land on the page
// underneath. It is hidden when the drag leaves the window (relatedTarget null),
// on drop, and by a watchdog in case the browser never sends either.
const hideDrop = () => { $("#dropOverlay").hidden = true; };
let dropWatchdog = null;
const isFileDrag = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragover", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  $("#dropOverlay").hidden = false;
  clearTimeout(dropWatchdog);
  dropWatchdog = setTimeout(hideDrop, 400); // dragover fires continuously while dragging
});
window.addEventListener("dragleave", (e) => { if (!e.relatedTarget) hideDrop(); });
window.addEventListener("dragend", hideDrop);
window.addEventListener("drop", (e) => {
  e.preventDefault();
  clearTimeout(dropWatchdog);
  hideDrop();
  const f = [...(e.dataTransfer.files || [])].find((x) => x.type.startsWith("image/") || isVideoFile(x) || /\.(jpe?g|png|tiff?|dng|gpr|cr2|cr3|nef|arw|orf|rw2|raf)$/i.test(x.name));
  if (f) openFile(f);
});

/* ---------------------------------------------------------------- export */
$("#exportBtn").onclick = (e) => { e.stopPropagation(); $("#exportMenu").classList.toggle("open"); };
document.addEventListener("click", () => $("#exportMenu").classList.remove("open"));
for (const b of $$("#exportMenu button")) {
  b.onclick = async () => {
    $("#exportMenu").classList.remove("open");
    const kind = b.dataset.export;
    if (kind === "video") return startVideoExport(+b.dataset.scale, b.dataset.codec || "hevc");
    const isLut = kind === "lut";
    busy(true, isLut ? "Gerando LUT…" : "Processando em resolução máxima…");
    try {
      const url = isLut ? `/api/images/${state.id}/lut` : `/api/images/${state.id}/export`;
      const body = isLut ? { params: stripPrivate(state.params) } : { params: stripPrivate(state.params), format: kind, quality: 95 };
      const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const blob = await res.blob();
      const cd = res.headers.get("Content-Disposition") || "";
      const name = (cd.match(/filename="([^"]+)"/) || [])[1] || "foto_corrigida.jpg";
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      toast(isLut ? "LUT salvo. Ele aplica só a correção de cor global (sem remoção de névoa)." : `Salvo: ${name}`);
    } catch (e) {
      toast("Falha ao exportar: " + e.message, true);
    } finally {
      busy(false);
    }
  };
}

/* ------------------------------------------------------------------ init */
buildSliders();
buildPresets();
$("#panel").classList.add("disabled");
