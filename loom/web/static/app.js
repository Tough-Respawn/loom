// loom/web/static/app.js
// UI Loom — modèle déclaratif « état → vue » (Preact + htm, zéro build).
//
// Principe : UNE source de vérité (`state.timeline`, une liste d'items ordonnés).
// Les événements SSE mutent l'état (création/maj par `id`), puis `render(App)`.
// Plus aucune manipulation DOM manuelle → la classe de bugs (doublons/fantômes,
// pills non rattachées) disparaît par construction. Validé par `node --check`.

import {
  html,
  render,
  useState,
  useEffect,
  useRef,
} from "./preact-htm.js";

// marked + DOMPurify sont chargés en global (scripts classiques avant ce module).
marked.setOptions({ breaks: true });
// Protège les spans LaTeX ($…$, $$…$$, \(…\), \[…\]) du parseur markdown — sinon il mange
// \lim_{x}, \frac, etc. On remplace par des jetons avant marked, on restaure (échappés
// HTML pour bloquer toute injection) après. MathJax les rend en SVG dans enhance().
function _protectMath(raw) {
  const maths = [];
  const stash = (delim, tex) => {
    maths.push(delim + tex + delim);
    return `@@MATH${maths.length - 1}@@`;
  };
  let s = raw;
  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash("$$", t));
  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => stash("$$", t.trim()));
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => stash("$", t.trim()));
  s = s.replace(/(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)/g, (_, t) => stash("$", t));
  return { text: s, maths };
}
const _escHtml = (s) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
const md = (raw) => {
  const { text, maths } = _protectMath(raw || "");
  const out = DOMPurify.sanitize(marked.parse(text));
  return out.replace(/@@MATH(\d+)@@/g, (_, i) => _escHtml(maths[+i] || ""));
};
const esc = (s) => (s || "").replace(/</g, "&lt;");
const fmtSecs = (ms) => (ms / 1000).toFixed(1) + "s";

const INIT = JSON.parse(
  document.getElementById("loom-init")?.textContent || "{}",
);

// ----------------------------------------------------------------------------
// État + rendu
// ----------------------------------------------------------------------------
const state = {
  timeline: [], // items : {kind, id, ...}
  pin: true, // auto-scroll collant
};
let _seq = 0;
const uid = () => "i" + ++_seq;

const root = document.getElementById("messages");
const scrollWrap = document.getElementById("messages-wrap");
let _raf = false;
function scheduleRender() {
  if (_raf) return;
  _raf = true;
  requestAnimationFrame(() => {
    _raf = false;
    render(html`<${App} />`, root);
    if (state.pin && scrollWrap)
      scrollWrap.scrollTop = scrollWrap.scrollHeight;
  });
}

// Mutations de la timeline (toujours suivies d'un re-render).
function push(item) {
  item.id = item.id || uid();
  state.timeline.push(item);
  scheduleRender();
  return item;
}
function get(id) {
  return state.timeline.find((i) => i.id === id);
}
function patch(id, fields) {
  const it = get(id);
  if (it) {
    Object.assign(it, fields);
    scheduleRender();
  }
  return it;
}

// ----------------------------------------------------------------------------
// Composants
// ----------------------------------------------------------------------------
function Think({ it }) {
  if (!it.text) return null;
  return html`<details class=${"think" + (it.active ? " active" : "")} open=${it.active}>
    <summary>réflexion${it.role ? " " + it.role : ""}</summary>
    <div>${it.text}</div>
  </details>`;
}

function IORow({ tag, text }) {
  // Une ligne IN ou OUT : aperçu 1 ligne, clic pour déplier le bloc complet.
  const [open, setOpen] = useState(false);
  const full = text == null ? "" : String(text);
  if (!full) return null;
  const first = full.split("\n")[0];
  const multi = full.indexOf("\n") >= 0 || first.length > 90;
  return html`<div class="tool-io">
    <div class="tool-io-row" onClick=${() => multi && setOpen(!open)}>
      <span class=${"io-tag io-" + tag.toLowerCase()}>${tag}</span>
      <span class="io-line">${open ? "" : first}</span>
      <span class="tool-caret">${multi ? (open ? "▾" : "▸") : ""}</span>
    </div>
    ${open ? html`<pre class="tool-io-body">${full}</pre>` : null}
  </div>`;
}

function ToolPill({ it }) {
  // Vue façon IN/OUT : en-tête = nom du tool + état ; puis la commande/entrée (IN) et la
  // sortie (OUT), chacune dépliable pour voir tout, bloc par bloc.
  const inText = it.in_full != null ? it.in_full : it.cmd || it.path || "";
  const outText = it.out_full != null ? it.out_full : it.preview || "";
  const status = it.pending ? (it.chars ? it.chars + " car." : "…") : it.ok ? "✓" : "✕";
  return html`<div class=${"tool-chip" + (it.pending ? "" : it.ok ? " ok" : " ko")}>
    <div class="tool-head">
      <span class="tool-name">${it.name || "outil"}</span>
      <span class="tool-status">${status}</span>
    </div>
    <${IORow} tag="IN" text=${inText} />
    ${it.pending ? null : html`<${IORow} tag="OUT" text=${outText} />`}
    ${it.stream && it.pending
      ? html`<pre class="tool-stream">${it.stream.slice(-1400)}</pre>`
      : null}
  </div>`;
}

function PermAsk({ it }) {
  const decide = (approve) => {
    const fd = new FormData();
    fd.append("id", it.callId);
    fd.append("approve", approve ? "1" : "0");
    fetch("/tool_decision", { method: "POST", body: fd });
    patch(it.id, { decided: true, approved: approve });
  };
  return html`<div class=${"perm-ask" + (it.decided ? " decided" : "")}>
    <div>
      <b>${it.name || "outil"}</b> veut s'exécuter${it.summary
        ? html` : <code>${it.summary}</code>`
        : ""}${it.decided
        ? it.approved
          ? "  → autorisé"
          : "  → refusé"
        : ""}
    </div>
    ${!it.decided
      ? html`<div class="perm-btns">
          <button onClick=${() => decide(true)}>Autoriser</button>
          <button onClick=${() => decide(false)}>Refuser</button>
        </div>`
      : null}
  </div>`;
}

function UserMsg({ it, userIndex }) {
  // Bouton "repartir de la" : tronque la conversation apres ce message et
  // pre-remplit l'input pour re-editer. Comme l'edition de message ChatGPT/Claude.
  const doFork = async (e) => {
    e.stopPropagation();
    if (userIndex == null) return;
    const fd = new FormData();
    fd.append("user_index", String(userIndex));
    try {
      const r = await fetch("/fork", { method: "POST", body: fd });
      if (!r.ok) return;
      const j = await r.json();
      // Tronque la timeline UI : garde jusqu'a CET item user (inclus)
      const idx = state.timeline.indexOf(it);
      if (idx >= 0) state.timeline = state.timeline.slice(0, idx + 1);
      // Pre-remplit l'input
      const inp = document.getElementById("input");
      if (inp) {
        inp.value = j.text || "";
        inp.focus();
      }
      scheduleRender();
    } catch (err) {
      /* best-effort */
    }
  };

  const forkBtn = html`<button class="msg-fork" type="button" title="Repartir de ce message" onClick=${doFork}>↩ repartir</button>`;

  const parts = Array.isArray(it.content) ? it.content : null;
  if (!parts) {
    return html`<div class="msg user">${it.content}${forkBtn}</div>`;
  }
  return html`<div class="msg user">
    ${parts.map((p) =>
      p.type === "image_url"
        ? html`<img src=${p.image_url.url} alt="image" />`
        : html`<span>${p.text}</span>`,
    )}
    ${forkBtn}
  </div>`;
}

// Bulle assistant : texte brut pendant le stream, markdown sanitizé à la fin
// (+ boutons copier injectés une fois rendu, via effet).
function Assistant({ it }) {
  const ref = useRef(null);
  useEffect(() => {
    if (it.done && ref.current) enhance(ref.current, it.raw);
  }, [it.done, it.raw]);
  if (it.done) {
    return html`<div
      class="msg assistant"
      ref=${ref}
      dangerouslySetInnerHTML=${{ __html: md(it.raw) }}
    ></div>`;
  }
  return html`<div class="msg assistant streaming">${it.raw}</div>`;
}

function enhance(el, raw) {
  el.querySelectorAll("pre").forEach((pre) => {
    if (pre.querySelector(".copy-btn")) return;
    // Capture le texte du code AVANT d'ajouter le bouton : sinon innerText inclut
    // « copier » et l'ancien hack regex rognait un code finissant vraiment par « copier ».
    const code = (pre.querySelector("code") || pre).innerText;
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "copier";
    btn.onclick = () => {
      navigator.clipboard.writeText(code);
      btn.textContent = "copié ✓";
      setTimeout(() => (btn.textContent = "copier"), 1200);
    };
    pre.appendChild(btn);
  });
  if (!el.querySelector(".msg-copy")) {
    const btn = document.createElement("button");
    btn.className = "msg-copy";
    btn.type = "button";
    btn.textContent = "copier";
    btn.title = "Copier la réponse";
    btn.onclick = () => {
      navigator.clipboard.writeText(raw);
      btn.textContent = "✓";
      setTimeout(() => (btn.textContent = "copier"), 1200);
    };
    el.appendChild(btn);
  }
  // Rend les formules LaTeX en SVG sur le DOM réel (MathJax tex-svg), après le markdown.
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([el]).catch(() => {});
  }
}

function Item({ it, userIndex }) {
  switch (it.kind) {
    case "user":
      return html`<${UserMsg} it=${it} userIndex=${userIndex} />`;
    case "assistant":
      return html`<${Assistant} it=${it} />`;
    case "think":
      return html`<${Think} it=${it} />`;
    case "tool":
      return html`<${ToolPill} it=${it} />`;
    case "perm":
      return html`<${PermAsk} it=${it} />`;
    case "error":
      return html`<div class="msg assistant err">${it.message}</div>`;
    case "phase":
      return html`<div class="phase-sep">&#9658; ${it.name}${it.detail ? " — " + it.detail : ""}</div>`;
    default:
      return null;
  }
}

function App() {
  if (!state.timeline.length) {
    return html`<div class="empty-state">
      Écris une demande. Loom agit avec ses outils (lire, écrire, exécuter, chercher).
    </div>`;
  }
  let _ui = 0;
  return state.timeline.map((it) => {
    let userIndex = null;
    if (it.kind === "user") {
      userIndex = _ui++;
    }
    return html`<${Item} key=${it.id} it=${it} userIndex=${userIndex} />`;
  });
}

// ----------------------------------------------------------------------------
// Client SSE
// ----------------------------------------------------------------------------
async function streamSSE(url, fd, onEvent, signal) {
  const resp = await fetch(url, { method: fd ? "POST" : "GET", body: fd || undefined, signal });
  if (resp.status === 429) {
    onEvent({ type: "error", message: "Occupé : un échange est déjà en cours." });
    return;
  }
  if (resp.status === 400) {
    onEvent({ type: "error", message: "Requête invalide." });
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      onEvent(JSON.parse(part.slice(6)));
    }
  }
}

// ----------------------------------------------------------------------------
// Chat (1 tour, boucle tool-use)
// ----------------------------------------------------------------------------
let currentAbort = null;

async function sendChat(text, images) {
  state.pin = true;
  if (currentAbort) currentAbort.abort();
  const ac = new AbortController();
  currentAbort = ac;

  if (images && images.length) {
    const parts = [{ type: "text", text }];
    for (const im of images) parts.push({ type: "image_url", image_url: { url: URL.createObjectURL(im) } });
    push({ kind: "user", content: parts });
  } else {
    push({ kind: "user", content: text });
  }
  const tools = {}; // callId -> item id
  let thinkId = null;
  let asstId = null;
  let lastSent = 0,
    lastRecv = 0,
    lastTokS = null; // derniers compteurs envoyé/reçu/débit (pour figer à la fin)

  const fd = new FormData();
  fd.append("message", text);
  for (const im of images || []) fd.append("image", im); // multi-images : le back fait getlist

  setMetrics(0, 0, null); // "↑0 ↓0" pulsant dès l'envoi -> liveness immédiate avant le 1er token

  const onEvent = (evt) => {
    switch (evt.type) {
      case "reasoning":
        // Une NOUVELLE étape de réflexion clôt le texte précédent -> bulle séparée
        // (sinon le texte d'après s'empile dans la 1re bulle, en haut, au lieu d'en bas).
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        if (!thinkId) thinkId = push({ kind: "think", role: "", text: "", active: true }).id;
        patch(thinkId, { text: get(thinkId).text + evt.text, active: true });
        break;
      case "text":
        // Le texte clôt l'étape de raisonnement courante (et la prochaine en ouvrira une
        // neuve) -> on ferme la bulle au lieu de tout empiler dans une seule.
        if (thinkId) {
          patch(thinkId, { active: false });
          thinkId = null;
        }
        if (!asstId) asstId = push({ kind: "assistant", raw: "", done: false }).id;
        patch(asstId, { raw: get(asstId).raw + evt.text });
        break;
      case "tool_begin":
      case "tool_call": {
        // Un outil se déclenche = fin de l'étape de réflexion en cours. On clôt la bulle
        // pour que le raisonnement du PROCHAIN tour démarre une bulle SÉPARÉE (une par
        // étape : réfléchir -> agir -> réfléchir…), au lieu d'une seule qui grossit.
        if (thinkId) {
          patch(thinkId, { active: false });
          thinkId = null;
        }
        // …et le TEXTE en cours aussi : un outil = fin de l'étape, la narration d'après
        // doit démarrer une bulle neuve SOUS l'outil (pas remonter dans la 1re bulle).
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        tools[evt.id] = tid;
        break;
      }
      case "tool_args": {
        // Deltas d'arguments d'un tool_call (contenu de write_file, params de tout outil) :
        // on cumule la taille pour la voir grossir sur la pastille pendant la génération.
        // Le compteur ↓ global, lui, est piloté par l'event "metrics" du backend.
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        patch(tid, { chars: (get(tid).chars || 0) + (evt.n || 0) });
        break;
      }
      case "tool_stream": {
        // Activité live d'un outil streamant (sous-agent) : on accumule dans `stream`,
        // affiché dans la pastille tant qu'elle n'est pas dépliée.
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        patch(tid, { stream: (get(tid).stream || "") + evt.text });
        break;
      }
      case "tool_result": {
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name });
        patch(tid, { name: evt.name, path: evt.path, cmd: evt.cmd, ok: evt.ok, preview: evt.preview, detail: evt.detail, in_full: evt.in_full, out_full: evt.out_full, pending: false });
        break;
      }
      case "phase":
        // Séparateur de phase du harnais de réflexion. Si on ignore cet event, le run
        // reste lisible (les lignes du texte narrent déjà l'avancement) -> non bloquant.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "phase", name: evt.name, task: evt.task, detail: evt.detail });
        break;
      case "tool_request":
        push({ kind: "perm", callId: evt.id, name: evt.name, summary: evt.summary });
        break;
      case "workspace":
        // Le serveur a adopté le dossier de travail désigné dans le message : on
        // reflète la nouvelle pastille (l'utilisateur n'a rien eu à pointer).
        loomWorkdir = evt.path;
        try {
          localStorage.loomWorkdir = loomWorkdir;
        } catch (e) {
          /* localStorage indispo : sans effet */
        }
        reflectWorkdir();
        break;
      case "session_title": {
        // Le serveur a inféré un titre pour la session : on l'écrit dans la sidebar en
        // direct (sinon il n'apparaîtrait qu'au prochain rechargement). La liste est du
        // HTML rendu serveur -> on cible le bouton par son data-id.
        const btn = document.querySelector(`.sess-pick[data-id="${evt.id}"]`);
        if (btn) btn.textContent = evt.title;
        break;
      }
      case "metrics":
        // Compteur live piloté par le backend : envoyés ↑ (prompt réel) et reçus ↓
        // (génération, tool-calls inclus, réconciliés sur l'usage) + débit mesuré.
        lastSent = evt.sent;
        lastRecv = evt.recv;
        lastTokS = evt.tok_s;
        setMetrics(lastSent, lastRecv, lastTokS, {});
        break;
      case "error":
        push({ kind: "error", message: "Erreur : " + evt.message + " (Loom est-il lancé ?)" });
        break;
    }
  };

  try {
    await streamSSE("/chat", fd, onEvent, ac.signal);
    if (asstId) patch(asstId, { done: true });
  } catch (err) {
    if (err.name === "AbortError") {
      if (asstId) patch(asstId, { done: true });
    } else {
      push({ kind: "error", message: "Erreur : " + err.message + " (Loom est-il lancé ?)" });
    }
  } finally {
    if (thinkId) patch(thinkId, { active: false });
    // Ne fige le compteur que si CETTE génération est encore l'active (une nouvelle
    // soumission a déjà remis le compteur à zéro -> ne pas l'écraser depuis l'ancienne).
    if (ac === currentAbort) {
      currentAbort = null;
      setMetrics(lastSent, lastRecv, lastTokS, { done: true });
    }
  }
}

// ----------------------------------------------------------------------------
// Hydratation + câblage des contrôles (formulaires, image, historique, toggles)
// ----------------------------------------------------------------------------
(INIT.messages || []).forEach((m) =>
  state.timeline.push({
    kind: m.role === "user" ? "user" : "assistant",
    id: uid(),
    content: m.content,
    raw: typeof m.content === "string" ? m.content : "",
    done: true,
  }),
);
// Les messages assistant persistés portent leur markdown dans `content` (string).
state.timeline.forEach((it) => {
  if (it.kind === "assistant" && typeof it.content === "string") it.raw = it.content;
});
scheduleRender();

// MathJax charge en async : re-typeset les bulles déjà rendues (historique) une fois prêt.
function _typesetAll() {
  if (!(window.MathJax && window.MathJax.typesetPromise)) return;
  document
    .querySelectorAll(".msg.assistant")
    .forEach((el) => window.MathJax.typesetPromise([el]).catch(() => {}));
}
if (window.__mathjaxReady) _typesetAll();
else document.addEventListener("mathjax-ready", _typesetAll);

const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const fileInput = document.getElementById("file");
let pendingImages = []; // images jointes au prochain message (max MAX_IMAGES)
const MAX_IMAGES = 6;
const previewWrap = document.getElementById("previewWrap");

// ----------------------------------------------------------------------------
// Dossier de travail des outils (où read/write/shell agissent), persisté
// ----------------------------------------------------------------------------
const workdirPath = document.getElementById("workdir-path");
const workdirChip = document.getElementById("workdir-chip");

let loomWorkdir =
  localStorage.loomWorkdir ||
  INIT.workspace_dir ||
  (workdirPath && workdirPath.textContent.trim()) ||
  "";

function reflectWorkdir() {
  if (workdirPath) workdirPath.textContent = loomWorkdir;
}
reflectWorkdir();

// --- Compteur live de génération (tokens réels + débit mesuré, piloté par le backend) ---
const genMetrics = document.getElementById("gen-metrics");
const gmText = document.getElementById("gm-text");
function setMetrics(sent, recv, tokS, opts) {
  if (!genMetrics || !gmText) return;
  if (sent == null && recv == null) {
    genMetrics.hidden = true;
    genMetrics.classList.remove("done");
    return;
  }
  genMetrics.hidden = false;
  genMetrics.classList.toggle("done", !!(opts && opts.done));
  const rate = tokS != null ? ` · ${tokS} tok/s` : "";
  gmText.textContent = `↑ ${sent || 0} · ↓ ${recv || 0}${rate}`;
}

// --- drawer réglages ---
const settingsBtn = document.getElementById("settings-btn");
const drawer = document.getElementById("settings-drawer");
const drawerScrim = document.getElementById("drawer-scrim");
const drawerClose = document.getElementById("drawer-close");
function openDrawer() {
  if (drawer) drawer.hidden = false;
  if (drawerScrim) drawerScrim.hidden = false;
}
function closeDrawer() {
  if (drawer) drawer.hidden = true;
  if (drawerScrim) drawerScrim.hidden = true;
}
if (settingsBtn) settingsBtn.addEventListener("click", openDrawer);
if (drawerClose) drawerClose.addEventListener("click", closeDrawer);
if (drawerScrim) drawerScrim.addEventListener("click", closeDrawer);
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawer && !drawer.hidden) closeDrawer();
});

// --- sessions (liste / nouvelle / switch / suppression) ---
// Changer de session change toute la conversation et la timeline : on recharge la page,
// qui se re-rend avec la session active côté serveur (KISS et fiable).
async function postForm(url, data) {
  const fd = new FormData();
  for (const k in data) fd.append(k, data[k]);
  return fetch(url, { method: "POST", body: fd });
}
const sessionNew = document.getElementById("session-new");
if (sessionNew) {
  sessionNew.addEventListener("click", async () => {
    const wd = document.getElementById("workdir-path");
    await postForm("/session/new", { workspace: wd ? wd.textContent.trim() : "" });
    location.reload();
  });
}
document.querySelectorAll(".sess-pick").forEach((b) => {
  b.addEventListener("click", async () => {
    await postForm("/session/activate", { id: b.dataset.id });
    location.reload();
  });
});
document.querySelectorAll(".sess-del").forEach((b) => {
  b.addEventListener("click", (e) => {
    e.stopPropagation();
    showToast("Supprimer cette session ?", [
      {
        label: "Supprimer",
        kind: "danger",
        onClick: async () => {
          await postForm("/session/delete", { id: b.dataset.id });
          location.reload();
        },
      },
      { label: "Annuler" },
    ]);
  });
});

// --- sélecteur de dossier natif ---
const pickFolderBtn = document.getElementById("pick-folder-btn");
if (pickFolderBtn) {
  pickFolderBtn.addEventListener("click", async () => {
    try {
      const r = await fetch("/pick-folder", { method: "POST" });
      const j = await r.json();
      if (j.path) {
        loomWorkdir = j.path;
        localStorage.loomWorkdir = loomWorkdir;
        reflectWorkdir();
        // Applique le dossier à la SESSION ACTIVE tout de suite (sinon il ne
        // s'appliquerait qu'à la prochaine « Nouvelle session »).
        await postForm("/session/workspace", { workspace: j.path });
      } else if (j.error) {
        console.warn("pick-folder:", j.error);
        if (workdirChip) {
          workdirChip.classList.add("err");
          setTimeout(() => workdirChip.classList.remove("err"), 1500);
        }
      }
    } catch (err) {
      console.warn("pick-folder:", err);
    }
  });
}

function renderPreviews() {
  previewWrap.innerHTML = "";
  pendingImages.forEach((file, i) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = file.name || "image";
    const rm = document.createElement("button");
    rm.type = "button";
    rm.textContent = "✕";
    rm.title = "Retirer";
    rm.addEventListener("click", () => removeImage(i));
    wrap.append(img, rm);
    previewWrap.append(wrap);
  });
  previewWrap.style.display = pendingImages.length ? "flex" : "none";
}
function addImages(files) {
  for (const f of files) {
    if (!f || !f.type.startsWith("image/")) continue;
    if (pendingImages.length >= MAX_IMAGES) break; // limite : les suivantes sont ignorées
    pendingImages.push(f);
  }
  renderPreviews();
}
function removeImage(i) {
  pendingImages.splice(i, 1);
  renderPreviews();
}
function clearImages() {
  pendingImages = [];
  renderPreviews();
}

// Scroll collant : on suit le bas seulement si l'utilisateur y est déjà.
function nearBottom() {
  if (!scrollWrap) return true;
  return scrollWrap.scrollTop + scrollWrap.clientHeight >= scrollWrap.scrollHeight - 120;
}
if (scrollWrap) scrollWrap.addEventListener("scroll", () => (state.pin = nearBottom()), { passive: true });

// --- toggle sidebar mobile ---
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarEl = document.getElementById("sidebar");
if (sidebarToggle && sidebarEl) {
  sidebarToggle.addEventListener("click", () => sidebarEl.classList.toggle("open"));
}

// --- formulaire : un seul chemin, l'agent tool-use (/chat) ---
const chatForm = document.getElementById("chat");

async function submitChat() {
  const text = input.value.trim();
  if (!text) return;
  history.push(text);
  histIdx = -1;
  const imgs = pendingImages.slice();
  input.value = "";
  clearImages();
  sendBtn.textContent = "Stop";
  try {
    await sendChat(text, imgs);
  } finally {
    sendBtn.textContent = "Envoyer";
    input.focus();
  }
}

// Stop : coupe l'affichage côté client (abort) ET la génération côté serveur
// (/cancel pose cancel_event). Pas de message vide envoyé -> plus de 429.
function stopChat() {
  if (currentAbort) currentAbort.abort();
  fetch("/cancel", { method: "POST" }).catch(() => {});
}
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  // Pendant une génération, le bouton « Stop » arrête au lieu de soumettre.
  if (currentAbort) {
    stopChat();
    return;
  }
  submitChat();
});

// historique ↑/↓ + Entrée pour envoyer
const history = [];
let histIdx = -1;
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitChat();
    return;
  }
  if (e.key === "ArrowUp" && input.selectionStart === 0 && history.length) {
    e.preventDefault();
    if (histIdx === -1) histIdx = history.length;
    if (histIdx > 0) histIdx--;
    input.value = history[histIdx];
  } else if (e.key === "ArrowDown" && histIdx !== -1) {
    e.preventDefault();
    if (histIdx < history.length - 1) {
      histIdx++;
      input.value = history[histIdx];
    } else {
      histIdx = -1;
      input.value = "";
    }
  }
});

// --- images (plusieurs, jusqu'à MAX_IMAGES) ---
if (fileInput)
  fileInput.addEventListener("change", () => {
    addImages([...fileInput.files]);
    fileInput.value = ""; // permet de re-sélectionner le même fichier ensuite
  });
const fileBtn = document.getElementById("fileBtn");
if (fileBtn) fileBtn.addEventListener("click", () => fileInput.click());
window.addEventListener("paste", (e) => {
  const files = [];
  for (const item of e.clipboardData.items) {
    if (item.type.startsWith("image/")) files.push(item.getAsFile());
  }
  if (files.length) addImages(files);
});

// --- toggle réflexion ---
const thinkingCb = document.getElementById("thinking-cb");
if (thinkingCb) {
  thinkingCb.addEventListener("change", () => {
    const fd = new FormData();
    fd.append("thinking", thinkingCb.checked ? "1" : "0");
    fetch("/thinking", { method: "POST", body: fd });
  });
}

// --- reset (vide la conversation côté serveur + client) ---
const resetBtn = document.getElementById("reset-btn");
if (resetBtn) {
  resetBtn.addEventListener("click", async () => {
    await fetch("/reset", { method: "POST" });
    state.timeline = [];
    scheduleRender();
  });
}

// --- skills : activer/désactiver + éditer ---
// Toggle : POST /skills avec la liste des skills COCHÉS -> le serveur re-render le panneau,
// on remplace le noeud. La case « tous » coche/décoche l'ensemble. Délégation sur document
// pour survivre au remplacement du panneau.
function syncSkillsMaster() {
  const cbs = [...document.querySelectorAll("#skills-panel .skill-cb")];
  const master = document.getElementById("skills-all");
  if (!master) return;
  const on = cbs.filter((c) => c.checked).length;
  master.checked = cbs.length > 0 && on === cbs.length;
  master.indeterminate = on > 0 && on < cbs.length;
}
async function postSkillsToggle() {
  const cbs = [...document.querySelectorAll("#skills-panel .skill-cb")];
  const fd = new FormData();
  cbs.filter((c) => c.checked).forEach((c) => fd.append("skill", c.dataset.name));
  const r = await fetch("/skills", { method: "POST", body: fd });
  const html = await r.text();
  const panel = document.getElementById("skills-panel");
  const tmp = document.createElement("div");
  tmp.innerHTML = html.trim();
  const fresh = tmp.firstElementChild;
  if (fresh && panel) panel.replaceWith(fresh);
  syncSkillsMaster();
}
document.addEventListener("change", (e) => {
  const t = e.target;
  if (t.classList && t.classList.contains("skill-cb")) {
    postSkillsToggle();
  } else if (t.id === "skills-all") {
    const on = t.checked;
    document
      .querySelectorAll("#skills-panel .skill-cb")
      .forEach((c) => (c.checked = on));
    postSkillsToggle();
  } else if (t.id === "model-select") {
    // Changer de modèle déclenche le load/unload côté serveur (POST /model via HTMX).
    // On suit l'état machine ; scheduleMachineRefresh re-sonde le temps que le POST persiste.
    scheduleMachineRefresh();
  }
});
syncSkillsMaster();

// --- état du modèle sur la machine (chargé / chargement / libre / serveur off) ---
let machineTimer = null;
async function refreshMachineState() {
  const chip = document.getElementById("machine-chip");
  if (!chip) return "";
  let d;
  try {
    d = await (await fetch("/machine_state")).json();
  } catch {
    chip.textContent = "";
    return "";
  }
  // Moniteur système : visible UNIQUEMENT pour un modèle local (home), pas pour le cloud.
  setSysmonVisible(d.mode === "home");
  let cls = "",
    text = "";
  if (d.mode === "remote") {
    // Modèle distant : la machine ne porte pas le cerveau. On la veut libre.
    if (d.reachable && d.any_loaded) {
      cls = "busy";
      text = "machine · libération…";
    } else {
      cls = "free";
      text = "machine · libre (modèle distant)";
    }
  } else if (!d.reachable) {
    cls = "off";
    text = "machine · serveur local éteint";
  } else if (d.model_loaded) {
    cls = "on";
    text = "machine · " + d.model + " chargé";
  } else {
    cls = "busy";
    text = "machine · " + d.model + " chargement…";
  }
  chip.className = "machine-chip " + cls;
  chip.textContent = text;
  return cls;
}
// Rafraîchit maintenant puis re-sonde : les 3 premiers passages (à 800 ms) laissent le POST
// /model se persister avant de figer l'état ; ensuite on continue tant que c'est TRANSITOIRE
// (chargement / libération) car un gros modèle met du temps à (dé)charger. Borné à ~40 s.
function scheduleMachineRefresh() {
  if (machineTimer) clearTimeout(machineTimer);
  let tries = 0;
  const tick = async () => {
    const cls = await refreshMachineState();
    tries += 1;
    if ((tries < 3 || cls === "busy") && tries < 14) {
      machineTimer = setTimeout(tick, tries < 3 ? 800 : 3000);
    }
  };
  tick();
}
scheduleMachineRefresh();

// Éditeur de skill (drawer latéral)
const skDrawer = document.getElementById("skill-drawer");
const skScrim = document.getElementById("skill-scrim");
const skName = document.getElementById("skdr-name");
const skDesc = document.getElementById("skdr-desc");
const skBody = document.getElementById("skdr-body");
const skStatus = document.getElementById("skdr-status");
let skCurrent = null;

function openSkillDrawer() {
  if (skDrawer) skDrawer.hidden = false;
  if (skScrim) skScrim.hidden = false;
}
function closeSkillDrawer() {
  if (skDrawer) skDrawer.hidden = true;
  if (skScrim) skScrim.hidden = true;
  skCurrent = null;
}
async function openSkillEditor(name) {
  try {
    const r = await fetch("/skill?name=" + encodeURIComponent(name));
    const d = await r.json();
    if (!r.ok || d.error) {
      skStatus && (skStatus.textContent = d.error || "chargement impossible");
      return;
    }
    skCurrent = d.name;
    if (skName) skName.textContent = d.name;
    if (skDesc) skDesc.textContent = d.description || "";
    if (skBody) skBody.value = d.source || "";
    if (skStatus)
      skStatus.textContent = d.has_override
        ? "override de session actif"
        : d.editable_on_disk
          ? ""
          : "skill sans fichier sur disque";
    const gbtn = document.getElementById("skdr-save-global");
    if (gbtn) gbtn.disabled = !d.editable_on_disk;
    openSkillDrawer();
    if (skBody) skBody.focus();
  } catch (err) {
    if (skStatus) skStatus.textContent = "erreur : " + err;
  }
}
async function saveSkill(scope) {
  if (!skCurrent) return;
  const fd = new FormData();
  fd.append("name", skCurrent);
  fd.append("body", skBody ? skBody.value : "");
  fd.append("scope", scope);
  if (skStatus) skStatus.textContent = "enregistrement…";
  try {
    const r = await fetch("/skill/save", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || d.error) {
      if (skStatus) skStatus.textContent = d.error || "échec";
      return;
    }
    if (skStatus)
      skStatus.textContent =
        scope === "global"
          ? "enregistré pour toutes les sessions ✓"
          : "appliqué à cette session ✓";
    // Rafraîchit le panneau (badge override) sans changer les cases cochées.
    postSkillsToggle();
  } catch (err) {
    if (skStatus) skStatus.textContent = "erreur : " + err;
  }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest(".skill-name");
  if (b) openSkillEditor(b.dataset.name);
});
document
  .getElementById("skdr-close")
  ?.addEventListener("click", closeSkillDrawer);
if (skScrim) skScrim.addEventListener("click", closeSkillDrawer);
document
  .getElementById("skdr-save-session")
  ?.addEventListener("click", () => saveSkill("session"));
document
  .getElementById("skdr-save-global")
  ?.addEventListener("click", () => saveSkill("global"));
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && skDrawer && !skDrawer.hidden) closeSkillDrawer();
});

// --- toasts (remplacent confirm()/alert() natifs) ---
// showToast(message, actions?) : actions = [{label, kind?, onClick?}]. Sans action, c'est
// une simple notification auto-dismiss. Clic sur une action -> exécute onClick puis ferme.
function showToast(message, actions = [], { timeout = 8000 } = {}) {
  let host = document.getElementById("toast-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "toast-host";
    document.body.appendChild(host);
  }
  const toast = document.createElement("div");
  toast.className = "toast";
  const msg = document.createElement("span");
  msg.className = "toast-msg";
  msg.textContent = message;
  toast.appendChild(msg);
  let timer = null;
  const dismiss = () => {
    if (timer) clearTimeout(timer);
    toast.classList.add("leaving");
    setTimeout(() => toast.remove(), 160);
  };
  if (actions.length) {
    const wrap = document.createElement("div");
    wrap.className = "toast-actions";
    for (const a of actions) {
      const btn = document.createElement("button");
      btn.textContent = a.label;
      if (a.kind) btn.classList.add(a.kind);
      btn.addEventListener("click", () => {
        dismiss();
        if (a.onClick) a.onClick();
      });
      wrap.appendChild(btn);
    }
    toast.appendChild(wrap);
  }
  host.appendChild(toast);
  if (timeout) timer = setTimeout(dismiss, timeout);
  return dismiss;
}

// --- moniteur système live (CPU / RAM / GPU) : visible seulement pour un modèle LOCAL ---
// Piloté par refreshMachineState (mode 'home'). Sparklines dessinées sur <canvas>, barres
// pour la mémoire. Poll ~1,2 s seulement tant que visible (aucune charge quand caché/cloud).
const SM_N = 48; // points d'historique par sparkline
const smHist = { cpu: [], gpu: [] };
let sysmonTimer = null;

function smPush(arr, v) {
  arr.push(v);
  if (arr.length > SM_N) arr.shift();
}
function smDrawSpark(canvas, data, color) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  const step = w / (SM_N - 1);
  const x0 = w - (data.length - 1) * step;
  const y = (v) => h - 1 - (Math.max(0, Math.min(100, v)) / 100) * (h - 2);
  const path = () => {
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = x0 + i * step;
      i ? ctx.lineTo(x, y(v)) : ctx.moveTo(x, y(v));
    });
  };
  // aire sous la courbe (dégradé) + ligne
  path();
  ctx.lineTo(x0 + (data.length - 1) * step, h);
  ctx.lineTo(x0, h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + "40");
  grad.addColorStop(1, color + "00");
  ctx.fillStyle = grad;
  ctx.fill();
  path();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.3;
  ctx.lineJoin = "round";
  ctx.stroke();
}

const SM_GB = 1073741824;
const setTxt = (id, v) => {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
};
async function sysmonTick() {
  let d;
  try {
    d = await (await fetch("/sysmon")).json();
  } catch {
    return;
  }
  if (d.cpu != null) {
    smPush(smHist.cpu, d.cpu);
    setTxt("sm-cpu-val", Math.round(d.cpu) + "%");
  }
  smDrawSpark(document.getElementById("sm-cpu-spark"), smHist.cpu, "#5b9d7e");
  if (d.ram) {
    const fill = document.getElementById("sm-ram-fill");
    if (fill) fill.style.width = d.ram.percent + "%";
    setTxt(
      "sm-ram-val",
      (d.ram.used / SM_GB).toFixed(1) + "/" + Math.round(d.ram.total / SM_GB) + "G",
    );
  }
  const g = d.gpu;
  if (g) {
    setTxt("sm-gpu-name", (g.name || "GPU").replace(/^NVIDIA GeForce /, ""));
    if (g.util != null) {
      smPush(smHist.gpu, g.util);
      setTxt("sm-gpu-val", Math.round(g.util) + "%");
    }
    smDrawSpark(document.getElementById("sm-gpu-spark"), smHist.gpu, "#e8c07a");
    if (g.mem_total) {
      const fill = document.getElementById("sm-vram-fill");
      if (fill) fill.style.width = (g.mem_used / g.mem_total) * 100 + "%";
      setTxt(
        "sm-vram-val",
        (g.mem_used / 1024).toFixed(1) + "/" + (g.mem_total / 1024).toFixed(1) + "G",
      );
    }
    setTxt("sm-temp", g.temp != null ? Math.round(g.temp) + "°C" : "");
    setTxt("sm-power", g.power != null ? Math.round(g.power) + " W" : "");
  } else {
    setTxt("sm-gpu-name", "GPU indisponible");
  }
}
function setSysmonVisible(on) {
  const el = document.getElementById("sysmon");
  if (!el) return;
  if (on) {
    el.hidden = false;
    if (!sysmonTimer) {
      sysmonTick();
      sysmonTimer = setInterval(sysmonTick, 1200);
    }
  } else {
    el.hidden = true;
    if (sysmonTimer) {
      clearInterval(sysmonTimer);
      sysmonTimer = null;
    }
  }
}

