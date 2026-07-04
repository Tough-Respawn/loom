// loom/web/static/app.js
// UI Loom â€” modÃ¨le dÃ©claratif Â« Ã©tat â†’ vue Â» (Preact + htm, zÃ©ro build).
//
// Principe : UNE source de vÃ©ritÃ© (`state.timeline`, une liste d'items ordonnÃ©s).
// Les Ã©vÃ©nements SSE mutent l'Ã©tat (crÃ©ation/maj par `id`), puis `render(App)`.
// Plus aucune manipulation DOM manuelle â†’ la classe de bugs (doublons/fantÃ´mes,
// pills non rattachÃ©es) disparaÃ®t par construction. ValidÃ© par `node --check`.

import {
  html,
  render,
  useState,
  useEffect,
  useRef,
} from "./preact-htm.js";

// marked + DOMPurify sont chargÃ©s en global (scripts classiques avant ce module).
marked.setOptions({ breaks: true });
// ProtÃ¨ge les spans LaTeX ($â€¦$, $$â€¦$$, \(â€¦\), \[â€¦\]) du parseur markdown â€” sinon il mange
// \lim_{x}, \frac, etc. On remplace par des jetons avant marked, on restaure (Ã©chappÃ©s
// HTML pour bloquer toute injection) aprÃ¨s. MathJax les rend en SVG dans enhance().
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
// Ã‰tat + rendu
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
    <summary>rÃ©flexion${it.role ? " " + it.role : ""}</summary>
    <div>${it.text}</div>
  </details>`;
}

function IORow({ tag, text }) {
  // Une ligne IN ou OUT : aperÃ§u 1 ligne, clic pour dÃ©plier le bloc complet.
  const [open, setOpen] = useState(false);
  const full = text == null ? "" : String(text);
  if (!full) return null;
  const first = full.split("\n")[0];
  const multi = full.indexOf("\n") >= 0 || first.length > 90;
  return html`<div class="tool-io">
    <div class="tool-io-row" onClick=${() => multi && setOpen(!open)}>
      <span class=${"io-tag io-" + tag.toLowerCase()}>${tag}</span>
      <span class="io-line">${open ? "" : first}</span>
      <span class="tool-caret">${multi ? (open ? "â–¾" : "â–¸") : ""}</span>
    </div>
    ${open ? html`<pre class="tool-io-body">${full}</pre>` : null}
  </div>`;
}

function ToolPill({ it }) {
  // Vue faÃ§on IN/OUT : en-tÃªte = nom du tool + Ã©tat ; puis la commande/entrÃ©e (IN) et la
  // sortie (OUT), chacune dÃ©pliable pour voir tout, bloc par bloc.
  const inText = it.in_full != null ? it.in_full : it.cmd || it.path || "";
  const outText = it.out_full != null ? it.out_full : it.preview || "";
  const status = it.pending ? (it.chars ? it.chars + " car." : "â€¦") : it.ok ? "âœ“" : "âœ•";
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
      <b>${it.name || "outil"}</b> veut s'exÃ©cuter${it.summary
        ? html` : <code>${it.summary}</code>`
        : ""}${it.decided
        ? it.approved
          ? "  â†’ autorisÃ©"
          : "  â†’ refusÃ©"
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

// Bulle assistant : texte brut pendant le stream, markdown sanitizÃ© Ã  la fin
// (+ boutons copier injectÃ©s une fois rendu, via effet).
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
    // Â« copier Â» et l'ancien hack regex rognait un code finissant vraiment par Â« copier Â».
    const code = (pre.querySelector("code") || pre).innerText;
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "copier";
    btn.onclick = () => {
      navigator.clipboard.writeText(code);
      btn.textContent = "copiÃ© âœ“";
      setTimeout(() => (btn.textContent = "copier"), 1200);
    };
    pre.appendChild(btn);
  });
  if (!el.querySelector(".msg-copy")) {
    const btn = document.createElement("button");
    btn.className = "msg-copy";
    btn.type = "button";
    btn.textContent = "copier";
    btn.title = "Copier la rÃ©ponse";
    btn.onclick = () => {
      navigator.clipboard.writeText(raw);
      btn.textContent = "âœ“";
      setTimeout(() => (btn.textContent = "copier"), 1200);
    };
    el.appendChild(btn);
  }
  // Rend les formules LaTeX en SVG sur le DOM rÃ©el (MathJax tex-svg), aprÃ¨s le markdown.
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
      return html`<div class="phase-sep">&#9658; ${it.name}${it.detail ? " â€” " + it.detail : ""}</div>`;
    default:
      return null;
  }
}

function App() {
  if (!state.timeline.length) {
    return html`<div class="empty-state">
      Ã‰cris une demande. Loom agit avec ses outils (lire, Ã©crire, exÃ©cuter, chercher).
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
    onEvent({ type: "error", message: "OccupÃ© : un Ã©change est dÃ©jÃ  en cours." });
    return;
  }
  if (resp.status === 400) {
    onEvent({ type: "error", message: "RequÃªte invalide." });
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
    lastTokS = null; // derniers compteurs envoyÃ©/reÃ§u/dÃ©bit (pour figer Ã  la fin)

  const fd = new FormData();
  fd.append("message", text);
  for (const im of images || []) fd.append("image", im); // multi-images : le back fait getlist

  setMetrics(0, 0, null); // "â†‘0 â†“0" pulsant dÃ¨s l'envoi -> liveness immÃ©diate avant le 1er token

  const onEvent = (evt) => {
    switch (evt.type) {
      case "reasoning":
        // Une NOUVELLE Ã©tape de rÃ©flexion clÃ´t le texte prÃ©cÃ©dent -> bulle sÃ©parÃ©e
        // (sinon le texte d'aprÃ¨s s'empile dans la 1re bulle, en haut, au lieu d'en bas).
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        if (!thinkId) thinkId = push({ kind: "think", role: "", text: "", active: true }).id;
        patch(thinkId, { text: get(thinkId).text + evt.text, active: true });
        break;
      case "text":
        // Le texte clÃ´t l'Ã©tape de raisonnement courante (et la prochaine en ouvrira une
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
        // Un outil se dÃ©clenche = fin de l'Ã©tape de rÃ©flexion en cours. On clÃ´t la bulle
        // pour que le raisonnement du PROCHAIN tour dÃ©marre une bulle SÃ‰PARÃ‰E (une par
        // Ã©tape : rÃ©flÃ©chir -> agir -> rÃ©flÃ©chirâ€¦), au lieu d'une seule qui grossit.
        if (thinkId) {
          patch(thinkId, { active: false });
          thinkId = null;
        }
        // â€¦et le TEXTE en cours aussi : un outil = fin de l'Ã©tape, la narration d'aprÃ¨s
        // doit dÃ©marrer une bulle neuve SOUS l'outil (pas remonter dans la 1re bulle).
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        tools[evt.id] = tid;
        break;
      }
      case "tool_args": {
        // Deltas d'arguments d'un tool_call (contenu de write_file, params de tout outil) :
        // on cumule la taille pour la voir grossir sur la pastille pendant la gÃ©nÃ©ration.
        // Le compteur â†“ global, lui, est pilotÃ© par l'event "metrics" du backend.
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        patch(tid, { chars: (get(tid).chars || 0) + (evt.n || 0) });
        break;
      }
      case "tool_stream": {
        // ActivitÃ© live d'un outil streamant (sous-agent) : on accumule dans `stream`,
        // affichÃ© dans la pastille tant qu'elle n'est pas dÃ©pliÃ©e.
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
        // SÃ©parateur de phase du harnais de rÃ©flexion. Si on ignore cet event, le run
        // reste lisible (les lignes du texte narrent dÃ©jÃ  l'avancement) -> non bloquant.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "phase", name: evt.name, task: evt.task, detail: evt.detail });
        break;
      case "tool_request":
        push({ kind: "perm", callId: evt.id, name: evt.name, summary: evt.summary });
        break;
      case "workspace":
        // Le serveur a adoptÃ© le dossier de travail dÃ©signÃ© dans le message : on
        // reflÃ¨te la nouvelle pastille (l'utilisateur n'a rien eu Ã  pointer).
        loomWorkdir = evt.path;
        try {
          localStorage.loomWorkdir = loomWorkdir;
        } catch (e) {
          /* localStorage indispo : sans effet */
        }
        reflectWorkdir();
        break;
      case "session_title": {
        // Le serveur a infÃ©rÃ© un titre pour la session : on l'Ã©crit dans la sidebar en
        // direct (sinon il n'apparaÃ®trait qu'au prochain rechargement). La liste est du
        // HTML rendu serveur -> on cible le bouton par son data-id.
        const btn = document.querySelector(`.sess-pick[data-id="${evt.id}"]`);
        if (btn) btn.textContent = evt.title;
        break;
      }
      case "metrics":
        // Compteur live pilotÃ© par le backend : envoyÃ©s â†‘ (prompt rÃ©el) et reÃ§us â†“
        // (gÃ©nÃ©ration, tool-calls inclus, rÃ©conciliÃ©s sur l'usage) + dÃ©bit mesurÃ©.
        lastSent = evt.sent;
        lastRecv = evt.recv;
        lastTokS = evt.tok_s;
        setMetrics(lastSent, lastRecv, lastTokS, {});
        break;
      case "error":
        push({ kind: "error", message: "Erreur : " + evt.message + " (Loom est-il lancÃ© ?)" });
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
      push({ kind: "error", message: "Erreur : " + err.message + " (Loom est-il lancÃ© ?)" });
    }
  } finally {
    if (thinkId) patch(thinkId, { active: false });
    // Ne fige le compteur que si CETTE gÃ©nÃ©ration est encore l'active (une nouvelle
    // soumission a dÃ©jÃ  remis le compteur Ã  zÃ©ro -> ne pas l'Ã©craser depuis l'ancienne).
    if (ac === currentAbort) {
      currentAbort = null;
      setMetrics(lastSent, lastRecv, lastTokS, { done: true });
    }
  }
}

// ----------------------------------------------------------------------------
// Hydratation + cÃ¢blage des contrÃ´les (formulaires, image, historique, toggles)
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
// Les messages assistant persistÃ©s portent leur markdown dans `content` (string).
state.timeline.forEach((it) => {
  if (it.kind === "assistant" && typeof it.content === "string") it.raw = it.content;
});
scheduleRender();

// MathJax charge en async : re-typeset les bulles dÃ©jÃ  rendues (historique) une fois prÃªt.
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
// Dossier de travail des outils (oÃ¹ read/write/shell agissent), persistÃ©
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

// --- Compteur live de gÃ©nÃ©ration (tokens rÃ©els + dÃ©bit mesurÃ©, pilotÃ© par le backend) ---
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
  const rate = tokS != null ? ` Â· ${tokS} tok/s` : "";
  gmText.textContent = `â†‘ ${sent || 0} Â· â†“ ${recv || 0}${rate}`;
}

// --- drawer rÃ©glages ---
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
// qui se re-rend avec la session active cÃ´tÃ© serveur (KISS et fiable).
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
  b.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!confirm("Supprimer cette session ?")) return;
    await postForm("/session/delete", { id: b.dataset.id });
    location.reload();
  });
});

// --- sÃ©lecteur de dossier natif ---
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
        // Applique le dossier Ã  la SESSION ACTIVE tout de suite (sinon il ne
        // s'appliquerait qu'Ã  la prochaine Â« Nouvelle session Â»).
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
    rm.textContent = "âœ•";
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
    if (pendingImages.length >= MAX_IMAGES) break; // limite : les suivantes sont ignorÃ©es
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

// Scroll collant : on suit le bas seulement si l'utilisateur y est dÃ©jÃ .
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

// Stop : coupe l'affichage cÃ´tÃ© client (abort) ET la gÃ©nÃ©ration cÃ´tÃ© serveur
// (/cancel pose cancel_event). Pas de message vide envoyÃ© -> plus de 429.
function stopChat() {
  if (currentAbort) currentAbort.abort();
  fetch("/cancel", { method: "POST" }).catch(() => {});
}
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  // Pendant une gÃ©nÃ©ration, le bouton Â« Stop Â» arrÃªte au lieu de soumettre.
  if (currentAbort) {
    stopChat();
    return;
  }
  submitChat();
});

// historique â†‘/â†“ + EntrÃ©e pour envoyer
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

// --- images (plusieurs, jusqu'Ã  MAX_IMAGES) ---
if (fileInput)
  fileInput.addEventListener("change", () => {
    addImages([...fileInput.files]);
    fileInput.value = ""; // permet de re-sÃ©lectionner le mÃªme fichier ensuite
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

// --- toggle rÃ©flexion ---
const thinkingCb = document.getElementById("thinking-cb");
if (thinkingCb) {
  thinkingCb.addEventListener("change", () => {
    const fd = new FormData();
    fd.append("thinking", thinkingCb.checked ? "1" : "0");
    fetch("/thinking", { method: "POST", body: fd });
  });
}

// --- reset (vide la conversation cÃ´tÃ© serveur + client) ---
const resetBtn = document.getElementById("reset-btn");
if (resetBtn) {
  resetBtn.addEventListener("click", async () => {
    await fetch("/reset", { method: "POST" });
    state.timeline = [];
    scheduleRender();
  });
}

