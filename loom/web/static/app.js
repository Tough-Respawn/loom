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
const md = (raw) => DOMPurify.sanitize(marked.parse(raw || ""));
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
let _raf = false;
function scheduleRender() {
  if (_raf) return;
  _raf = true;
  requestAnimationFrame(() => {
    _raf = false;
    render(html`<${App} />`, root);
    if (state.pin) window.scrollTo(0, document.body.scrollHeight);
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

function ToolPill({ it }) {
  const [open, setOpen] = useState(false);
  const hasDetail = !!it.detail;
  const status = it.pending
    ? "…"
    : (it.ok ? "✓ " : "✕ ") + (it.preview || "").split("\n")[0];
  return html`<div class=${"tool-chip" + (it.pending ? "" : it.ok ? " ok" : " ko") + (hasDetail ? " has-detail" : "")}>
    <div class="tool-row" onClick=${() => hasDetail && setOpen(!open)}>
      <span class="tool-main">${it.name || "outil"}${it.path ? " → " + it.path : ""}</span>
      <span class="tool-status">${status}</span>
      <span class="tool-caret">${hasDetail ? (open ? "▾" : "▸") : ""}</span>
    </div>
    ${it.stream && !open
      ? html`<pre class="tool-stream">${it.stream.slice(-1400)}</pre>`
      : null}
    ${hasDetail && open
      ? html`<pre class="tool-detail">${it.detail}</pre>`
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

function UserMsg({ it }) {
  const parts = Array.isArray(it.content) ? it.content : null;
  if (!parts) {
    return html`<div class="msg user">${it.content}</div>`;
  }
  return html`<div class="msg user">
    ${parts.map((p) =>
      p.type === "image_url"
        ? html`<img src=${p.image_url.url} alt="image" />`
        : html`<span>${p.text}</span>`,
    )}
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
}

function Item({ it }) {
  switch (it.kind) {
    case "user":
      return html`<${UserMsg} it=${it} />`;
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
  return state.timeline.map((it) => html`<${Item} key=${it.id} it=${it} />`);
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

async function sendChat(text, image) {
  state.pin = true;
  if (currentAbort) currentAbort.abort();
  const ac = new AbortController();
  currentAbort = ac;

  if (image) {
    const url = URL.createObjectURL(image);
    push({ kind: "user", content: [{ type: "text", text }, { type: "image_url", image_url: { url } }] });
  } else {
    push({ kind: "user", content: text });
  }
  const tools = {}; // callId -> item id
  let thinkId = null;
  let asstId = null;

  const fd = new FormData();
  fd.append("message", text);
  if (image) fd.append("image", image);

  const onEvent = (evt) => {
    switch (evt.type) {
      case "reasoning":
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
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        tools[evt.id] = tid;
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
        patch(tid, { name: evt.name, path: evt.path, ok: evt.ok, preview: evt.preview, detail: evt.detail, pending: false });
        break;
      }
      case "tool_request":
        push({ kind: "perm", callId: evt.id, name: evt.name, summary: evt.summary });
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
    if (ac === currentAbort) currentAbort = null;
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

const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const fileInput = document.getElementById("file");
let pendingImage = null;
const preview = document.getElementById("preview");
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
  b.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!confirm("Supprimer cette session ?")) return;
    await postForm("/session/delete", { id: b.dataset.id });
    location.reload();
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

function setImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  pendingImage = file;
  preview.src = URL.createObjectURL(file);
  previewWrap.style.display = "flex";
}
function clearImage() {
  pendingImage = null;
  previewWrap.style.display = "none";
}

// Scroll collant : on suit le bas seulement si l'utilisateur y est déjà.
function nearBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 120;
}
window.addEventListener("scroll", () => (state.pin = nearBottom()), { passive: true });

// --- formulaire : un seul chemin, l'agent tool-use (/chat) ---
const chatForm = document.getElementById("chat");

async function submitChat() {
  const text = input.value.trim();
  if (!text) return;
  history.push(text);
  histIdx = -1;
  const img = pendingImage;
  input.value = "";
  clearImage();
  sendBtn.textContent = "Stop";
  try {
    await sendChat(text, img);
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

// --- image ---
if (fileInput) fileInput.addEventListener("change", () => setImage(fileInput.files[0]));
const fileBtn = document.getElementById("fileBtn");
if (fileBtn) fileBtn.addEventListener("click", () => fileInput.click());
const clearImgBtn = document.getElementById("clearImgBtn");
if (clearImgBtn) clearImgBtn.addEventListener("click", clearImage);
window.addEventListener("paste", (e) => {
  for (const item of e.clipboardData.items) {
    if (item.type.startsWith("image/")) setImage(item.getAsFile());
  }
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

