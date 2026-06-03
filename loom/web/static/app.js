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

// Ticker : tant qu'une étape ou un verify tourne, on rafraîchit pour le timer live.
setInterval(() => {
  const live = state.timeline.some(
    (i) =>
      (i.kind === "step" && !i.done) ||
      (i.kind === "verify" && i.status === "pending"),
  );
  if (live) scheduleRender();
}, 100);

// ----------------------------------------------------------------------------
// Helpers d'étape (timer + tokens)
// ----------------------------------------------------------------------------
function metaText(s) {
  const secs = fmtSecs((s.done ? s.tEnd : Date.now()) - s.t0);
  // tokens = total RÉEL (somme des usage) + estimation LIVE du tour en cours
  // (≈1/chunk). `~` tant que le tour n'est pas confirmé par un usage.
  const total = s.tokens + s.live;
  const tok = total ? (s.live ? `~${total} tok` : `${total} tok`) : "";
  return "⏱ " + secs + (tok ? "  ·  " + tok : "");
}
function bumpLive(step) {
  // Incrémente l'estimation du tour courant (le ticker 100ms / les patches de
  // contenu déclenchent le re-render — pas besoin d'en forcer un ici).
  if (step) step.live++;
}
function addUsage(step, evt) {
  if (!step) return;
  // Tour confirmé : on entérine les vrais tokens et on remet l'estimation à zéro.
  step.tokens += evt.completion_tokens || 0;
  step.live = 0;
  scheduleRender();
}
function endStep(step) {
  if (step && !step.done) {
    step.done = true;
    step.tEnd = Date.now();
    scheduleRender();
  }
}

// ----------------------------------------------------------------------------
// Composants
// ----------------------------------------------------------------------------
function Step({ it }) {
  return html`<div class=${"step-hdr" + (it.done ? "" : " active")}>
    <span class="step-name">${it.label}</span>
    <span class="step-meta">${metaText(it)}</span>
  </div>`;
}

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
    ${hasDetail && open
      ? html`<pre class="tool-detail">${it.detail}</pre>`
      : null}
  </div>`;
}

function Verify({ it }) {
  if (it.status === "pending") {
    return html`<div class="verify-panel pending">
      <span class="verify-head">Vérification en cours…</span>
      <span class="vtimer"> ${fmtSecs(Date.now() - it.t0)}</span>
    </div>`;
  }
  const ok = it.status === "ok";
  const secs = fmtSecs((it.tEnd || it.t0) - it.t0);
  return html`<div class=${"verify-panel " + (ok ? "ok" : "ko")}>
    <div class="verify-head">
      ${ok
        ? "✓ Vérificateur : aucun défaut"
        : `✕ Vérificateur : ${(it.defects || []).length} défaut(s)`}
      ${"  ·  " + secs}
    </div>
    ${!ok && (it.defects || []).length
      ? html`<ul class="verify-list">
          ${it.defects.map(
            (d) => html`<li>
              <code>${d.location || ""}</code>
              <span class="vk"> [${d.kind || ""}] </span>${d.evidence || ""}
            </li>`,
          )}
        </ul>`
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

// Badge de routage (auto) : « → Build/Chat » + bouton pour forcer l'autre mode.
function RouteBadge({ it }) {
  const other = it.mode === "build" ? "chat" : "build";
  return html`<div class="run-info" style="align-self:center">
    ${it.mode === "build" ? "→ Build" : "→ Chat"}
    ${!it.overridden
      ? html`<button
          type="button"
          style="font-size:11px;padding:2px 8px;margin-left:8px;background:#1d2a22"
          onClick=${() => {
            patch(it.id, { overridden: true });
            dispatch(it.text, it.img, other);
          }}
        >
          plutôt ${other}
        </button>`
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
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.type = "button";
    btn.textContent = "copier";
    btn.onclick = () => {
      navigator.clipboard.writeText(pre.innerText.replace(/copier$/, ""));
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
    case "step":
      return html`<${Step} it=${it} />`;
    case "tool":
      return html`<${ToolPill} it=${it} />`;
    case "verify":
      return html`<${Verify} it=${it} />`;
    case "perm":
      return html`<${PermAsk} it=${it} />`;
    case "route":
      return html`<${RouteBadge} it=${it} />`;
    case "runinfo":
      return html`<div class="run-info">dossier de travail : ${it.workspace}</div>`;
    case "revision":
      return html`<div class="tool-chip rev">Révision ${it.n}, correction en cours</div>`;
    case "error":
      return html`<div class="msg assistant err">${it.message}</div>`;
    default:
      return null;
  }
}

function App() {
  if (!state.timeline.length) {
    return html`<div class="empty-state">
      Écris ci-dessous. En Auto, Loom route vers chat ou build (ou force le mode).
    </div>`;
  }
  return state.timeline.map((it) => html`<${Item} key=${it.id} it=${it} />`);
}

// ----------------------------------------------------------------------------
// Client SSE
// ----------------------------------------------------------------------------
async function streamSSE(url, fd, onEvent, signal) {
  const resp = await fetch(url, { method: "POST", body: fd, signal });
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
  const step = push({ kind: "step", label: "Réponse", t0: Date.now(), tokens: 0, live: 0, done: false });
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
        bumpLive(step);
        break;
      case "text":
        if (thinkId) patch(thinkId, { active: false });
        if (!asstId) asstId = push({ kind: "assistant", raw: "", done: false }).id;
        patch(asstId, { raw: get(asstId).raw + evt.text });
        bumpLive(step);
        break;
      case "tool_begin":
      case "tool_call": {
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        tools[evt.id] = tid;
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
      case "usage":
        addUsage(step, evt);
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
    endStep(step);
    if (ac === currentAbort) currentAbort = null;
  }
}

// ----------------------------------------------------------------------------
// Pipeline multi-agent
// ----------------------------------------------------------------------------
async function runPipeline(task, workspace) {
  state.pin = true;
  push({ kind: "user", content: task });

  const agents = {}; // agent -> { stepId, thinkId, asstId }
  const tools = {};
  let curVerifyId = null;
  const stepOf = (agent) => agents[agent] && get(agents[agent].stepId);

  const ensure = (agent, role, model) => {
    if (agents[agent]) return agents[agent];
    const stepId = push({
      kind: "step",
      label: role + " (" + model + ")",
      t0: Date.now(),
      tokens: 0,
      live: 0,
      done: false,
    }).id;
    agents[agent] = { stepId, thinkId: null, asstId: null, role };
    return agents[agent];
  };

  const fd = new FormData();
  fd.append("task", task);
  fd.append("mode", "build");
  if (workspace) fd.append("workspace", workspace);

  const onEvent = (evt) => {
    const a = evt.agent ? agents[evt.agent] : null;
    switch (evt.type) {
      case "run_info":
        push({ kind: "runinfo", workspace: evt.workspace });
        break;
      case "agent_start":
        ensure(evt.agent, evt.role, evt.model);
        break;
      case "reasoning": {
        const ag = a || ensure(evt.agent, evt.role || "?", evt.model || "?");
        if (!ag.thinkId) ag.thinkId = push({ kind: "think", role: ag.role, text: "", active: true }).id;
        patch(ag.thinkId, { text: get(ag.thinkId).text + evt.text, active: true });
        bumpLive(stepOf(evt.agent));
        break;
      }
      case "content": {
        const ag = a || ensure(evt.agent, "?", "?");
        if (ag.thinkId) patch(ag.thinkId, { active: false });
        if (!ag.asstId) ag.asstId = push({ kind: "assistant", raw: "", done: false }).id;
        patch(ag.asstId, { raw: get(ag.asstId).raw + evt.text });
        bumpLive(stepOf(evt.agent));
        break;
      }
      case "agent_done":
        if (a) {
          if (a.thinkId) patch(a.thinkId, { active: false });
          if (a.asstId) patch(a.asstId, { done: true });
          endStep(get(a.stepId));
        }
        break;
      case "usage":
        addUsage(stepOf(evt.agent), evt);
        break;
      case "tool_begin":
      case "tool_call": {
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        tools[evt.id] = tid;
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
      case "verify_start":
        curVerifyId = push({ kind: "verify", status: "pending", t0: Date.now() }).id;
        break;
      case "verify":
        if (curVerifyId)
          patch(curVerifyId, { status: evt.ok ? "ok" : "ko", defects: evt.defects || [], tEnd: Date.now() });
        curVerifyId = null;
        break;
      case "revision":
        push({ kind: "revision", n: evt.n });
        break;
      case "error":
        push({ kind: "error", message: "Erreur : " + evt.message });
        break;
    }
  };

  try {
    await streamSSE("/run", fd, onEvent);
  } catch (err) {
    push({ kind: "error", message: "Erreur : " + err.message });
  } finally {
    Object.values(agents).forEach((ag) => endStep(get(ag.stepId)));
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
// Mode (auto/chat/build) + dossier de travail (persistés en localStorage)
// ----------------------------------------------------------------------------
const modeSeg = document.getElementById("mode-seg");
const workdirPath = document.getElementById("workdir-path");
const workdirChip = document.getElementById("workdir-chip");

let loomMode = localStorage.loomMode || "auto";
// Défaut = workspace_dir rendu par le serveur dans #workdir-path (sinon init_json).
let loomWorkdir =
  localStorage.loomWorkdir ||
  INIT.workspace_dir ||
  (workdirPath && workdirPath.textContent.trim()) ||
  "";

function reflectMode() {
  if (!modeSeg) return;
  modeSeg.querySelectorAll("button[data-mode]").forEach((b) =>
    b.classList.toggle("on", b.dataset.mode === loomMode),
  );
}
function reflectWorkdir() {
  if (workdirPath) workdirPath.textContent = loomWorkdir;
}
reflectMode();
reflectWorkdir();

// segmented control : clic → mode actif persisté
if (modeSeg) {
  modeSeg.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-mode]");
    if (!btn) return;
    loomMode = btn.dataset.mode;
    localStorage.loomMode = loomMode;
    reflectMode();
  });
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

// --- formulaire unifié (auto-route chat vs build) ---
const chatForm = document.getElementById("chat");

// Route un message déjà lu vers /run (build) ou /chat (chat).
// `forced` non nul court-circuite le classifieur (boutons segmented / « plutôt … »).
async function dispatch(text, img, forced) {
  let mode = forced || loomMode;
  if (mode === "auto") {
    try {
      const r = await fetch("/classify", {
        method: "POST",
        body: new URLSearchParams({ message: text }),
      });
      mode = (await r.json()).mode;
    } catch {
      mode = "chat"; // défaut sûr si le classifieur échoue
    }
    // Badge déclaratif (item timeline) : « → mode » + « plutôt {autre} ».
    push({ kind: "route", mode, text, img });
  }
  if (mode === "build") {
    await runPipeline(text, loomWorkdir);
  } else {
    await sendChat(text, img);
  }
}

async function submitChat() {
  const text = input.value.trim();
  if (!text) return;
  history.push(text);
  histIdx = -1;
  const img = pendingImage;
  input.value = "";
  clearImage();
  sendBtn.textContent = "…";
  try {
    await dispatch(text, img);
  } finally {
    sendBtn.textContent = "Envoyer";
    input.focus();
  }
}
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
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
