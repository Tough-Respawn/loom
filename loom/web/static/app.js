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
// Multi-ONGLETS : chaque session ouverte = un onglet avec SA timeline, SA génération (flux
// concurrent), SON Stop et SON compteur. La sidebar garde TOUTES les sessions ; fermer un
// onglet ne supprime pas la session. `state.active` = onglet affiché.
const state = {
  tabs: {}, // sid -> { sid, title, timeline:[], streaming, abort, model, thinking, workspace, meter, metrics }
  order: [], // ordre des onglets ouverts
  active: null, // sid de l'onglet affiché
  pin: true, // auto-scroll collant
};
let _seq = 0;
const uid = () => "i" + ++_seq;
function tab(sid) {
  return state.tabs[sid];
}
function activeTab() {
  return state.tabs[state.active] || null;
}

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
    renderMsgNav(); // navigateur de prompts (liste des messages user de l'onglet actif)
  });
}

// --- Navigateur de prompts : liste les messages UTILISATEUR de l'onglet actif à droite.
// Survol = texte complet (title + panneau élargi), clic = saut vers le message. ---
function _userMsgTexts() {
  const t = activeTab();
  if (!t) return [];
  const out = [];
  for (const it of t.timeline) {
    if (it.kind !== "user") continue;
    const c = it.content;
    out.push(
      typeof c === "string"
        ? c
        : Array.isArray(c)
          ? (c.find((p) => p.type === "text") || {}).text || ""
          : "",
    );
  }
  return out;
}
function renderMsgNav() {
  const nav = document.getElementById("msg-nav");
  if (!nav) return;
  const texts = _userMsgTexts();
  if (texts.length < 2) {
    // inutile pour 0-1 message : on cache
    nav.hidden = true;
    nav.innerHTML = "";
    return;
  }
  nav.hidden = false;
  nav.innerHTML = "";
  texts.forEach((txt, i) => {
    const clean = (txt || "(vide)").replace(/\s+/g, " ").trim();
    const b = document.createElement("button");
    b.type = "button";
    b.className = "mn-item";
    b.title = clean;
    // Replié = juste un point ; le texte n'apparaît qu'au survol du panneau (déplié).
    const span = document.createElement("span");
    span.className = "mn-text";
    span.textContent = clean;
    const dot = document.createElement("span");
    dot.className = "mn-dot";
    b.append(span, dot);
    b.addEventListener("click", () => {
      const el = document.querySelectorAll("#messages .msg.user")[i];
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      nav.querySelectorAll(".mn-item.active").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      el.classList.add("mn-flash");
      setTimeout(() => el.classList.remove("mn-flash"), 900);
    });
    nav.append(b);
  });
}

// Mutations liées à la timeline d'un ONGLET précis. Un re-render n'a lieu que si cet onglet
// est actif (les onglets en arrière-plan génèrent sans redessiner l'écran). Deux flux
// concurrents mutent chacun SA timeline sans se marcher dessus.
function opsFor(sid) {
  const tl = () => (state.tabs[sid] ? state.tabs[sid].timeline : []);
  const get = (id) => tl().find((i) => i.id === id);
  const push = (item) => {
    item.id = item.id || uid();
    tl().push(item);
    if (sid === state.active) scheduleRender();
    return item;
  };
  const patch = (id, fields) => {
    const it = get(id);
    if (it) {
      Object.assign(it, fields);
      if (sid === state.active) scheduleRender();
    }
    return it;
  };
  return { get, push, patch };
}
// Ops de l'onglet ACTIF (hydratation, rendu direct).
const push = (item) => opsFor(state.active).push(item);
const get = (id) => opsFor(state.active).get(id);
const patch = (id, fields) => opsFor(state.active).patch(id, fields);

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
      // Tronque la timeline de l'onglet actif : garde jusqu'a CET item user (inclus)
      const at = activeTab();
      if (at) {
        const idx = at.timeline.indexOf(it);
        if (idx >= 0) at.timeline = at.timeline.slice(0, idx + 1);
      }
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
    case "notice":
      return html`<div class="notice-line">${it.text}</div>`;
    default:
      return null;
  }
}

function App() {
  const t = activeTab();
  if (!t || !t.timeline.length) {
    return html`<div class="empty-state">
      Écris une demande. Loom agit avec ses outils (lire, écrire, exécuter, chercher).
    </div>`;
  }
  let _ui = 0;
  return t.timeline.map((it) => {
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
async function sendChat(sid, text, images) {
  const t = tab(sid);
  if (!t) return;
  state.pin = true;
  // Interrompt UNIQUEMENT la génération de CET onglet (les autres continuent).
  if (t.abort) t.abort.abort();
  const ac = new AbortController();
  t.abort = ac;
  t.streaming = true;
  renderTabs();
  if (sid === state.active) syncComposer();

  // Ops liées à la timeline de CET onglet (les events du flux mutent SA timeline, même s'il
  // n'est pas à l'écran).
  const { push, get, patch } = opsFor(sid);

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
  fd.append("session_id", sid); // cible la session de l'onglet (génération concurrente)
  for (const im of images || []) fd.append("image", im); // multi-images : le back fait getlist

  if (sid === state.active) setMetrics(0, 0, null); // liveness immédiate si onglet affiché

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
      case "notice":
        // Signal du harnais (ex. « modèle local occupé, mise en file ») : ligne discrète.
        push({ kind: "notice", text: evt.text });
        break;
      case "workspace":
        // Le serveur a adopté le dossier de travail : mémorisé sur l'onglet, pastille MAJ
        // seulement si c'est l'onglet affiché.
        t.workspace = evt.path;
        if (sid === state.active) {
          loomWorkdir = evt.path;
          try {
            localStorage.loomWorkdir = loomWorkdir;
          } catch (e) {
            /* localStorage indispo : sans effet */
          }
          reflectWorkdir();
        }
        break;
      case "session_title": {
        // Titre inféré : MAJ de l'onglet, de la sidebar et de la barre d'onglets.
        if (t) t.title = evt.title;
        const btn = document.querySelector(`.sess-pick[data-id="${evt.id}"]`);
        if (btn) btn.textContent = evt.title;
        renderTabs();
        break;
      }
      case "metrics":
        // Compteur live du backend : mémorisé sur l'onglet, affiché seulement si actif.
        lastSent = evt.sent;
        lastRecv = evt.recv;
        lastTokS = evt.tok_s;
        t.metrics = { sent: lastSent, recv: lastRecv, tokS: lastTokS };
        if (sid === state.active) setMetrics(lastSent, lastRecv, lastTokS, {});
        break;
      case "totals":
        // Cumul RÉEL de la session : mémorisé sur l'onglet, compteur affiché si actif.
        t.meter = evt;
        if (sid === state.active) updateUsageMeter(evt);
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
    // Fin de CE flux : si c'est encore le flux courant de l'onglet (pas remplacé par une
    // nouvelle soumission), on marque l'onglet non-générant et on fige son compteur.
    if (t.abort === ac) {
      t.abort = null;
      t.streaming = false;
      renderTabs();
      if (sid === state.active) {
        setMetrics(lastSent, lastRecv, lastTokS, { done: true });
        syncComposer();
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Onglets (barre navigateur) : ouvrir / activer / fermer. La sidebar garde TOUTES les
// sessions ; un onglet = une session ouverte ; fermer un onglet ne supprime pas la session.
// ----------------------------------------------------------------------------
const tabbarEl = document.getElementById("tabbar");

function renderTabs() {
  if (!tabbarEl) return;
  tabbarEl.hidden = state.order.length === 0;
  tabbarEl.innerHTML = "";
  for (const sid of state.order) {
    const t = state.tabs[sid];
    if (!t) continue;
    const el = document.createElement("div");
    el.className = "tab" + (sid === state.active ? " active" : "");
    el.title = t.title || "session";
    const dot = document.createElement("span");
    dot.className = "tab-dot " + (t.streaming ? "gen" : "idle");
    const title = document.createElement("span");
    title.className = "tab-title";
    title.textContent = t.title || "session";
    const x = document.createElement("button");
    x.className = "tab-x";
    x.type = "button";
    x.textContent = "✕";
    x.title = "Fermer l'onglet (la session reste dans la sidebar)";
    x.addEventListener("click", (e) => {
      e.stopPropagation();
      closeTab(sid);
    });
    el.append(dot, title, x);
    el.addEventListener("click", () => activateTab(sid));
    tabbarEl.append(el);
  }
  const plus = document.createElement("button");
  plus.className = "tab-new";
  plus.type = "button";
  plus.textContent = "+";
  plus.title = "Nouvelle session";
  plus.addEventListener("click", newSessionTab);
  tabbarEl.append(plus);
}

// Textes des messages UTILISATEUR d'une conversation -> amorce l'historique ↑/↓ de l'onglet
// (pour rappeler les prompts déjà envoyés, pas seulement ceux tapés depuis le chargement).
function _userTexts(messages) {
  const out = [];
  for (const m of messages || []) {
    if (m.role !== "user" && m.kind !== "user") continue;
    const c = m.content;
    if (typeof c === "string") out.push(c);
    else if (Array.isArray(c)) {
      const tp = c.find((p) => p.type === "text");
      if (tp && tp.text) out.push(tp.text);
    }
  }
  return out;
}

function _hydrateTimeline(t, messages) {
  (messages || []).forEach((m) =>
    t.timeline.push({
      kind: m.role === "user" ? "user" : "assistant",
      id: uid(),
      content: m.content,
      raw: typeof m.content === "string" ? m.content : "",
      done: true,
    }),
  );
}

async function openTab(sid) {
  if (!state.tabs[sid]) {
    let d;
    try {
      d = await (await fetch("/session_state?id=" + encodeURIComponent(sid))).json();
    } catch {
      return;
    }
    const t = {
      sid,
      title: d.title || "session",
      timeline: [],
      streaming: false,
      abort: null,
      model: d.model || "",
      thinking: d.thinking !== false,
      workspace: d.workspace || "",
      meter: d.usage_totals || null,
      metrics: null,
      draft: "",
      history: _userTexts(d.messages),
      histIdx: -1,
    };
    _hydrateTimeline(t, d.messages);
    state.tabs[sid] = t;
    state.order.push(sid);
  }
  activateTab(sid);
}

function activateTab(sid) {
  const t = state.tabs[sid];
  if (!t) return;
  // Brouillon PAR ONGLET : on sauve le texte NON envoyé de l'onglet qu'on quitte, on
  // chargera celui de l'onglet ouvert (plus bas). Sinon la zone de saisie est partagée.
  const inputEl = document.getElementById("input");
  const prev = state.tabs[state.active];
  if (prev && inputEl) prev.draft = inputEl.value;
  state.active = sid;
  // Le serveur suit ce focus (pour /model, /tools, /reset, /pick-folder qui opèrent sur _cur).
  postForm("/session/activate", { id: sid }).catch(() => {});
  // Synchronise les contrôles de la sidebar/topbar à cet onglet.
  const sel = document.getElementById("model-select");
  if (sel && t.model) sel.value = t.model;
  const think = document.getElementById("thinking-cb");
  if (think) think.checked = !!t.thinking;
  if (t.workspace) {
    loomWorkdir = t.workspace;
    try {
      localStorage.loomWorkdir = t.workspace;
    } catch (e) {
      /* sans effet */
    }
    reflectWorkdir();
  }
  if (t.meter) updateUsageMeter(t.meter);
  if (t.metrics)
    setMetrics(t.metrics.sent, t.metrics.recv, t.metrics.tokS, { done: !t.streaming });
  else setMetrics(null, null, null);
  // Surligne la session active dans la sidebar.
  document
    .querySelectorAll(".session-item")
    .forEach((li) => li.classList.remove("active"));
  const li = document
    .querySelector(`.sess-pick[data-id="${sid}"]`)
    ?.closest(".session-item");
  if (li) li.classList.add("active");
  // Charge le brouillon de l'onglet ouvert (vide s'il n'a rien tapé).
  if (inputEl) {
    inputEl.value = t.draft || "";
    autosize();
  }
  state.pin = true;
  scheduleRender();
  renderTabs();
  syncComposer();
}

function closeTab(sid) {
  const t = state.tabs[sid];
  if (t && t.abort) t.abort.abort(); // stoppe le flux de l'onglet fermé (la session reste)
  delete state.tabs[sid];
  state.order = state.order.filter((s) => s !== sid);
  if (state.active === sid) {
    state.active = state.order[state.order.length - 1] || null;
    if (state.active) activateTab(state.active);
    else {
      scheduleRender();
      renderTabs();
    }
  } else {
    renderTabs();
  }
}

async function newSessionTab() {
  const wd = document.getElementById("workdir-path");
  const r = await postForm("/session/new", {
    workspace: wd ? wd.textContent.trim() : "",
  });
  const d = await r.json();
  state.tabs[d.id] = {
    sid: d.id,
    title: d.title || "session",
    timeline: [],
    streaming: false,
    abort: null,
    model: "",
    thinking: true,
    workspace: d.workspace || "",
    meter: null,
    metrics: null,
    draft: "",
    history: [],
    histIdx: -1,
  };
  state.order.push(d.id);
  addSidebarSession(d);
  activateTab(d.id);
}

function addSidebarSession(d) {
  const list = document.querySelector(".session-list");
  if (!list || list.querySelector(`.sess-pick[data-id="${d.id}"]`)) return;
  const li = document.createElement("li");
  li.className = "session-item";
  li.innerHTML =
    `<button type="button" class="sess-pick" data-id="${d.id}" title="${d.workspace || ""}">${d.title || "session"}</button>` +
    `<button type="button" class="sess-del" data-id="${d.id}" title="Supprimer">✕</button>`;
  list.prepend(li);
}

// ----------------------------------------------------------------------------
// Hydratation + câblage des contrôles (formulaires, image, historique, toggles)
// ----------------------------------------------------------------------------
// Onglet de départ = la session active (hydratée depuis INIT). La sidebar rendue par le
// serveur liste toutes les sessions ; on n'ouvre QUE l'active au chargement.
{
  const sid = INIT.active_session;
  if (sid) {
    const t0 = {
      sid,
      title: INIT.title || "session",
      timeline: [],
      streaming: false,
      abort: null,
      model: INIT.model || "",
      thinking: INIT.thinking !== false,
      workspace: INIT.workspace || "",
      meter: INIT.usage_totals || null,
      metrics: null,
      draft: "",
      history: _userTexts(INIT.messages),
      histIdx: -1,
    };
    _hydrateTimeline(t0, INIT.messages);
    state.tabs[sid] = t0;
    state.order.push(sid);
    state.active = sid;
  }
}
renderTabs();
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

// --- Compteur CUMULÉ de la session : la VRAIE somme facturée (input rejoué à chaque appel
// d'outil + output), distincte du live per-tour ci-dessus. Persistée côté serveur. ---
function fmtTok(n) {
  n = n || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
  return String(n);
}
const usageMeter = document.getElementById("usage-meter");
function updateUsageMeter(t) {
  if (!usageMeter || !t) return;
  const inEl = document.getElementById("um-in");
  const outEl = document.getElementById("um-out");
  if (inEl) inEl.textContent = fmtTok(t.tokens_in);
  if (outEl) outEl.textContent = fmtTok(t.tokens_out);
  // Taux de cache = part de l'input servie par le cache de préfixe. C'est LA mesure : haut
  // (vert) = le prompt caching mord ; ~0 (rouge) = préfixe instable, on repaie tout plein pot.
  const cacheEl = document.getElementById("um-cache");
  if (cacheEl) {
    const pct = t.cache_pct || 0;
    cacheEl.textContent = t.tokens_in > 0 ? "· cache " + pct + "%" : "";
    cacheEl.classList.toggle("miss", t.tokens_in > 0 && pct < 20);
  }
  // Nombre d'appels API = le MULTIPLICATEUR (le contexte est rejoué à chaque appel) : le
  // levier d'optimisation n°1. « · 58× » se lit « input payé 58 fois ».
  const callsEl = document.getElementById("um-calls");
  if (callsEl) callsEl.textContent = t.api_calls > 0 ? "· " + t.api_calls + "×" : "";
  // Jauge de contexte : occupation COURANTE (prompt du dernier appel) / fenêtre du modèle.
  // Répond à « à quel point le contexte est plein ». Ambre >=70%, rouge >=85% (la
  // microcompaction élague les vieux résultats d'outils au-delà, cf. seuil serveur).
  const ctxEl = document.getElementById("um-ctx");
  if (ctxEl) {
    const used = t.context_tokens || 0;
    const win = t.context_window || 0;
    if (win > 0 && used > 0) {
      const pct = Math.min(100, Math.round((used / win) * 100));
      const fill = document.getElementById("um-ctx-fill");
      if (fill) fill.style.width = pct + "%";
      const pctEl = document.getElementById("um-ctx-pct");
      if (pctEl) pctEl.textContent = pct + "%";
      // Source de la fenêtre : provider = le modèle distant fait autorité ; config = déclaré
      // (le provider ne publie pas sa fenêtre, ex. Z.ai) ; local = notre limite d'allocation.
      const src =
        t.context_source === "provider"
          ? "fenêtre lue du provider (fait autorité)"
          : t.context_source === "local"
          ? "fenêtre allouée localement (notre limite)"
          : "fenêtre déclarée (le provider ne publie pas la sienne)";
      ctxEl.title =
        "Fenêtre de contexte : " + fmtTok(used) + " / " + fmtTok(win) +
        " tokens (" + pct + "%). " + src +
        ". Au-delà de ~85%, la microcompaction élague les vieux résultats d'outils.";
      ctxEl.classList.toggle("warn", pct >= 70 && pct < 85);
      ctxEl.classList.toggle("crit", pct >= 85);
      ctxEl.hidden = false;
    } else {
      ctxEl.hidden = true;
    }
  }
  // Coût volontairement PAS affiché : sans tarif réel il polluerait. Le backend le cumule
  // toujours (cost_usd), prêt à réafficher quand on aura les vrais chiffres du provider.
  usageMeter.hidden = !(t.api_calls > 0 || t.tokens_in > 0 || t.tokens_out > 0);
}
updateUsageMeter(INIT.usage_totals);

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
// Délégation (les items de la sidebar peuvent être ajoutés dynamiquement) : cliquer une
// session l'OUVRE comme onglet (ou l'active si déjà ouverte) — plus de rechargement de page.
const sessionNew = document.getElementById("session-new");
if (sessionNew) sessionNew.addEventListener("click", newSessionTab);
document.querySelector(".session-list")?.addEventListener("click", (e) => {
  // Suppression : un petit toast de confirmation SORT À DROITE de la session cliquée (pas en
  // bas de l'écran) -> confirmation là où est la session.
  const del = e.target.closest(".sess-del");
  if (del) {
    e.stopPropagation();
    openDeleteConfirm(del.dataset.id, del.closest(".session-item"));
    return;
  }
  const pick = e.target.closest(".sess-pick");
  if (pick) openTab(pick.dataset.id);
});

// Popover de confirmation de suppression, ancré à DROITE de la ligne de session.
function _outsideDelClose(e) {
  if (!e.target.closest("#sess-del-pop")) closeDeleteConfirm();
}
function closeDeleteConfirm() {
  document.getElementById("sess-del-pop")?.remove();
  document.removeEventListener("click", _outsideDelClose, true);
}
function openDeleteConfirm(sid, row) {
  closeDeleteConfirm();
  if (!row) return;
  const r = row.getBoundingClientRect();
  const pop = document.createElement("div");
  pop.id = "sess-del-pop";
  pop.className = "sess-del-pop";
  // Aligné sur la ligne : même hauteur, collé juste à sa droite (prolongement visuel).
  pop.style.top = r.top + "px";
  pop.style.height = r.height + "px";
  pop.style.left = r.right + 3 + "px";
  const label = document.createElement("span");
  label.className = "sdp-label";
  label.textContent = "Supprimer cette session ?";
  const yes = document.createElement("button");
  yes.type = "button";
  yes.className = "sdp-yes";
  yes.textContent = "Supprimer";
  const no = document.createElement("button");
  no.type = "button";
  no.className = "sdp-no";
  no.textContent = "Annuler";
  yes.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    await postForm("/session/delete", { id: sid });
    if (state.tabs[sid]) closeTab(sid); // ferme l'onglet si ouvert
    row.remove();
    closeDeleteConfirm();
  });
  no.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeDeleteConfirm();
  });
  pop.append(label, yes, no);
  document.body.appendChild(pop);
  // Ferme au clic ailleurs (en phase capture pour ne pas rater les clics dans la sidebar).
  setTimeout(() => document.addEventListener("click", _outsideDelClose, true), 0);
}

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

// Le bouton reflète l'état de l'onglet ACTIF (Stop si SA génération tourne, sinon Envoyer).
function syncComposer() {
  if (sendBtn) sendBtn.textContent = activeTab()?.streaming ? "Stop" : "Envoyer";
}

function submitChat() {
  const text = input.value.trim();
  if (!text || !state.active) return;
  const t = activeTab();
  if (t) {
    (t.history || (t.history = [])).push(text); // historique ↑/↓ de CET onglet
    t.histIdx = -1;
    t.draft = ""; // message parti -> brouillon vidé
  }
  const imgs = pendingImages.slice();
  input.value = "";
  autosize(); // revient à 1 ligne après envoi
  clearImages();
  // Envoi sur l'onglet actif (flux concurrent) ; le bouton passe à Stop via syncComposer.
  sendChat(state.active, text, imgs).finally(() => input.focus());
  syncComposer();
}

// Stop : coupe l'affichage de l'onglet actif (abort) ET sa génération serveur (/cancel avec
// son session_id). Les autres onglets ne sont pas touchés.
function stopChat() {
  const t = activeTab();
  if (t && t.abort) t.abort.abort();
  if (state.active) postForm("/cancel", { session_id: state.active }).catch(() => {});
  syncComposer();
}
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  // Si l'onglet actif génère, le bouton « Stop » arrête au lieu de soumettre.
  if (activeTab()?.streaming) {
    stopChat();
    return;
  }
  submitChat();
});

// Auto-dimensionnement du textarea : grandit avec le contenu jusqu'à max-height (CSS 200px),
// puis scroll interne. Appelé à la frappe ET après toute écriture programmatique (brouillon
// d'onglet, rappel d'historique, reset) pour que l'utilisateur voie/édite tout son prompt.
function autosize() {
  if (!input) return;
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}
input.addEventListener("input", autosize);

// historique ↑/↓ + Entrée pour envoyer — l'historique est PAR ONGLET (t.history/t.histIdx),
// amorcé avec les prompts déjà envoyés de la conversation.
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitChat();
    return;
  }
  const t = activeTab();
  if (!t) return;
  const hist = t.history || (t.history = []);
  if (e.key === "ArrowUp" && input.selectionStart === 0 && hist.length) {
    e.preventDefault();
    if (t.histIdx == null || t.histIdx === -1) t.histIdx = hist.length;
    if (t.histIdx > 0) t.histIdx--;
    input.value = hist[t.histIdx];
    autosize();
  } else if (e.key === "ArrowDown" && t.histIdx != null && t.histIdx !== -1) {
    e.preventDefault();
    if (t.histIdx < hist.length - 1) {
      t.histIdx++;
      input.value = hist[t.histIdx];
    } else {
      t.histIdx = -1;
      input.value = "";
    }
    autosize();
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
    if (activeTab()) activeTab().thinking = thinkingCb.checked;
    const fd = new FormData();
    fd.append("thinking", thinkingCb.checked ? "1" : "0");
    fetch("/thinking", { method: "POST", body: fd });
  });
}

// --- reset (vide la conversation côté serveur + client) ---
const resetBtn = document.getElementById("reset-btn");
if (resetBtn) {
  resetBtn.addEventListener("click", async () => {
    // /reset opère sur la session focus (_cur = onglet actif via /session/activate).
    await fetch("/reset", { method: "POST" });
    const at = activeTab();
    if (at) {
      at.timeline = [];
      at.meter = null;
      at.metrics = null;
      updateUsageMeter({});
      setMetrics(null, null, null);
    }
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
    // Mémorise le modèle sur l'onglet actif (sinon figé au switch), puis suit l'état machine.
    if (activeTab()) activeTab().model = t.value;
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
    setTxt(
      "sm-gpu-name",
      (g.name || "GPU").replace(/^(NVIDIA GeForce |AMD Radeon |Intel\(R\) )/, ""),
    );
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

// --- Gestionnaire de modèles distants (panneau engrenage) : ajout/édition/suppression À CHAUD
// (le backend monte la route sans redémarrer). Reconstruit le <select> après chaque mutation. ---
(function () {
  const rmList = document.getElementById("rm-list");
  const rmForm = document.getElementById("rm-form");
  const rmAddBtn = document.getElementById("rm-add-btn");
  if (!rmList || !rmForm || !rmAddBtn) return;
  const $ = (id) => document.getElementById(id);

  function rebuildModelSelect(payload) {
    const sel = document.getElementById("model-select");
    if (!sel || !payload) return;
    const cur = sel.value;
    sel.innerHTML = payload
      .map(
        (m) =>
          '<option value="' +
          m.id +
          '"' +
          (m.id === cur ? " selected" : "") +
          ">" +
          (m.remote ? "remote · " : "home · ") +
          m.id +
          "</option>",
      )
      .join("");
  }

  function setMsg(txt, kind) {
    const m = $("rm-msg");
    if (m) {
      m.textContent = txt || "";
      m.className = "rm-msg" + (kind ? " " + kind : "");
    }
  }
  const esc = (s) =>
    String(s == null ? "" : s).replace(
      /[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
    );

  function renderList(remotes) {
    if (!remotes || !remotes.length) {
      rmList.innerHTML = '<div class="rm-empty">aucun modèle distant</div>';
      return;
    }
    rmList.innerHTML = remotes
      .map(function (m) {
        const host = String(m.base_url || "")
          .replace(/^https?:\/\//, "")
          .split("/")[0];
        const sub =
          esc(host) +
          " · " +
          esc(m.model) +
          (m.context ? " · " + fmtTok(m.context) + " ctx" : "") +
          (m.has_key ? "" : ' · <span class="rm-nokey">sans clé</span>');
        const tag = m.managed ? "" : '<span class="rm-tag">config</span>';
        const btns = m.managed
          ? '<button type="button" class="rm-ic rm-edit" data-id="' +
            esc(m.id) +
            '" title="Éditer">✎</button><button type="button" class="rm-ic rm-del" data-id="' +
            esc(m.id) +
            '" title="Supprimer">✕</button>'
          : "";
        return (
          '<div class="rm-item"><div class="rm-item-main"><span class="rm-id">' +
          esc(m.id) +
          "</span>" +
          tag +
          '<div class="rm-sub">' +
          sub +
          "</div></div>" +
          btns +
          "</div>"
        );
      })
      .join("");
  }

  let cache = [];
  async function load() {
    try {
      const d = await (await fetch("/models/config")).json();
      cache = d.remotes || [];
      renderList(cache);
    } catch {}
  }

  function bodyFromForm() {
    return {
      id: $("rm-id").value.trim(),
      base_url: $("rm-base").value.trim(),
      model: $("rm-model").value.trim(),
      api_key: $("rm-key").value,
      context: $("rm-ctx").value.trim() || null,
      vision: $("rm-vision").checked,
    };
  }

  function openForm(rec) {
    $("rm-id").value = rec ? rec.id : "";
    $("rm-id").disabled = !!rec; // l'id ne se renomme pas en édition
    $("rm-base").value = rec ? rec.base_url : "";
    $("rm-model").value = rec ? rec.model : "";
    $("rm-key").value = "";
    $("rm-key").placeholder =
      rec && rec.has_key ? "clé inchangée (laisser vide)" : "clé API";
    $("rm-ctx").value = rec && rec.context ? rec.context : "";
    $("rm-vision").checked = !!(rec && rec.vision);
    setMsg("");
    rmForm.hidden = false;
    rmAddBtn.hidden = true;
  }
  function closeForm() {
    rmForm.hidden = true;
    rmAddBtn.hidden = false;
  }

  rmAddBtn.addEventListener("click", () => openForm(null));
  $("rm-cancel").addEventListener("click", closeForm);

  rmList.addEventListener("click", async (e) => {
    const ed = e.target.closest(".rm-edit");
    const dl = e.target.closest(".rm-del");
    if (ed) {
      const rec = cache.find((x) => x.id === ed.dataset.id);
      if (rec) openForm(rec);
    } else if (dl) {
      if (!confirm("Supprimer le modèle « " + dl.dataset.id + " » ?")) return;
      const r = await fetch(
        "/models/remote/" + encodeURIComponent(dl.dataset.id),
        { method: "DELETE" },
      );
      const j = await r.json();
      if (j.ok) {
        cache = j.remotes;
        renderList(cache);
        rebuildModelSelect(j.models);
      }
    }
  });

  $("rm-test").addEventListener("click", async () => {
    const b = bodyFromForm();
    if (!b.base_url || !b.model) {
      setMsg("base_url et modèle requis", "err");
      return;
    }
    setMsg("test en cours…");
    try {
      const r = await fetch("/models/remote/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(b),
      });
      const j = await r.json();
      setMsg(j.ok ? "connexion OK" : "échec : " + (j.message || ""), j.ok ? "ok" : "err");
    } catch {
      setMsg("test impossible", "err");
    }
  });

  rmForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const b = bodyFromForm();
    if (!b.id || !b.base_url || !b.model) {
      setMsg("id, base_url et modèle requis", "err");
      return;
    }
    setMsg("enregistrement…");
    try {
      const r = await fetch("/models/remote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(b),
      });
      const j = await r.json();
      if (j.ok) {
        cache = j.remotes;
        renderList(cache);
        rebuildModelSelect(j.models);
        closeForm();
      } else {
        setMsg(j.error || "erreur", "err");
      }
    } catch {
      setMsg("enregistrement impossible", "err");
    }
  });

  // Rafraîchit la liste à l'ouverture du panneau engrenage (et une fois au chargement).
  const gear = document.getElementById("gear-btn");
  if (gear) gear.addEventListener("click", load);
  load();
})();

// --- Console de configuration (modal) : tous les paramètres réels, deux couches
// commun/système, édition en direct des vrais fichiers TOML (commentaires préservés backend). ---
(function () {
  const modal = document.getElementById("config-modal");
  const body = document.getElementById("config-body");
  const openBtn = document.getElementById("cfg-open");
  if (!modal || !body || !openBtn) return;
  const esc = (s) =>
    String(s == null ? "" : s).replace(
      /[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
    );

  function provLabel(src) {
    return src === "systeme"
      ? "surchargé ici"
      : src === "commun"
        ? "défaut commun"
        : "défaut";
  }

  function control(p) {
    const v = p.value;
    if (p.type === "bool") {
      return (
        '<input type="checkbox" class="cfg-cb cfg-ctl"' + (v ? " checked" : "") + ">"
      );
    }
    if (p.type === "select") {
      const opts = (p.options || [])
        .map(
          (o) =>
            '<option value="' + esc(o) + '"' + (o === v ? " selected" : "") + ">" + esc(o) + "</option>",
        )
        .join("");
      return '<select class="cfg-sel cfg-ctl">' + opts + "</select>";
    }
    const itype = p.type === "secret" ? "password" : p.type === "int" ? "number" : "text";
    const ph = p.value == null ? ' placeholder="auto"' : "";
    return (
      '<input class="cfg-in cfg-ctl" type="' +
      itype +
      '" value="' +
      esc(v == null ? "" : v) +
      '"' +
      ph +
      ">"
    );
  }

  function rowHtml(p) {
    const resetVis = p.source !== "defaut" ? "" : ' style="display:none"';
    const applies =
      p.applies === "restart"
        ? '<span class="cfg-applies-restart">redémarrage</span>'
        : '<span class="cfg-applies-live">live</span>';
    return (
      '<div class="cfg-row" data-section="' +
      esc(p.section) +
      '" data-key="' +
      esc(p.key) +
      '" data-type="' +
      esc(p.type) +
      '">' +
      '<div class="cfg-row-main"><div class="cfg-label-line">' +
      '<span class="cfg-label">' +
      esc(p.label) +
      "</span>" +
      '<span class="cfg-i" title="' +
      esc(p.help) +
      '">i</span>' +
      '<span class="cfg-badge ' +
      esc(p.layer) +
      '">' +
      (p.layer === "systeme" ? "système" : "commun") +
      "</span>" +
      '<span class="cfg-nat">' +
      esc(p.nature) +
      "</span></div>" +
      '<div class="cfg-sub"><span class="cfg-prov ' +
      esc(p.source) +
      '">' +
      provLabel(p.source) +
      "</span> · " +
      applies +
      "</div></div>" +
      '<div class="cfg-control">' +
      control(p) +
      '<button type="button" class="cfg-reset"' +
      resetVis +
      ">réinitialiser</button>" +
      '<span class="cfg-saved">enregistré</span>' +
      "</div></div>"
    );
  }

  function render(data) {
    body.innerHTML = (data.sections || [])
      .map(
        (s) =>
          '<div class="cfg-section"><div class="cfg-sec-head">' +
          esc(s.label) +
          '</div><div class="cfg-rows">' +
          s.params.map(rowHtml).join("") +
          "</div></div>",
      )
      .join("");
  }

  async function loadCfg() {
    try {
      const d = await (await fetch("/config")).json();
      render(d);
    } catch {
      body.innerHTML = '<div class="rm-empty">config indisponible</div>';
    }
  }

  function open() {
    modal.hidden = false;
    loadCfg();
  }
  function close() {
    modal.hidden = true;
  }
  openBtn.addEventListener("click", open);
  modal.addEventListener("click", (e) => {
    if (e.target.hasAttribute("data-cfg-close")) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) close();
  });

  function valueOf(ctl, type) {
    if (type === "bool") return ctl.checked;
    return ctl.value;
  }
  function flashSaved(row) {
    const s = row.querySelector(".cfg-saved");
    if (!s) return;
    s.classList.add("show");
    setTimeout(() => s.classList.remove("show"), 1400);
  }

  async function save(row) {
    const ctl = row.querySelector(".cfg-ctl");
    const section = row.dataset.section;
    const key = row.dataset.key;
    const value = valueOf(ctl, row.dataset.type);
    const r = await fetch("/config/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section, key, value }),
    });
    const j = await r.json();
    if (!j.ok) return;
    // Provenance mise à jour selon la couche écrite (ou 'defaut' si valeur vidée = reset).
    const src = j.source === "commun" || j.source === "systeme" ? j.source : "defaut";
    const prov = row.querySelector(".cfg-prov");
    if (prov) {
      prov.className = "cfg-prov " + src;
      prov.textContent = provLabel(src);
    }
    const rb = row.querySelector(".cfg-reset");
    if (rb) rb.style.display = src === "defaut" ? "none" : "";
    flashSaved(row);
  }

  // Édition : change (blur pour les inputs, immédiat pour cases/selects) -> sauve.
  body.addEventListener("change", (e) => {
    const row = e.target.closest(".cfg-row");
    if (row && e.target.classList.contains("cfg-ctl")) save(row);
  });
  // Reset : retire la clé du fichier de couche, puis recharge (valeur effective inférieure).
  body.addEventListener("click", async (e) => {
    const rb = e.target.closest(".cfg-reset");
    if (!rb) return;
    const row = e.target.closest(".cfg-row");
    await fetch("/config/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section: row.dataset.section, key: row.dataset.key }),
    });
    loadCfg();
  });
})();

