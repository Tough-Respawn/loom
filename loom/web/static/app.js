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
    // € n'existe pas dans les polices TeX du bundle SVG : le glyphe manquant fait
    // crasher MathJax (« Math input error », vécu sur les montants). En \text{},
    // il retombe sur la police système et affiche un vrai €.
    maths.push(delim + tex.replace(/€/g, "\\text{€}") + delim);
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
// Multi-ONGLETS + PANNEAUX (split view) : chaque session ouverte = un onglet avec SA
// timeline, SA génération (flux concurrent), SON Stop et SON compteur. L'AFFICHAGE passe
// par 1 à 4 panneaux (state.panes), chacun un chat complet (timeline + saisie) lié à un
// onglet — la vue simple est le cas à 1 panneau. `state.active` = sid du panneau FOCUS :
// les singletons (topbar, msg-nav, sélecteur modèle, moniteur) suivent ce focus.
const state = {
  tabs: {}, // sid -> { sid, title, timeline:[], streaming, abort, model, thinking, workspace, meter, metrics }
  order: [], // ordre des onglets ouverts
  panes: [], // Pane[] (créés par createPane : élément cloné de #pane-tpl + refs + état)
  focusedIdx: 0, // index du panneau focus dans panes
  active: null, // sid du panneau focus (maintenu par focusPane/activateTab/setPaneSid)
};
let _seq = 0;
const uid = () => "i" + ++_seq;
function tab(sid) {
  return state.tabs[sid];
}
function activeTab() {
  return state.tabs[state.active] || null;
}
function focusedPane() {
  return state.panes[state.focusedIdx] || state.panes[0] || null;
}
function panesFor(sid) {
  return state.panes.filter((p) => p.sid === sid);
}
function paneShowing(sid) {
  return state.panes.find((p) => p.sid === sid) || null;
}

// Rendu PAR PANNEAU (rAF coalescé par panneau) : seuls les panneaux qui montrent la
// timeline mutée redessinent — les onglets d'arrière-plan génèrent sans toucher l'écran.
function renderPane(pane) {
  if (!pane || pane._raf) return;
  pane._raf = true;
  requestAnimationFrame(() => {
    pane._raf = false;
    render(html`<${App} sid=${pane.sid} />`, pane.root);
    if (pane.pin) pane.wrap.scrollTop = pane.wrap.scrollHeight;
    if (pane === focusedPane()) renderMsgNav(); // navigateur de prompts (panneau focus)
  });
}
function scheduleRenderFor(sid) {
  panesFor(sid).forEach(renderPane);
}
function scheduleRender() {
  state.panes.forEach(renderPane);
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
      const fp = focusedPane();
      const el = fp && fp.root.querySelectorAll(".msg.user")[i];
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

// Mutations liées à la timeline d'un ONGLET précis. Un re-render n'a lieu que dans les
// panneaux qui MONTRENT cet onglet (les flux d'arrière-plan génèrent sans redessiner).
// Deux flux concurrents mutent chacun SA timeline sans se marcher dessus.
function opsFor(sid) {
  const tl = () => (state.tabs[sid] ? state.tabs[sid].timeline : []);
  const get = (id) => tl().find((i) => i.id === id);
  const push = (item) => {
    item.id = item.id || uid();
    tl().push(item);
    scheduleRenderFor(sid);
    return item;
  };
  const patch = (id, fields) => {
    const it = get(id);
    if (it) {
      Object.assign(it, fields);
      scheduleRenderFor(sid);
    }
    return it;
  };
  return { get, push, patch };
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

function IORow({ tag, text, force }) {
  // Une ligne IN ou OUT : aperçu 1 ligne, clic pour déplier le bloc complet. `force` rend la
  // ligne toujours dépliable (arène d'agents : on veut TOUJOURS pouvoir lire l'entrée/sortie).
  const [open, setOpen] = useState(false);
  const full = text == null ? "" : String(text);
  if (!full) return null;
  const first = full.split("\n")[0];
  const multi = force || full.indexOf("\n") >= 0 || first.length > 90;
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

// Chaque agent parallèle prend un nom de FOOTBALLEUR international (pour la blague) au lieu de
// "dispatch_agent". Déterministe (stable au re-render/rejeu), varié d'un groupe à l'autre via seed.
const AGENT_NAMES = [
  "Zidane", "Messi", "Ronaldo", "Mbappé", "Ronaldinho", "Maradona",
  "Benzema", "Mahrez", "Salah", "Modrić", "Iniesta", "Neymar",
];
function _agentName(seed, i) {
  let h = 0;
  for (const c of String(seed || "")) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return AGENT_NAMES[(h + i) % AGENT_NAMES.length];
}

// Carte d'un agent : comme une pastille d'outil (avec IN/OUT dépliables, `force`), mais l'entête
// porte un AVATAR + le NOM de l'agent. Empilées VERTICALEMENT dans le bloc (plus lisible).
function AgentLane({ it, name }) {
  const cls = it.pending ? " working" : it.ok ? " ok" : " ko";
  const status = it.pending ? (it.chars ? it.chars + " car." : "…") : it.ok ? "✓" : "✕";
  const initial = (name || "A").trim().charAt(0);
  const inText = it.in_full != null ? it.in_full : it.cmd || it.path || "";
  const outText = it.out_full != null ? it.out_full : it.preview || "";
  return html`<div class=${"agent-card" + cls}>
    <div class="agent-head">
      <span class="lane-avatar">${initial}</span>
      <span class="lane-name">${name}</span>
      <span class="lane-status">${status}</span>
    </div>
    <${IORow} tag="IN" text=${inText} force=${true} />
    ${it.pending ? null : html`<${IORow} tag="OUT" text=${outText} force=${true} />`}
    ${it.stream && it.pending
      ? html`<pre class="tool-stream">${it.stream.slice(-1400)}</pre>`
      : null}
  </div>`;
}

// Bloc « N agents en parallèle » : les cartes d'agents EMPILÉES verticalement (une sous l'autre),
// pas côte à côte. Chaque carte est consommée du flux principal -> pas de double affichage.
function ParallelArena({ lanes }) {
  const running = lanes.some((l) => l.pending);
  const seed = (lanes[0] && lanes[0].id) || "";
  return html`<div class=${"agent-arena" + (running ? " running" : " done")}>
    <div class="arena-tag">${lanes.length} agents en parallèle</div>
    <div class="arena-lanes">
      ${lanes.map(
        (l, i) => html`<${AgentLane}
          key=${l.id}
          it=${l}
          name=${_agentName(seed, i)}
        />`,
      )}
    </div>
  </div>`;
}

function PermAsk({ it, sid }) {
  const decide = (approve) => {
    const fd = new FormData();
    fd.append("id", it.callId);
    fd.append("approve", approve ? "1" : "0");
    fetch("/tool_decision", { method: "POST", body: fd });
    opsFor(sid).patch(it.id, { decided: true, approved: approve });
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

// Boutons de réponse d'un wizard (/add-model, /remove-model, /rebench) : purs
// raccourcis de frappe — le clic envoie le libellé comme un message tapé. Non
// persistés : au rechargement on répond au clavier (l'état wizard est côté serveur).
function WizChoices({ it, sid }) {
  const pick = (label) => {
    if (it.decided) return;
    opsFor(sid).patch(it.id, { decided: true, picked: label });
    // Envoie la réponse DEPUIS le panneau qui montre cette session (sinon le focus).
    const pane = paneShowing(sid) || focusedPane();
    if (!pane) return;
    pane.input.value = label;
    submitPane(pane);
  };
  return html`<div class=${"wiz-choices" + (it.decided ? " decided" : "")}>
    ${it.options.map(
      (o) => html`<button key=${o} disabled=${!!it.decided}
        class=${it.picked === o ? "picked" : ""} onClick=${() => pick(o)}>${o}</button>`,
    )}
  </div>`;
}

function UserMsg({ it, userIndex, sid }) {
  // Bouton "repartir de la" : tronque la conversation apres ce message et
  // pre-remplit l'input pour re-editer. Comme l'edition de message ChatGPT/Claude.
  const doFork = async (e) => {
    e.stopPropagation();
    if (userIndex == null) return;
    const fd = new FormData();
    fd.append("user_index", String(userIndex));
    // Contenu du message : permet au serveur de le retrouver même si l'index a
    // glissé (compaction), et d'expliquer sinon — plus d'échec muet.
    const itText = Array.isArray(it.content)
      ? it.content
          .filter((p) => p.type === "text")
          .map((p) => p.text)
          .join(" ")
          .trim()
      : String(it.content || "").trim();
    fd.append("text", itText);
    const t = tab(sid);
    try {
      const r = await fetch("/fork", { method: "POST", body: fd });
      if (!r.ok) {
        if (t) {
          t.timeline.push({
            kind: "error",
            message: (await r.text()) || "repartir impossible",
          });
          scheduleRenderFor(sid);
        }
        return;
      }
      const j = await r.json();
      // Tronque la timeline de CET onglet : garde jusqu'a CET item user (inclus)
      if (t) {
        const idx = t.timeline.indexOf(it);
        if (idx >= 0) t.timeline = t.timeline.slice(0, idx + 1);
      }
      // Pre-remplit la saisie du panneau qui montre cette session
      const pane = paneShowing(sid) || focusedPane();
      if (pane) {
        pane.input.value = j.text || "";
        pane.input.focus();
      }
      scheduleRenderFor(sid);
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

function Item({ it, userIndex, sid }) {
  switch (it.kind) {
    case "user":
      return html`<${UserMsg} it=${it} userIndex=${userIndex} sid=${sid} />`;
    case "assistant":
      return html`<${Assistant} it=${it} />`;
    case "think":
      return html`<${Think} it=${it} />`;
    case "tool":
      return html`<${ToolPill} it=${it} />`;
    case "perm":
      return html`<${PermAsk} it=${it} sid=${sid} />`;
    case "choices":
      return html`<${WizChoices} it=${it} sid=${sid} />`;
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

function App({ sid }) {
  const t = tab(sid);
  if (!t || !t.timeline.length) {
    return html`<div class="empty-state">
      Écris une demande. Loom agit avec ses outils (lire, écrire, exécuter, chercher).
    </div>`;
  }
  let _ui = 0;
  const items = t.timeline;
  // PRÉ-PASSE : toute carte d'outil appartenant à un groupe parallèle est consommée par le bloc
  // « N agents » -> on ne la rend JAMAIS en plus dans le flux vertical (pas de double affichage).
  // Indépendant de l'ordre des events (robuste au streaming live comme au rejeu).
  const consumed = new Set();
  for (const it of items) {
    if (it.kind === "parallel") {
      (it.lanes || []).forEach((lid) => consumed.add("tool:" + lid));
    }
  }
  const out = [];
  for (const it of items) {
    if (it.kind === "parallel") {
      const lanes = (it.lanes || [])
        .map((lid) => items.find((x) => x.id === "tool:" + lid))
        .filter(Boolean);
      out.push(html`<${ParallelArena} key=${it.id} lanes=${lanes} />`);
      continue;
    }
    if (consumed.has(it.id)) continue; // carte d'agent : déjà dans son bloc
    let userIndex = null;
    if (it.kind === "user") userIndex = _ui++;
    out.push(html`<${Item} key=${it.id} it=${it} userIndex=${userIndex} sid=${sid} />`);
  }
  return out;
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
  panesFor(sid).forEach((p) => (p.pin = true));
  // Interrompt UNIQUEMENT la génération de CET onglet (les autres continuent).
  if (t.abort) t.abort.abort();
  const ac = new AbortController();
  t.abort = ac;
  t.streaming = true;
  renderTabs();
  syncComposersFor(sid);

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

  // Le serveur modèle peut être DÉMARRÉ PAR CE MESSAGE (auto-start côté back) : on
  // relance le suivi du chip machine, sinon il reste figé sur « serveur local éteint »
  // pendant que le modèle tourne (contradiction constatée le 2026-07-10 — le poller
  // s'arrête quand l'état n'est pas transitoire et rien ne le relançait ici).
  scheduleMachineRefresh();

  // Suivi des SILENCES du flux : au-delà de 2,5 s sans événement alors que ça génère,
  // on affiche la ligne d'activité (label selon la phase : avant le 1er token = le
  // contexte s'ingère ; après = le modèle mouline sans streamer).
  let lastEvtAt = Date.now();
  let sawToken = false;

  const onEvent = (evt) => {
    lastEvtAt = Date.now();
    if (evt.type === "text" || evt.type === "reasoning" || evt.type === "tool_args")
      sawToken = true;
    // Le vrai flux reprend -> lève un éventuel label forcé (ex. « compaction… »), sauf
    // l'event `status` lui-même qui le pose/efface.
    if (
      evt.type === "text" ||
      evt.type === "reasoning" ||
      evt.type === "tool_result" ||
      evt.type === "tool_call"
    )
      t.forcedActivity = null;
    switch (evt.type) {
      case "note":
        // Note en vol INJECTÉE (le modèle vient de la recevoir) : bulle utilisateur
        // à sa vraie position dans le fil, les bulles en cours sont clôturées.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "user", content: evt.text });
        break;
      case "parallel":
        // Groupe d'outils lancés EN PARALLÈLE (distant) : on clôt les bulles en cours et on
        // pose un marqueur « arène ». Les tool_call/tool_result suivants, dont l'id est dans
        // `lanes`, seront rendus CÔTE À CÔTE (animation), pas empilés.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "parallel", lanes: evt.ids || [] });
        break;
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
      case "status":
        // Signal d'activité explicite (ex. compaction en cours) : label FORCÉ, prioritaire
        // sur la détection de silence du timer, affiché comme « le modèle tourne ». Effacé
        // par un label vide, ou dès que le vrai flux reprend (content/reasoning/tool_result).
        t.forcedActivity = evt.label || null;
        setActivityFor(sid, t.forcedActivity);
        break;
      case "models":
        // Un modèle vient d'être ajouté/monté (wizard /add-model) : recharge le
        // sélecteur et la liste du panneau engrenage sans rechargement de page.
        if (window.refreshModels) window.refreshModels();
        break;
      case "choices":
        // Boutons de réponse du wizard (oui/annuler, types de modèle…) : bloc
        // cliquable sous le dernier message — cf. WizChoices.
        push({ kind: "choices", options: evt.options || [] });
        break;
      case "error":
        push({ kind: "error", message: "Erreur : " + evt.message + " (connexion à loom.web perdue ?)" });
        break;
    }
  };

  // L'indicateur d'activité vit DANS chaque panneau qui montre cet onglet (les flux
  // d'arrière-plan sans panneau n'affichent rien) ; nettoyé au finally.
  const gaTimer = setInterval(() => {
    // Label FORCÉ (compaction…) prioritaire sur la détection de silence.
    if (t.forcedActivity) {
      setActivityFor(sid, t.forcedActivity);
      return;
    }
    const quiet = Date.now() - lastEvtAt > 2500;
    setActivityFor(
      sid,
      t.streaming && quiet
        ? sawToken
          ? "le modèle travaille"
          : "préparation du contexte (prefill)"
        : null,
    );
  }, 500);

  try {
    await streamSSE("/chat", fd, onEvent, ac.signal);
    if (asstId) patch(asstId, { done: true });
  } catch (err) {
    if (err.name === "AbortError") {
      if (asstId) patch(asstId, { done: true });
    } else {
      push({ kind: "error", message: "Erreur : " + err.message + " (connexion à loom.web perdue ?)" });
    }
  } finally {
    clearInterval(gaTimer);
    // État machine re-sondé en fin de flux : capture l'état final réel (chargé /
    // libéré) quel que soit ce que le chip racontait pendant la génération.
    scheduleMachineRefresh();
    setActivityFor(sid, null);
    if (thinkId) patch(thinkId, { active: false });
    // Fin de CE flux : si c'est encore le flux courant de l'onglet (pas remplacé par une
    // nouvelle soumission), on marque l'onglet non-générant et on fige son compteur.
    if (t.abort === ac) {
      t.abort = null;
      t.streaming = false;
      renderTabs();
      syncComposersFor(sid);
      if (sid === state.active) {
        setMetrics(lastSent, lastRecv, lastTokS, { done: true });
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
    el.className =
      "tab" +
      (sid === state.active ? " active" : "") +
      (sid !== state.active && paneShowing(sid) ? " shown" : "");
    el.title = t.title || "session";
    // Split view : clic droit = menu (ouvrir à droite/en dessous…), glisser = déplacer
    // l'onglet vers un panneau.
    el.addEventListener("contextmenu", (e) => openTabMenu(e, sid));
    el.draggable = true;
    el.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/loom-sid", sid);
      ev.dataTransfer.effectAllowed = "move";
    });
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

// Rejoue le JOURNAL d'affichage temps réel (timeline.jsonl) : reconstruit la vue EXACTEMENT
// comme en direct — raisonnement, texte, cartes d'outils (avec résultats). Utilisé au
// chargement/ouverture quand la session a un journal ; sinon on retombe sur les messages.
function _replayTimeline(t, events) {
  const byTool = {};
  let think = null,
    asst = null;
  const add = (item) => {
    item.id = item.id || uid();
    t.timeline.push(item);
    return item;
  };
  (events || []).forEach((e) => {
    const d = e.data || {};
    switch (e.event) {
      case "user":
        think = null;
        asst = null;
        add({
          kind: "user",
          content: d.content,
          raw: typeof d.content === "string" ? d.content : "",
          done: true,
        });
        break;
      case "parallel":
        think = null;
        asst = null;
        add({ kind: "parallel", lanes: d.ids || [] });
        break;
      case "reasoning":
        if (asst) asst = null;
        if (!think)
          think = add({ kind: "think", role: "", text: "", active: false, done: true });
        think.text += d.text || "";
        break;
      case "text":
        if (think) think = null;
        if (!asst) asst = add({ kind: "assistant", raw: "", done: true });
        asst.raw += d.text || "";
        break;
      case "tool_call": {
        think = null;
        asst = null;
        const tid = "tool:" + (d.id || d.name);
        if (!byTool[tid])
          byTool[tid] = add({ id: tid, kind: "tool", name: d.name, pending: true });
        break;
      }
      case "tool_result": {
        const tid = "tool:" + (d.id || d.name);
        let it = byTool[tid];
        if (!it) it = byTool[tid] = add({ id: tid, kind: "tool", name: d.name });
        Object.assign(it, {
          name: d.name,
          path: d.path,
          cmd: d.cmd,
          ok: d.ok,
          preview: d.preview,
          detail: d.detail,
          in_full: d.in_full,
          out_full: d.out_full,
          pending: false,
        });
        break;
      }
    }
  });
}

// Charge l'affichage d'une session dans son onglet : rejoue le journal temps réel s'il existe
// (vue riche), sinon retombe sur les messages persistés (sessions legacy sans journal).
async function _loadDisplay(t, sid, hasTimeline, messages) {
  if (hasTimeline) {
    try {
      const tl = await (
        await fetch("/session/" + encodeURIComponent(sid) + "/timeline")
      ).json();
      if (tl.events && tl.events.length) {
        _replayTimeline(t, tl.events);
        return;
      }
    } catch {}
  }
  _hydrateTimeline(t, messages);
}

// Charge un onglet (état + affichage) SANS l'activer — partagé par openTab et la split
// view (restauration de disposition, « ouvrir à droite » d'une session non ouverte).
async function ensureTab(sid) {
  if (state.tabs[sid]) return state.tabs[sid];
  let d;
  try {
    d = await (await fetch("/session_state?id=" + encodeURIComponent(sid))).json();
  } catch {
    return null;
  }
  const t = {
    sid,
    title: d.title || "session",
    timeline: [],
    streaming: false,
    abort: null,
    model: d.model || "",
    thinking: d.thinking !== false,
    localOnly: !!d.local_only,
    workspace: d.workspace || "",
    meter: d.usage_totals || null,
    metrics: null,
    draft: "",
    history: _userTexts(d.messages),
    histIdx: -1,
  };
  await _loadDisplay(t, sid, d.has_timeline, d.messages);
  state.tabs[sid] = t;
  state.order.push(sid);
  return t;
}

async function openTab(sid) {
  if (!(await ensureTab(sid))) return;
  activateTab(sid);
}

// Change l'onglet AFFICHÉ par un panneau (brouillon sauvé/restauré, composer resynchronisé).
function setPaneSid(pane, sid) {
  // Brouillon PAR ONGLET : on sauve le texte NON envoyé de l'onglet qu'on quitte,
  // on charge celui de l'onglet affiché. Sinon la zone de saisie serait partagée.
  const prev = state.tabs[pane.sid];
  if (prev) prev.draft = pane.input.value;
  pane.sid = sid;
  const t = state.tabs[sid];
  pane.input.value = (t && t.draft) || "";
  autosize(pane);
  pane.pin = true;
  hidePal(pane);
  // Ligne d'activité : propre à l'onglet quitté — le timer du flux du NOUVEL onglet la
  // repeindra dans les 500 ms s'il est en silence, sinon elle doit disparaître.
  setPaneActivity(pane, null);
  syncComposer(pane);
  renderPane(pane);
  if (pane === focusedPane()) state.active = sid;
}

// Synchronise les singletons (topbar, sélecteur modèle, moniteur, sidebar) sur l'onglet
// qui prend le FOCUS, et informe le serveur (session focus _cur).
function focusSingletonsFor(sid) {
  const t = state.tabs[sid];
  state.active = sid;
  if (!t) return;
  // Le serveur suit ce focus (pour /model, /tools, /reset, /pick-folder qui opèrent
  // sur _cur). La rafale /machine_state part APRÈS la réponse : un tick immédiat
  // lirait l'ANCIENNE session active côté serveur et re-basculerait le moniteur
  // avec un état périmé (course vécue au switch d'onglet).
  postForm("/session/activate", { id: sid })
    .then(() => scheduleMachineRefresh())
    .catch(() => {});
  // Synchronise les contrôles de la sidebar/topbar à cet onglet.
  const sel = document.getElementById("model-select");
  if (sel && t.model) sel.value = t.model;
  // Teinte du sélecteur ET visibilité du moniteur système suivent l'onglet FOCUS
  // immédiatement (un distant n'a pas besoin du moniteur).
  paintModelSelect();
  const think = document.getElementById("thinking-cb");
  if (think) think.checked = !!t.thinking;
  const lo = document.getElementById("local-only-cb");
  if (lo) lo.checked = !!t.localOnly;
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
  renderMsgNav();
}

// Donne le focus au panneau i : liseré, singletons, serveur — sans changer son onglet.
function focusPane(i) {
  const pane = state.panes[i];
  if (!pane) return;
  const changed = state.focusedIdx !== i || state.active !== pane.sid;
  state.focusedIdx = i;
  state.panes.forEach((p, k) => p.el.classList.toggle("focused", k === i));
  if (!changed) return;
  if (pane.sid) focusSingletonsFor(pane.sid);
  else state.active = null;
  renderTabs();
}

function activateTab(sid) {
  const t = state.tabs[sid];
  if (!t) return;
  // Déjà affiché dans un panneau -> on FOCUS ce panneau (pas de double affichage).
  const shown = paneShowing(sid);
  if (shown) {
    focusPane(state.panes.indexOf(shown));
    return;
  }
  const pane = focusedPane();
  if (!pane) return;
  setPaneSid(pane, sid);
  focusSingletonsFor(sid);
  renderTabs();
}

function closeTab(sid) {
  const t = state.tabs[sid];
  if (t && t.abort) t.abort.abort(); // stoppe le flux de l'onglet fermé (la session reste)
  delete state.tabs[sid];
  state.order = state.order.filter((s) => s !== sid);
  const showing = panesFor(sid);
  if (showing.length && state.panes.length > 1) {
    // Split : fermer l'onglet ferme son (ses) panneau(x), les autres continuent.
    showing.forEach((p) => removePane(p));
  } else if (showing.length) {
    // Vue simple : on bascule sur le dernier onglet restant (comportement historique).
    const pane = showing[0];
    pane.sid = null;
    const next = state.order[state.order.length - 1] || null;
    if (next) activateTab(next);
    else {
      state.active = null;
      pane.input.value = "";
      renderPane(pane);
      renderTabs();
    }
  } else {
    renderTabs();
  }
  savePanesLayout();
}

// ----------------------------------------------------------------------------
// Panneaux : fabrique d'un chat complet (timeline + composer) cloné de #pane-tpl.
// Toute la mécanique de saisie (envoi, stop, palette « / », historique ↑/↓, images,
// activité, scroll collant) est PAR PANNEAU — c'est ce qui rend la split view possible.
// ----------------------------------------------------------------------------
const panesEl = document.getElementById("panes");
const paneTpl = document.getElementById("pane-tpl");
const MAX_IMAGES = 6;
let splitDir = "cols"; // orientation d'un split à 2 panneaux (à droite vs en dessous)
// Disposition sauvegardée, capturée AVANT l'hydratation : le premier applyPaneLayout()
// (1 panneau) réécrit localStorage — sans cette capture, la restauration lirait du vide.
const SAVED_PANES = (() => {
  try {
    return JSON.parse(localStorage.loomPanes || "null");
  } catch (e) {
    return null;
  }
})();
let CMDS = []; // commandes « / » (source de vérité serveur), partagées par les palettes
fetch("/commands")
  .then((r) => r.json())
  .then((d) => {
    CMDS = d.commands || [];
  })
  .catch(() => {}); // pas de palette = pas de casse, le chat marche sans

function createPane() {
  const el = paneTpl.content.firstElementChild.cloneNode(true);
  const q = (sel) => el.querySelector(sel);
  const pane = {
    sid: null,
    el,
    wrap: q(".messages-wrap"),
    root: q(".messages"),
    form: q(".chat-form"),
    input: q("textarea"),
    sendBtn: q(".send-btn"),
    fileInput: q('input[type="file"]'),
    fileBtn: q(".file-btn"),
    previewWrap: q(".preview-wrap"),
    activityEl: q(".gen-activity"),
    activityLabel: q(".ga-label"),
    palEl: q(".cmd-palette"),
    pendingImages: [], // images jointes au prochain message (max MAX_IMAGES)
    palIdx: 0,
    pin: true, // auto-scroll collant
    _raf: false,
  };
  wirePane(pane);
  return pane;
}

function applyPaneLayout() {
  if (!panesEl) return;
  panesEl.className =
    "panes layout-" +
    state.panes.length +
    (state.panes.length > 1 ? " multi" : "") +
    (state.panes.length === 2 && splitDir === "rows" ? " rows" : "");
  savePanesLayout();
}

function savePanesLayout() {
  try {
    localStorage.loomPanes = JSON.stringify({
      sids: state.panes.map((p) => p.sid).filter(Boolean),
      dir: splitDir,
    });
  } catch (e) {
    /* localStorage indispo : la disposition ne survivra pas au rechargement */
  }
}

function addPane(sid, dir) {
  if (state.panes.length >= 4) {
    showToast("4 panneaux maximum");
    return null;
  }
  const shown = sid && paneShowing(sid);
  if (shown) {
    focusPane(state.panes.indexOf(shown));
    return null;
  }
  if (dir) splitDir = dir;
  const pane = createPane();
  state.panes.push(pane);
  panesEl.appendChild(pane.el);
  if (sid && state.tabs[sid]) setPaneSid(pane, sid);
  applyPaneLayout();
  focusPane(state.panes.length - 1);
  renderPane(pane);
  renderTabs();
  return pane;
}

function removePane(pane) {
  const i = state.panes.indexOf(pane);
  if (i < 0 || state.panes.length <= 1) return;
  // Brouillon sauvé avant de perdre le panneau (l'onglet reste ouvert dans la barre).
  const t = state.tabs[pane.sid];
  if (t) t.draft = pane.input.value;
  state.panes.splice(i, 1);
  pane.el.remove();
  if (state.focusedIdx >= state.panes.length)
    state.focusedIdx = state.panes.length - 1;
  applyPaneLayout();
  focusPane(state.focusedIdx);
  renderTabs();
}

// ---- Indicateur d'activité (silences du flux) : par panneau ----
function setPaneActivity(pane, label) {
  if (!pane || !pane.activityEl) return;
  if (!label) {
    pane.activityEl.hidden = true;
    return;
  }
  pane.activityEl.hidden = false;
  if (pane.activityLabel.textContent !== label)
    pane.activityLabel.textContent = label;
}
function setActivityFor(sid, label) {
  panesFor(sid).forEach((p) => setPaneActivity(p, label));
}

// Le bouton reflète l'état de l'onglet du panneau (Stop si SA génération tourne).
function syncComposer(pane) {
  if (pane && pane.sendBtn)
    pane.sendBtn.textContent = tab(pane.sid)?.streaming ? "Stop" : "Envoyer";
}
function syncComposersFor(sid) {
  panesFor(sid).forEach(syncComposer);
}

// Auto-dimensionnement du textarea : grandit avec le contenu jusqu'à max-height (CSS
// 200px), puis scroll interne. Appelé à la frappe ET après toute écriture programmatique.
function autosize(pane) {
  if (!pane || !pane.input) return;
  pane.input.style.height = "auto";
  pane.input.style.height = Math.min(pane.input.scrollHeight, 200) + "px";
}

// ---- images jointes (par panneau, jusqu'à MAX_IMAGES) ----
function renderPreviews(pane) {
  pane.previewWrap.innerHTML = "";
  pane.pendingImages.forEach((file, i) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = file.name || "image";
    const rm = document.createElement("button");
    rm.type = "button";
    rm.textContent = "✕";
    rm.title = "Retirer";
    rm.addEventListener("click", () => removeImage(pane, i));
    wrap.append(img, rm);
    pane.previewWrap.append(wrap);
  });
  pane.previewWrap.style.display = pane.pendingImages.length ? "flex" : "none";
}
function addImages(pane, files) {
  for (const f of files) {
    if (!f || !f.type.startsWith("image/")) continue;
    if (pane.pendingImages.length >= MAX_IMAGES) break; // les suivantes sont ignorées
    pane.pendingImages.push(f);
  }
  renderPreviews(pane);
}
function removeImage(pane, i) {
  pane.pendingImages.splice(i, 1);
  renderPreviews(pane);
}
function clearImages(pane) {
  pane.pendingImages = [];
  renderPreviews(pane);
}

// ---- Palette de commandes « / » : liste globale (CMDS), filtrage par panneau ----
function palMatches(pane) {
  const v = pane.input.value;
  // Palette uniquement sur le PREMIER mot d'un message qui commence par « / » :
  // dès qu'un espace ou un retour ligne arrive, on est dans les arguments.
  if (!v.startsWith("/") || /[\s\n]/.test(v)) return [];
  const tok = v.slice(1).toLowerCase();
  return CMDS.filter((c) => c.name.slice(1).toLowerCase().startsWith(tok));
}
function hidePal(pane) {
  if (pane && pane.palEl) pane.palEl.hidden = true;
}
function renderPal(pane) {
  if (!pane.palEl) return;
  const m = palMatches(pane);
  if (!m.length) {
    hidePal(pane);
    return;
  }
  pane.palIdx = Math.min(pane.palIdx, m.length - 1);
  pane.palEl.replaceChildren(
    ...m.map((c, i) => {
      const item = document.createElement("div");
      item.className = "cmd-item" + (i === pane.palIdx ? " sel" : "");
      item.setAttribute("role", "option");
      const line = document.createElement("div");
      line.className = "cmd-line";
      const name = document.createElement("span");
      name.className = "cmd-name";
      name.textContent = c.name;
      const usage = document.createElement("span");
      usage.className = "cmd-usage";
      usage.textContent = c.usage;
      line.append(name, usage);
      const desc = document.createElement("div");
      desc.className = "cmd-desc";
      desc.textContent = c.description;
      item.append(line, desc);
      // mousedown (pas click) : ne pas voler le focus du textarea avant l'insertion
      item.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        palPick(pane, c);
      });
      item.addEventListener("mouseenter", () => {
        pane.palIdx = i;
        renderPal(pane);
      });
      return item;
    }),
  );
  pane.palEl.hidden = false;
}
function palPick(pane, c) {
  pane.input.value = c.name + " ";
  autosize(pane);
  hidePal(pane);
  pane.input.focus();
}

// ---- envoi / stop (par panneau, vers l'onglet du panneau) ----
function submitPane(pane) {
  const sid = pane.sid;
  const text = pane.input.value.trim();
  if (!text || !sid) return;
  // Ferme la palette « / » : le clear programmatique du champ n'émet pas d'événement
  // input, elle resterait ouverte sur du vide.
  hidePal(pane);
  const t = tab(sid);
  // Une réponse part (tapée OU cliquée) : les blocs de boutons en attente sont
  // consommés — des boutons périmés ne doivent pas rester cliquables.
  if (t)
    for (const it of t.timeline)
      if (it.kind === "choices" && !it.decided)
        opsFor(sid).patch(it.id, { decided: true });
  // Pièces jointes pendant une génération : la file /note est TEXTE-ONLY. On bloque
  // net (rien n'est consommé) plutôt que de laisser l'image en attente partir avec
  // le PROCHAIN message, auquel elle ne correspondrait plus.
  if (t?.streaming && pane.pendingImages.length) {
    showToast(
      "images impossibles pendant une génération — retire-les pour envoyer une note texte, ou Stop d'abord",
    );
    return;
  }
  if (t) {
    (t.history || (t.history = [])).push(text); // historique ↑/↓ de CET onglet
    t.histIdx = -1;
    t.draft = ""; // message parti -> brouillon vidé
  }
  // NOTE EN VOL (« btw » natif) : pendant une génération, un message avec du texte
  // NE l'interrompt plus — il part en file (/note) et la boucle l'injecte au prochain
  // point d'arrêt (la bulle apparaît à ce moment-là, à sa vraie position). Pour
  // interrompre : bouton Stop, puis envoyer.
  if (t?.streaming) {
    pane.input.value = "";
    autosize(pane);
    postForm("/note", { session_id: sid, text })
      .then(async (r) => {
        if (r.ok)
          showToast(
            "note transmise — prise en compte au prochain point d'arrêt du modèle",
          );
        else {
          const err = await r.json().catch(() => null);
          showToast(err?.error || "note refusée (session ?)");
        }
      })
      .catch(() => showToast("note perdue : loom.web injoignable"));
    pane.input.focus();
    return;
  }
  const imgs = pane.pendingImages.slice();
  pane.input.value = "";
  autosize(pane); // revient à 1 ligne après envoi
  clearImages(pane);
  // Envoi sur l'onglet du panneau (flux concurrent) ; le bouton passe à Stop.
  sendChat(sid, text, imgs).finally(() => pane.input.focus());
  syncComposer(pane);
}

// Stop : coupe l'affichage de cet onglet (abort) ET sa génération serveur (/cancel avec
// son session_id). Les autres onglets ne sont pas touchés.
function stopPane(pane) {
  const t = tab(pane.sid);
  if (t && t.abort) t.abort.abort();
  if (pane.sid) postForm("/cancel", { session_id: pane.sid }).catch(() => {});
  syncComposer(pane);
}

function wirePane(pane) {
  const input = pane.input;
  // Focus du panneau au premier geste dedans (split view) — capture, avant les boutons.
  pane.el.addEventListener(
    "pointerdown",
    () => {
      const i = state.panes.indexOf(pane);
      if (i >= 0 && i !== state.focusedIdx) focusPane(i);
    },
    true,
  );
  // Scroll collant : on suit le bas seulement si l'utilisateur y est déjà.
  pane.wrap.addEventListener(
    "scroll",
    () => {
      pane.pin =
        pane.wrap.scrollTop + pane.wrap.clientHeight >=
        pane.wrap.scrollHeight - 120;
    },
    { passive: true },
  );
  // Si l'onglet du panneau génère, SEUL un appui explicite sur le bouton « Stop »
  // arrête (focus sur le bouton = clic ou activation clavier DU bouton). Un
  // Entrée depuis le champ de saisie — surtout vide — ne doit JAMAIS couper une
  // réflexion en cours (vécu 2026-07-10 : minutes de travail perdues).
  pane.form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (tab(pane.sid)?.streaming) {
      if (document.activeElement === pane.sendBtn) stopPane(pane);
      return;
    }
    submitPane(pane);
  });
  input.addEventListener("input", () => {
    autosize(pane);
    pane.palIdx = 0;
    renderPal(pane);
  });
  input.addEventListener("blur", () => setTimeout(() => hidePal(pane), 120));
  // historique ↑/↓ + Entrée pour envoyer — l'historique est PAR ONGLET (t.history).
  input.addEventListener("keydown", (e) => {
    // La palette ouverte capte la navigation AVANT l'historique et l'envoi.
    if (pane.palEl && !pane.palEl.hidden) {
      const m = palMatches(pane);
      if (e.key === "ArrowDown" && m.length) {
        e.preventDefault();
        pane.palIdx = (pane.palIdx + 1) % m.length;
        renderPal(pane);
        return;
      }
      if (e.key === "ArrowUp" && m.length) {
        e.preventDefault();
        pane.palIdx = (pane.palIdx - 1 + m.length) % m.length;
        renderPal(pane);
        return;
      }
      if (e.key === "Escape") {
        hidePal(pane);
        return;
      }
      if ((e.key === "Tab" || e.key === "Enter") && m.length) {
        const sel = m[Math.min(pane.palIdx, m.length - 1)];
        // Entrée sur une commande DÉJÀ complète = envoi normal ; sinon on complète.
        if (e.key === "Tab" || input.value.trim() !== sel.name) {
          e.preventDefault();
          palPick(pane, sel);
          return;
        }
        hidePal(pane);
      }
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitPane(pane);
      return;
    }
    const t = tab(pane.sid);
    if (!t) return;
    const hist = t.history || (t.history = []);
    if (e.key === "ArrowUp" && input.selectionStart === 0 && hist.length) {
      e.preventDefault();
      if (t.histIdx == null || t.histIdx === -1) t.histIdx = hist.length;
      if (t.histIdx > 0) t.histIdx--;
      input.value = hist[t.histIdx];
      autosize(pane);
    } else if (e.key === "ArrowDown" && t.histIdx != null && t.histIdx !== -1) {
      e.preventDefault();
      if (t.histIdx < hist.length - 1) {
        t.histIdx++;
        input.value = hist[t.histIdx];
      } else {
        t.histIdx = -1;
        input.value = "";
      }
      autosize(pane);
    }
  });
  pane.fileInput.addEventListener("change", () => {
    addImages(pane, [...pane.fileInput.files]);
    pane.fileInput.value = ""; // permet de re-sélectionner le même fichier ensuite
  });
  pane.fileBtn.addEventListener("click", () => pane.fileInput.click());
  // Déposer un ONGLET (glissé depuis la barre) sur ce panneau : la session s'y affiche
  // (échange si elle était déjà dans un autre panneau — jamais de doublon).
  pane.el.addEventListener("dragover", (e) => {
    if (![...e.dataTransfer.types].includes("text/loom-sid")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    pane.el.classList.add("pane-drop");
  });
  pane.el.addEventListener("dragleave", () =>
    pane.el.classList.remove("pane-drop"),
  );
  pane.el.addEventListener("drop", (e) => {
    pane.el.classList.remove("pane-drop");
    const sid = e.dataTransfer.getData("text/loom-sid");
    if (!sid || !state.tabs[sid]) return;
    e.preventDefault();
    dropTabOnPane(sid, pane);
  });
}

// ----------------------------------------------------------------------------
// Split view : commandes (menu contextuel d'onglet, glisser-déposer, raccourcis).
// ----------------------------------------------------------------------------
function dropTabOnPane(sid, pane) {
  if (pane.sid === sid) return;
  const other = paneShowing(sid);
  if (other) {
    // Déplacer = ÉCHANGER les onglets des deux panneaux (pas de double affichage).
    const mine = pane.sid;
    setPaneSid(other, mine);
    setPaneSid(pane, sid);
  } else {
    setPaneSid(pane, sid);
  }
  focusPane(state.panes.indexOf(pane));
  savePanesLayout();
  renderTabs();
}

// « Ouvrir à droite / en dessous » : l'onglet visé s'affiche dans un NOUVEAU panneau.
// S'il est déjà affiché, le nouveau panneau prend le plus récent onglet NON affiché
// (scinder n'a de sens qu'avec deux contenus différents — pas de doublon).
async function splitWith(sid, dir) {
  if (!sid || !(await ensureTab(sid))) return;
  if (!paneShowing(sid)) {
    addPane(sid, dir);
    savePanesLayout();
    return;
  }
  const free = state.order.filter((s) => !paneShowing(s));
  if (!free.length) {
    showToast("rien à scinder : ouvre un autre onglet d'abord");
    return;
  }
  addPane(free[free.length - 1], dir);
  savePanesLayout();
}

// Vue simple : ne garde que le panneau FOCUS (les onglets restent dans la barre).
function singleView() {
  if (state.panes.length <= 1) return;
  const keep = focusedPane();
  [...state.panes].forEach((p) => {
    if (p !== keep) removePane(p);
  });
  savePanesLayout();
}

// ---- menu contextuel d'onglet (clic droit) ----
let _ctxCloser = null;
function closeTabMenu() {
  document.getElementById("tab-ctx")?.remove();
  if (_ctxCloser) {
    document.removeEventListener("click", _ctxCloser, true);
    _ctxCloser = null;
  }
}
function openTabMenu(e, sid) {
  e.preventDefault();
  closeTabMenu();
  const m = document.createElement("div");
  m.className = "ctx-menu";
  m.id = "tab-ctx";
  const add = (label, fn, disabled) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.disabled = !!disabled;
    b.addEventListener("click", () => {
      closeTabMenu();
      fn();
    });
    m.appendChild(b);
  };
  const full = state.panes.length >= 4;
  add("Ouvrir à droite", () => splitWith(sid, "cols"), full);
  add("Ouvrir en dessous", () => splitWith(sid, "rows"), full);
  const sep = document.createElement("div");
  sep.className = "ctx-sep";
  m.appendChild(sep);
  add("Remettre en vue simple", () => singleView(), state.panes.length <= 1);
  document.body.appendChild(m);
  m.style.left = Math.min(e.clientX, window.innerWidth - m.offsetWidth - 6) + "px";
  m.style.top = Math.min(e.clientY, window.innerHeight - m.offsetHeight - 6) + "px";
  _ctxCloser = (ev) => {
    if (!ev.target.closest("#tab-ctx")) closeTabMenu();
  };
  setTimeout(() => document.addEventListener("click", _ctxCloser, true), 0);
}

// ---- raccourcis : Ctrl+\ scinder, Ctrl+Maj+\ vue simple, Alt+1..4 focus panneau.
// e.code (position physique) plutôt que e.key : indépendant de la disposition AZERTY.
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeTabMenu();
  const backslash = e.code === "Backslash" || e.code === "IntlBackslash";
  if (e.ctrlKey && !e.shiftKey && backslash) {
    e.preventDefault();
    splitWith(state.active, "cols");
  } else if (e.ctrlKey && e.shiftKey && backslash) {
    e.preventDefault();
    singleView();
  } else if (e.altKey && /^Digit[1-4]$/.test(e.code)) {
    const i = +e.code.slice(5) - 1;
    if (state.panes[i]) {
      e.preventDefault();
      focusPane(i);
      state.panes[i].input.focus();
    }
  }
});

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
    localOnly: false,
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
// Premier PANNEAU + onglet de départ = la session active (hydratée depuis INIT). La
// sidebar rendue par le serveur liste toutes les sessions ; on n'ouvre QUE l'active.
{
  const pane = createPane();
  state.panes = [pane];
  state.focusedIdx = 0;
  panesEl.appendChild(pane.el);
  pane.el.classList.add("focused");
  applyPaneLayout();
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
      localOnly: !!INIT.local_only,
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
    pane.sid = sid;
    state.active = sid;
    // Async : si la session active a un journal temps réel, on remplace l'affichage par le
    // REJEU (raisonnement + cartes d'outils, comme en direct) au lieu des simples bulles.
    (async () => {
      try {
        const tl = await (
          await fetch("/session/" + encodeURIComponent(sid) + "/timeline")
        ).json();
        if (tl.events && tl.events.length) {
          t0.timeline = [];
          _replayTimeline(t0, tl.events);
          scheduleRenderFor(sid);
        }
      } catch {}
    })();
  }
  pane.input.focus(); // l'ancien autofocus du textarea unique est mort avec lui
}
renderTabs();
scheduleRender();

// Restaure la DISPOSITION de la dernière fois (split 2-4 panneaux) : uniquement des
// sessions qui existent encore (INIT.sessions fait foi), en fond — la page reste
// utilisable pendant le chargement des onglets.
(async () => {
  const saved = SAVED_PANES;
  if (!saved || !Array.isArray(saved.sids) || saved.sids.length < 2) return;
  const known = new Set((INIT.sessions || []).map((s) => s.id));
  const sids = saved.sids.filter((s) => known.has(s));
  if (sids.length < 2) return;
  splitDir = saved.dir === "rows" ? "rows" : "cols";
  const first = state.panes[0];
  if (first && first.sid !== sids[0] && (await ensureTab(sids[0]))) {
    if (!paneShowing(sids[0])) setPaneSid(first, sids[0]);
  }
  for (const sid of sids.slice(1)) {
    if (state.panes.length >= 4) break;
    if (paneShowing(sid)) continue;
    if (await ensureTab(sid)) addPane(sid);
  }
  focusPane(0);
  renderTabs();
  savePanesLayout();
})();

// MathJax charge en async : re-typeset les bulles déjà rendues (historique) une fois prêt.
function _typesetAll() {
  if (!(window.MathJax && window.MathJax.typesetPromise)) return;
  document
    .querySelectorAll(".msg.assistant")
    .forEach((el) => window.MathJax.typesetPromise([el]).catch(() => {}));
}
if (window.__mathjaxReady) _typesetAll();
else document.addEventListener("mathjax-ready", _typesetAll);

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

// Bouton « compacter » (près de la jauge de contexte) : DÉTERMINISTE et instantané (aucun
// appel modèle) — allège les vieux résultats d'outils et clippe l'historique pour libérer
// du contexte sans attendre la saturation (le résumé LLM dense, lui, est réservé à la
// compaction AUTO en cours de conversation). Cible la session active de l'onglet ; met à
// jour la jauge avec les compteurs renvoyés.
const compactBtn = document.getElementById("um-compact");
if (compactBtn) {
  compactBtn.addEventListener("click", async () => {
    const prev = compactBtn.textContent;
    compactBtn.disabled = true;
    compactBtn.textContent = "…";
    // Feedback LISIBLE : un refus (429 = génération en cours) ou un « rien à compacter »
    // clignotait 1,5 s et repartait -> on le ratait, d'où l'impression que « rien ne se
    // passe ». On garde le message plus longtemps (surtout pour 429/erreur) et on le rend
    // explicite. Pendant une génération, la compaction AUTO gère déjà le contexte.
    let msg = "erreur",
      hold = 3000;
    try {
      const data = state.active ? { session_id: state.active } : {};
      const r = await postForm("/compact", data);
      if (r.ok) {
        const d = await r.json();
        updateUsageMeter(d);
        if (d.collapsed) {
          msg = "✓ −" + d.collapsed;
          hold = 1800;
        } else {
          msg = "déjà compact";
        }
      } else if (r.status === 429) {
        msg = "génération en cours";
      }
    } catch (_e) {
      msg = "erreur";
    }
    compactBtn.textContent = msg;
    setTimeout(() => {
      compactBtn.textContent = prev;
      compactBtn.disabled = false;
    }, hold);
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
  if (!pick) return;
  // Ctrl+clic (ou Cmd) : multi-sélection pour suppression groupée, sans ouvrir l'onglet.
  if (e.ctrlKey || e.metaKey) {
    e.preventDefault();
    toggleMultiSel(pick.dataset.id, pick.closest(".session-item"));
    return;
  }
  clearMultiSel();
  openTab(pick.dataset.id);
});

// --- multi-sélection de sessions (Ctrl+clic) : barre « supprimer (N) » sous le label. ---
const _multiSel = new Set();
function clearMultiSel() {
  _multiSel.clear();
  document
    .querySelectorAll(".session-item.sel")
    .forEach((el) => el.classList.remove("sel"));
  document.getElementById("sess-multi")?.remove();
}
function toggleMultiSel(sid, row) {
  if (!sid || !row) return;
  if (_multiSel.has(sid)) {
    _multiSel.delete(sid);
    row.classList.remove("sel");
  } else {
    _multiSel.add(sid);
    row.classList.add("sel");
  }
  renderMultiBar();
}
function renderMultiBar() {
  document.getElementById("sess-multi")?.remove();
  if (!_multiSel.size) return;
  const list = document.querySelector(".session-list");
  if (!list) return;
  const bar = document.createElement("div");
  bar.id = "sess-multi";
  bar.className = "sess-multi";
  const label = document.createElement("span");
  label.textContent = _multiSel.size + " sélectionnée" + (_multiSel.size > 1 ? "s" : "");
  const del = document.createElement("button");
  del.type = "button";
  del.className = "sm-del";
  del.textContent = "supprimer";
  del.addEventListener("click", async () => {
    for (const sid of [..._multiSel]) {
      await postForm("/session/delete", { id: sid });
      if (state.tabs[sid]) closeTab(sid);
      document
        .querySelector('.sess-pick[data-id="' + sid + '"]')
        ?.closest(".session-item")
        ?.remove();
    }
    clearMultiSel();
  });
  const no = document.createElement("button");
  no.type = "button";
  no.className = "sm-no";
  no.textContent = "annuler";
  no.addEventListener("click", clearMultiSel);
  bar.append(label, del, no);
  list.before(bar);
}
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && _multiSel.size) clearMultiSel();
});

// Popover de confirmation de suppression : le MÊME geste pour les sessions et les
// skills (harmonie UI). Ancré à droite de la ligne (align="right") ou déplié vers la
// gauche quand l'ancre touche le bord droit (align="left", ex. drawer de skill).
function _outsideDelClose(e) {
  if (!e.target.closest("#sess-del-pop")) closeDeleteConfirm();
}
function closeDeleteConfirm() {
  document.getElementById("sess-del-pop")?.remove();
  document.removeEventListener("click", _outsideDelClose, true);
}
function openConfirmPop(row, labelText, onYes, align = "right") {
  closeDeleteConfirm();
  if (!row) return;
  const r = row.getBoundingClientRect();
  const pop = document.createElement("div");
  pop.id = "sess-del-pop";
  pop.className = "sess-del-pop";
  pop.style.top = r.top + "px";
  pop.style.height = Math.max(r.height, 26) + "px";
  const label = document.createElement("span");
  label.className = "sdp-label";
  label.textContent = labelText;
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
    await onYes();
    closeDeleteConfirm();
  });
  no.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeDeleteConfirm();
  });
  pop.append(label, yes, no);
  document.body.appendChild(pop);
  if (align === "left") {
    // Déplié vers la gauche de l'ancre (le drawer colle au bord droit de l'écran).
    pop.style.left = Math.max(6, r.left - pop.offsetWidth - 6) + "px";
  } else {
    // Aligné sur la ligne : collé juste à sa droite (prolongement visuel).
    pop.style.left = r.right + 3 + "px";
  }
  // Ferme au clic ailleurs (en phase capture pour ne pas rater les clics dans la sidebar).
  setTimeout(() => document.addEventListener("click", _outsideDelClose, true), 0);
}
function openDeleteConfirm(sid, row) {
  openConfirmPop(row, "Supprimer cette session ?", async () => {
    await postForm("/session/delete", { id: sid });
    if (state.tabs[sid]) closeTab(sid); // ferme l'onglet si ouvert
    row.remove();
  });
}

// --- sélecteur de dossier natif ---
const pickFolderBtn = document.getElementById("pick-folder-btn");
if (pickFolderBtn) {
  pickFolderBtn.addEventListener("click", async () => {
    try {
      const r = await fetch("/pick-folder", { method: "POST" });
      const j = await r.json();
      if (j.path) {
        // Applique le dossier à la session de CET onglet — session_id ciblé, comme
        // /chat et /cancel : sans lui le back écrit dans la session focus globale,
        // pas forcément celle de l'onglet (bug du 2026-07-10). Et la puce ne se met
        // à jour qu'APRÈS confirmation serveur : avant, elle affichait le nouveau
        // dossier même si le POST échouait -> le tour partait sur l'ancien dossier
        // pendant que l'UI en montrait un autre.
        const wsData = { workspace: j.path };
        if (state.active) wsData.session_id = state.active;
        const resp = await postForm("/session/workspace", wsData);
        if (!resp.ok) throw new Error("session/workspace HTTP " + resp.status);
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
      // Échec du POST : la puce N'A PAS changé (source de vérité = serveur) ;
      // on la fait clignoter pour que l'échec soit VISIBLE, plus de console muette.
      console.warn("pick-folder:", err);
      if (workdirChip) {
        workdirChip.classList.add("err");
        setTimeout(() => workdirChip.classList.remove("err"), 1500);
      }
    }
  });
}

// --- toggle sidebar mobile ---
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarEl = document.getElementById("sidebar");
if (sidebarToggle && sidebarEl) {
  sidebarToggle.addEventListener("click", () => sidebarEl.classList.toggle("open"));
}

// --- images collées (Ctrl+V n'a pas de panneau : on cible le panneau FOCUS) ---
window.addEventListener("paste", (e) => {
  const files = [];
  for (const item of e.clipboardData.items) {
    if (item.type.startsWith("image/")) files.push(item.getAsFile());
  }
  const fp = focusedPane();
  if (files.length && fp) addImages(fp, files);
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

// --- toggle session privée (local_only : les sous-agents restent sur le modèle de la session) ---
const localOnlyCb = document.getElementById("local-only-cb");
if (localOnlyCb) {
  localOnlyCb.addEventListener("change", () => {
    if (activeTab()) activeTab().localOnly = localOnlyCb.checked;
    const fd = new FormData();
    fd.append("local_only", localOnlyCb.checked ? "1" : "0");
    fetch("/local_only", { method: "POST", body: fd });
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
    machineUnloaded = false; // re-sélection = warmup relancé côté serveur
    scheduleMachineRefresh();
  }
});
syncSkillsMaster();

// --- état du modèle sur la machine (chargé / chargement / déchargé / libre / serveur off) ---
let machineTimer = null;
// Déchargement MANUEL (bouton) : sans ce flag, un modèle local sélectionné mais non
// chargé s'afficherait « chargement… » à tort. Levé dès que le modèle recharge.
let machineUnloaded = false;
async function refreshMachineState() {
  const chip = document.getElementById("machine-chip");
  const unloadBtn = document.getElementById("machine-unload");
  const startBtn = document.getElementById("machine-server-start");
  const stopBtn = document.getElementById("machine-server-stop");
  const hideActions = () => {
    for (const b of [unloadBtn, startBtn, stopBtn]) if (b) b.hidden = true;
  };
  if (!chip) return "";
  let d;
  try {
    d = await (await fetch("/machine_state")).json();
  } catch {
    chip.textContent = "";
    hideActions();
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
    if (d.starting) {
      // Serveur lancé par loom.web (auto ou bouton), pas encore joignable.
      cls = "busy";
      text = "machine · démarrage du serveur…";
    } else {
      cls = "off";
      text = "machine · serveur local éteint";
    }
  } else if (d.model_loaded) {
    if (machineUnloaded) {
      // Unload demandé mais llama-swap n'a pas fini de tuer le llama-server (~2 s) :
      // état transitoire BUSY pour que le suivi continue jusqu'au vrai état final.
      cls = "busy";
      text = "machine · libération…";
    } else {
      cls = "on";
      text = "machine · " + d.model + " chargé";
    }
  } else if (d.loading) {
    // état « starting » réel de llama-swap (le chargement peut prendre 1-3 min).
    // Un chargement en cours invalide un déchargement manuel antérieur.
    machineUnloaded = false;
    cls = "busy";
    text = "machine · " + d.model + " chargement…";
  } else if (machineUnloaded) {
    cls = "free";
    text = "machine · modèle déchargé (VRAM libre)";
  } else {
    cls = "busy";
    text = "machine · " + d.model + " chargement…";
  }
  chip.className = "machine-chip " + cls;
  chip.textContent = text;
  // Actions contextuelles : chaque bouton n'apparaît que quand il a du sens.
  // « décharger » : un modèle local occupe réellement la VRAM — PAS pendant un
  // chargement (llama-swap ignore l'unload d'un modèle en état « starting »).
  if (unloadBtn)
    unloadBtn.hidden = !(
      d.mode === "home" &&
      d.reachable &&
      d.any_loaded &&
      !d.loading &&
      !machineUnloaded
    );
  // « démarrer le serveur » : modèle local sélectionné, serveur éteint, pas déjà en route.
  if (startBtn)
    startBtn.hidden = !(d.mode === "home" && !d.reachable && !d.starting);
  // « éteindre le serveur » : seulement l'instance GÉRÉE par loom.web (jamais une stack
  // lancée à la main dans un terminal).
  if (stopBtn) stopBtn.hidden = !(d.reachable && d.managed);
  return cls;
}
// Rafraîchit maintenant puis re-sonde : les 3 premiers passages (à 800 ms) laissent le POST
// /model se persister avant de figer l'état ; ensuite on continue tant que c'est TRANSITOIRE
// (démarrage serveur / chargement / libération). Borné à ~2 min : un démarrage à froid
// (llama-swap + chargement d'un 35B) dépasse largement les 40 s de l'ancienne borne.
function scheduleMachineRefresh() {
  if (machineTimer) clearTimeout(machineTimer);
  let tries = 0;
  const tick = async () => {
    const cls = await refreshMachineState();
    tries += 1;
    if ((tries < 3 || cls === "busy") && tries < 42) {
      machineTimer = setTimeout(tick, tries < 3 ? 800 : 3000);
    }
  };
  tick();
}
scheduleMachineRefresh();
// Boutons machine : chaque action POST puis re-suit l'état (le chip raconte la suite).
// « décharger le modèle » : libère la VRAM sans changer de sélection ni quitter Loom
// (llama-swap rechargera à la prochaine requête).
document.getElementById("machine-unload")?.addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    await fetch("/machine/unload", { method: "POST" });
    machineUnloaded = true;
  } catch {
    /* serveur local injoignable : l'état re-sondé ci-dessous l'affichera */
  } finally {
    btn.disabled = false;
  }
  scheduleMachineRefresh();
});
// « démarrer le serveur » : trigger manuel du serveur modèle (sinon il part tout seul à
// la sélection d'un modèle local ou à la première interaction).
document
  .getElementById("machine-server-start")
  ?.addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
      await fetch("/machine/server/start", { method: "POST" });
    } catch {
      /* loom.web injoignable : rien à faire de plus ici */
    } finally {
      btn.disabled = false;
    }
    scheduleMachineRefresh();
  });
// « éteindre le serveur » : tue l'arbre complet (llama-swap + llama-server) -> RAM/VRAM
// rendues. Ne s'affiche que pour l'instance gérée par loom.web.
document
  .getElementById("machine-server-stop")
  ?.addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
      await fetch("/machine/server/stop", { method: "POST" });
      machineUnloaded = false;
    } catch {
      /* idem : l'état re-sondé fera foi */
    } finally {
      btn.disabled = false;
    }
    scheduleMachineRefresh();
  });

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
    setSkillDrawerMode("edit");
    const r = await fetch("/skill?name=" + encodeURIComponent(name));
    const d = await r.json();
    if (!r.ok || d.error) {
      skStatus && (skStatus.textContent = d.error || "chargement impossible");
      return;
    }
    skCurrent = d.name;
    if (skName) skName.textContent = d.name;
    if (skDesc)
      skDesc.textContent =
        (d.origin ? "[" + d.origin + "] " : "") + (d.description || "");
    // Bouton Supprimer : seulement pour les skills gérés par l'utilisateur
    // (appris / ajoutés) — jamais le package ni les plugins.
    const dbtn = document.getElementById("skdr-delete");
    if (dbtn) dbtn.hidden = !d.deletable;
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
// Suppression d'un skill (appris / ajouté user) : MÊME popover que les sessions.
// Délégation sur document : le panneau skills est REMPLACÉ à chaque toggle.
async function doDeleteSkill(name, fromDrawer) {
  const fd = new FormData();
  fd.append("name", name);
  try {
    const r = await fetch("/skill/delete", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || d.error) {
      if (fromDrawer && skStatus)
        skStatus.textContent = d.error || "suppression impossible";
      else console.warn("skill/delete:", d.error);
      return;
    }
    if (fromDrawer) closeSkillDrawer();
    postSkillsToggle();
  } catch (err) {
    if (fromDrawer && skStatus) skStatus.textContent = "erreur : " + err;
  }
}
document.addEventListener("click", (e) => {
  const del = e.target.closest && e.target.closest(".skill-del");
  if (!del) return;
  openConfirmPop(del.closest(".skill-row") || del, "Supprimer ce skill ?", () =>
    doDeleteSkill(del.dataset.name, false),
  );
});
document.getElementById("skdr-delete")?.addEventListener("click", (e) => {
  if (!skCurrent) return;
  openConfirmPop(
    e.currentTarget,
    "Supprimer ce skill ?",
    () => doDeleteSkill(skCurrent, true),
    "left",
  );
});

// --- création d'un skill (mode « create » du drawer) ---
// « + new » (entête Skills, comme les sessions) ouvre le drawer VIDE : nom éditable +
// description ; « Générer » fait rédiger le SKILL.md complet par le modèle ; « Créer »
// écrit dans var/skills_user et bascule le drawer en mode édition normal.
const skNameInput = document.getElementById("skdr-name-input");
const skGenRow = document.getElementById("skdr-genrow");
const skGenDesc = document.getElementById("skdr-gen-desc");
let skMode = "edit";
// BROUILLON de création persistant : rouvrir + new (ou fermer/rouvrir le drawer)
// REPREND où on en était — saisie conservée, génération en fond retrouvée — au lieu
// de repartir d'une page blanche (vécu : perte de la demande + état illisible).
const skDraft = { name: "", desc: "", body: "" };
let skGenBusy = false;
skNameInput?.addEventListener("input", () => {
  if (skMode === "create") skDraft.name = skNameInput.value;
});
skGenDesc?.addEventListener("input", () => {
  skDraft.desc = skGenDesc.value;
});
document.getElementById("skdr-body")?.addEventListener("input", () => {
  if (skMode === "create" && skBody) skDraft.body = skBody.value;
});

function setSkillDrawerMode(mode) {
  skMode = mode;
  const create = mode === "create";
  if (skName) skName.hidden = create;
  if (skNameInput) skNameInput.hidden = !create;
  if (skGenRow) skGenRow.hidden = !create;
  const el = (id) => document.getElementById(id);
  if (el("skdr-create")) el("skdr-create").hidden = !create;
  if (el("skdr-save-session")) el("skdr-save-session").hidden = create;
  if (el("skdr-save-global")) el("skdr-save-global").hidden = create;
  if (el("skdr-delete") && create) el("skdr-delete").hidden = true;
}
function openSkillCreator() {
  skCurrent = null;
  setSkillDrawerMode("create");
  if (skName) skName.textContent = "";
  if (skDesc)
    skDesc.textContent =
      "[user] nouveau skill — décris-le puis Générer, ou écris le SKILL.md";
  // Reprise du brouillon (jamais de page blanche imposée) + état de génération.
  if (skNameInput) skNameInput.value = skDraft.name;
  if (skGenDesc) skGenDesc.value = skDraft.desc;
  if (skBody) skBody.value = skDraft.body;
  if (skStatus) skStatus.textContent = "";
  const genStatus = document.getElementById("skdr-gen-status");
  if (genStatus) genStatus.hidden = !skGenBusy;
  const gbtn = document.getElementById("skdr-generate");
  if (gbtn) gbtn.disabled = skGenBusy;
  openSkillDrawer();
  if (skNameInput && !skGenBusy) skNameInput.focus();
}
document.getElementById("skill-new")?.addEventListener("click", (e) => {
  // Dans le <summary> repliable : ne pas déclencher le pli/dépli de la section.
  e.preventDefault();
  e.stopPropagation();
  openSkillCreator();
});
document.getElementById("skdr-generate")?.addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  const genStatus = document.getElementById("skdr-gen-status");
  const description = skGenDesc ? skGenDesc.value.trim() : "";
  if (!description) {
    if (genStatus) {
      genStatus.hidden = false;
      genStatus.textContent = "décris d'abord le skill dans le champ ci-dessus";
      setTimeout(() => (genStatus.hidden = true), 2500);
    }
    return;
  }
  const fd = new FormData();
  fd.append("name", skNameInput ? skNameInput.value.trim() : "");
  fd.append("description", description);
  // Feedback VISIBLE dans la zone de génération (pas relégué au pied du drawer) :
  // point pulsant + texte, bouton neutralisé le temps de l'appel.
  btn.disabled = true;
  const oldLabel = btn.textContent;
  btn.textContent = "génération…";
  if (genStatus) {
    genStatus.hidden = false;
    genStatus.textContent =
      "le modèle rédige le SKILL.md… (peut prendre ~1 min si le serveur vient de démarrer)";
  }
  if (skStatus) skStatus.textContent = "";
  skGenBusy = true;
  // La génération peut DÉMARRER le serveur modèle (auto-start) : suivre le chip machine.
  scheduleMachineRefresh();
  // Le drawer peut être FERMÉ pendant la rédaction (elle continue en fond) : à la
  // fin on range le brouillon et on prévient par un toast — jamais de résultat perdu
  // ni d'attente muette.
  const drawerVisible = () =>
    skDrawer && !skDrawer.hidden && skMode === "create";
  try {
    const r = await fetch("/skill/generate", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || d.error) {
      const msg = d.error || "génération impossible";
      if (drawerVisible() && genStatus) genStatus.textContent = msg;
      else showToast("skill : " + msg);
      return;
    }
    skDraft.body = d.source || "";
    if (drawerVisible()) {
      if (skBody) skBody.value = skDraft.body;
      if (genStatus) genStatus.hidden = true;
      if (skStatus)
        skStatus.textContent = "brouillon généré — relis, ajuste, puis Créer";
    } else {
      showToast("skill : brouillon prêt", [
        { label: "ouvrir", onClick: () => openSkillCreator() },
      ]);
    }
  } catch (err) {
    if (drawerVisible() && genStatus) genStatus.textContent = "erreur : " + err;
    else showToast("skill : erreur de génération — " + err);
  } finally {
    skGenBusy = false;
    btn.disabled = false;
    btn.textContent = oldLabel;
    const gs = document.getElementById("skdr-gen-status");
    if (gs && !drawerVisible()) gs.hidden = true;
  }
});
document.getElementById("skdr-create")?.addEventListener("click", async () => {
  const name = skNameInput ? skNameInput.value.trim() : "";
  if (!name) {
    if (skStatus) skStatus.textContent = "donne un nom au skill";
    return;
  }
  const fd = new FormData();
  fd.append("name", name);
  fd.append("description", skGenDesc ? skGenDesc.value.trim() : "");
  fd.append("body", skBody ? skBody.value : "");
  if (skStatus) skStatus.textContent = "création…";
  try {
    const r = await fetch("/skill/create", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || d.error) {
      if (skStatus) skStatus.textContent = d.error || "création impossible";
      return;
    }
    // Créé : le brouillon a rempli son office, on repart propre au prochain + new.
    skDraft.name = skDraft.desc = skDraft.body = "";
    await postSkillsToggle();
    setSkillDrawerMode("edit");
    openSkillEditor(d.name);
  } catch (err) {
    if (skStatus) skStatus.textContent = "erreur : " + err;
  }
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
    smRestorePos();
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

// --- moniteur déplaçable : on attrape la boîte n'importe où (aucun élément interactif
// dedans), on la pose où on veut ; position mémorisée et re-clampée dans la fenêtre
// à chaque affichage (changement de résolution / fenêtre redimensionnée). ---
const SM_POS_KEY = "loomSysmonPos";
function smPlace(el, l, t) {
  l = Math.max(0, Math.min(l, window.innerWidth - el.offsetWidth));
  t = Math.max(0, Math.min(t, window.innerHeight - el.offsetHeight));
  el.style.left = l + "px";
  el.style.top = t + "px";
  el.style.right = "auto";
}
function smRestorePos() {
  const el = document.getElementById("sysmon");
  if (!el) return;
  try {
    const saved = JSON.parse(localStorage.getItem(SM_POS_KEY) || "null");
    if (saved) smPlace(el, saved.l, saved.t);
  } catch {
    /* position corrompue : on garde le coin par défaut */
  }
}
(function () {
  const el = document.getElementById("sysmon");
  if (!el) return;
  let grab = null; // décalage curseur -> coin de la boîte pendant le drag
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const r = el.getBoundingClientRect();
    grab = { x: e.clientX - r.left, y: e.clientY - r.top };
    el.classList.add("sm-drag");
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!grab) return;
    smPlace(el, e.clientX - grab.x, e.clientY - grab.y);
  });
  const drop = () => {
    if (!grab) return;
    grab = null;
    el.classList.remove("sm-drag");
    const r = el.getBoundingClientRect();
    localStorage.setItem(SM_POS_KEY, JSON.stringify({ l: r.left, t: r.top }));
  };
  el.addEventListener("pointerup", drop);
  el.addEventListener("lostpointercapture", drop);
})();

// --- Gestionnaire de modèles distants (panneau engrenage) : ajout/édition/suppression À CHAUD
// (le backend monte la route sans redémarrer). Reconstruit le <select> après chaque mutation. ---
(function () {
  const rmList = document.getElementById("rm-list");
  const rmForm = document.getElementById("rm-form");
  const rmAddBtn = document.getElementById("rm-add-btn");
  if (!rmList || !rmForm || !rmAddBtn) return;
  const $ = (id) => document.getElementById(id);

  // DEUX blocs (machine vs distant), EN PHASE avec le rendu serveur (_models.html) :
  // la couleur par type (classe opt-<type>) distingue texte/image/vidéo DANS le bloc
  // machine ; re-teinte du select fermé après reconstruction.
  function rebuildModelSelect(payload) {
    const sel = document.getElementById("model-select");
    if (!sel || !payload) return;
    const cur = sel.value;
    const typ = (m) =>
      m.remote ? "remote" : m.video ? "video" : m.image ? "image" : "home";
    // Bloc home ordonné par type (texte, image, vidéo) : couleurs contiguës.
    const rank = { home: 0, image: 1, video: 2, remote: 0 };
    const groups = [
      ["home", (m) => !m.remote],
      ["distant · api", (m) => m.remote],
    ];
    sel.innerHTML = groups
      .map(([label, keep]) => {
        const opts = payload
          .filter(keep)
          .sort((a, b) => rank[typ(a)] - rank[typ(b)]);
        if (!opts.length) return "";
        return (
          '<optgroup label="' + label + '">' +
          opts
            .map(
              (m) =>
                '<option class="opt-' + typ(m) + '" value="' + esc(m.id) + '"' +
                (m.desc ? ' title="' + esc(m.desc) + '"' : "") +
                (m.id === cur ? " selected" : "") +
                ">" + esc(m.id) + "</option>",
            )
            .join("") +
          "</optgroup>"
        );
      })
      .join("");
    paintModelSelect();
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
          (m.has_key
            ? m.key_hint
              ? ' · clé ' + esc(m.key_hint)
              : ""
            : ' · <span class="rm-nokey">sans clé</span>');
        // config = défini dans local.toml (éditable ici aussi) ; sinon géré par l'UI.
        const tag = m.managed ? "" : '<span class="rm-tag">config</span>';
        // Édition pour TOUS (config inclus) ; suppression réservée aux modèles gérés par l'UI.
        const btns =
          '<button type="button" class="rm-ic rm-edit" data-id="' +
          esc(m.id) +
          '" title="Éditer">✎</button>' +
          (m.managed
            ? '<button type="button" class="rm-ic rm-del" data-id="' +
              esc(m.id) +
              '" title="Supprimer">✕</button>'
            : "");
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

  // Rafraîchissement GLOBAL (sélecteur + panneau) : appelé par le flux SSE quand le
  // wizard /add-model monte un modèle à chaud — sans ça, « disponible dans le
  // sélecteur » n'était vrai qu'après un rechargement de page.
  window.refreshModels = async function () {
    try {
      const d = await (await fetch("/models/config")).json();
      cache = d.remotes || [];
      renderList(cache);
      rebuildModelSelect(d.models);
    } catch {}
  };

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
      rec && rec.has_key
        ? "clé actuelle " + (rec.key_hint || "•••") + " — vide = inchangée"
        : "clé API (propre à cette machine)";
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

  // Rafraîchit la liste à l'ouverture de la console de config (le bloc modèles y vit
  // désormais, plus dans l'engrenage) + une fois au chargement.
  const cfgOpen = document.getElementById("cfg-open");
  if (cfgOpen) cfgOpen.addEventListener("click", load);
  load();
})();

// --- Console de configuration (modal) : tous les paramètres réels, deux couches
// commun/système, édition en direct des vrais fichiers TOML (commentaires préservés backend). ---
(function () {
  const modal = document.getElementById("config-modal");
  const body = document.getElementById("config-body");
  const openBtn = document.getElementById("cfg-open");
  if (!modal || !body || !openBtn) return;

  // Onglets : une seule section affichée (commun | systeme | modeles), pas de scroll global.
  let activeTab = "commun";
  function showTab(tab) {
    activeTab = tab;
    document
      .querySelectorAll("#cfg-tabs .cfg-tab")
      .forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    // Panneaux hors #config-body (modèles locaux, distants) togglés par leur data-tab.
    document
      .querySelectorAll("#config-modal .cfg-panel")
      .forEach((p) => (p.hidden = p.dataset.tab !== tab));
    // Groupes de couche rendus dans #config-body (commun / systeme).
    body
      .querySelectorAll(".cfg-group")
      .forEach((g) => (g.hidden = g.dataset.tab !== tab));
  }
  document
    .querySelectorAll("#cfg-tabs .cfg-tab")
    .forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
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

  // Ligne LÉGÈRE : libellé + (i) + champ. La méta (nature/effet) va en infobulle ; « modifié »
  // se signale par un liseré discret + le lien réinitialiser (rien sur les lignes au défaut).
  function rowHtml(p) {
    // "Modifié" = surcharge PROPRE À LA MACHINE (local.toml) uniquement. Une valeur qui vient
    // de defaults.toml est le défaut livré commun -> pas marquée (sinon tout est "modifié").
    const customized = p.source === "systeme";
    const restart =
      p.applies === "restart"
        ? ' <span class="cfg-restart" title="Prise en compte au redémarrage de Loom">redémarrage</span>'
        : "";
    // Infobulle = explication en clair, sans jargon interne (plus de "(libre)"/"(override)").
    const tip = esc(p.help);
    return (
      '<div class="cfg-row' +
      (customized ? " customized" : "") +
      '" data-section="' +
      esc(p.section) +
      '" data-key="' +
      esc(p.key) +
      '" data-type="' +
      esc(p.type) +
      '">' +
      '<div class="cfg-row-main"><span class="cfg-label">' +
      esc(p.label) +
      "</span>" +
      '<span class="cfg-i" title="' +
      tip +
      '">i</span>' +
      restart +
      "</div>" +
      '<div class="cfg-control">' +
      control(p) +
      '<button type="button" class="cfg-reset"' +
      (customized ? "" : " hidden") +
      ">réinitialiser</button>" +
      '<span class="cfg-saved">✓</span>' +
      "</div></div>"
    );
  }

  function sectionsHtml(list) {
    const order = [];
    const bySec = {};
    list.forEach((p) => {
      if (!bySec[p.sectionLabel]) {
        bySec[p.sectionLabel] = [];
        order.push(p.sectionLabel);
      }
      bySec[p.sectionLabel].push(p);
    });
    return order
      .map(
        (lbl) =>
          '<div class="cfg-section"><div class="cfg-sec-head">' +
          esc(lbl) +
          "</div>" +
          bySec[lbl].map(rowHtml).join("") +
          "</div>",
      )
      .join("");
  }

  const LAYER_GROUPS = [{ layer: "commun" }, { layer: "systeme" }];

  function render(data) {
    const params = [];
    (data.sections || []).forEach((s) =>
      s.params.forEach((p) =>
        params.push(Object.assign({ sectionLabel: s.label }, p)),
      ),
    );
    body.innerHTML = LAYER_GROUPS.map((g) => {
      const gp = params.filter((p) => p.layer === g.layer);
      if (!gp.length) return "";
      // Essentiel visible d'emblée ; le reste replié sous « Réglages avancés ».
      const ess = gp.filter((p) => !p.advanced);
      const adv = gp.filter((p) => p.advanced);
      const advHtml = adv.length
        ? '<details class="cfg-adv"><summary>Réglages avancés (' +
          adv.length +
          ")</summary>" +
          sectionsHtml(adv) +
          "</details>"
        : "";
      return (
        '<div class="cfg-group" data-tab="' +
        g.layer +
        '">' +
        sectionsHtml(ess) +
        advHtml +
        "</div>"
      );
    }).join("");
    showTab(activeTab); // applique la visibilité après (re)construction du contenu
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
    // "Modifié" (liseré + reset) seulement pour une surcharge MACHINE (local.toml). Un édit
    // commun va dans defaults.toml (le défaut livré) -> reste discret.
    const customized = j.source === "systeme";
    row.classList.toggle("customized", customized);
    const rb = row.querySelector(".cfg-reset");
    if (rb) rb.hidden = !customized;
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

// --- Onglet Modèles locaux : liste des modèles servis sur cette machine + édition du tuning
// d'offload GPU (context / n_gpu_layers / cpu_moe / n_cpu_moe) dans leur model.toml. ---
(function () {
  const list = document.getElementById("local-list");
  const cfgOpen = document.getElementById("cfg-open");
  if (!list) return;
  const esc = (s) =>
    String(s == null ? "" : s).replace(
      /[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
    );

  function numField(key, label, val, ph) {
    return (
      '<label class="lm-field"><span>' +
      label +
      '</span><input data-key="' +
      key +
      '" type="number" value="' +
      (val == null ? "" : esc(val)) +
      '"' +
      (ph ? ' placeholder="' + ph + '"' : "") +
      "></label>"
    );
  }

  function render(models) {
    if (!models || !models.length) {
      list.innerHTML = '<div class="rm-empty">aucun modèle local découvert</div>';
      return;
    }
    list.innerHTML = models
      .map(function (m) {
        const go = m.size_mb ? Math.round((m.size_mb / 1024) * 10) / 10 + " Go" : "";
        const sub =
          esc(m.filename) +
          (go ? " · " + go : "") +
          (m.repo ? " · " + esc(m.repo) : "") +
          (m.n_layers ? " · " + m.n_layers + " couches" : "") +
          (m.vision ? " · vision" : "");
        const tune =
          numField("context", "Contexte (tokens)", m.context, "") +
          numField("n_gpu_layers", "Couches sur GPU", m.n_gpu_layers, "auto") +
          numField("n_cpu_moe", "Experts MoE en RAM (n)", m.n_cpu_moe, "—") +
          '<label class="lm-field lm-bool"><input data-key="cpu_moe" type="checkbox"' +
          (m.cpu_moe ? " checked" : "") +
          "><span>Tous les experts en RAM (cpu_moe)</span></label>";
        return (
          '<div class="lm-item" data-id="' +
          esc(m.id) +
          '"><div class="lm-head"><span class="lm-id">' +
          esc(m.id) +
          '</span><span class="lm-tag">local</span><span class="lm-saved">enregistré</span></div>' +
          '<div class="lm-sub">' +
          sub +
          '</div><div class="lm-tune">' +
          tune +
          "</div></div>"
        );
      })
      .join("");
  }

  async function load() {
    try {
      const d = await (await fetch("/models/local")).json();
      render(d.models);
    } catch {}
  }

  async function save(item, ctl) {
    const value = ctl.type === "checkbox" ? ctl.checked : ctl.value;
    const r = await fetch("/models/local/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: item.dataset.id, key: ctl.dataset.key, value }),
    });
    const j = await r.json();
    if (j.ok) {
      const s = item.querySelector(".lm-saved");
      if (s) {
        s.classList.add("show");
        setTimeout(() => s.classList.remove("show"), 1400);
      }
    }
  }

  list.addEventListener("change", (e) => {
    const item = e.target.closest(".lm-item");
    if (item && e.target.dataset.key) save(item, e.target);
  });
  if (cfgOpen) cfgOpen.addEventListener("click", load);
  load();
})();


// --- Teinte du sélecteur de modèle FERMÉ : les optgroups/couleurs ne sont visibles
// que dans le popup ouvert -> le select fermé reprend la couleur du TYPE sélectionné
// (classe sel-<type>, lue sur l'option choisie ; home = neutre, pas de classe). ---
function paintModelSelect() {
  const sel = document.getElementById("model-select");
  if (!sel || !sel.options.length) return;
  const opt = sel.options[sel.selectedIndex] || sel.options[0];
  const m = opt.className.match(/opt-(\w+)/);
  sel.className = m && m[1] !== "home" ? "sel-" + m[1] : "";
  // Moniteur système : utile quand la MACHINE travaille (local/image/vidéo),
  // superflu sur un distant — bascule immédiate, /machine_state confirme après.
  if (typeof setSysmonVisible === "function")
    setSysmonVisible(!(m && m[1] === "remote"));
}
document.addEventListener("change", (e) => {
  if (e.target && e.target.id === "model-select") paintModelSelect();
});
// Le POST /model (htmx) REMPLACE le <select> (hx-swap outerHTML) : re-teinter après
// chaque swap — l'écouteur ci-dessus survit (délégué au document), pas les classes.
document.body.addEventListener("htmx:afterSwap", () => paintModelSelect());
paintModelSelect();
