// loom/web/static/components.js — issu du decoupage de app.js (comportement constant).
import { html, useEffect, useRef, useState } from "./preact-htm.js";
import { opsFor, paneShowing, scheduleRenderFor, tab } from "./state.js";
import { renderPreviews, submitPane } from "./panes.js";
import { md } from "./render.js";

export function Think({ it }) {
  if (!it.text) return null;
  return html`<details class=${"think" + (it.active ? " active" : "")} open=${it.active}>
    <summary>réflexion${it.role ? " " + it.role : ""}</summary>
    <div>${it.text}</div>
  </details>`;
}

export function IORow({ tag, text, force }) {
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

export function ToolPill({ it }) {
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

export const AGENT_NAMES = [
  "Zidane", "Messi", "Ronaldo", "Mbappé", "Ronaldinho", "Maradona",
  "Benzema", "Mahrez", "Salah", "Modrić", "Iniesta", "Neymar",
];

export function _agentName(seed, i) {
  let h = 0;
  for (const c of String(seed || "")) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return AGENT_NAMES[(h + i) % AGENT_NAMES.length];
}

export function AgentLane({ it, name }) {
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

export function ParallelArena({ lanes }) {
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

export function PermAsk({ it, sid }) {
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

export function WizChoices({ it, sid }) {
  const pick = (label) => {
    if (it.decided) return;
    // STRICTEMENT le panneau qui montre cette session : jamais de reroutage vers le
    // focus (un « oui » de wizard posté dans une autre session serait un message chat).
    const pane = paneShowing(sid);
    if (!pane) return;
    opsFor(sid).patch(it.id, { decided: true, picked: label });
    // La réponse wizard part SEULE : les images en attente du composer ne s'invitent
    // pas dans un « oui » — elles restent pour le prochain vrai message.
    const staged = pane.pendingImages;
    pane.pendingImages = [];
    pane.input.value = label;
    submitPane(pane);
    pane.pendingImages = staged;
    renderPreviews(pane);
  };
  return html`<div class=${"wiz-choices" + (it.decided ? " decided" : "")}>
    ${it.options.map(
      (o) => html`<button key=${o} disabled=${!!it.decided}
        class=${it.picked === o ? "picked" : ""} onClick=${() => pick(o)}>${o}</button>`,
    )}
  </div>`;
}

export function UserMsg({ it, userIndex, sid }) {
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
    // Cible la session de CE panneau : sans session_id, le serveur tronque la session
    // focus (_cur) — qui peut être un AUTRE panneau (course avec /session/activate).
    fd.append("session_id", sid);
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
      // Pre-remplit la saisie du panneau qui montre cette session — JAMAIS un autre
      // panneau (le texte forké de A ne doit pas atterrir dans le composer de B).
      const pane = paneShowing(sid);
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

export function Assistant({ it }) {
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

export function enhance(el, raw) {
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

export function Item({ it, userIndex, sid }) {
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
    case "harness":
      // 3e voix : le garde-fou Loom (ni toi ni le modèle). Identité visuelle propre.
      return html`<div class="harness-line"><span class="harness-tag">Loom${it.hkind ? html` <span class="harness-kind">· ${it.hkind}</span>` : ""}</span><span class="harness-text">${it.text}</span></div>`;
    default:
      return null;
  }
}

export function App({ sid }) {
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
