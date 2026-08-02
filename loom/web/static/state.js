import { html, render } from "./preact-htm.js";
import { App } from "./components.js";

export const state = {
  tabs: {}, // sid -> { sid, title, timeline:[], streaming, abort, model, thinking, workspace, meter, metrics }
  order: [], // ordre des onglets ouverts
  panes: [], // Pane[] (créés par createPane : élément cloné de #pane-tpl + refs + état)
  focusedIdx: 0, // index du panneau focus dans panes
  active: null, // sid du panneau focus (écrit par focusSingletonsFor ; null si panneau vide)
  layoutRoot: null, // arbre de disposition (nœuds split dir/ratio, feuilles pane)
  maximized: null, // panneau en plein écran (bascule ⛶) ou null
};

export let _seq = 0;

export const uid = () => "i" + ++_seq;

export function tab(sid) {
  return state.tabs[sid];
}

export function activeTab() {
  return state.tabs[state.active] || null;
}

export function focusedPane() {
  return state.panes[state.focusedIdx] || state.panes[0] || null;
}

export function panesFor(sid) {
  return state.panes.filter((p) => p.sid === sid);
}

export function paneShowing(sid) {
  return state.panes.find((p) => p.sid === sid) || null;
}

export function renderPane(pane) {
  if (!pane || pane._raf) return;
  pane._raf = true;
  requestAnimationFrame(() => {
    pane._raf = false;
    render(html`<${App} sid=${pane.sid} />`, pane.root);
    if (pane.pin) pane.wrap.scrollTop = pane.wrap.scrollHeight;
    if (pane === focusedPane()) renderMsgNav(); // navigateur de prompts (panneau focus)
  });
}

export function scheduleRenderFor(sid) {
  panesFor(sid).forEach(renderPane);
}

export function _userMsgTexts() {
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

export function renderMsgNav() {
  const nav = document.getElementById("msg-nav");
  if (!nav) return;
  // En split, ce navigateur GLOBAL (fixé au bord droit de la fenêtre) se superpose au
  // panneau de droite et intercepte les clics sur son bandeau (⛶/✕ inaccessibles,
  // vécu 2026-07-19) : il n'a de sens qu'en vue simple.
  if (state.panes.length > 1) {
    nav.hidden = true;
    nav._sig = "";
    return;
  }
  // Signature AVANT toute extraction (session + nb de messages user) : appelé à chaque
  // frame du streaming, il ne doit pas extraire les textes tant que rien ne change.
  const t = activeTab();
  let count = 0;
  if (t) for (const it of t.timeline) if (it.kind === "user") count++;
  const sig = (state.active || "") + "|" + count;
  if (nav._sig === sig) return;
  nav._sig = sig;
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

export function opsFor(sid) {
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
