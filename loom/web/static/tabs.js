// loom/web/static/tabs.js — issu du decoupage de app.js (comportement constant).
import {
  focusedPane,
  paneShowing,
  panesFor,
  renderMsgNav,
  renderPane,
  state,
  uid,
} from "./state.js";
import {
  importSessionFile,
  newSessionTab,
  openTabMenu,
  paintModelSelect,
  postForm,
  reflectWorkdir,
  scheduleMachineRefresh,
  setMetrics,
  updateUsageMeter,
  wireSidDrag,
} from "./panels.js";
import {
  autosize,
  hidePal,
  removePane,
  savePanesLayout,
  setPaneActivity,
  syncComposer,
} from "./panes.js";
import { set_loomWorkdir } from "./shared.js";

export const tabbarEl = document.getElementById("tabbar");

export function modelType(modelId) {
  if (!modelId) return "";
  const opt = document.querySelector(
    `#model-select option[value="${CSS.escape(modelId)}"]`,
  );
  const m = opt && opt.className.match(/opt-(\w+)/);
  return m ? m[1] : "";
}

export function renderPaneHeads() {
  for (const p of state.panes) {
    const t = state.tabs[p.sid];
    p.headTitle.textContent = t ? t.title || "session" : "—";
    p.headDot.className = "pane-dot " + (t && t.streaming ? "gen" : "idle");
    // La classe de type vit sur le PANNEAU : ses variables d'accent re-thèment tout
    // l'intérieur (composer, outils, focus…), pas seulement le bandeau.
    const mt = t ? modelType(t.model) : "";
    for (const c of [...p.el.classList])
      if (c.startsWith("mdl-")) p.el.classList.remove(c);
    if (mt) p.el.classList.add("mdl-" + mt);
  }
}

export function renderTabs() {
  if (!tabbarEl) return;
  renderPaneHeads();
  tabbarEl.hidden = state.order.length === 0;
  tabbarEl.innerHTML = "";
  for (const sid of state.order) {
    const t = state.tabs[sid];
    if (!t) continue;
    const el = document.createElement("div");
    const mt = modelType(t.model);
    el.className =
      "tab" +
      (sid === state.active ? " active" : "") +
      (sid !== state.active && paneShowing(sid) ? " shown" : "") +
      (mt ? " mdl-" + mt : "");
    el.title =
      (t.title || "session") +
      (t.model ? " — " + t.model : " — modèle par défaut");
    // Split view : clic droit = menu (ouvrir à droite/en dessous…), glisser = déplacer
    // l'onglet vers un panneau.
    el.addEventListener("contextmenu", (e) => openTabMenu(e, sid));
    wireSidDrag(el, () => sid);
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
  plus.title = "Nouvelle session (clic droit : importer une conversation .zip)";
  plus.addEventListener("click", newSessionTab);
  // Clic droit = importer un export de session (le sélecteur de fichier s'ouvre).
  plus.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    let inp = document.getElementById("session-import-file");
    if (!inp) {
      inp = document.createElement("input");
      inp.type = "file";
      inp.accept = ".zip";
      inp.id = "session-import-file";
      inp.hidden = true;
      inp.addEventListener("change", () => {
        if (inp.files && inp.files[0]) importSessionFile(inp.files[0]);
        inp.value = "";
      });
      document.body.appendChild(inp);
    }
    inp.click();
  });
  tabbarEl.append(plus);
}

export function _userTexts(messages) {
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

export function _hydrateTimeline(t, messages) {
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

export function _replayTimeline(t, events) {
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
      case "harness":
        think = null;
        asst = null;
        add({ kind: "harness", hkind: d.kind, text: d.text });
        break;
      case "reasoning":
        if (asst) asst = null;
        if (!think)
          think = add({
            kind: "think",
            role: "",
            text: "",
            active: false,
            done: true,
          });
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
          byTool[tid] = add({
            id: tid,
            kind: "tool",
            name: d.name,
            pending: true,
          });
        break;
      }
      case "tool_result": {
        const tid = "tool:" + (d.id || d.name);
        let it = byTool[tid];
        if (!it)
          it = byTool[tid] = add({ id: tid, kind: "tool", name: d.name });
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

export async function _loadDisplay(t, sid, hasTimeline, messages) {
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

export const _tabLoads = {};

export function ensureTab(sid) {
  if (state.tabs[sid]) return Promise.resolve(state.tabs[sid]);
  if (_tabLoads[sid]) return _tabLoads[sid];
  const load = (async () => {
    let d;
    try {
      d = await (
        await fetch("/session_state?id=" + encodeURIComponent(sid))
      ).json();
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
    if (state.tabs[sid]) return state.tabs[sid]; // course perdue : le 1er arrivé fait foi
    state.tabs[sid] = t;
    state.order.push(sid);
    return t;
  })().finally(() => delete _tabLoads[sid]);
  _tabLoads[sid] = load;
  return load;
}

export async function openTab(sid) {
  if (!(await ensureTab(sid))) return;
  activateTab(sid);
}

export function setPaneSid(pane, sid) {
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
  // state.active n'est PAS écrit ici : focusSingletonsFor est le SEUL écrivain du
  // focus (bug du drop sur le panneau focus, même famille que 000d094).
  // La disposition persistée porte les sids des feuilles : tout changement d'onglet
  // affiché se sauve ici, LE point de mutation (idempotent, gestes utilisateur only).
  savePanesLayout();
}

export let _lastActivated = null;

export function focusSingletonsFor(sid) {
  const t = state.tabs[sid];
  state.active = sid;
  if (!t) return;
  // Le serveur suit ce focus (pour /model, /tools, /reset, /pick-folder qui opèrent
  // sur _cur). La rafale /machine_state part APRÈS la réponse : un tick immédiat
  // lirait l'ANCIENNE session active côté serveur et re-basculerait le moniteur
  // avec un état périmé (course vécue au switch d'onglet).
  if (_lastActivated !== sid) {
    _lastActivated = sid;
    postForm("/session/activate", { id: sid })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        // Vérité serveur à la bascule : le path affiché doit suivre CET onglet,
        // pas un cache périmé. /session/activate renvoie l'état complet (workspace,
        // model, title) précisément pour ça.
        if (d && d.workspace !== undefined) {
          t.workspace = d.workspace;
          if (d.title) t.title = d.title;
          if (d.model) t.model = d.model;
          set_loomWorkdir(t.workspace);
          try {
            localStorage.loomWorkdir = t.workspace;
          } catch (e) {
            /* sans effet */
          }
          reflectWorkdir();
        }
        scheduleMachineRefresh();
      })
      .catch(() => {});
  }
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
  // Le path suit CET onglet, même vide : sinon on garde l'affichage du path de
  // l'onglet PRÉCÉDENT (vécu à la bascule d'onglet : le modèle suivait, le path non).
  set_loomWorkdir(t.workspace || "");
  try {
    localStorage.loomWorkdir = t.workspace || "";
  } catch (e) {
    /* sans effet */
  }
  reflectWorkdir();
  if (t.meter) updateUsageMeter(t.meter);
  if (t.metrics)
    setMetrics(t.metrics.sent, t.metrics.recv, t.metrics.tokS, {
      done: !t.streaming,
    });
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

export function focusPane(target) {
  const i = typeof target === "number" ? target : state.panes.indexOf(target);
  const pane = state.panes[i];
  if (!pane) return;
  state.focusedIdx = i;
  state.panes.forEach((p, k) => p.el.classList.toggle("focused", k === i));
  if (pane.sid) focusSingletonsFor(pane.sid);
  else state.active = null;
  renderTabs();
}

export function activateTab(sid) {
  const t = state.tabs[sid];
  if (!t) return;
  // Déjà affiché dans un panneau -> on FOCUS ce panneau (pas de double affichage).
  const shown = paneShowing(sid);
  if (shown) {
    focusPane(shown);
    return;
  }
  const pane = focusedPane();
  if (!pane) return;
  setPaneSid(pane, sid);
  focusPane(pane);
}

export function closeTab(sid) {
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
    // Via setPaneSid (pas de mutation à la main) : palette fermée, activité effacée,
    // composer resynchronisé — le chemin manuel oubliait tout ça (bouton figé sur Stop).
    const pane = showing[0];
    const next = state.order[state.order.length - 1] || null;
    if (next) activateTab(next);
    else {
      setPaneSid(pane, null);
      state.active = null;
      renderTabs();
    }
  } else {
    renderTabs();
  }
}
