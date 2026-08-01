// loom/web/static/panes.js — issu du decoupage de app.js (comportement constant).
import { focusedPane, opsFor, paneShowing, panesFor, renderMsgNav, renderPane, state, tab } from "./state.js";
import { focusPane, setPaneSid } from "./tabs.js";
import { SID_MIME, dropTabOnPane, openTabMenu, postForm, showToast, wireSidDrag } from "./panels.js";
import { CMDS } from "./shared.js";
import { sendChat, sendHandoff } from "./chat.js";

export const panesEl = document.getElementById("panes");

export const paneTpl = document.getElementById("pane-tpl");

export const MAX_IMAGES = 6;

export function createPane() {
  const el = paneTpl.content.firstElementChild.cloneNode(true);
  const q = (sel) => el.querySelector(sel);
  const pane = {
    sid: null,
    el,
    headTitle: q(".pane-title"),
    headDot: q(".pane-dot"),
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

export function leafOf(pane, node, parent, key) {
  node = node || state.layoutRoot;
  if (!node) return null;
  if (node.type === "pane")
    return node.pane === pane ? { node, parent, key } : null;
  return (
    leafOf(pane, node.a, node, "a") || leafOf(pane, node.b, node, "b")
  );
}

export function splitPaneNode(target, dir, newPane, before) {
  const loc = leafOf(target);
  const leaf = { type: "pane", pane: newPane };
  if (!loc) {
    state.layoutRoot = leaf;
    return;
  }
  const split = {
    type: "split",
    dir,
    ratio: 0.5,
    a: before ? leaf : loc.node,
    b: before ? loc.node : leaf,
  };
  if (loc.parent) loc.parent[loc.key] = split;
  else state.layoutRoot = split;
}

export function detachLeaf(loc) {
  const sibling = loc.key === "a" ? loc.parent.b : loc.parent.a;
  for (const k of Object.keys(loc.parent)) delete loc.parent[k];
  Object.assign(loc.parent, sibling);
}

export const clampRatio = (r) => Math.min(0.85, Math.max(0.15, r));

export function renderLayout() {
  if (!panesEl || !state.layoutRoot) return;
  panesEl.classList.toggle("multi", state.panes.length > 1);
  if (state.maximized && !state.panes.includes(state.maximized))
    state.maximized = null;
  if (state.maximized) {
    state.maximized.el.style.flex = "1 1 0";
    panesEl.replaceChildren(state.maximized.el);
    requestAnimationFrame(updatePlaceholders);
    return;
  }
  const build = (node) => {
    if (node.type === "pane") {
      node.pane.el.style.flex = "1 1 0";
      return node.pane.el;
    }
    const box = document.createElement("div");
    box.className = "split-box " + (node.dir === "row" ? "srow" : "scol");
    const a = build(node.a);
    const b = build(node.b);
    a.style.flex = `${node.ratio} 1 0`;
    b.style.flex = `${1 - node.ratio} 1 0`;
    box.append(a, makeSash(node, a, b), b);
    return box;
  };
  panesEl.replaceChildren(build(state.layoutRoot));
  requestAnimationFrame(updatePlaceholders);
  renderMsgNav(); // suit le nb de panneaux (masqué en split, restauré en vue simple)
}

export function makeSash(node, aEl, bEl) {
  const sash = document.createElement("div");
  const vertical = node.dir === "row"; // colonnes côte à côte -> séparateur vertical
  sash.className = "sash " + (vertical ? "sash-v" : "sash-h");
  sash.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const box = sash.parentElement;
    const r = box.getBoundingClientRect();
    sash.classList.add("dragging");
    sash.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const frac = vertical
        ? (ev.clientX - r.left) / r.width
        : (ev.clientY - r.top) / r.height;
      node.ratio = clampRatio(frac);
      aEl.style.flex = `${node.ratio} 1 0`;
      bEl.style.flex = `${1 - node.ratio} 1 0`;
    };
    const up = () => {
      sash.classList.remove("dragging");
      sash.removeEventListener("pointermove", move);
      sash.removeEventListener("pointerup", up);
      sash.removeEventListener("lostpointercapture", up);
      updatePlaceholders();
      savePanesLayout();
    };
    sash.addEventListener("pointermove", move);
    sash.addEventListener("pointerup", up);
    sash.addEventListener("lostpointercapture", up);
  });
  return sash;
}

export function updatePlaceholders() {
  const widths = state.panes.map((p) => p.el.clientWidth);
  state.panes.forEach((p, i) => {
    const w = widths[i];
    const ph =
      w > 0 && w < 340
        ? "Écris une demande…"
        : w > 0 && w < 560
          ? "Écris une demande… (« / » commandes)"
          : "Écris une demande — Loom agit avec ses outils… (« / » pour les commandes)";
    if (p.input.placeholder !== ph) p.input.placeholder = ph;
  });
}

export function movePaneToEdge(src, target, dir, before) {
  if (src === target) return;
  const loc = leafOf(src);
  if (!loc || !loc.parent) return; // src est la racine seule : rien à déplacer
  detachLeaf(loc);
  splitPaneNode(target, dir, src, before);
  renderLayout();
  focusPane(src);
  savePanesLayout();
}

export function toggleMaximize(pane) {
  state.maximized = state.maximized === pane ? null : pane;
  renderLayout();
  focusPane(pane);
}

export function savePanesLayout() {
  const ser = (n) =>
    n.type === "pane"
      ? { sid: n.pane.sid }
      : { dir: n.dir, ratio: n.ratio, a: ser(n.a), b: ser(n.b) };
  try {
    localStorage.loomPanes = JSON.stringify(
      state.layoutRoot ? { tree: ser(state.layoutRoot) } : null,
    );
  } catch (e) {
    /* localStorage indispo : la disposition ne survivra pas au rechargement */
  }
}

export function addPane(sid, dir, target, before) {
  if (state.panes.length >= 4) {
    showToast("4 panneaux maximum");
    return null;
  }
  const shown = sid && paneShowing(sid);
  if (shown) {
    focusPane(shown);
    return null;
  }
  const pane = createPane();
  state.panes.push(pane);
  splitPaneNode(target || focusedPane(), dir || "row", pane, before);
  if (sid && state.tabs[sid]) setPaneSid(pane, sid);
  else renderPane(pane); // panneau vide : peindre l'état « écris une demande »
  state.maximized = null;
  renderLayout();
  focusPane(pane);
  savePanesLayout();
  return pane;
}

export function removePane(pane) {
  const i = state.panes.indexOf(pane);
  if (i < 0 || state.panes.length <= 1) return;
  // Le focus suit l'IDENTITÉ du panneau focus, pas son index : retirer un panneau
  // situé AVANT lui décalait tout et basculait focus + _cur serveur sur la mauvaise
  // session (repro : A|B|C focus B, fermer A -> focus sautait sur C).
  const keep = focusedPane();
  // Brouillon sauvé avant de perdre le panneau (l'onglet reste ouvert dans la barre).
  const t = state.tabs[pane.sid];
  if (t) t.draft = pane.input.value;
  const loc = leafOf(pane);
  if (loc && loc.parent) detachLeaf(loc);
  state.panes.splice(i, 1);
  pane.el.remove();
  const ki = state.panes.indexOf(keep);
  renderLayout();
  focusPane(ki >= 0 ? ki : Math.min(state.focusedIdx, state.panes.length - 1));
  savePanesLayout();
}

export function setPaneActivity(pane, label) {
  if (!pane || !pane.activityEl) return;
  if (!label) {
    pane.activityEl.hidden = true;
    return;
  }
  pane.activityEl.hidden = false;
  if (pane.activityLabel.textContent !== label)
    pane.activityLabel.textContent = label;
}

export function setActivityFor(sid, label) {
  panesFor(sid).forEach((p) => setPaneActivity(p, label));
}

export function syncComposer(pane) {
  if (pane && pane.sendBtn)
    pane.sendBtn.textContent = tab(pane.sid)?.streaming ? "Stop" : "Envoyer";
}

export function syncComposersFor(sid) {
  panesFor(sid).forEach(syncComposer);
}

export function autosize(pane) {
  if (!pane || !pane.input) return;
  pane.input.style.height = "auto";
  pane.input.style.height = Math.min(pane.input.scrollHeight, 200) + "px";
}

export const _thumbUrls = new Map();

export function _thumbUrl(file) {
  let u = _thumbUrls.get(file);
  if (!u) {
    u = URL.createObjectURL(file);
    _thumbUrls.set(file, u);
  }
  return u;
}

export function _dropThumb(file) {
  const u = _thumbUrls.get(file);
  if (u) {
    URL.revokeObjectURL(u);
    _thumbUrls.delete(file);
  }
}

export function renderPreviews(pane) {
  pane.previewWrap.innerHTML = "";
  pane.pendingImages.forEach((file, i) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.src = _thumbUrl(file);
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

export function addImages(pane, files) {
  for (const f of files) {
    if (!f || !f.type.startsWith("image/")) continue;
    if (pane.pendingImages.length >= MAX_IMAGES) break; // les suivantes sont ignorées
    pane.pendingImages.push(f);
  }
  renderPreviews(pane);
}

export function removeImage(pane, i) {
  const [f] = pane.pendingImages.splice(i, 1);
  if (f) _dropThumb(f);
  renderPreviews(pane);
}

export function clearImages(pane) {
  pane.pendingImages.forEach(_dropThumb);
  pane.pendingImages = [];
  renderPreviews(pane);
}

export function palMatches(pane) {
  const v = pane.input.value;
  // Palette uniquement sur le PREMIER mot d'un message qui commence par « / » :
  // dès qu'un espace ou un retour ligne arrive, on est dans les arguments.
  if (!v.startsWith("/") || /[\s\n]/.test(v)) return [];
  const tok = v.slice(1).toLowerCase();
  return CMDS.filter((c) => c.name.slice(1).toLowerCase().startsWith(tok));
}

export function hidePal(pane) {
  if (pane && pane.palEl) pane.palEl.hidden = true;
}

export function renderPal(pane) {
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

export function palPick(pane, c) {
  pane.input.value = c.name + " ";
  autosize(pane);
  hidePal(pane);
  pane.input.focus();
}

export function submitPane(pane) {
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
  sendChat(sid, text, imgs).finally(() => {
    // Ne re-focus que si l'utilisateur n'est pas parti taper AILLEURS : la fin d'un
    // flux (parfois minutes plus tard) ne doit jamais voler le clavier d'un autre
    // panneau — sinon la suite de sa phrase partait dans la mauvaise session.
    const ae = document.activeElement;
    if (
      focusedPane() === pane &&
      (!ae || ae === document.body || pane.el.contains(ae))
    )
      pane.input.focus();
  });
  syncComposer(pane);
}

export function handoffMessage(sourceSid, targetSid, text, provenance) {
  return sendHandoff(sourceSid, targetSid, text, provenance);
}

export function stopPane(pane) {
  const t = tab(pane.sid);
  if (t && t.abort) t.abort.abort();
  if (pane.sid) postForm("/cancel", { session_id: pane.sid }).catch(() => {});
  syncComposer(pane);
}

export function wirePane(pane) {
  const input = pane.input;
  // Focus du panneau au premier geste dedans (split view) — capture, avant les boutons.
  const takeFocus = (e) => {
    // Cliquer « envoyer » dans un panneau source NON focus doit conserver le focus
    // courant : c'est précisément lui qui désigne la cible nominale du handoff.
    if (e?.target?.closest?.(".msg-send")) return;
    if (state.panes[state.focusedIdx] !== pane) focusPane(pane);
  };
  pane.el.addEventListener("pointerdown", takeFocus, true);
  // Le focus panneau SUIT aussi le focus CLAVIER (Tab entre panneaux) : sinon coller
  // une image ou les singletons topbar ciblaient un autre panneau que celui où on tape.
  pane.el.addEventListener("focusin", takeFocus);
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
  // ✕ du bandeau : ferme le PANNEAU (l'onglet et la session restent dans la barre).
  pane.el.querySelector(".pane-x")?.addEventListener("click", (e) => {
    e.stopPropagation();
    removePane(pane);
  });
  // ⛶ / double-clic sur le bandeau : bascule plein écran de CE panneau.
  pane.el.querySelector(".pane-max")?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleMaximize(pane);
  });
  const head = pane.el.querySelector(".pane-head");
  if (head) {
    head.addEventListener("dblclick", (e) => {
      if (e.target.closest("button")) return;
      toggleMaximize(pane);
    });
    // Le bandeau est la POIGNÉE du panneau : le saisir-glisser déplace la zone de
    // chat entière vers les zones directionnelles d'un autre panneau.
    wireSidDrag(head, () => pane.sid);
    // Clic droit sur le bandeau : même menu que l'onglet (split, chemin réel copiable).
    head.addEventListener("contextmenu", (ev) => openTabMenu(ev, pane.sid));
  }
  // Déposer un ONGLET (glissé depuis la barre) ou un PANNEAU (par son bandeau) :
  // zones DIRECTIONNELLES à la VS Code. L'overlay montre où il atterrit : centre =
  // remplacer/échanger, bord = scinder de ce côté. Jamais de double affichage.
  const overlay = pane.el.querySelector(".pane-overlay");
  // Rect capturé à l'ENTRÉE du drag (il ne bouge pas pendant), zone repeinte
  // seulement quand elle change — dragover tire à la cadence de la souris.
  let dragRect = null;
  let lastZone = null;
  let dragDepth = 0; // enter/leave comptés : les enfants traversés émettent des
  // dragleave intermédiaires qui tuaient l'overlay en plein panneau (« il faut
  // forcer pour qu'il propose », vécu 2026-07-19)
  const zoneAt = (e) => {
    const r = dragRect || pane.el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top) / r.height;
    // Centre = seulement le CŒUR du panneau (40 % médians) ; partout ailleurs, le
    // bord le PLUS PROCHE est proposé directement — plus besoin d'aller chercher
    // l'extrême bord pour obtenir un split.
    if (x > 0.3 && x < 0.7 && y > 0.3 && y < 0.7) return "center";
    const d = { left: x, right: 1 - x, top: y, bottom: 1 - y };
    return Object.keys(d).reduce((a, b) => (d[a] < d[b] ? a : b));
  };
  const ZONE_GEO = {
    center: [0, 0, 100, 100],
    left: [0, 0, 50, 100],
    right: [50, 0, 50, 100],
    top: [0, 0, 100, 50],
    bottom: [0, 50, 100, 50],
  };
  const endDrag = () => {
    overlay.hidden = true;
    dragRect = null;
    lastZone = null;
    dragDepth = 0;
  };
  pane.el.addEventListener("dragenter", (e) => {
    if (![...e.dataTransfer.types].includes(SID_MIME)) return;
    dragDepth++;
    if (!dragRect) {
      dragRect = pane.el.getBoundingClientRect();
      lastZone = null;
    }
  });
  pane.el.addEventListener("dragover", (e) => {
    if (!dragRect) return; // pas un drag d'onglet/panneau
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    const z = zoneAt(e);
    if (z === lastZone) return;
    lastZone = z;
    const g = ZONE_GEO[z];
    overlay.hidden = false;
    overlay.style.left = g[0] + "%";
    overlay.style.top = g[1] + "%";
    overlay.style.width = g[2] + "%";
    overlay.style.height = g[3] + "%";
  });
  // Ne clore le drag QUE quand on quitte vraiment le panneau (compteur à zéro) —
  // pas à chaque frontière d'enfant traversée.
  pane.el.addEventListener("dragleave", () => {
    if (dragDepth > 0) dragDepth--;
    if (dragDepth === 0) endDrag();
  });
  pane.el.addEventListener("drop", (e) => {
    const z = zoneAt(e);
    endDrag();
    const sid = e.dataTransfer.getData(SID_MIME);
    if (!sid || !state.tabs[sid]) return;
    e.preventDefault();
    const src = paneShowing(sid); // panneau existant glissé (par bandeau ou onglet)
    if (src === pane) return; // déposé sur lui-même
    if (z === "center") {
      dropTabOnPane(sid, pane, src); // centre : remplacer / échanger
      return;
    }
    const dir = z === "left" || z === "right" ? "row" : "col";
    const before = z === "left" || z === "top";
    if (src) movePaneToEdge(src, pane, dir, before); // panneau : regreffe de ce côté
    else addPane(sid, dir, pane, before); // onglet d'arrière-plan : nouveau panneau
  });
}
