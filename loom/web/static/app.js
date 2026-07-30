// loom/web/static/app.js
// UI Loom — modèle déclaratif « état → vue » (Preact + htm, zéro build).
//
// Principe : UNE source de vérité (`state.timeline`, une liste d'items ordonnés).
// Les événements SSE mutent l'état (création/maj par `id`), puis `render(App)`.
// Plus aucune manipulation DOM manuelle → la classe de bugs (doublons/fantômes,
// pills non rattachées) disparaît par construction. Validé par `node --check`.

import { _phRaf, loomWorkdir, set_CMDS, set__phRaf, set_loomWorkdir, set_machineUnloaded, set_skGenBusy } from "./shared.js";
import { addImages, clampRatio, createPane, renderLayout, savePanesLayout, updatePlaceholders } from "./panes.js";
import { SM_POS_KEY, _multiSel, _typesetAll, clearMultiSel, closeDrawer, closeSkillDrawer, closeTabMenu, compactBtn, doDeleteSkill, drawer, drawerClose, drawerScrim, fmtTok, localOnlyCb, newSessionTab, openConfirmPop, openDeleteConfirm, openDrawer, openSkillCreator, openSkillEditor, paintModelSelect, pickFolderBtn, postForm, postSkillsToggle, reflectWorkdir, resetBtn, saveSkill, scheduleMachineRefresh, sessionNew, setMetrics, setSkillDrawerMode, settingsBtn, showToast, sidebarEl, sidebarToggle, singleView, skBody, skCurrent, skDraft, skDrawer, skGenDesc, skMode, skNameInput, skScrim, skStatus, smPlace, splitWith, syncSkillsMaster, thinkingCb, toggleMultiSel, updateUsageMeter, workdirChip } from "./panels.js";
import { activeTab, focusedPane, renderPane, scheduleRenderFor, state, tab } from "./state.js";
import { _hydrateTimeline, _replayTimeline, _userTexts, ensureTab, focusPane, openTab, renderTabs, setPaneSid } from "./tabs.js";
import { INIT } from "./render.js";
import { render } from "./preact-htm.js";

marked.setOptions({ breaks: true });

fetch("/commands")
  .then((r) => r.json())
  .then((d) => {
    set_CMDS(d.commands || []);
  })
  .catch(() => {});

window.addEventListener("resize", () => {
  if (_phRaf) return;
  set__phRaf(true);
  requestAnimationFrame(() => {
    set__phRaf(false);
    updatePlaceholders();
  });
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeTabMenu();
  const backslash = e.code === "Backslash" || e.code === "IntlBackslash";
  if (e.ctrlKey && !e.shiftKey && backslash) {
    e.preventDefault();
    splitWith(state.active);
  } else if (e.ctrlKey && e.shiftKey && backslash) {
    e.preventDefault();
    singleView();
  } else if (e.altKey && !e.ctrlKey && /^Digit[1-4]$/.test(e.code)) {
    // !ctrlKey : AltGr (Windows) allume ctrl+alt à la fois — sans ce garde, taper
    // ~ # { (AltGr+2/3/4 en AZERTY) volait le focus au lieu d'écrire le caractère.
    const i = +e.code.slice(5) - 1;
    if (state.panes[i]) {
      e.preventDefault();
      focusPane(i);
      state.panes[i].input.focus();
    }
  }
});

{
  const pane = createPane();
  state.panes = [pane];
  state.focusedIdx = 0;
  state.maximized = null;
  state.layoutRoot = { type: "pane", pane };
  pane.el.classList.add("focused");
  renderLayout();
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

renderPane(state.panes[0]);

(async () => {
  let saved = null;
  try {
    saved = JSON.parse(localStorage.loomPanes || "null");
  } catch (e) {
    saved = null;
  }
  const tree = saved && saved.tree;
  if (!tree || !tree.dir) return; // rien de sauvegardé, ou vue simple
  const known = new Set((INIT.sessions || []).map((s) => s.id));
  const sids = [];
  (function collect(n) {
    if (!n) return;
    if (n.dir) {
      collect(n.a);
      collect(n.b);
    } else if (n.sid && known.has(n.sid) && !sids.includes(n.sid)) sids.push(n.sid);
  })(tree);
  if (sids.length < 2) return;
  // Chargements en PARALLÈLE : 4 panneaux = le max des aller-retours, pas leur somme.
  await Promise.all(sids.slice(0, 4).map(ensureTab));
  const seen = new Set();
  let reuse = state.panes[0]; // le panneau du boot est recyclé pour la 1re feuille
  const build = (n) => {
    if (n.dir) {
      const a = build(n.a);
      const b = build(n.b);
      if (a && b)
        return {
          type: "split",
          dir: n.dir === "col" ? "col" : "row",
          ratio: clampRatio(+n.ratio || 0.5),
          a,
          b,
        };
      return a || b;
    }
    if (!n.sid || !state.tabs[n.sid] || seen.has(n.sid) || seen.size >= 4)
      return null;
    seen.add(n.sid);
    let pane = reuse;
    reuse = null;
    if (!pane) {
      pane = createPane();
      state.panes.push(pane);
    }
    setPaneSid(pane, n.sid);
    return { type: "pane", pane };
  };
  const root = build(tree);
  if (!root) return;
  state.layoutRoot = root;
  renderLayout();
  focusPane(0); // complet par contrat : resynchronise serveur/topbar + renderTabs
  savePanesLayout();
})();

if (window.__mathjaxReady) _typesetAll();
else document.addEventListener("mathjax-ready", _typesetAll);

reflectWorkdir();

updateUsageMeter(INIT.usage_totals);

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

if (settingsBtn) settingsBtn.addEventListener("click", openDrawer);

if (drawerClose) drawerClose.addEventListener("click", closeDrawer);

if (drawerScrim) drawerScrim.addEventListener("click", closeDrawer);

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && drawer && !drawer.hidden) closeDrawer();
});

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

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && _multiSel.size) clearMultiSel();
});

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
        set_loomWorkdir(j.path);
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

if (sidebarToggle && sidebarEl) {
  sidebarToggle.addEventListener("click", () => sidebarEl.classList.toggle("open"));
}

window.addEventListener("paste", (e) => {
  const files = [];
  for (const item of e.clipboardData.items) {
    if (item.type.startsWith("image/")) files.push(item.getAsFile());
  }
  const fp = focusedPane();
  if (files.length && fp) addImages(fp, files);
});

if (thinkingCb) {
  thinkingCb.addEventListener("change", () => {
    if (activeTab()) activeTab().thinking = thinkingCb.checked;
    const fd = new FormData();
    fd.append("thinking", thinkingCb.checked ? "1" : "0");
    fetch("/thinking", { method: "POST", body: fd });
  });
}

if (localOnlyCb) {
  localOnlyCb.addEventListener("change", () => {
    if (activeTab()) activeTab().localOnly = localOnlyCb.checked;
    const fd = new FormData();
    fd.append("local_only", localOnlyCb.checked ? "1" : "0");
    fetch("/local_only", { method: "POST", body: fd });
  });
}

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
    // Seuls les panneaux montrant CETTE session repeignent (pas les 3 autres du split).
    scheduleRenderFor(state.active);
  });
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
    set_machineUnloaded(false); // re-sélection = warmup relancé côté serveur
    scheduleMachineRefresh();
    renderTabs(); // la teinte local/distant de l'onglet suit le nouveau modèle
  }
});

syncSkillsMaster();

scheduleMachineRefresh();

document.getElementById("machine-unload")?.addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    await fetch("/machine/unload", { method: "POST" });
    set_machineUnloaded(true);
  } catch {
    /* serveur local injoignable : l'état re-sondé ci-dessous l'affichera */
  } finally {
    btn.disabled = false;
  }
  scheduleMachineRefresh();
});

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

document
  .getElementById("machine-server-stop")
  ?.addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
      await fetch("/machine/server/stop", { method: "POST" });
      set_machineUnloaded(false);
    } catch {
      /* idem : l'état re-sondé fera foi */
    } finally {
      btn.disabled = false;
    }
    scheduleMachineRefresh();
  });

document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest(".skill-name");
  if (b) openSkillEditor(b.dataset.name);
});

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

skNameInput?.addEventListener("input", () => {
  if (skMode === "create") skDraft.name = skNameInput.value;
});

skGenDesc?.addEventListener("input", () => {
  skDraft.desc = skGenDesc.value;
});

document.getElementById("skdr-body")?.addEventListener("input", () => {
  if (skMode === "create" && skBody) skDraft.body = skBody.value;
});

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
  set_skGenBusy(true);
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
    set_skGenBusy(false);
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
        // Tous les distants vivent dans config/local.toml (source unique) :
        // édition ET suppression pour tous, plus de distinction config/UI.
        const btns =
          '<button type="button" class="rm-ic rm-edit" data-id="' +
          esc(m.id) +
          '" title="Éditer">✎</button>' +
          '<button type="button" class="rm-ic rm-del" data-id="' +
          esc(m.id) +
          '" title="Supprimer">✕</button>';
        return (
          '<div class="rm-item"><div class="rm-item-main"><span class="rm-id">' +
          esc(m.id) +
          "</span>" +
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

(function () {
  const $ = (id) => document.getElementById(id);
  const panel = $("cfg-ame");
  if (!panel) return;
  const esc = (s) =>
    String(s == null ? "" : s).replace(
      /[&<>"]/g,
      (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
    );
  const msg = (txt, kind) => {
    const m = $("ame-msg");
    m.textContent = txt || "";
    m.className = "rm-msg" + (kind ? " " + kind : "");
  };

  // Liste des sessions (cases cochées par défaut = tout), rechargée à chaque
  // ouverture de l'onglet (les sessions bougent).
  function loadSessions() {
    fetch("/soul/sessions")
      .then((r) => r.json())
      .then((d) => {
        $("ame-sessions").innerHTML = (d.sessions || [])
          .map(
            (s) =>
              '<label><input type="checkbox" class="ame-sess" value="' + esc(s.id) +
              '" checked> ' + esc(s.title || s.id) +
              '<span class="ame-date">' + esc((s.updated_at || "").slice(0, 10)) + "</span></label>",
          )
          .join("");
      });
  }
  document.querySelectorAll('#cfg-tabs [data-tab="ame"]').forEach((b) =>
    b.addEventListener("click", loadSessions),
  );

  // Sélecteur Exporter | Importer : un seul bloc visible à la fois (panneau moins chargé).
  function seg(exporting) {
    $("ame-seg-exp").classList.toggle("on", exporting);
    $("ame-seg-imp").classList.toggle("on", !exporting);
    $("ame-exp").hidden = !exporting;
    $("ame-imp").hidden = exporting;
    msg("");
    if (exporting) loadSessions();
  }
  $("ame-seg-exp").addEventListener("click", () => seg(true));
  $("ame-seg-imp").addEventListener("click", () => seg(false));
  $("ame-all").addEventListener("change", (e) => {
    panel.querySelectorAll(".ame-sess").forEach((c) => (c.checked = e.target.checked));
  });

  // Jauge de force : POST débouncé vers le serveur (zxcvbn Python = source unique
  // du verdict ; le bouton Exporter suit `ok` ET la confirmation, et le serveur
  // re-vérifie de toute façon). La confirmation protège du champ masqué : une typo
  // invisible rendrait l'archive indéchiffrable à jamais (GCM authentifie).
  let debTimer = null;
  function gauge() {
    clearTimeout(debTimer);
    debTimer = setTimeout(() => {
      const p = $("ame-pass").value;
      const p2 = $("ame-pass2").value;
      const g = $("ame-gauge");
      if (!p) {
        g.textContent = "";
        $("ame-export").disabled = true;
        return;
      }
      fetch("/soul/passphrase/check", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "passphrase=" + encodeURIComponent(p),
      })
        .then((r) => r.json())
        .then((d) => {
          const match = p === p2;
          g.className = "ame-gauge " + (d.ok && match ? "ok" : "ko");
          g.textContent = !d.ok
            ? "trop faible (" + d.score + "/4) — allonge ou clique générer"
            : !match
              ? "force " + d.score + "/4 — les deux champs ne correspondent pas"
              : "force " + d.score + "/4 — crack estimé : " + d.crack_display;
          $("ame-export").disabled = !(d.ok && match);
        });
    }, 250);
  }
  $("ame-pass").addEventListener("input", gauge);
  $("ame-pass2").addEventListener("input", gauge);

  // Générer : remplit les DEUX champs (pas de risque de typo) SANS les révéler
  // (le user décide via l'œil).
  $("ame-gen").addEventListener("click", () => {
    fetch("/soul/passphrase/generate", { method: "POST" })
      .then((r) => r.json())
      .then((d) => {
        $("ame-pass").value = d.passphrase;
        $("ame-pass2").value = d.passphrase;
        gauge();
        msg("phrase générée — clique l'œil pour la lire et la mémoriser", "");
      });
  });
  const eye = (inputId, btnId) =>
    $(btnId).addEventListener("click", () => {
      const i = $(inputId);
      i.type = i.type === "password" ? "text" : "password";
    });
  eye("ame-pass", "ame-eye");
  eye("ame-ipass", "ame-ieye");

  $("ame-export").addEventListener("click", () => {
    const ids = Array.from(panel.querySelectorAll(".ame-sess:checked")).map((c) => c.value);
    msg("export en cours…");
    fetch("/soul/export", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body:
        "dest_dir=" + encodeURIComponent($("ame-dest").value) +
        "&passphrase=" + encodeURIComponent($("ame-pass").value) +
        "&session_ids=" + encodeURIComponent(ids.join(",")),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok) return msg(d.error || "échec de l'export", "err");
        msg(
          "exporté : " + d.path + " (" + Math.round(d.size / 1024) + " Ko, " +
          d.sessions + " session(s))",
          "ok",
        );
      })
      .catch(() => msg("échec de l'export (réseau)", "err"));
  });

  $("ame-import").addEventListener("click", () => {
    msg("import en cours…");
    fetch("/soul/import", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body:
        "file=" + encodeURIComponent($("ame-file").value) +
        "&passphrase=" + encodeURIComponent($("ame-ipass").value),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (!ok) return msg(d.error || "échec de l'import", "err");
        const s = d.report.sessions;
        msg(
          "importé : " + s.ajoutees + " session(s) ajoutée(s), " + s.remplacees +
          " remplacée(s), " + s.ignorees + " ignorée(s) ; skills +" +
          (d.report.skills_learned.ajoutes + d.report.skills_user.ajoutes) +
          " ; mémoire +" + d.report.memoire.ajoutes,
          "ok",
        );
        // La barre des sessions est rendue par le serveur (pas de fonction de
        // rechargement côté JS) : les sessions importées apparaissent au prochain
        // chargement de la page. On rafraîchit la liste du panneau Âme, elle.
        loadSessions();
      })
      .catch(() => msg("échec de l'import (réseau)", "err"));
  });
})();

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

document.addEventListener("change", (e) => {
  if (e.target && e.target.id === "model-select") paintModelSelect();
});

document.body.addEventListener("htmx:afterSwap", () => paintModelSelect());

paintModelSelect();
