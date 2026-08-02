import { focusedPane, paneShowing, state } from "./state.js";
import { activateTab, closeTab, ensureTab, focusPane, setPaneSid } from "./tabs.js";
import { addPane, removePane } from "./panes.js";
import { INIT } from "./render.js";
import { loomWorkdir, machineUnloaded, set_machineUnloaded, skGenBusy } from "./shared.js";
import { html } from "./preact-htm.js";

export const SID_MIME = "text/loom-sid";

export function wireSidDrag(el, getSid) {
  el.draggable = true;
  el.addEventListener("dragstart", (ev) => {
    const sid = getSid();
    if (!sid) {
      ev.preventDefault();
      return;
    }
    ev.dataTransfer.setData(SID_MIME, sid);
    ev.dataTransfer.effectAllowed = "move";
  });
}

export function dropTabOnPane(sid, pane, other) {
  // Sauvegarder le brouillon avant l'échange empêche sa perte entre deux panneaux.
  const mineTab = state.tabs[pane.sid];
  if (mineTab) mineTab.draft = pane.input.value;
  if (other) setPaneSid(other, pane.sid);
  setPaneSid(pane, sid);
  focusPane(pane);
}

export async function splitWith(sid) {
  if (!sid || !(await ensureTab(sid))) return;
  if (!paneShowing(sid)) {
    addPane(sid, "row");
    return;
  }
  const free = state.order.filter((s) => !paneShowing(s));
  if (!free.length) {
    showToast("rien à scinder : ouvre un autre onglet d'abord");
    return;
  }
  addPane(free[free.length - 1], "row");
}

export function singleView() {
  if (state.panes.length <= 1) return;
  const keep = focusedPane();
  [...state.panes].forEach((p) => {
    if (p !== keep) removePane(p);
  });
}

export let _ctxCloser = null;

export function closeTabMenu() {
  document.getElementById("tab-ctx")?.remove();
  if (_ctxCloser) {
    document.removeEventListener("click", _ctxCloser, true);
    _ctxCloser = null;
  }
}

export function sessionDirPath(sid) {
  const root = (INIT.sessions_root || "").replace(/[\\/]+$/, "");
  if (!root || !sid) return "";
  return root + (root.includes("\\") ? "\\" : "/") + sid;
}

export function openTabMenu(e, sid) {
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
  // Ouvrir à droite par défaut; le glisser-déposer ajuste ensuite la disposition.
  add("Ouvrir en parallèle", () => splitWith(sid), full);
  const sep = document.createElement("div");
  sep.className = "ctx-sep";
  m.appendChild(sep);
  add("Remettre en vue simple", () => singleView(), state.panes.length <= 1);
  // Garder le chemin sélectionnable sans fermer le menu contextuel.
  const dir = sessionDirPath(sid);
  if (dir) {
    const sep2 = document.createElement("div");
    sep2.className = "ctx-sep";
    m.appendChild(sep2);
    const p = document.createElement("div");
    p.className = "ctx-path";
    p.textContent = dir;
    p.title = "Dossier réel de la session (session.json, timeline.jsonl, debug.log)";
    m.appendChild(p);
    add("Copier le chemin de la session", () => {
      navigator.clipboard.writeText(dir);
      showToast("chemin copié");
    });
  }
  add("Exporter la conversation (.zip)", () => {
    window.location = "/session/" + encodeURIComponent(sid) + "/export";
  });
  document.body.appendChild(m);
  m.style.left = Math.min(e.clientX, window.innerWidth - m.offsetWidth - 6) + "px";
  m.style.top = Math.min(e.clientY, window.innerHeight - m.offsetHeight - 6) + "px";
  _ctxCloser = (ev) => {
    if (!ev.target.closest("#tab-ctx")) closeTabMenu();
  };
  setTimeout(() => document.addEventListener("click", _ctxCloser, true), 0);
}

export async function newSessionTab() {
  const wd = document.getElementById("workdir-path");
  const r = await postForm("/session/new", {
    workspace: wd ? wd.textContent.trim() : "",
  });
  openSessionTab(await r.json());
}

export function openSessionTab(d) {
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

export async function importSessionFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  let d = null;
  try {
    const r = await fetch("/session/import", { method: "POST", body: fd });
    d = await r.json();
    if (!r.ok) {
      showToast(d && d.error ? d.error : "import impossible");
      return;
    }
  } catch {
    showToast("import impossible (serveur injoignable ?)");
    return;
  }
  openSessionTab(d);
  showToast("conversation importée" + (d.model ? " (modèle : " + d.model + ")" : ""));
}

export function addSidebarSession(d) {
  const list = document.querySelector(".session-list");
  if (!list || list.querySelector(`.sess-pick[data-id="${d.id}"]`)) return;
  const li = document.createElement("li");
  li.className = "session-item";
  li.innerHTML =
    `<button type="button" class="sess-pick" data-id="${d.id}" title="${d.workspace || ""}">${d.title || "session"}</button>` +
    `<button type="button" class="sess-del" data-id="${d.id}" title="Supprimer">✕</button>`;
  list.prepend(li);
}

export function _typesetAll() {
  if (!(window.MathJax && window.MathJax.typesetPromise)) return;
  document
    .querySelectorAll(".msg.assistant")
    .forEach((el) => window.MathJax.typesetPromise([el]).catch(() => {}));
}

export const workdirPath = document.getElementById("workdir-path");

export const workdirChip = document.getElementById("workdir-chip");

export function reflectWorkdir() {
  if (workdirPath) workdirPath.textContent = loomWorkdir;
}

export const genMetrics = document.getElementById("gen-metrics");

export const gmText = document.getElementById("gm-text");

export function setMetrics(sent, recv, tokS, opts) {
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

export function fmtTok(n) {
  n = n || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
  return String(n);
}

export const usageMeter = document.getElementById("usage-meter");

export function updateUsageMeter(t) {
  if (!usageMeter || !t) return;
  const inEl = document.getElementById("um-in");
  const outEl = document.getElementById("um-out");
  if (inEl) inEl.textContent = fmtTok(t.tokens_in);
  if (outEl) outEl.textContent = fmtTok(t.tokens_out);
  // Le taux de cache mesure la part de l'entrée réellement servie depuis le préfixe.
  const cacheEl = document.getElementById("um-cache");
  if (cacheEl) {
    const pct = t.cache_pct || 0;
    cacheEl.textContent = t.tokens_in > 0 ? "· cache " + pct + "%" : "";
    cacheEl.classList.toggle("miss", t.tokens_in > 0 && pct < 20);
  }
  // Le nombre d'appels multiplie le coût d'un contexte rejoué.
  const callsEl = document.getElementById("um-calls");
  if (callsEl) callsEl.textContent = t.api_calls > 0 ? "· " + t.api_calls + "×" : "";
  // La jauge compare le dernier prompt à la fenêtre du modèle.
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
      // Le provider fait autorité lorsqu'il publie sa fenêtre, sinon utiliser la config.
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
  // Masquer le coût sans tarif fiable, tout en continuant à le cumuler.
  usageMeter.hidden = !(t.api_calls > 0 || t.tokens_in > 0 || t.tokens_out > 0);
}

export const compactBtn = document.getElementById("um-compact");

export const settingsBtn = document.getElementById("settings-btn");

export const drawer = document.getElementById("settings-drawer");

export const drawerScrim = document.getElementById("drawer-scrim");

export const drawerClose = document.getElementById("drawer-close");

export function openDrawer() {
  if (drawer) drawer.hidden = false;
  if (drawerScrim) drawerScrim.hidden = false;
}

export function closeDrawer() {
  if (drawer) drawer.hidden = true;
  if (drawerScrim) drawerScrim.hidden = true;
}

export async function postForm(url, data) {
  const fd = new FormData();
  for (const k in data) fd.append(k, data[k]);
  return fetch(url, { method: "POST", body: fd });
}

export const sessionNew = document.getElementById("session-new");

export const _multiSel = new Set();

export function clearMultiSel() {
  _multiSel.clear();
  document
    .querySelectorAll(".session-item.sel")
    .forEach((el) => el.classList.remove("sel"));
  document.getElementById("sess-multi")?.remove();
}

export function toggleMultiSel(sid, row) {
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

export function renderMultiBar() {
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

export function _outsideDelClose(e) {
  if (!e.target.closest("#sess-del-pop")) closeDeleteConfirm();
}

export function closeDeleteConfirm() {
  document.getElementById("sess-del-pop")?.remove();
  document.removeEventListener("click", _outsideDelClose, true);
}

export function openConfirmPop(row, labelText, onYes, align = "right") {
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
    pop.style.left = Math.max(6, r.left - pop.offsetWidth - 6) + "px";
  } else {
    pop.style.left = r.right + 3 + "px";
  }
  // La phase capture garantit la fermeture même depuis la sidebar.
  setTimeout(() => document.addEventListener("click", _outsideDelClose, true), 0);
}

export function openDeleteConfirm(sid, row) {
  openConfirmPop(row, "Supprimer cette session ?", async () => {
    await postForm("/session/delete", { id: sid });
    if (state.tabs[sid]) closeTab(sid); // ferme l'onglet si ouvert
    row.remove();
  });
}

export const pickFolderBtn = document.getElementById("pick-folder-btn");

export const sidebarToggle = document.getElementById("sidebar-toggle");

export const sidebarEl = document.getElementById("sidebar");

export const thinkingCb = document.getElementById("thinking-cb");

export const localOnlyCb = document.getElementById("local-only-cb");

export const resetBtn = document.getElementById("reset-btn");

export function syncSkillsMaster() {
  const cbs = [...document.querySelectorAll("#skills-panel .skill-cb")];
  const master = document.getElementById("skills-all");
  if (!master) return;
  const on = cbs.filter((c) => c.checked).length;
  master.checked = cbs.length > 0 && on === cbs.length;
  master.indeterminate = on > 0 && on < cbs.length;
}

export async function postSkillsToggle() {
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

export let machineTimer = null;

export async function refreshMachineState() {
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
  // Le moniteur machine n'a de sens que pour un modèle local.
  setSysmonVisible(d.mode === "home");
  let cls = "",
    text = "";
  if (d.mode === "remote") {
    if (d.reachable && d.any_loaded) {
      cls = "busy";
      text = "machine · libération…";
    } else {
      cls = "free";
      text = "machine · libre (modèle distant)";
    }
  } else if (!d.reachable) {
    if (d.starting) {
      cls = "busy";
      text = "machine · démarrage du serveur…";
    } else {
      cls = "off";
      text = "machine · serveur local éteint";
    }
  } else if (d.model_loaded) {
    if (machineUnloaded) {
      // Garder un état transitoire jusqu'à la fin réelle du déchargement.
      cls = "busy";
      text = "machine · libération…";
    } else {
      cls = "on";
      text = "machine · " + d.model + " chargé";
    }
  } else if (d.loading) {
    // Un chargement actif annule visuellement un ancien déchargement demandé.
    set_machineUnloaded(false);
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
  // Ne proposer le déchargement que lorsque llama-swap peut réellement l'honorer.
  if (unloadBtn)
    unloadBtn.hidden = !(
      d.mode === "home" &&
      d.reachable &&
      d.any_loaded &&
      !d.loading &&
      !machineUnloaded
    );
  if (startBtn)
    startBtn.hidden = !(d.mode === "home" && !d.reachable && !d.starting);
  // Ne jamais proposer d'éteindre une stack lancée hors de Loom.
  if (stopBtn) stopBtn.hidden = !(d.reachable && d.managed);
  return cls;
}

export function scheduleMachineRefresh() {
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

export const skDrawer = document.getElementById("skill-drawer");

export const skScrim = document.getElementById("skill-scrim");

export const skName = document.getElementById("skdr-name");

export const skDesc = document.getElementById("skdr-desc");

export const skBody = document.getElementById("skdr-body");

export const skStatus = document.getElementById("skdr-status");

export let skCurrent = null;

export function openSkillDrawer() {
  if (skDrawer) skDrawer.hidden = false;
  if (skScrim) skScrim.hidden = false;
}

export function closeSkillDrawer() {
  if (skDrawer) skDrawer.hidden = true;
  if (skScrim) skScrim.hidden = true;
  skCurrent = null;
}

export async function openSkillEditor(name) {
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
    // Seuls les skills gérés par l'utilisateur sont supprimables.
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

export async function saveSkill(scope) {
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
    postSkillsToggle();
  } catch (err) {
    if (skStatus) skStatus.textContent = "erreur : " + err;
  }
}

export async function doDeleteSkill(name, fromDrawer) {
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

export const skNameInput = document.getElementById("skdr-name-input");

export const skGenRow = document.getElementById("skdr-genrow");

export const skGenDesc = document.getElementById("skdr-gen-desc");

export let skMode = "edit";

export const skDraft = { name: "", desc: "", body: "" };

export function setSkillDrawerMode(mode) {
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

export function openSkillCreator() {
  skCurrent = null;
  setSkillDrawerMode("create");
  if (skName) skName.textContent = "";
  if (skDesc)
    skDesc.textContent =
      "[user] nouveau skill — décris-le puis Générer, ou écris le SKILL.md";
  // Restaurer le brouillon et son état de génération.
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

export function showToast(message, actions = [], { timeout = 8000 } = {}) {
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

export const SM_N = 48;

export const smHist = { cpu: [], gpu: [] };

export let sysmonTimer = null;

export function smPush(arr, v) {
  arr.push(v);
  if (arr.length > SM_N) arr.shift();
}

export function smDrawSpark(canvas, data, color) {
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

export const SM_GB = 1073741824;

export const setTxt = (id, v) => {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
};

export async function sysmonTick() {
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

export function setSysmonVisible(on) {
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

export const SM_POS_KEY = "loomSysmonPos";

export function smPlace(el, l, t) {
  l = Math.max(0, Math.min(l, window.innerWidth - el.offsetWidth));
  t = Math.max(0, Math.min(t, window.innerHeight - el.offsetHeight));
  el.style.left = l + "px";
  el.style.top = t + "px";
  el.style.right = "auto";
}

export function smRestorePos() {
  const el = document.getElementById("sysmon");
  if (!el) return;
  try {
    const saved = JSON.parse(localStorage.getItem(SM_POS_KEY) || "null");
    if (saved) smPlace(el, saved.l, saved.t);
  } catch {
    /* position corrompue : on garde le coin par défaut */
  }
}

export function paintModelSelect() {
  const sel = document.getElementById("model-select");
  if (!sel || !sel.options.length) return;
  const opt = sel.options[sel.selectedIndex] || sel.options[0];
  const m = opt.className.match(/opt-(\w+)/);
  sel.className = m && m[1] !== "home" ? "sel-" + m[1] : "";
  // Basculer immédiatement le moniteur; l'état machine confirmera ensuite.
  if (typeof setSysmonVisible === "function")
    setSysmonVisible(!(m && m[1] === "remote"));
}
