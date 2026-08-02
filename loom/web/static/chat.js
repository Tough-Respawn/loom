import { opsFor, panesFor, scheduleRenderFor, state, tab } from "./state.js";
import { renderTabs } from "./tabs.js";
import { setActivityFor, syncComposersFor } from "./panes.js";
import { reflectWorkdir, scheduleMachineRefresh, setMetrics, updateUsageMeter } from "./panels.js";
import { loomWorkdir, set_loomWorkdir } from "./shared.js";
import { streamSSE } from "./sse.js";

export async function sendChat(sid, text, images, options = {}) {
  const t = tab(sid);
  if (!t) return;
  panesFor(sid).forEach((p) => (p.pin = true));
  // Interrompre seulement la génération de cet onglet.
  if (t.abort) t.abort.abort();
  const ac = new AbortController();
  t.abort = ac;
  t.streaming = true;
  renderTabs();
  syncComposersFor(sid);

  // Les événements mutent la timeline de leur onglet, même hors écran.
  const { push, get, patch } = opsFor(sid);

  let userId;
  const userFields = options.userFields || {};
  if (images && images.length) {
    const parts = [{ type: "text", text }];
    for (const im of images) parts.push({ type: "image_url", image_url: { url: URL.createObjectURL(im) } });
    userId = push({ kind: "user", content: parts, ...userFields }).id;
  } else {
    userId = push({ kind: "user", content: text, ...userFields }).id;
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
  for (const [key, value] of Object.entries(options.form || {}))
    fd.append(key, String(value));

  if (sid === state.active) setMetrics(0, 0, null); // liveness immédiate si onglet affiché

  // Relancer le suivi machine car ce message peut démarrer le serveur.
  scheduleMachineRefresh();

  // Un silence prolongé affiche la phase d'activité au lieu d'un flux apparemment figé.
  let lastEvtAt = Date.now();
  let sawToken = false;

  const onEvent = (evt) => {
    lastEvtAt = Date.now();
    if (evt.type === "text" || evt.type === "reasoning" || evt.type === "tool_args")
      sawToken = true;
    // La reprise du flux efface tout label forcé, sauf si `status` le pilote.
    if (
      evt.type === "text" ||
      evt.type === "reasoning" ||
      evt.type === "tool_result" ||
      evt.type === "tool_call"
    )
      t.forcedActivity = null;
    switch (evt.type) {
      case "queued":
        // Une réponse 202 signifie que le backend a mis le message en file.
        if (userId) patch(userId, { queued: true });
        push({
          kind: "notice",
          text: evt.message || "Message mis en file d'attente : pris au prochain point d'arrêt.",
        });
        break;
      case "note":
        // Afficher une note injectée à la position réellement vue par le modèle.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        if (evt.handoff_id) {
          const existing = t.timeline.find(
            (item) => item.handoffId === evt.handoff_id,
          );
          if (existing)
            patch(existing.id, {
              content: evt.text,
              provenance: evt.provenance || existing.provenance || [],
              queued: false,
            });
          else
            push({
              kind: "user",
              content: evt.text,
              provenance: evt.provenance || [],
              handoffId: evt.handoff_id,
            });
        } else push({ kind: "user", content: evt.text });
        break;
      case "harness":
        // Distinguer visuellement la voix du garde-fou.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "harness", hkind: evt.kind, text: evt.text });
        break;
      case "monitor_event":
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({
          kind: "monitor_event",
          monitorId: evt.monitor_id,
          description: evt.description,
          text: evt.text,
          final: !!evt.final,
        });
        break;
      case "parallel":
        // Les appels d'un même groupe parallèle se rendent côte à côte.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "parallel", lanes: evt.ids || [] });
        break;
      case "reasoning":
        // Une nouvelle réflexion clôt le texte précédent pour préserver l'ordre visuel.
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        if (!thinkId) thinkId = push({ kind: "think", role: "", text: "", active: true }).id;
        patch(thinkId, { text: get(thinkId).text + evt.text, active: true });
        break;
      case "text":
        // Le texte clôt la bulle de raisonnement courante.
        if (thinkId) {
          patch(thinkId, { active: false });
          thinkId = null;
        }
        if (!asstId)
          asstId = push({
            kind: "assistant",
            raw: "",
            done: false,
            provenance: evt.provenance || [],
          }).id;
        else if (evt.provenance)
          patch(asstId, { provenance: evt.provenance });
        patch(asstId, { raw: get(asstId).raw + evt.text });
        break;
      case "tool_begin":
      case "tool_call": {
        // Un outil clôt l'étape de réflexion courante.
        if (thinkId) {
          patch(thinkId, { active: false });
          thinkId = null;
        }
        // La narration après l'outil doit commencer sous sa carte.
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        tools[evt.id] = tid;
        break;
      }
      case "tool_args": {
        // Cumuler les arguments rend visible la progression d'un gros appel d'outil.
        const tid = "tool:" + (evt.id || evt.name);
        if (!get(tid)) push({ id: tid, kind: "tool", name: evt.name, pending: true });
        patch(tid, { chars: (get(tid).chars || 0) + (evt.n || 0) });
        break;
      }
      case "tool_stream": {
        // Accumuler l'activité live dans la carte repliée de l'outil.
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
        // Le séparateur de phase reste décoratif et non bloquant.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "phase", name: evt.name, task: evt.task, detail: evt.detail });
        break;
      case "tool_request":
        push({ kind: "perm", callId: evt.id, name: evt.name, summary: evt.summary });
        break;
      case "notice":
        push({ kind: "notice", text: evt.text });
        break;
      case "workspace":
        // Mémoriser le workspace par onglet et n'afficher que celui qui est actif.
        t.workspace = evt.path;
        if (sid === state.active) {
          set_loomWorkdir(evt.path);
          try {
            localStorage.loomWorkdir = loomWorkdir;
          } catch (e) {
            /* localStorage indispo : sans effet */
          }
          reflectWorkdir();
        }
        break;
      case "session_title": {
        if (t) t.title = evt.title;
        const btn = document.querySelector(`.sess-pick[data-id="${evt.id}"]`);
        if (btn) btn.textContent = evt.title;
        renderTabs();
        break;
      }
      case "metrics":
        lastSent = evt.sent;
        lastRecv = evt.recv;
        lastTokS = evt.tok_s;
        t.metrics = { sent: lastSent, recv: lastRecv, tokS: lastTokS };
        if (sid === state.active) setMetrics(lastSent, lastRecv, lastTokS, {});
        break;
      case "totals":
        t.meter = evt;
        if (sid === state.active) updateUsageMeter(evt);
        break;
      case "status":
        // Un statut explicite a priorité sur le label déduit du silence.
        t.forcedActivity = evt.label || null;
        setActivityFor(sid, t.forcedActivity);
        break;
      case "models":
        // Recharger les modèles après un montage à chaud.
        if (window.refreshModels) window.refreshModels();
        break;
      case "choices":
        push({ kind: "choices", options: evt.options || [] });
        break;
      case "error":
        push({ kind: "error", message: "Erreur : " + evt.message + " (connexion à loom.web perdue ?)" });
        break;
      case "request_error": {
        if (options.handoffId) {
          t.timeline = t.timeline.filter((item) => item.id !== userId);
          scheduleRenderFor(sid);
          opsFor(options.errorSid || sid).push({
            kind: "error",
            message:
              "Transfert impossible : " +
              (evt.message || "session cible introuvable"),
          });
        } else {
          push({ kind: "error", message: evt.message || "Requête refusée." });
        }
        break;
      }
    }
  };

  // Afficher l'attente seulement dans les panneaux concernés, avec le temps écoulé.
  const sentAt = Date.now();
  const gaTimer = setInterval(() => {
    // Ne rien repeindre pour un flux sans panneau visible.
    if (!panesFor(sid).length) return;
    // Un label forcé reste prioritaire sur la détection de silence.
    if (t.forcedActivity) {
      setActivityFor(sid, t.forcedActivity);
      return;
    }
    const quiet = Date.now() - lastEvtAt > 2500;
    const elapsed = Math.round((Date.now() - sentAt) / 1000);
    // Distinguer le chargement du modèle du prefill avant le premier token.
    const chipTxt =
      (document.getElementById("machine-chip") || {}).textContent || "";
    const phase = chipTxt.includes("chargement")
      ? "chargement du modèle"
      : "préparation du contexte (prefill)";
    setActivityFor(
      sid,
      t.streaming && quiet
        ? sawToken
          ? "le modèle travaille"
          : `${phase} · ${elapsed} s`
        : null,
    );
  }, 500);

  try {
    await streamSSE(options.endpoint || "/chat", fd, onEvent, ac.signal);
    if (asstId) patch(asstId, { done: true });
  } catch (err) {
    if (err.name === "AbortError") {
      if (asstId) patch(asstId, { done: true });
    } else {
      if (options.handoffId) {
        t.timeline = t.timeline.filter((item) => item.id !== userId);
        scheduleRenderFor(sid);
        opsFor(options.errorSid || sid).push({
          kind: "error",
          message: "Transfert impossible : " + err.message,
        });
      } else {
        push({ kind: "error", message: "Erreur : " + err.message + " (connexion à loom.web perdue ?)" });
      }
    }
  } finally {
    clearInterval(gaTimer);
    // Resonder l'état machine en fin de flux pour corriger les états transitoires.
    scheduleMachineRefresh();
    setActivityFor(sid, null);
    if (thinkId) patch(thinkId, { active: false });
    // Ne clôturer l'état que si ce flux n'a pas été remplacé.
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

function _handoffId() {
  if (globalThis.crypto?.randomUUID) return "handoff:" + crypto.randomUUID();
  return "handoff:" + Date.now() + ":" + Math.random().toString(36).slice(2);
}

function _handoffForm(sourceSid, previous, handoffId, queueOnly = false) {
  return {
    source_session_id: sourceSid,
    provenance: JSON.stringify(Array.isArray(previous) ? previous : []),
    handoff_id: handoffId,
    ...(queueOnly ? { queue_only: "1" } : {}),
  };
}

export async function sendHandoff(sourceSid, targetSid, text, previous) {
  const source = tab(sourceSid);
  const target = tab(targetSid);
  if (!source || !target) {
    if (source)
      opsFor(sourceSid).push({
        kind: "error",
        message: "Transfert impossible : session cible introuvable",
      });
    return;
  }

  const handoffId = _handoffId();
  const chain = [
    ...(Array.isArray(previous) ? previous : []),
    {
      session_id: sourceSid,
      title: source.title || "session",
      model: source.model || "modèle par défaut",
    },
  ];
  const userFields = { provenance: chain, handoffId };

  // Si la cible génère, laisser le backend arbitrer atomiquement entre file et flux direct.
  if (target.streaming) {
    panesFor(targetSid).forEach((pane) => (pane.pin = true));
    const queuedItem = opsFor(targetSid).push({
      kind: "user",
      content: text,
      ...userFields,
      queued: true,
    });
    const fd = new FormData();
    fd.append("message", text);
    fd.append("session_id", targetSid);
    for (const [key, value] of Object.entries(
      _handoffForm(sourceSid, previous, handoffId, true),
    ))
      fd.append(key, String(value));
    try {
      const response = await fetch("/handoff", { method: "POST", body: fd });
      if (response.status === 202) return;
      target.timeline = target.timeline.filter((item) => item.id !== queuedItem.id);
      scheduleRenderFor(targetSid);
      if (response.status !== 409) {
        const detail = (await response.text()).trim();
        opsFor(sourceSid).push({
          kind: "error",
          message:
            "Transfert impossible : " +
            (detail || "session cible introuvable"),
        });
        return;
      }
    } catch (err) {
      target.timeline = target.timeline.filter((item) => item.id !== queuedItem.id);
      scheduleRenderFor(targetSid);
      opsFor(sourceSid).push({
        kind: "error",
        message: "Transfert impossible : " + err.message,
      });
      return;
    }
  }

  return sendChat(targetSid, text, [], {
    endpoint: "/handoff",
    form: _handoffForm(sourceSid, previous, handoffId),
    userFields,
    handoffId,
    errorSid: sourceSid,
  });
}
