// loom/web/static/chat.js — issu du decoupage de app.js (comportement constant).
import { opsFor, panesFor, state, tab } from "./state.js";
import { renderTabs } from "./tabs.js";
import { setActivityFor, syncComposersFor } from "./panes.js";
import { reflectWorkdir, scheduleMachineRefresh, setMetrics, updateUsageMeter } from "./panels.js";
import { loomWorkdir, set_loomWorkdir } from "./shared.js";
import { streamSSE } from "./sse.js";

export async function sendChat(sid, text, images) {
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
      case "harness":
        // 3e voix : intervention du garde-fou Loom, bulle distincte.
        if (thinkId) { patch(thinkId, { active: false }); thinkId = null; }
        if (asstId) { patch(asstId, { done: true }); asstId = null; }
        push({ kind: "harness", hkind: evt.kind, text: evt.text });
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
  // Minuteur : le prefill (et un éventuel démarrage à froid devant lui) peut durer
  // de 20 s à 2 min — les secondes écoulées rendent l'attente lisible (retour user
  // 2026-07-19 : « 1min52 » vécu à l'aveugle, c'était chargement 60 s + prefill 27 s).
  const sentAt = Date.now();
  const gaTimer = setInterval(() => {
    // Aucun panneau ne montre ce flux : rien à peindre (N flux d'arrière-plan ne
    // doivent pas scanner les panneaux toutes les 500 ms pour rien).
    if (!panesFor(sid).length) return;
    // Label FORCÉ (compaction…) prioritaire sur la détection de silence.
    if (t.forcedActivity) {
      setActivityFor(sid, t.forcedActivity);
      return;
    }
    const quiet = Date.now() - lastEvtAt > 2500;
    const elapsed = Math.round((Date.now() - sentAt) / 1000);
    // Avant le 1er token, DEUX phases distinctes (observation 2026-07-19 : 56 s de
    // chargement étiquetées « prefill » = illisible) : tant que le chip machine dit
    // « chargement… », c'est le CHARGEMENT du modèle, pas le prefill.
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
