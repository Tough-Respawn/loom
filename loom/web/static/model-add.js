// Panneaux « Ajouter un modèle » et « Recalibrer un modèle » : même mécanique (état
// serveur relu par GET, étapes par POST), jamais sendChat. `prefix` nomme les ids du
// DOM, `endpoint` la route, `command` la commande qui ouvre le parcours.
import { md } from "./render.js";

export function initWizardPanel({ prefix, endpoint, command, eventKey, getSessionId, postForm, showToast }) {
  const panel = document.getElementById(prefix + "-panel");
  if (!panel) return null;
  const openButton = document.getElementById(prefix + "-open");
  const reply = document.getElementById(prefix + "-reply");
  const status = document.getElementById(prefix + "-status");
  const choices = document.getElementById(prefix + "-choices");
  const form = document.getElementById(prefix + "-form");
  const input = document.getElementById(prefix + "-input");
  const cancel = document.getElementById(prefix + "-cancel");
  const log = document.getElementById(prefix + "-log");
  let sid = null, timer = null, revision = 0, sending = false, wasBusy = false;

  function paint(data) {
    openButton.hidden = !!(data.active || data.busy);
    reply.innerHTML = md(data.reply || "");
    reply.querySelectorAll("a").forEach((link) => { link.target = "_blank"; link.rel = "noopener noreferrer"; });
    if (wasBusy && !data.busy) document.dispatchEvent(new CustomEvent("loom:models-changed"));
    wasBusy = !!data.busy;
    status.textContent = data.status || (data.busy ? "Chargement…" : "");
    if (log) {
      // Journal de progression (recalibration) : chaque étape horodatée, défilé en bas.
      const lines = data.log || [];
      log.hidden = !lines.length;
      log.textContent = lines.join("\n");
      log.scrollTop = log.scrollHeight;
    }
    input.type = data.secret ? "password" : "text";
    input.value = "";
    form.hidden = !data.active;
    choices.replaceChildren();
    for (const label of data.choices || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "rm-btn";
      button.textContent = label;
      button.onclick = () => send(label);
      choices.append(button);
    }
    for (const button of panel.querySelectorAll("button, input")) button.disabled = !!data.busy;
    clearTimeout(timer);
    if (data.busy) timer = setTimeout(() => refresh(), 1500);
    else if (data.active) input.focus();
  }

  async function refresh() {
    const version = revision;
    try {
      const response = await fetch(endpoint + "?session_id=" + encodeURIComponent(sid));
      const data = await response.json();
      if (version !== revision) return null;
      if (!response.ok) throw new Error(data.error);
      paint(data);
      return data;
    } catch (e) { if (version === revision) status.textContent = e.message; }
    return null;
  }

  async function send(message) {
    if (sending || !sid) return;
    sending = true;
    const version = revision;
    clearTimeout(timer);
    input.value = "";
    for (const button of panel.querySelectorAll("button, input")) button.disabled = true;
    status.textContent = "Chargement…";
    try {
      const response = await postForm(endpoint, {session_id: sid, message});
      const data = await response.json();
      if (version !== revision) return;
      if (!response.ok) throw new Error(data.error || "Opération impossible");
      if (data.ui_action === "cancel_interface") {
        showToast(data.message);
        await refresh();
      } else paint(data);
      document.dispatchEvent(new CustomEvent("loom:models-changed"));
    } catch (e) {
      if (version === revision) {
        await refresh();
        status.textContent = e.message;
      }
    } finally { sending = false; }
  }

  async function open(sessionId, argument = "") {
    if (sending) { showToast("Une étape est déjà en cours."); return; }
    sid = sessionId || getSessionId();
    if (!sid) return;
    revision++;
    wasBusy = false;
    panel.hidden = false;
    const data = await refresh();
    if (data && !data.active && !data.busy) await send(command + (argument ? " " + argument : ""));
  }
  openButton.onclick = () => open(getSessionId());
  document.addEventListener("loom:manage-models", (event) => {
    if (event.detail?.[eventKey]) open(event.detail.sid, event.detail.argument);
  });
  document.addEventListener("loom:interface-cancelled", (event) => {
    if (sid && event.detail.sid === sid) refresh();
  });
  form.onsubmit = (event) => { event.preventDefault(); if (input.value.trim()) send(input.value.trim()); };
  cancel.onclick = () => send("/cancel");
  return { open };
}
