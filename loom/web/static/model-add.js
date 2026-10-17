// Model installation has its own state and requests; it never uses sendChat.
import { md } from "./render.js";

export function initModelAdd({ getSessionId, postForm, showToast }) {
  const panel = document.getElementById("model-add-panel");
  if (!panel) return;
  const openButton = document.getElementById("model-add-open");
  const reply = document.getElementById("model-add-reply");
  const status = document.getElementById("model-add-status");
  const choices = document.getElementById("model-add-choices");
  const form = document.getElementById("model-add-form");
  const input = document.getElementById("model-add-input");
  const cancel = document.getElementById("model-add-cancel");
  let sid = null, timer = null, revision = 0, sending = false, wasBusy = false;

  function paint(data) {
    openButton.hidden = !!(data.active || data.busy);
    reply.innerHTML = md(data.reply || "");
    reply.querySelectorAll("a").forEach((link) => { link.target = "_blank"; link.rel = "noopener noreferrer"; });
    if (wasBusy && !data.busy) document.dispatchEvent(new CustomEvent("loom:models-changed"));
    wasBusy = !!data.busy;
    status.textContent = data.status || (data.busy ? "Chargement…" : "");
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
      const response = await fetch("/models/add?session_id=" + encodeURIComponent(sid));
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
      const response = await postForm("/models/add", {session_id: sid, message});
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
    if (sending) { showToast("Une étape d’ajout est déjà en cours."); return; }
    sid = sessionId || getSessionId();
    if (!sid) return;
    revision++;
    wasBusy = false;
    panel.hidden = false;
    const data = await refresh();
    if (data && !data.active && !data.busy) await send("/add-model" + (argument ? " " + argument : ""));
  }
  openButton.onclick = () => open(getSessionId());
  document.addEventListener("loom:manage-models", (event) => {
    if (event.detail?.add) open(event.detail.sid, event.detail.argument);
  });
  document.addEventListener("loom:add-model-cancelled", (event) => {
    if (event.detail.sid === sid) refresh();
  });
  form.onsubmit = (event) => { event.preventDefault(); if (input.value.trim()) send(input.value.trim()); };
  cancel.onclick = () => send("/cancel");
}
