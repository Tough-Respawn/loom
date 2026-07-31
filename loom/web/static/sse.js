// loom/web/static/sse.js — issu du decoupage de app.js (comportement constant).
export async function streamSSE(url, fd, onEvent, signal) {
  const resp = await fetch(url, { method: fd ? "POST" : "GET", body: fd || undefined, signal });
  if (resp.status === 429) {
    onEvent({ type: "error", message: "Occupé : un échange est déjà en cours." });
    return;
  }
  if (resp.status === 400) {
    onEvent({ type: "error", message: "Requête invalide." });
    return;
  }
  if (resp.status === 202) {
    // Message légitimement MIS EN FILE (génération de la session en cours) : le back
    // le drainera au prochain point d'arrêt. Réponse TEXTE (pas un flux SSE) -> on la
    // lit et on signale l'état « en file » à l'appelant, au lieu de l'avaler en silence
    // (sinon la bulle utilisateur reste affichée mais jamais persistée -> disparaît au
    // rechargement).
    let msg = "";
    try {
      msg = (await resp.text()).trim();
    } catch (e) {
      /* corps illisible : on garde le libellé par défaut côté appelant */
    }
    onEvent({ type: "queued", message: msg });
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      onEvent(JSON.parse(part.slice(6)));
    }
  }
}
