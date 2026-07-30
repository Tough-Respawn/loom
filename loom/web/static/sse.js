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
