from __future__ import annotations

from __future__ import annotations
import httpx
from openai import APIConnectionError, APIError, APITimeoutError




def _classify_api_error(exc: APIError) -> str:
    """Range une erreur du SDK openai en catégorie d'ACTION (pas en code HTTP brut).

    Le piège historique : tout `APIError` était traité comme un overflow (« écris plus
    petit »), y compris un 404 « modèle inconnu » ou un serveur éteint -> diagnostic
    trompeur + retries inutiles. On discrimine :
    - 'timeout' / 'connection' : transport (serveur lent ou pas lancé) -> stop, pas de retry ;
    - 'model_not_found' : 404 (llama-swap n'a pas le modèle demandé) -> stop ;
    - 'other' : erreur cliente 4xx (auth, requête invalide) -> stop, on remonte la cause ;
    - 'overflow' : 5xx OU erreur sans statut (tool_call vraisemblablement tronqué par
      max_tokens) -> seul cas où « écris plus court » + retry borné a un sens.
    """
    if isinstance(exc, APITimeoutError):  # sous-classe d'APIConnectionError -> avant
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return "connection"
    status = getattr(exc, "status_code", None)
    if status == 404:
        return "model_not_found"
    # Débordement de la FENÊTRE DE CONTEXTE (entrée) : llama.cpp/llama-swap le renvoie en
    # 400 « request (N tokens) exceeds the available context size ». C'est RÉCUPÉRABLE (on
    # compacte l'historique et on relance), à ne pas confondre avec une vraie 4xx cliente.
    msg = str(getattr(exc, "message", "") or exc).lower()
    if "context" in msg and (
        "exceed" in msg or "size" in msg or "length" in msg or "too long" in msg
    ):
        return "context_overflow"
    if status is not None and status < 500:
        return "other"
    return "overflow"


def _classify_stream_error(exc: Exception) -> str:
    """Erreur pendant le STREAM : le SDK openai n'enrobe que la phase de requête ; en
    pleine itération, httpx fuit À NU (ReadTimeout vécu en prod : prefill post-compaction
    plus long que le timeout de lecture -> traceback brut au lieu du message propre).
    On range ces exceptions dans les mêmes catégories d'action que les APIError."""
    if isinstance(exc, APIError):
        return _classify_api_error(exc)
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return "connection"
