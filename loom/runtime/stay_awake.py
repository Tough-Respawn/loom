"""Empêche la mise en veille du système tant qu'une génération tourne.

Quand Windows se met en veille, TOUS les processus sont suspendus (CPU coupé) : loom.web, le
serveur modèle llama.cpp et le relais SSE gèlent -> la génération s'arrête et le navigateur
perd la connexion (« network error »). On demande à Windows de ne pas endormir le SYSTÈME
(ES_SYSTEM_REQUIRED) tant qu'au moins une génération est active ; l'ÉCRAN peut toujours
s'éteindre (on ne pose PAS ES_DISPLAY_REQUIRED) -> le travail continue EN ARRIÈRE-PLAN.

Un thread dédié RÉ-AFFIRME l'état périodiquement : `SetThreadExecutionState` est lié au
thread appelant (l'état tombe si ce thread meurt), or les requêtes Flask tournent sur des
threads éphémères -> on centralise l'état dans un thread persistant. No-op hors Windows.

NOTE : ES_SYSTEM_REQUIRED bloque la veille par INACTIVITÉ (le cas courant : on s'éloigne, le
minuteur endort la machine). Il NE bloque PAS une veille FORCÉE (fermeture du capot selon la
config d'alimentation, ou veille manuelle). Pour couvrir la fermeture du capot il faut
changer le plan d'alim (powercfg) — hors scope de ce garde-fou.
"""

from __future__ import annotations

import sys
import threading

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class StayAwake:
    """Compteur de générations actives -> maintient le système éveillé tant que > 0."""

    def __init__(self) -> None:
        self._n = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._win = sys.platform == "win32"
        if self._win:
            threading.Thread(
                target=self._loop, daemon=True, name="loom-stay-awake"
            ).start()

    def _apply(self, keep: bool) -> None:
        import ctypes

        flags = _ES_CONTINUOUS | (_ES_SYSTEM_REQUIRED if keep else 0)
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
        except Exception:  # noqa: BLE001 - best-effort : jamais casser une génération
            pass

    def _loop(self) -> None:
        while True:
            with self._lock:
                keep = self._n > 0
            self._apply(keep)
            # Réaffirmer périodiquement couvre les OS qui expirent cette demande.
            self._wake.wait(30)
            self._wake.clear()

    def acquire(self) -> None:
        """Une génération démarre : garder le système éveillé."""
        with self._lock:
            self._n += 1
        self._wake.set()

    def release(self) -> None:
        """Une génération finit : libérer si plus aucune n'est active."""
        with self._lock:
            self._n = max(0, self._n - 1)
        self._wake.set()

    def active(self) -> int:
        with self._lock:
            return self._n
