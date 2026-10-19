"""Suivi du journal du serveur modèle (llama-server) via llama-swap.

llama-swap lance llama-server lui-même et garde sa sortie dans ses tampons : rien n'en
arrive dans serve.log (cf. serve.capture_upstream_log). Il l'expose en flux sur
`GET /logs/stream/upstream` (text/plain : le tampon d'abord, puis le direct). Ce module
tient ce flux ouvert dans un fil de fond et conserve les dernières lignes numérotées,
pour que l'interface montre dans le fil ce que « voit le terminal » pendant qu'un
modèle se charge (chargement, ajustement mémoire, avancement du prefill), sans que la
génération ait à s'en occuper.
"""

from __future__ import annotations

import re
import threading
from collections import deque

_PREFIX = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] ")
_BACKLOG = 2000


def strip_prefix(line: str) -> str:
    """Retire l'horodatage relatif et le niveau de llama-server (`0.01.139.576 I `)."""
    return _PREFIX.sub("", line.rstrip("\r\n"))


class UpstreamLogTail:
    """Tampon circulaire des lignes du serveur modèle, nourri par un fil de fond.

    `since(seq)` rend (seq courant, lignes de numéro > seq). À la (re)connexion,
    llama-swap rejoue son tampon : les lignes déjà vues sont ignorées jusqu'à la
    première inconnue, pour ne pas dupliquer l'historique après une coupure."""

    def __init__(self, root: str, reconnect_s: float = 2.0):
        self.root = root.rstrip("/")
        self.reconnect_s = reconnect_s
        self._lines: deque[tuple[int, str]] = deque(maxlen=_BACKLOG)
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ---- lecture (interface) ----
    @property
    def seq(self) -> int:
        return self._seq

    def since(self, seq: int) -> tuple[int, list[str]]:
        with self._lock:
            return self._seq, [ln for n, ln in self._lines if n > seq]

    # ---- alimentation ----
    def feed(self, raw_lines, resync: bool = False) -> int:
        """Ajoute des lignes (brutes) au tampon ; renvoie le nombre retenu. `resync` :
        début d'un rejeu de tampon, les lignes déjà connues sont sautées."""
        added = 0
        with self._lock:
            known = {ln for _, ln in self._lines} if resync else None
            for raw in raw_lines:
                line = strip_prefix(raw)
                if not line.strip():
                    continue
                if known is not None:
                    if line in known:
                        continue
                    known = None  # première ligne inconnue : on est en direct
                self._seq += 1
                self._lines.append((self._seq, line))
                added += 1
        return added

    def start(self) -> UpstreamLogTail:
        threading.Thread(
            target=self._run, daemon=True, name="loom-upstream-log"
        ).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import urllib.request

        while not self._stop.is_set():
            try:
                # Un long délai d'inactivité : le flux se tait quand le modèle ne fait
                # rien ; une coupure (serveur arrêté) ferme la socket et on se reconnecte.
                with urllib.request.urlopen(
                    self.root + "/logs/stream/upstream", timeout=3600
                ) as resp:
                    resync = True
                    while not self._stop.is_set():
                        raw = resp.readline()
                        if not raw:
                            break
                        self.feed([raw.decode("utf-8", "replace")], resync=resync)
                        resync = False
            except Exception:  # noqa: BLE001 - serveur éteint, réseau : on réessaie
                pass
            self._stop.wait(self.reconnect_s)
