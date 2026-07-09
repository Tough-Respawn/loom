# loom/runtime/comfy.py
"""Moteur ComfyUI géré par Loom : démarrage (Job Object kill-on-close), soumission
d'un workflow API et récupération du PNG. HTTP uniquement (urllib), aucune dépendance.

VRAM (6 Go) : UN modèle à la fois — l'appelant décharge le LLM (unload_local) AVANT
generate(), et appelle free() quand on rebascule sur un modèle texte. RAM : free()
peut GARDER les poids image en cache (keep_ram=True) quand la machine est assez
large pour LLM + cache image ensemble — la prochaine image évite le rechargement
disque. L'arbitrage (RAM disponible vs taille du LLM) vit côté web/app.py."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from loom.runtime.manager import (
    _assign_to_job,
    _terminate_tree,
    _win_job_kill_on_close,
)


class ComfyError(RuntimeError):
    """Erreur montrable dans le chat (jamais de stacktrace brute)."""


class ComfyEngine:
    def __init__(self, comfy_dir: str, port: int = 8188) -> None:
        self.dir = Path(comfy_dir)
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None
        self._job = None
        self._lock = threading.Lock()

    # --- processus --------------------------------------------------------
    def is_up(self, timeout: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/system_stats", timeout=timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def ensure_up(self, timeout: float = 180.0) -> bool:
        """Démarre ComfyUI si besoin (le python de SON venv privé, cwd = son install)
        et attend le port. Ne tue jamais une instance lancée hors Loom (on ne gère que
        la nôtre — même règle que le serveur modèle)."""
        if self.is_up():
            return True
        py = self.dir / ".venv" / "Scripts" / "python.exe"
        if not py.is_file():
            raise ComfyError(
                f"ComfyUI introuvable ({py}) — vérifie comfy_dir dans model.toml."
            )
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                try:
                    self._proc = subprocess.Popen(
                        [str(py), "main.py", "--port", str(self.port)],
                        cwd=str(self.dir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=(sys.platform != "win32"),
                    )
                except OSError as exc:
                    raise ComfyError(f"démarrage ComfyUI impossible : {exc}") from exc
                if sys.platform == "win32":
                    if self._job is None:
                        self._job = _win_job_kill_on_close()
                    if self._job is not None:
                        _assign_to_job(self._job, self._proc.pid)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_up():
                return True
            time.sleep(1.0)
        raise ComfyError(
            f"ComfyUI ne répond pas sur :{self.port} après {int(timeout)} s "
            "(premier démarrage lent ? relance ; sinon lance-le à la main pour voir l'erreur)."
        )

    def stop(self) -> None:
        with self._lock:
            p, self._proc = self._proc, None
            if p is not None and p.poll() is None:
                _terminate_tree(p)

    # --- génération -------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float = 30.0) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        return json.loads(body) if body.strip() else {}

    def upload_image(self, path: str) -> str:
        """Envoie une image d'ENTRÉE à ComfyUI (POST /upload/image, multipart minimal
        en urllib — toujours zéro dépendance) ; renvoie le nom sous lequel un nœud
        LoadImage peut la charger."""
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise ComfyError(f"photo d'entrée illisible ({path}) : {exc}") from exc
        boundary = "----loom" + hex(random.getrandbits(64))[2:]
        part = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{p.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
        tail = (
            f"\r\n--{boundary}\r\n"
            'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f"true\r\n--{boundary}--\r\n"
        ).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/upload/image",
            data=part + data + tail,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read().decode("utf-8", "replace") or "{}")
        except (urllib.error.URLError, OSError) as exc:
            raise ComfyError(f"upload de la photo vers ComfyUI échoué : {exc}") from exc
        return str(out.get("name") or p.name)

    def generate(
        self,
        workflow_template: str,
        prompt: str,
        timeout: float = 600.0,
        image_path: str | None = None,
    ) -> tuple[bytes, str]:
        """Injecte prompt+seed dans le template, soumet, attend, renvoie le média
        produit : (bytes, extension) — « .png » pour une image, « .webm »/« .mp4 »
        pour une vidéo (workflows Wan). L'appelant nomme le fichier avec l'extension.

        Le prompt est injecté via json.dumps (jamais de collage brut : guillemets et
        retours à la ligne restent un JSON valide). {SEED} : entier aléatoire 63 bits
        -> chaque message donne une image différente, comme le « randomize » de l'UI.
        {IMAGE} (édition/i2v, ex. Kontext ou Wan i2v) : photo d'entrée uploadée puis
        référencée par son nom — requise si le template la déclare."""
        # Sweep des doublons RÉSIDUELS d'anciennes générations (purge immédiate ratée :
        # handle Windows tenu par ComfyUI au moment du unlink). Tout loom_* présent ici
        # prédate ce job (générations sérialisées) -> suppression sûre, best-effort.
        try:
            for stale in (self.dir / "output").glob("loom_*"):
                try:
                    stale.unlink()
                except OSError:
                    pass
        except OSError:
            pass
        wf = workflow_template.replace(
            '"{PROMPT}"', json.dumps(prompt, ensure_ascii=False)
        )
        wf = wf.replace('"{SEED}"', str(random.getrandbits(63)))
        if '"{IMAGE}"' in wf:
            if not image_path:
                raise ComfyError(
                    "ce modèle ÉDITE une photo existante : mets le chemin du fichier "
                    "image (ex. C:/photos/moi.jpg) dans ton message, avec l'instruction."
                )
            wf = wf.replace(
                '"{IMAGE}"',
                json.dumps(self.upload_image(image_path), ensure_ascii=False),
            )
        try:
            graph = json.loads(wf)
        except json.JSONDecodeError as exc:
            raise ComfyError(f"workflow.json invalide après injection : {exc}") from exc
        try:
            sub = self._post("/prompt", {"prompt": graph})
        except (urllib.error.URLError, OSError) as exc:
            raise ComfyError(f"soumission à ComfyUI échouée : {exc}") from exc
        if "error" in sub or "prompt_id" not in sub:
            # Nœud manquant / entrée invalide : ComfyUI détaille dans node_errors.
            detail = json.dumps(sub, ensure_ascii=False)[:300]
            raise ComfyError(f"workflow refusé par ComfyUI : {detail}")
        pid = sub["prompt_id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(1.5)
            try:
                with urllib.request.urlopen(
                    f"{self.base}/history/{pid}", timeout=10
                ) as resp:
                    hist = json.loads(resp.read().decode("utf-8", "replace"))
            except (urllib.error.URLError, OSError):
                continue  # transitoire : ComfyUI charge le modèle
            entry = hist.get(pid)
            if not entry:
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [
                    m[1].get("exception_message", "")
                    for m in status.get("messages", [])
                    if m and m[0] == "execution_error" and isinstance(m[1], dict)
                ]
                raise ComfyError(
                    "génération échouée côté ComfyUI : "
                    + (msgs[0] if msgs else "?")[:200]
                )
            # Sortie = premier fichier produit, quel que soit le nœud d'écriture :
            # SaveImage range sous "images", SaveWEBM/PreviewVideo sous d'autres clés —
            # on scanne toute liste de dicts porteurs d'un "filename".
            for out in entry.get("outputs", {}).values():
                for lst in out.values():
                    if not isinstance(lst, list):
                        continue
                    for im in lst:
                        if not (isinstance(im, dict) and im.get("filename")):
                            continue
                        q = urllib.parse.urlencode(
                            {
                                "filename": im["filename"],
                                "subfolder": im.get("subfolder", ""),
                                "type": im.get("type", "output"),
                            }
                        )
                        with urllib.request.urlopen(
                            f"{self.base}/view?{q}", timeout=120
                        ) as resp:
                            ext = Path(im["filename"]).suffix or ".png"
                            payload = resp.read()
                        # ComfyUI a écrit SA copie dans <comfy>/output : doublon inutile
                        # (l'unique copie vit dans le dossier de session Loom) -> purgée
                        # aussitôt récupérée. ComfyUI peut encore tenir un handle Windows
                        # quelques instants -> retries ; en dernier recours le sweep du
                        # prochain generate() (ci-dessus) rattrape le résidu.
                        dup = (
                            self.dir
                            / "output"
                            / im.get("subfolder", "")
                            / im["filename"]
                        )
                        for _ in range(4):
                            try:
                                dup.unlink(missing_ok=True)
                                break
                            except OSError:
                                time.sleep(0.7)
                        else:
                            print(f"[loom] doublon ComfyUI non purgé : {dup}")
                        return payload, ext
        raise ComfyError(f"génération sans réponse après {int(timeout)} s (timeout).")

    def free(self, keep_ram: bool = False) -> None:
        """Rend la VRAM (déchargement des modèles image) — best-effort, jamais bloquant.

        keep_ram=True garde les poids en cache RAM : la prochaine image repart du
        cache au lieu du disque. À réserver aux machines où LLM + cache tiennent
        ensemble (l'appelant décide, cf. _free_image_engines dans web/app.py)."""
        try:
            self._post(
                "/free",
                {"unload_models": True, "free_memory": not keep_ram},
                timeout=10,
            )
        except (urllib.error.URLError, OSError, ComfyError):
            pass
