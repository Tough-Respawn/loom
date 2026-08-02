"""Cycle de vie du serveur modèle GÉRÉ PAR loom.web (démarrage auto + boutons UI).

Le serveur (serve.py -> llama-swap -> llama-server) est lancé comme ENFANT de
loom.web : il démarre à la demande (sélection d'un modèle local, interaction, ou
bouton), reste vivant tant que Loom tourne, et meurt AVEC loom.web — crash compris —
grâce à un Job Object Windows « kill-on-close » (sur POSIX : groupe de session +
atexit). L'utilisateur reste maître : bouton « éteindre le serveur » dans l'UI.
"""

from __future__ import annotations

import atexit
import subprocess
import sys
import threading

from loom.runtime.serve import _terminate_tree


def _win_job_kill_on_close():
    """Job Object Windows avec KILL_ON_JOB_CLOSE : tout process assigné (et ses
    descendants, llama-server compris) meurt quand le handle se ferme — c.-à-d. quand
    le process loom.web disparaît, quelle qu'en soit la cause. C'est la SEULE garantie
    « pas d'orphelin qui tient 20 Go de RAM » sur Windows (atexit ne couvre pas un
    crash). Renvoie le handle à garder vivant, ou None (POSIX / API indisponible)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _Io(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _Io),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job,
        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        kernel32.CloseHandle(job)
        return None
    return job


def _assign_to_job(job, pid: int) -> bool:
    """Attache un process fraîchement lancé au job (ses futurs enfants suivront)."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001
    handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    if not handle:
        return False
    ok = bool(kernel32.AssignProcessToJobObject(job, handle))
    kernel32.CloseHandle(handle)
    return ok


class ModelServerManager:
    """Une seule instance gérée à la fois ; start() concurrents dédupliqués (verrou).

    Ne gère QUE le serveur qu'elle a elle-même lancé : une stack démarrée à la main
    dans un terminal reste sous le contrôle de son terminal (on ne tue pas ce qu'on
    n'a pas créé)."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._job = None
        self._lock = threading.Lock()
        self._starting = False
        atexit.register(self.stop)

    def owns_running(self) -> bool:
        """Vrai si NOTRE serve.py tourne encore (peu importe que llama-swap réponde déjà)."""
        p = self._proc
        return p is not None and p.poll() is None

    @property
    def starting(self) -> bool:
        """Démarrage en cours : lancé par nous, pas encore confirmé joignable par l'UI."""
        return self._starting and self.owns_running()

    def confirm_started(self) -> None:
        """À appeler quand le serveur a répondu (sonde /machine_state) : fin du démarrage."""
        self._starting = False

    def start(self) -> bool:
        """Lance `python -m loom.runtime.serve` (télécharge les GGUF manquants, génère le
        yaml, lance llama-swap). Idempotent. Sa sortie détaillée va déjà dans
        var/logs/serve.log (serve.py s'en charge)."""
        with self._lock:
            if self.owns_running():
                return True
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, "-m", "loom.runtime.serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=(sys.platform != "win32"),
                )
            except OSError:
                self._proc = None
                return False
            if sys.platform == "win32":
                if self._job is None:
                    self._job = _win_job_kill_on_close()
                if self._job is not None:
                    _assign_to_job(self._job, self._proc.pid)
            self._starting = True
            return True

    def stop(self) -> bool:
        """Éteint l'arbre complet (serve.py + llama-swap + llama-server) et libère
        RAM/VRAM. Sans effet sur un serveur non géré (lancé hors Loom)."""
        with self._lock:
            p = self._proc
            self._proc = None
            self._starting = False
            if p is None or p.poll() is not None:
                return False
            _terminate_tree(p)
            return True
