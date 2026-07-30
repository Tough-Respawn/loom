from __future__ import annotations

from __future__ import annotations
import re
import subprocess
import sys
from flask import Response, render_template, request

from loom.web.routes.helpers import _ctx
from loom.web.routes.skills import _index_context




# ---- Routes : socle (index, statiques, toggles) ---------------------------------------


def _register_misc_routes(app, S):
    @app.get("/")
    def index() -> str:
        return render_template("index.html", **_index_context(S))

    @app.get("/genimg/<sid>/<name>")
    def genimg_session(sid: str, name: str):
        # Sert les médias générés depuis le dossier de LA session (unique copie).
        # `sid` compose le CHEMIN de base : validé strictement (12 hex, format
        # uuid4.hex[:12] de SessionStore.create) sinon 404 — un sid forgé (`..`,
        # antislash Windows) ferait pointer la base hors de var/sessions.
        # send_from_directory protège `name` ; mimetype déduit du nom (png/webm).
        from flask import send_from_directory

        if not re.fullmatch(r"[0-9a-f]{12}", sid):
            return Response("session invalide", status=404)
        return send_from_directory(S.session_store.root / sid / "generated", name)

    @app.get("/genimg/<name>")
    def genimg(name: str):
        # LEGACY : messages d'avant 2026-07-09, servis depuis var/generated.
        from flask import send_from_directory

        return send_from_directory(S.generated_dir, name)

    @app.get("/favicon.ico")
    def favicon():
        # Requête par défaut du navigateur (silence le 404) : on sert le SVG de la trame.
        # Le <link rel="icon" type="image/svg+xml"> reste la source primaire de l'onglet.
        from flask import send_from_directory

        return send_from_directory(
            app.static_folder, "favicon.svg", mimetype="image/svg+xml"
        )

    @app.post("/tool_decision")
    def tool_decision():
        pend = S.pending.get(request.form.get("id", ""))

        if pend is not None:
            pend["approved"] = request.form.get("approve") == "1"

            pend["event"].set()

        return Response("", status=204)

    @app.post("/tools")
    def tools_update():
        conv, save = _ctx(S)

        conv.set_tools(request.form.getlist("tool"))

        save()

        return render_template(
            "_tools.html",
            available_tools=S.available_tools,
            active_tools=conv.active_tools,
        )

    @app.post("/thinking")
    def thinking_update():
        conv, save = _ctx(S)

        conv.set_thinking(request.form.get("thinking") == "1")

        save()

        return Response(str(int(conv.thinking)), mimetype="text/plain")

    @app.post("/local_only")
    def local_only_update():
        # Session PRIVÉE : coupe tout routage distant des sous-agents (chaîne
        # dispatch_models ignorée). Décision humaine par session, persistée.
        conv, save = _ctx(S)

        conv.set_local_only(request.form.get("local_only") == "1")

        save()

        return Response(str(int(conv.local_only)), mimetype="text/plain")

    @app.route("/pick-folder", methods=["POST"])
    def pick_folder():
        # Sous-processus : évite les soucis tkinter hors du thread principal de Flask.

        script = (
            "import tkinter, tkinter.filedialog as fd;"
            "r=tkinter.Tk(); r.withdraw(); r.attributes('-topmost', True);"
            "p=fd.askdirectory(); print(p if p else '')"
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
            )

        except Exception as exc:  # noqa: BLE001 - tkinter absent / timeout
            return {"path": "", "error": str(exc)[:200]}

        path = (proc.stdout or "").strip()

        if proc.returncode != 0 and not path:
            return {
                "path": "",
                "error": (proc.stderr or "sélecteur indisponible")[:200],
            }

        return {"path": path}

    @app.get("/sysmon")
    def sysmon_metrics():
        # Métriques système LIVE (CPU/RAM/GPU) pour le moniteur affiché avec un modèle LOCAL.
        # nvidia-smi + psutil ; champs à None si une source manque (le front s'adapte).
        from loom.runtime.sysmon import read_metrics

        return read_metrics()
