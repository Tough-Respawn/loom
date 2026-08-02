# loom/agent/client.py
"""Client modèle : parle à l'endpoint OpenAI-compatible de Loom via le SDK openai."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from loom.agent.compaction import (
    _SUMMARY_MARKER,
    _SUMMARY_SYSTEM,
    _ctx_estimate,
    _flatten_for_summary,
    _force_fit,
    _inject_notes,
    _microcompact_tools,
    _msg_chars,
)
from loom.agent.debuglog import _debug, _debug_messages, log_event, tools_fingerprint
from loom.agent.debuglog import context_fingerprint as context_fingerprint
from loom.agent.debuglog import set_debug_log_path as set_debug_log_path
from loom.agent.errors import _classify_stream_error
from loom.agent.guards import (
    _REFOCUS_NOTE,
    _check_no_progress,
    _dispatch_no_tool_calls,
    _loom_nudge,
)
from loom.agent.guards import _claims_missing_artifact as _claims_missing_artifact
from loom.agent.streaming import (
    _close,
    _estimate_usage,
    _iter_events,
    _salvage_tool_calls,
    _stream_model_turn,
    build_create_kwargs,
)
from loom.agent.streaming import _turn_timing_fields as _turn_timing_fields
from loom.agent.toolrun import _run_tools_parallel, _run_tools_sequential, _safe_args
from loom.agent.toolsets import _DEBUG_FORCE, _PARALLEL_SAFE


def _inject_monitor_events(provider, convo: list[dict]):
    """Injecte les événements asynchrones comme de vrais résultats d'outil."""
    if provider is None:
        return 0
    count = 0
    for event in provider() or []:
        call_id = f"monitor_event_{event['id']}"
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "monitor",
                        "arguments": json.dumps(
                            {
                                "action": "event",
                                "monitor_id": event["monitor_id"],
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": event["model_content"],
        }
        convo.extend((assistant_message, tool_message))
        yield (
            "monitor_event",
            {
                **event,
                "tool_call_id": call_id,
                "assistant_message": assistant_message,
                "tool_message": tool_message,
            },
        )
        count += 1
    return count


class LoomClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "loom-local",
        model: str = "local",
        timeout: int = 120,
        max_retries: int = 6,
        routes: dict | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # Serveur LOCAL : timeout de LECTURE long. Pendant un gros prefill (contexte
        # recalculé après compaction : plusieurs minutes à ~200 t/s) ou un chargement de
        # modèle, llama-server n'émet RIEN — c'est du travail légitime, pas une panne.
        # Un read=120s coupait ces phases (ReadTimeout vécu). Connexion/écriture restent
        # au timeout court : un serveur éteint doit échouer vite.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(float(timeout), read=max(600.0, float(timeout))),
            max_retries=max_retries,
        )
        # Routes vers des modèles DISTANTS (API OpenAI-compatible) : id de modèle ->
        # {base_url, api_key, model, enable_thinking_param}. Tout modèle absent de cette
        # table part vers l'endpoint LOCAL (_client). Un client openai par endpoint, monté
        # une fois ici. Le reste du code appelle toujours `model=<id>` : le routage est interne.
        self._routes: dict[str, dict] = {}
        # Disjoncteur slot KV : modèles dont le save/restore a PENDU (hang serveur,
        # cf. _slot_action) -> on n'essaie plus jusqu'au restart du process.
        self._slot_broken: set[str] = set()
        for rid, spec in (routes or {}).items():
            self._routes[rid] = {
                "client": OpenAI(
                    base_url=spec["base_url"],
                    api_key=spec.get("api_key") or "none",
                    timeout=timeout,
                    max_retries=max_retries,
                ),
                "base_url": spec["base_url"],
                "api_key": spec.get("api_key") or "",
                "model": spec.get("model") or rid,
                "enable_thinking_param": bool(spec.get("enable_thinking_param", False)),
            }
        # Cache de la fenêtre de contexte découverte par provider (id Loom -> int|None).
        # None mémorisé = provider interrogé mais muet -> on ne re-frappe pas l'API.
        self._ctx_cache: dict[str, int | None] = {}

    def _resolve(self, model: str | None):
        """(client_openai, model_api, native_extras) pour le modèle demandé. Un modèle
        distant (présent dans routes) part vers son endpoint, SANS les extra_body llama.cpp ;
        sinon l'endpoint local avec les extras natifs."""
        if model and model in self._routes:
            r = self._routes[model]
            return r["client"], r["model"], r["enable_thinking_param"]
        return self._client, (model or self.model), True

    def summarize_slice(
        self,
        old_messages: list[dict],
        model: str | None = None,
        budget_chars: int = 30000,
    ) -> str:
        """PRIMITIVE UNIQUE de résumé, partagée par TOUS les chemins de compaction : l'étage
        de la boucle d'outils, le résumé pré-tour (context.summarize) et le bouton manuel.
        Aplatit les vieux tours (borné), appelle le modèle en NON-stream, retire le <think>.
        Renvoie le texte du résumé, ou '' si rien à résumer / réponse vide / appel en échec.
        FAIL-SOFT : ne lève jamais (un résumé raté ne doit jamais crasher l'appelant)."""
        body = _flatten_for_summary(list(old_messages), budget_chars)
        if not body.strip():
            return ""
        oai, api_model, _ = self._resolve(model)
        try:
            resp = oai.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": body},
                ],
                # 700 (et non 2000) : un résumé dense/télégraphique n'a pas besoin de plus,
                # et sur un modèle local lent (~8 tok/s) 2000 tokens = plusieurs MINUTES.
                max_tokens=700,
                temperature=0.2,
            )
            summary = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - best-effort : jamais crasher l'appelant
            log_event(
                "summary.error",
                level="WARN",
                msg=f"{type(exc).__name__}: {str(exc)[:120]}",
            )
            return ""
        # Modèle « thinking » (Qwen/local) : le raisonnement peut précéder le contenu -> on
        # ne garde que l'après-</think>.
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()
        return summary.strip()

    def summarize_old_turns(
        self,
        convo: list[dict],
        model: str | None = None,
        keep_recent: int = 6,
        budget_chars: int = 30000,
    ) -> int:
        """Résume les vieux tours EN PLACE via `summarize_slice` (garde les `keep_recent`
        derniers intacts). Renvoie le nb de messages remplacés (0 = trop peu à résumer,
        réponse vide, ou appel en échec — rien n'est touché)."""
        cut = len(convo) - keep_recent
        if cut < 2:
            return 0  # trop peu de vieux tours (ou convo plus courte que keep_recent)
        # Ne pas orpheliner un résultat d'outil : si la queue conservée débute par un
        # role:tool dont l'appel part au résumé, on pousse ces tool vers le bloc résumé
        # (certains providers rejettent un message tool sans tool_calls le précédant).
        while cut < len(convo) and convo[cut].get("role") == "tool":
            cut += 1
        summary = self.summarize_slice(convo[:cut], model, budget_chars)
        if not summary:
            return 0
        convo[:cut] = [{"role": "user", "content": f"{_SUMMARY_MARKER}\n{summary}"}]
        return cut

    def compact_conversation(
        self,
        messages: list[dict],
        system_prompt: str = "",
        target_chars: int | None = None,
        keep_recent_tools: int = 2,
    ) -> tuple[list[dict], int]:
        """Compaction MANUELLE (bouton UI) : DÉTERMINISTE et INSTANTANÉE sur une COPIE.

        AUCUN appel modèle : un résumé LLM sur un modèle local lent (~8 tok/s × 2000 tokens)
        bloquerait le bouton PLUSIEURS MINUTES (observé live : /compact figé à « … », verrou
        tenu, 429 sur les clics suivants). Ici c'est purement local/instantané :
        1. vide les vieux résultats d'outils (`_microcompact_tools`) ;
        2. si `target_chars` est donné, CLIPPE le contexte vivant pour tenir dessous
           (`_force_fit`) — libère la conversation jusqu'au plancher. Le prompt système +
           les schémas d'outils, eux, sont INCOMPRESSIBLES (souvent l'essentiel du ctx).
        Renvoie (messages, tokens_libérés_estimés) ; 0 = déjà au plus bas."""
        convo = list(messages)

        def _tot() -> int:
            return len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)

        before = _tot()
        _microcompact_tools(convo, keep_recent_tools)
        if target_chars:
            _force_fit(convo, system_prompt, target_chars)
        return convo, max(0, (before - _tot()) // 3)

    def local_server_root(self) -> str:
        """Racine du serveur LOCAL (llama-swap) : la base_url SANS le suffixe `/v1`.
        Sert à joindre l'API de management (`/running`, `/api/models/unload`)."""
        b = self.base_url
        return b[:-3] if b.endswith("/v1") else b.rstrip("/")

    def is_remote(self, model: str | None) -> bool:
        """Vrai si le modèle est servi par une API DISTANTE (route montée), pas en local."""
        return bool(model and model in self._routes)

    def add_remote_route(self, model_id: str, spec: dict) -> None:
        """Monte (ou remplace) À CHAUD la route d'un modèle distant : un client OpenAI de plus,
        sans redémarrer. Un distant = juste une URL + une clé, rien à charger en VRAM -> l'ajout
        est immédiat. Invalide le cache de contexte pour ce modèle."""
        self._routes[model_id] = {
            "client": OpenAI(
                base_url=spec["base_url"],
                api_key=spec.get("api_key") or "none",
                timeout=self.timeout,
                max_retries=self.max_retries,
            ),
            "base_url": spec["base_url"],
            "api_key": spec.get("api_key") or "",
            "model": spec.get("model") or model_id,
            "enable_thinking_param": bool(spec.get("enable_thinking_param", False)),
        }
        self._ctx_cache.pop(model_id, None)

    def remove_remote_route(self, model_id: str) -> None:
        """Démonte à chaud la route d'un modèle distant (l'id disparaît du sélecteur)."""
        self._routes.pop(model_id, None)
        self._ctx_cache.pop(model_id, None)

    def remote_api_key(self, model_id: str) -> str:
        """Clé brute d'une route distante — USAGE SERVEUR uniquement (préserver la clé lors
        d'une édition sans la re-saisir, indice masqué). NE JAMAIS renvoyer telle quelle au client."""
        return (self._routes.get(model_id) or {}).get("api_key", "")

    def remote_route_info(self, model_id: str) -> dict:
        """Infos SÛRES d'une route distante pour l'UI (jamais la clé en clair) : base_url,
        modèle côté provider, présence d'une clé."""
        r = self._routes.get(model_id) or {}
        return {
            "base_url": r.get("base_url", ""),
            "model": r.get("model", model_id),
            "has_key": bool(r.get("api_key")),
            "enable_thinking_param": bool(r.get("enable_thinking_param", False)),
        }

    def ping_remote(
        self, base_url: str, api_key: str, model: str, timeout: float = 15.0
    ) -> tuple[bool, str]:
        """Test de connexion RÉEL d'un endpoint distant : 1 requête 1-token, non-stream,
        sans retry. (ok, message). Sert au bouton « Tester » avant d'enregistrer un modèle."""
        try:
            oai = OpenAI(
                base_url=base_url,
                api_key=api_key or "none",
                timeout=timeout,
                max_retries=0,
            )
            oai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True, "OK"
        except Exception as e:  # noqa: BLE001 - remonte un message clair à l'UI
            return False, f"{type(e).__name__}: {str(e)[:160]}"

    def remote_context(self, model: str | None) -> int | None:
        """Fenêtre de contexte RÉELLE d'un modèle distant, lue DU PROVIDER (`GET /models`).

        C'est « le modèle lui-même » qui répond, pas la config Loom. Best-effort : renvoie
        l'entier si le provider publie un champ de contexte (OpenRouter `context_length`,
        vLLM `max_model_len`, `max_input_tokens`…), sinon None — beaucoup d'API (Z.ai, OpenAI)
        renvoient le schéma nu (id/object/created/owned_by) et NE publient rien : l'appelant
        retombe alors sur la valeur déclarée en config. Résultat mis en cache (l'absence aussi)
        pour ne pas re-frapper l'API à chaque page."""
        if not model or model not in self._routes:
            return None
        if model in self._ctx_cache:
            return self._ctx_cache[model]
        import json as _json
        import urllib.error
        import urllib.request

        r = self._routes[model]
        base = str(r.get("base_url") or "").rstrip("/")
        api_model = r.get("model") or model
        key = r.get("api_key") or ""
        keys = (
            "context_length",
            "context_window",
            "max_context_length",
            "max_model_len",
            "max_input_tokens",
        )
        found: int | None = None
        try:
            req = urllib.request.Request(
                base + "/models", headers={"Authorization": "Bearer " + key}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = _json.loads(resp.read().decode("utf-8", "replace"))
            items = data.get("data") if isinstance(data, dict) else data
            for m in items or []:
                if not isinstance(m, dict) or m.get("id") != api_model:
                    continue
                for k in keys:
                    v = m.get(k)
                    if isinstance(v, (int, float)) and v > 0:
                        found = int(v)
                        break
                tp = m.get("top_provider")  # OpenRouter niche le contexte ici
                if found is None and isinstance(tp, dict):
                    v = tp.get("context_length")
                    if isinstance(v, (int, float)) and v > 0:
                        found = int(v)
                break
        except (urllib.error.URLError, OSError, ValueError):
            found = None
        self._ctx_cache[model] = found
        return found

    def unload_local(self, model: str | None = None, timeout: float = 30.0) -> bool:
        """Décharge le(s) modèle(s) LOCAL(aux) via l'API llama-swap (libère la VRAM). Model
        None = tous. Best-effort : False si le serveur local est injoignable (non lancé, ou
        llama-server direct sans API de swap). Cf. `POST /api/models/unload[/:model]`."""
        import urllib.error
        import urllib.request

        root = self.local_server_root()
        path = f"/api/models/unload/{model}" if model else "/api/models/unload"
        try:
            req = urllib.request.Request(root + path, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def running_local(self, timeout: float = 5.0) -> tuple[bool, str]:
        """(joignable, texte brut JSON) du serveur LOCAL. `GET /running` est l'API
        de llama-swap (mode multi-modèles) ; en MONO-MODÈLE direct, llama-server
        répond 404 dessus alors qu'il est VIVANT — l'avaler comme « injoignable »
        faisait afficher « serveur éteint » et attendre ~90 s à CHAQUE message
        (vécu 2026-07-21, 1re machine mono-modèle du parc). Toute réponse HTTP =
        serveur vivant ; en direct, l'inventaire se lit sur /v1/models (le chemin
        du GGUF contient l'id du modèle -> les tests par sous-chaîne des appelants
        restent valides). Best-effort."""
        import urllib.error
        import urllib.request

        root = self.local_server_root()
        try:
            with urllib.request.urlopen(root + "/running", timeout=timeout) as resp:
                return True, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError:  # réponse HTTP -> vivant, mais pas llama-swap
            try:
                with urllib.request.urlopen(
                    root + "/v1/models", timeout=timeout
                ) as resp:
                    return True, resp.read().decode("utf-8", "replace")
            except (urllib.error.URLError, OSError, ValueError):
                return True, ""  # vivant (il vient de répondre en HTTP)
        except (urllib.error.URLError, OSError, ValueError):
            return False, ""

    def warmup_local(self, model: str) -> None:
        """Charge un modèle LOCAL en envoyant un ping 1-token : llama-swap charge à la 1re
        requête (et swap l'ancien si besoin). BLOQUANT le temps du chargement -> appeler dans
        un thread. Best-effort (avale toute erreur : serveur non lancé, etc.)."""
        try:
            for _ in self.stream_chat(
                [{"role": "user", "content": "ping"}],
                "",
                1,
                model=model,
                thinking=False,
            ):
                pass
        except Exception as e:  # noqa: BLE001 - warmup best-effort, jamais bloquant
            _debug("WARMUP_ERR", str(e))

    def _slot_action(
        self, model: str | None, action: str, name: str, force: bool = False
    ) -> bool:
        """POST /slots/0?action=save|restore sur le serveur LOCAL (via la route
        llama-swap /upstream/<modèle>/, repli /slots direct pour un llama-server
        sans swap). Nécessite --slot-save-path côté serveur. Best-effort.

        DISJONCTEUR : constaté le 2026-07-10 sur ornith q8 (CUDA + mmproj), le save
        PEND côté llama-server (502 après ~60 s) alors qu'il marche en CPU pur ->
        timeout court (20 s) et, au premier échec, on ARRÊTE d'essayer pour ce
        modèle (jusqu'au restart) : l'appelant retombe sur le ré-amorçage par
        re-prefill. Sans ça, chaque fin de tour perdait ~1 min à attendre le hang."""
        import urllib.request

        if self.is_remote(model):
            return False  # distant : cache géré par le provider, pas de slot local
        # Interrupteurs ([server] slot_kv / hot_resume, cf. config.py) :
        # - slot_kv (legacy, OFF par défaut) autorise save ET restore par tour ;
        # - hot_resume n'autorise que le SAVE en fin de tour — le restore n'y est
        #   permis qu'en one-shot explicite (force=True, via try_hot_resume) ;
        # - force court-circuite les deux (jamais le disjoncteur _slot_broken).
        allowed = (
            force
            or getattr(self, "slot_kv_enabled", False)
            or (action == "save" and getattr(self, "hot_resume_enabled", False))
        )
        if not allowed:
            return False
        key = model or "(local)"
        if key in self._slot_broken:
            return False
        root = self.local_server_root()
        payload = json.dumps({"filename": name}).encode()
        paths = [f"/upstream/{model}/slots/0?action={action}"] if model else []
        paths.append(f"/slots/0?action={action}")
        for i, path in enumerate(paths):
            req = urllib.request.Request(
                root + path,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    body = json.loads(resp.read().decode() or "{}")
                _debug(f"SLOT_{action.upper()}", {"name": name, **body}, terminal=False)
                return True
            except Exception as e:  # noqa: BLE001 - slot KV best-effort, jamais bloquant
                # DISJONCTE (plus d'essais pour ce modèle) sur les échecs DURABLES :
                # - timeout = hang serveur (vécu : ~60 s perdus par tour) ;
                # - HTTP 501 = llama-server refuse (constaté 2026-07-10 : slot save NON
                #   SUPPORTÉ quand un projecteur multimodal --mmproj est chargé — les
                #   modèles vision retombent définitivement sur le ré-amorçage).
                code = getattr(e, "code", None)
                if (
                    isinstance(e, TimeoutError)
                    or "timed out" in str(e).lower()
                    or code == 501
                ):
                    self._slot_broken.add(key)
                    if code == 501:
                        # Cause racine CONFIRMÉE (2026-07-13, A/B même binaire b9442 :
                        # qwen avec --mmproj -> 501, sans -> 200) : llama-server refuse
                        # le save/restore de slot quand un projecteur MULTIMODAL est
                        # chargé (« This feature is not supported by multimodal »).
                        # Limitation upstream — pas un flag manquant (--slot-save-path
                        # est bien passé). Le corps du 501 est joint pour distinguer
                        # les causes si le serveur change.
                        detail = ""
                        try:
                            detail = (e.read() or b"").decode("utf-8", "replace")[:120]
                        except Exception:  # noqa: BLE001 - diagnostic best-effort
                            pass
                        cause = (
                            "501 : mmproj (multimodal) chargé — limitation llama.cpp"
                            + (f" [{detail}]" if detail else "")
                        )
                    else:
                        cause = "hang serveur"
                    _debug(
                        f"SLOT_{action.upper()}_ERR",
                        f"{path} : {cause} -> save/restore DÉSACTIVÉ pour {key}",
                    )
                    print(
                        f"[slot] save/restore KV désactivé pour {key} ({cause}) — "
                        "repli sur le ré-amorçage par re-prefill",
                        flush=True,
                    )
                    return False
                # Échec d'un chemin INTERMÉDIAIRE (ex. route swap /upstream/… en
                # mono-modèle direct -> 404 ATTENDU avant le repli /slots) :
                # silencieux — un repli nominal loggé en ERR fait croire à un bug
                # (vécu 2026-07-21). ERR seulement si le DERNIER chemin échoue.
                if i == len(paths) - 1:
                    _debug(
                        f"SLOT_{action.upper()}_ERR", f"{path} : {e}", terminal=False
                    )
        return False

    def save_slot(
        self, model: str | None, name: str, session_id: str | None = None
    ) -> bool:
        """Sauve le cache KV du slot local dans <slot-save-path>/<name> (~ms). À faire
        PENDANT que le slot contient la conversation, AVANT tout appel qui l'écrase
        (titre, reflect, sous-agent). Cf. restore_slot.

        `session_id` (reprise à chaud) : écrit un sidecar <name>.meta.json
        {model, session} à côté du fichier — try_hot_resume ne restaurera JAMAIS
        un save d'une autre session ou d'un autre modèle (KV d'autrui = inutile
        au mieux). Le slot est marqué CHAUD (état vivant + copie disque)."""
        ok = self._slot_action(model, "save", name)
        if ok and session_id and getattr(self, "hot_resume_enabled", False):
            try:
                meta = self._slots_meta_path(name)
                meta.write_text(
                    json.dumps({"model": model or "", "session": session_id}),
                    encoding="utf-8",
                )
                self._warm_slots().add(model or "")
            except OSError as e:  # meta best-effort, le save reste valide sans lui
                _debug("HOT_RESUME_META_ERR", str(e), terminal=False)
        return ok

    def restore_slot(self, model: str | None, name: str, force: bool = False) -> bool:
        """Restaure un cache KV sauvé par save_slot (~ms au lieu de re-préfiller des
        minutes) : le prochain tour de la conversation ne préfille que son delta."""
        return self._slot_action(model, "restore", name, force=force)

    # ---- Reprise à CHAUD one-shot (roadmap 16) --------------------------------
    # Un restore n'a de sens qu'en ÉVÉNEMENT ponctuel — slot froid après un
    # (re)démarrage du serveur, un swap de modèle ou un boot de loom.web — jamais
    # par tour (le rythme de croisière est protégé par le prompt-cache natif +
    # l'isolation 2e slot ; un restore par tour ajoutait un rollback par tour,
    # mesuré 2026-07-23).

    def _warm_slots(self) -> set:
        """Modèles locaux dont le slot serveur contient (à notre connaissance) la
        conversation vivante. Vide au boot de loom.web = tout est froid."""
        warm = getattr(self, "_slot_warm", None)
        if warm is None:
            warm = self._slot_warm = set()
        return warm

    def _slots_meta_path(self, name: str):
        """Chemin du sidecar meta d'un fichier de slot (injectable via
        slots_dir_override pour les tests)."""

        override = getattr(self, "slots_dir_override", None)
        if override is not None:
            base = Path(override)
        else:
            from loom.runtime.serve import slots_dir

            base = Path(slots_dir())
        return base / f"{name}.meta.json"

    def mark_all_cold(self) -> None:
        """À appeler quand le serveur modèle s'arrête ou (re)démarre : plus aucun
        slot n'est présumé chaud — le prochain amorçage tentera la reprise."""
        self._warm_slots().clear()

    def try_hot_resume(self, model: str | None, session_id: str) -> bool:
        """Restore ONE-SHOT du slot d'une conversation sur slot FROID. Best-effort,
        jamais d'exception. True = un restore a réussi (le re-prefill d'amorçage
        qui suit ne paiera que le delta).

        Gardes, dans l'ordre : feature off -> non ; modèle distant -> non ; slot
        déjà chaud -> non (rien à faire) ; meta absent/autre session/autre modèle
        -> non ; modèle HYBRIDE (cache_isolation mesuré au bench) sans binaire
        restore-safe -> non (le restore y perd les checkpoints : re-prefill total
        de toute façon, cf. PR ggml-org #26004). Un seul essai par slot froid :
        échec -> marqué chaud quand même (le repli re-prefill prend la main,
        pas de tempête de retries)."""
        if not getattr(self, "hot_resume_enabled", False):
            return False
        if not model or self.is_remote(model):
            return False
        warm = self._warm_slots()
        # Swap sortant : cibler M décharge le modèle précédent (llama-swap tue
        # son process) -> son slot devient froid.
        last = getattr(self, "_last_local_model", None)
        if last not in (None, model):
            warm.discard(last)
        self._last_local_model = model
        if model in warm:
            return False
        warm.add(model)  # un seul essai par période froide, succès ou pas
        try:
            meta = json.loads(
                self._slots_meta_path("turnend.kv").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return False  # pas de save exploitable (ou illisible) : repli re-prefill
        if meta.get("model") != model or meta.get("session") != session_id:
            return False
        if model in getattr(self, "hybrid_models", set()) and not getattr(
            self, "restore_safe", False
        ):
            _debug(
                "HOT_RESUME_SKIP",
                f"{model} : hybride + binaire non restore-safe",
                terminal=False,
            )
            return False
        ok = self.restore_slot(model, "turnend.kv", force=True)
        _debug("HOT_RESUME", {"model": model, "ok": ok}, terminal=False)
        return ok

    def warm_context(
        self,
        messages: list[dict],
        system_prompt: str,
        model: str | None = None,
        registry=None,
        thinking: bool = True,
    ) -> bool:
        """Ré-amorce le cache KV du slot LOCAL : re-prefill silencieux (1 token) du
        MÊME préfixe que le prochain tour — system prompt + messages + schémas
        d'outils, car le chat template rend le tout : un écart dans N'IMPORTE quel
        élément = zéro réutilisation. Le slot llama-server est UNIQUE : tout appel
        intermédiaire (titre, reflect, ping) écrase le cache de la conversation ->
        sans ré-amorçage le message suivant re-préfillerait TOUT (des minutes en
        local). Best-effort (avale toute erreur), False si échec."""
        try:
            oai, api_model, native = self._resolve(model)
            kwargs = build_create_kwargs(
                api_model,
                messages,
                system_prompt,
                1,
                thinking,
                tools=registry.openai_tools() if registry else None,
                native_extras=native,
            )
            stream = oai.chat.completions.create(**kwargs)
            try:
                for _ in _iter_events(stream):
                    pass
            finally:
                _close(stream)
            return True
        except Exception as e:  # noqa: BLE001 - amorçage best-effort, jamais bloquant
            _debug("WARM_CTX_ERR", str(e))
            return False

    def infer_title(self, model: str | None, message: str) -> str:
        """Titre COURT (3-5 mots) d'une conversation, inféré par le modèle. NON streamé, tout
        petit budget.

        Point clé : on COUPE le raisonnement — sinon un modèle « thinking » (glm, Qwen…)
        épuise le budget en réflexion et ne sort jamais le titre (d'où le repli sur le début
        du message). AGNOSTIQUE au provider ET au modèle : on essaie les conventions connues
        pour désactiver le thinking, chacune ignorée/rejetée SANS CASSE si le backend ne la
        connaît pas ; on garde la 1re qui produit un vrai titre. Couvre le DISTANT (Z.ai/GLM,
        OpenRouter…) comme le LOCAL (llama.cpp/Qwen). Renvoie "" si rien d'exploitable ->
        l'appelant gère le repli (début du message)."""
        oai, api_model, _ = self._resolve(model)
        prompt = (
            "Donne un titre TRÈS court (3 à 5 mots) résumant cette demande, en français, "
            "sans guillemets ni ponctuation finale. Réponds UNIQUEMENT par le titre.\n\n"
            "Demande : " + (message or "")[:500]
        )
        base = {
            "model": api_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Tu génères des titres de conversation courts et clairs.",
                },
                {"role": "user", "content": prompt},
            ],
            # Petite marge : si un backend ne sait pas couper le thinking, il a quand même la
            # place de finir un raisonnement trivial ET d'émettre le titre.
            "max_tokens": 96,
            "temperature": 0.3,
        }
        # Conventions anti-thinking connues, dans l'ordre (chacune best-effort), puis appel nu.
        attempts = (
            {"extra_body": {"thinking": {"type": "disabled"}}},  # Z.ai / GLM
            {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            },  # llama.cpp / Qwen (local)
            {"extra_body": {"reasoning": {"enabled": False}}},  # OpenRouter
            {},  # modèle sans raisonnement / provider strict
        )
        # Titre = cosmétique : ÉCHEC RAPIDE si le backend est lent/éteint (pas de retries en
        # cascade qui feraient traîner la fin du tour). On retombe alors sur le repli message.
        fast = oai.with_options(max_retries=0, timeout=20)
        for extra in attempts:
            payload = {**base, **extra}
            # 2e passe SANS temperature : certains providers la FIGENT par modèle/mode
            # (Kimi/Moonshot : 400 « only 0.6/1 is allowed ») — la nôtre est cosmétique.
            for drop_temp in (False, True):
                if drop_temp:
                    payload = {k: v for k, v in payload.items() if k != "temperature"}
                try:
                    resp = fast.chat.completions.create(**payload)
                    txt = (resp.choices[0].message.content or "").strip()
                    txt = txt.strip('"').strip("'").strip()
                    if txt:
                        return txt.splitlines()[0][:60].strip()
                    break  # réponse vide : cette variante ne donnera rien -> suivante
                except (APIConnectionError, APITimeoutError):
                    # Backend down/lent : inutile de tenter les autres variantes de param
                    # (elles échoueront pareil) -> abandon, l'appelant fait le repli message.
                    return ""
                except Exception as e:  # noqa: BLE001 - param rejeté par ce backend
                    _debug("TITLE_ERR", str(e))
                    if not drop_temp and "temperature" in str(e).lower():
                        continue  # même variante, sans imposer notre température
                    break  # variante suivante
        return ""

    def describe_image(self, data_uri: str, question: str, model: str) -> str:
        """Fait décrire une image par un modèle VISION (`model`) pour un modèle de raisonnement
        qui ne voit pas (ex. glm-5.2). Appel court NON streamé. Renvoie une description texte
        (exhaustive, structurée). Sert au routage de read_image (approche « VLM comme outil » :
        le raisonneur interroge l'image à la demande). En cas d'erreur : message clair, jamais
        d'exception qui casserait la boucle."""
        oai, api_model, _ = self._resolve(model)
        sys_p = (
            "Tu décris une image pour un AUTRE modèle qui ne la voit pas. Sois exhaustif, "
            "structuré et FIDÈLE : transcris le texte lisible tel quel, décris le layout, la "
            "hiérarchie, les couleurs, les composants et leur position. Pas d'interprétation "
            "gratuite ni de conseil — juste ce qui est réellement dans l'image."
        )
        q = (question or "").strip() or (
            "Décris cette image exhaustivement (texte, layout, couleurs, éléments)."
        )
        content = [
            {"type": "text", "text": q},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
        try:
            resp = oai.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": content},
                ],
                max_tokens=1500,
                stream=False,
            )
            return (
                resp.choices[0].message.content or ""
            ).strip() or "(le VLM n'a rien renvoyé)"
        except Exception as exc:  # noqa: BLE001 - décrire une image ne doit jamais crasher
            _debug("DESCRIBE_IMAGE_ERR", str(exc))
            return f"[description d'image indisponible via le modèle vision : {str(exc)[:160]}]"

    def stream_chat(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 2048,
        model: str | None = None,
        thinking: bool = True,
        stream_holder: dict | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Yield les events (reasoning|content), system prompt injecté en tête."""
        oai, api_model, native = self._resolve(model)
        kwargs = build_create_kwargs(
            api_model,
            messages,
            system_prompt,
            max_tokens,
            thinking,
            native_extras=native,
        )
        _debug_messages(kwargs["model"], kwargs["messages"])
        stream = oai.chat.completions.create(**kwargs)
        # Exposé à /cancel pour fermer un stream figé (cf. _stream_model_turn).
        if stream_holder is not None:
            stream_holder["stream"] = stream
        reasoning, content = "", ""
        saw_usage = False
        try:
            for kind, chunk in _iter_events(stream):
                if kind == "reasoning":
                    reasoning += chunk
                elif kind == "content":
                    content += chunk
                elif kind == "usage":
                    saw_usage = True
                yield (kind, chunk)
            if not saw_usage:
                yield (
                    "usage",
                    _estimate_usage(system_prompt, messages, content, reasoning, []),
                )
        finally:
            if stream_holder is not None:
                stream_holder["stream"] = None
            _close(stream)
            _debug("REPONSE <- modele", {"reasoning": reasoning, "content": content})

    def _preventive_compaction(
        self,
        convo: list[dict],
        system_prompt: str,
        model: str | None,
        compact_after_tokens: int | None,
        keep_recent_tools: int,
        refocus_note: bool,
        st: dict,
    ) -> Iterator[tuple[str, object]]:
        """Compaction PRÉVENTIVE avant l'appel modèle : microcompact des vieux
        résultats d'outils, puis force-fit si un résultat RÉCENT est géant (local).
        Ne stoppe jamais le tour ; met à jour st["refocus_done"]."""
        # BACKSTOP TRONCATURE (session 2026-07-14) : 2 arguments d'outil tronqués de
        # suite = la génération est étranglée par la fenêtre (prompt+completion ≈
        # contexte), pas par la verbosité — « réémets plus court » ne débloquera
        # jamais. On compacte de FORCE, même sans seuil configuré et même si
        # l'estimation en caractères (qui peut sous-compter) reste sous le seuil.
        if st.get("truncated_streak", 0) >= 2 and not self.is_remote(model):
            st["truncated_streak"] = 0
            _microcompact_tools(convo, keep_recent_tools)
            _force_fit(
                convo,
                system_prompt,
                max((compact_after_tokens or 0) * 3, len(system_prompt) + 4000),
            )
            _debug(
                "COMPACT_TRONCATURE",
                "2 tool calls tronqués de suite -> compaction forcée "
                f"(~{_ctx_estimate(system_prompt, convo)} tokens).",
            )
            yield (
                "tool_result",
                {
                    "name": "(compaction sur troncature)",
                    "ok": True,
                    "preview": (
                        "Appels d'outil coupés deux fois de suite : contexte "
                        "compacté pour redonner de la place à la génération. "
                        "Je réémets l'appel."
                    ),
                },
            )
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
        # Microcompact : si le contexte vivant approche la fenêtre, vider les vieux
        # résultats d'outils AVANT d'appeler le modèle (évite l'overflow sur une
        # chaîne longue). Estimation grossière ~4 car./token, comme loom.context.
        if not compact_after_tokens:
            return
        # ~3 car./token (et non 4) : code/TSX/JSON tokenise plus dense que de la prose.
        # Surestimer fait déclencher la compaction PLUS TÔT — biais voulu (on ne vide
        # que les vieux résultats), pour ne pas heurter la fenêtre par sous-comptage.
        approx = (
            len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)
        ) // 3
        if approx <= compact_after_tokens:
            return
        cleared = _microcompact_tools(convo, keep_recent_tools)
        if cleared:
            _debug(
                "MICROCOMPACT",
                f"{cleared} résultat(s) d'outil allégé(s) (~{approx} tokens "
                f"> seuil {compact_after_tokens}).",
            )
            # Jauge à jour TOUT DE SUITE (sinon elle reste au pic tant que
            # l'appel suivant n'a pas rendu son usage réel).
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
        # ESCALADE PRÉVENTIVE : vider les vieux résultats ne suffit pas quand UN
        # résultat RÉCENT est géant (ex. read_file d'un gros JSON minifié -> 74k
        # car.) — le microcompact le GARDE (il fait partie des récents). Résultat :
        # avant, on touchait quand même l'overflow (400 / pic à 100 %) et on ne
        # rattrapait qu'en réactif. Ici, si ça déborde ENCORE, on FORCE-FIT AVANT
        # l'appel -> plus jamais de 400. Local uniquement (le distant gère sa
        # fenêtre lui-même).
        if (
            not self.is_remote(model)
            and _ctx_estimate(system_prompt, convo) > compact_after_tokens
        ):
            # Budget PLANCHER = prompt système (incompressible) + un jeu de
            # travail minimal. Sans lui, un seuil plus petit que le prompt
            # système rend le budget INATTEIGNABLE : le force-fit détruit
            # alors tout le travail récent À CHAQUE tour (résultat du dernier
            # read compris) -> le modèle relit en boucle -> repeat_stop
            # (vécu en éval, cas context_squeeze).
            _force_fit(
                convo,
                system_prompt,
                max(compact_after_tokens * 3, len(system_prompt) + 4000),
            )
            if refocus_note and not st["refocus_done"]:
                st["refocus_done"] = True
                yield _loom_nudge(convo, "refocus", _REFOCUS_NOTE)
            _debug(
                "FORCE_FIT_PREVENTIF",
                f"un résultat récent trop gros -> clip avant l'appel "
                f"(~{_ctx_estimate(system_prompt, convo)} tokens <= seuil "
                f"{compact_after_tokens}).",
            )
            yield (
                "tool_result",
                {
                    "name": "(compaction préventive)",
                    "ok": True,
                    "preview": (
                        "Contexte réduit AVANT saturation (un résultat récent "
                        "trop gros pour la fenêtre). Je continue."
                    ),
                },
            )
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})

    def _handle_stream_api_error(
        self,
        exc: Exception,
        convo: list[dict],
        system_prompt: str,
        model: str | None,
        compact_after_tokens: int | None,
        max_overflow_retries: int,
        max_summaries: int,
        refocus_note: bool,
        st: dict,
    ) -> Iterator[tuple[str, object]]:
        """Erreur pendant l'appel modèle : échelle context_overflow (étages 1-4),
        output_overflow, puis erreurs non récupérables. Issue via st["action"] :
        "continue" (relancer le même tour) ou "done" (l'event terminal ('done', …)
        a déjà été yieldé). Met à jour overflow_retries / summary_retries /
        force_fits / refocus_done dans st."""
        kind = _classify_stream_error(exc)
        log_event("api.error", level="WARN", kind=kind, msg=str(exc)[:140])
        # DÉBORDEMENT D'ENTRÉE : la requête (prompt + historique + résultats d'outils
        # accumulés) dépasse la fenêtre de contexte. On NE crashe PAS et on ne demande
        # PAS « écris plus court » (ça vise la sortie) : on COMPACTE DUR — on vide TOUS
        # les vieux résultats d'outils (pas seulement au-delà des 4 derniers), de plus
        # en plus agressivement à chaque retry — puis on RELANCE le même tour. Le modèle
        # reprend avec SES messages (ce qu'il a déjà fait) intacts ; les gros résultats
        # d'outils deviennent le placeholder _CLEARED_TOOL (qui dit de ne pas refaire).
        if kind == "context_overflow":
            # Jauge HONNÊTE : la requête qui vient d'échouer a dépassé la fenêtre,
            # mais un 400 ne rend pas d'usage -> la jauge serait restée au dernier
            # appel réussi (ex. 66 %), donnant l'impression qu'on compacte trop tôt.
            # On pousse l'estimation du contexte VIVANT (qui a débordé) : la jauge
            # monte au pic (~100 %) AVANT la compaction, puis redescend après.
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
            # ÉTAGE 1-2 : vider les vieux résultats d'outils (sans LLM, sûr, gratuit).
            if st["overflow_retries"] < max_overflow_retries:
                st["overflow_retries"] += 1
                keep = (
                    1 if st["overflow_retries"] == 1 else 0
                )  # 2e retry : on vide tout
                cleared = _microcompact_tools(convo, keep)
                log_event(
                    "guard",
                    level="WARN",
                    kind="context_overflow",
                    retry=st["overflow_retries"],
                    cleared=cleared,
                )
                _debug(
                    "CONTEXT_OVERFLOW",
                    f"compaction dure (keep={keep}) : {cleared} résultat(s) d'outil "
                    f"vidé(s), retry {st['overflow_retries']}/{max_overflow_retries}.",
                )
                yield (
                    "tool_result",
                    {
                        "name": "(compaction)",
                        "ok": True,
                        "preview": (
                            f"Contexte saturé : {cleared} ancien(s) résultat(s) "
                            "d'outil allégé(s) pour libérer de la place. Je reprends "
                            "où j'en étais."
                        ),
                    },
                )
                yield (
                    "context_estimate",
                    {"tokens": _ctx_estimate(system_prompt, convo)},
                )
                st["action"] = "continue"
                return
            # ÉTAGE 3 : les tool results sont tous vidés et ça déborde encore -> ce
            # sont les TOURS du modèle qui saturent. On RÉSUME les vieux tours en un
            # bloc dense (anglais) et on poursuit, au lieu d'abandonner. Une seule
            # fois (max_summaries) ; borné pour ne pas boucler.
            # LOCAL UNIQUEMENT : un distant a une grande fenêtre + gère son propre
            # contexte/cache ; on ne réécrit pas son historique (cache-bust). S'il
            # déborde vraiment (rare), on tombe sur l'arrêt propre ci-dessous.
            if not self.is_remote(model) and st["summary_retries"] < max_summaries:
                st["summary_retries"] += 1
                # VISIBLE pendant le résumé (appel modèle bloquant, muet) : label
                # d'activité « compaction… » comme « le modèle tourne », sinon
                # l'UI paraît figée. Effacé dès que le tool_result/texte reprend.
                yield ("status", {"label": "compaction du contexte (résumé)…"})
                collapsed = self.summarize_old_turns(
                    convo, model, keep_recent=6, budget_chars=30000
                )
                if collapsed:
                    st["overflow_retries"] = (
                        0  # convo réduit : le microcompact peut resservir
                    )
                    log_event(
                        "guard",
                        level="WARN",
                        kind="context_summarized",
                        collapsed=collapsed,
                    )
                    _debug(
                        "CONTEXT_SUMMARY",
                        f"{collapsed} ancien(s) tour(s) résumé(s) en un bloc, "
                        "reprise à partir du résumé.",
                    )
                    yield (
                        "tool_result",
                        {
                            "name": "(résumé de session)",
                            "ok": True,
                            "preview": (
                                f"Contexte saturé : {collapsed} anciens tours "
                                "résumés en un bloc dense pour libérer de la place. "
                                "Je reprends à partir du résumé."
                            ),
                        },
                    )
                    yield (
                        "context_estimate",
                        {"tokens": _ctx_estimate(system_prompt, convo)},
                    )
                    st["action"] = "continue"
                    return
            # ÉTAGE 4 : FORCE-FIT déterministe (AUCUN LLM) — on ne s'arrête JAMAIS
            # pour saturation. Le résumé a gardé des messages récents encore trop
            # gros ? On CLIPPE le contexte vivant sous un budget qui RÉTRÉCIT à chaque
            # passe (géométrique : contre l'erreur d'estimation chars/token et le cas
            # où même clippé ça re-déborde), puis on relance. Converge toujours.
            st["force_fits"] += 1
            shrink = max(0.12, 0.7 ** st["force_fits"])
            base = compact_after_tokens or _ctx_estimate(system_prompt, convo) or 8000
            # Même plancher « système + minimal » qu'au préventif : la pression
            # géométrique s'applique au CONVO, pas à l'incompressible. Si le
            # prompt système seul dépasse la vraie fenêtre, on finit sur l'arrêt
            # context_irreducible (cas anormal), pas sur une destruction stérile.
            budget = max(len(system_prompt) + 1500, int(base * 3 * shrink))
            _force_fit(convo, system_prompt, budget)
            if refocus_note and not st["refocus_done"]:
                st["refocus_done"] = True
                yield _loom_nudge(convo, "refocus", _REFOCUS_NOTE)
            log_event(
                "guard",
                level="WARN",
                kind="context_force_fit",
                force_fit=st["force_fits"],
                est_tokens=_ctx_estimate(system_prompt, convo),
            )
            _debug(
                "FORCE_FIT",
                f"passe {st['force_fits']} : contexte clippé sous ~{budget} car. "
                f"(~{_ctx_estimate(system_prompt, convo)} tokens), reprise.",
            )
            yield (
                "tool_result",
                {
                    "name": "(compaction forcée)",
                    "ok": True,
                    "preview": (
                        f"Contexte réduit de force (passe {st['force_fits']}) pour tenir "
                        "dans la fenêtre. Je continue."
                    ),
                },
            )
            yield ("context_estimate", {"tokens": _ctx_estimate(system_prompt, convo)})
            if st["force_fits"] < 8:
                st["action"] = "continue"
                return
            # Garde-fou ANTI-RUNAWAY : après 8 réductions géométriques (budget ~12 %
            # du seuil), si ça déborde ENCORE, c'est dégénéré (prompt système ~ la
            # fenêtre entière) — là seulement, on s'arrête pour ne pas boucler.
            yield (
                "content",
                "\n[génération interrompue : contexte irréductible même après "
                "compaction forcée — cas anormal (prompt système trop grand pour la "
                "fenêtre ?). Le travail déjà écrit est conservé.]",
            )
            yield ("done", {"reason": "context_irreducible"})
            st["action"] = "done"
            return
        # OVERFLOW : tool_call vraisemblablement tronqué par max_tokens (5xx ou
        # erreur sans statut). On NE crashe PAS : on demande de découper et on
        # relance (reprise bornée par max_overflow_retries), sinon stop propre.
        if kind == "overflow":
            if st["overflow_retries"] >= max_overflow_retries:
                yield (
                    "content",
                    f"\n[génération interrompue : {str(exc)[:160]}. "
                    "Fichiers déjà écrits conservés.]",
                )
                yield ("done", {"reason": "output_overflow"})
                st["action"] = "done"
                return
            st["overflow_retries"] += 1
            log_event(
                "guard",
                level="WARN",
                kind="output_overflow",
                retry=st["overflow_retries"],
            )
            note = (
                "Ta réponse précédente était trop longue et a été tronquée par "
                "la limite de tokens. Écris des fichiers PLUS PETITS : un seul "
                "fichier par appel write_file, et découpe tout contenu volumineux "
                "en plusieurs fichiers/appels successifs. Reprends, en plus court."
            )
            yield _loom_nudge(convo, "troncature", note)
            st["action"] = "continue"
            return
        # Erreurs NON récupérables : pas un overflow -> message clair et stop net,
        # PAS de « écris plus court » trompeur ni de retry voué à re-échouer.
        reason = {
            "timeout": (
                "le serveur a mis trop de temps à répondre (timeout) — souvent "
                "un long recalcul de contexte (après compaction) ; relance, le "
                "cache rend la reprise plus rapide."
            ),
            # « injoignable » : dire QUOI lancer — loom.web tourne forcément (il
            # affiche ce message), c'est le serveur MODÈLE qui manque (ou l'API).
            "connection": (
                "API distante injoignable (réseau ou base_url à vérifier)."
                if self.is_remote(model or self.model)
                else "serveur de modèle local injoignable — lance la stack "
                "modèle (llama-swap / serve) ou choisis un modèle distant."
            ),
            "model_not_found": (
                f"modèle « {model or self.model} » introuvable ou non chargé "
                "(vérifie le modèle sélectionné)."
            ),
            "other": f"erreur du serveur de modèle : {str(exc)[:160]}",
        }[kind]
        yield ("content", f"\n[génération interrompue : {reason}]")
        yield ("done", {"reason": "api_error", "kind": kind})
        st["action"] = "done"

    def stream_chat_tools(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = 2048,
        model: str | None = None,
        registry=None,
        thinking: bool = True,
        max_iters: int = 500,
        permission=None,
        confirm=None,
        max_overflow_retries: int = 2,
        max_summaries: int = 1,
        repeat_limit: int = 3,
        compact_after_tokens: int | None = None,
        keep_recent_tools: int = 4,
        max_act_nudges: int = 2,
        max_length_continues: int = 30,
        max_loop_breaks: int = 2,
        max_empty_retries: int = 2,
        strong: bool = False,
        notes_provider=None,
        monitor_events_provider=None,
        refocus_note: bool = True,
        stream_holder: dict | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Boucle tool-use : relaie le texte, exécute les outils, relance le modèle.

        `notes_provider` (optionnel) : callable sans argument renvoyant les REMARQUES
        de l'utilisateur arrivées PENDANT le tour (« notes en vol », façon Claude
        Code). Drainées avant CHAQUE appel modèle : chacune est injectée dans la
        conversation (role user, préfixe explicite) et ré-émise en event ('note',
        texte injecté) pour que l'appelant la persiste/affiche. Une note ne stoppe
        jamais le tour — elle l'infléchit au prochain point d'arrêt.

        `monitor_events_provider` draine les lignes de monitors aux mêmes points
        d'arrêt, sous forme de résultats d'outil marqués comme données externes.

        Yield les mêmes tuples que stream_chat — ('reasoning'|'content', str) —
        plus ('tool_call', {id,name,arguments}) et ('tool_result', {id,name,ok,
        preview}).

        L'ARRÊT est piloté par le modèle (stop naturel) : dès qu'il répond SANS
        tool_call, on sort. Par-dessus, deux garde-fous non-négociables (best
        practice agentic : le modèle, surtout petit, ne sait pas toujours s'arrêter) :
        - `max_iters` : backstop ANTI-RUNAWAY tres haut (defaut 500) — PAS un cap de
          progression ; l'arret normal vient du non-progres (repeat_limit) et du coupe-circuit anti-boucle ;
        - `repeat_limit` : non-progrès — si le modèle réémet `repeat_limit` fois de
          suite EXACTEMENT le même jeu d'appels (mêmes outils + mêmes args), il
          tourne en rond, on coupe. Chaque garde-fou émet un message d'arrêt EXPLICITE
          (on sait que c'est la sécurité, pas une fin normale).

        PAS de mur de temps : sur un modèle local lent, un chrono global décapitait
        la boucle en plein travail (cf. session démineur). Les bornes sont le NOMBRE
        de tours et le NON-PROGRÈS, jamais l'horloge.

        Chaque sortie émet un event terminal ('done', {'reason': ...}) : 'natural'
        (stop du modèle), 'repeat_stop', 'loop_degenerate', 'max_iters',
        'context_irreducible', 'output_overflow', 'api_error', 'empty_response'
        (réponse vide malgré les relances). Les consommateurs
        qui ne s'en servent pas l'ignorent (dispatch if/elif) ; les évals s'en
        servent comme stop_reason mesurable au lieu de pattern-matcher les textes.
        """
        convo = list(messages)
        # Résolu une fois : le modèle est fixe pour tout l'appel. Route vers l'endpoint
        # local ou distant selon l'id, et coupe les extra_body llama.cpp si distant.
        oai, api_model, native = self._resolve(model)
        tools = registry.openai_tools() if registry else None
        # État PARTAGÉ de la boucle : compteurs/garde-fous mutés par les sous-
        # générateurs privés. "action" porte l'issue posée par chaque helper :
        # "continue" (relancer le tour), "done" (sortie — l'event terminal
        # ('done', …) a déjà été yieldé) ou "proceed" (on poursuit le tour).
        st: dict = {
            "overflow_retries": 0,
            "summary_retries": 0,  # nb de compactions PAR RÉSUMÉ déjà tentées
            "force_fits": 0,  # nb de réductions DÉTERMINISTES forcées (dernier recours, jamais d'arrêt)
            "prev_sig_set": None,  # jeu d'appels du tour précédent (détecteur de non-progrès)
            "repeat_streak": 0,
            "executed": False,  # un run_shell / dispatch_agent a-t-il réellement tourné ce tour ?
            "files_written": set(),  # chemins écrits avec succès ce tour (couche A)
            "act_nudges": 0,  # nb de relances « passe de la parole à l'acte » déjà émises
            "length_continues": 0,  # nb de relances « continue » sur troncature max_tokens
            "loop_breaks": 0,  # nb de coupes « tu répètes la même phrase, agis » déjà émises
            "fail_count": 0,  # échecs cumulés d'outils d'exécution/vérif ce tour (cascade de bugs)
            "debug_forced": False,  # méthode debug déjà imposée ce tour ? (anti-nag)
            "refocus_done": False,  # note de recentrage post-force-fit déjà émise ? (une seule)
            "empty_retries": 0,  # nb de relances sur réponse VIDE (0 texte, 0 tool call)
            "truncated_streak": 0,  # troncatures d'arguments d'outil CONSÉCUTIVES
            "verify_streak": 0,  # checks navigateur verts consécutifs (anti sur-vérification)
            "text": "",  # texte accumulé du dernier appel modèle
            "reasoning": "",  # raisonnement accumulé du dernier appel modèle
            "action": "",  # issue posée par le dernier sous-générateur
        }

        for _ in range(max_iters):
            # Compaction préventive (microcompact + force-fit) AVANT l'appel modèle.
            yield from self._preventive_compaction(
                convo,
                system_prompt,
                model,
                compact_after_tokens,
                keep_recent_tools,
                refocus_note,
                st,
            )
            # Événements asynchrones + notes en vol, au même point d'arrêt.
            yield from _inject_monitor_events(monitor_events_provider, convo)
            yield from _inject_notes(notes_provider, convo)
            kwargs = build_create_kwargs(
                api_model,
                convo,
                system_prompt,
                max_tokens,
                thinking,
                tools=tools,
                native_extras=native,
            )
            _debug_messages(kwargs["model"], kwargs["messages"])
            # Empreinte des OUTILS envoyés : rendus en tête de prompt par le
            # template, toute variation entre tours casse le cache au token ~0.
            _debug(
                "TOOLS_EMPREINTE",
                tools_fingerprint(kwargs.get("tools")),
                terminal=False,
            )
            collector: dict = {"tool_calls": [], "finish_reason": None}
            try:
                yield from _stream_model_turn(
                    oai,
                    api_model,
                    kwargs,
                    system_prompt,
                    convo,
                    collector,
                    tools,
                    thinking,
                    st,
                    stream_holder=stream_holder,
                )
            except (APIError, httpx.HTTPError) as exc:
                yield from self._handle_stream_api_error(
                    exc,
                    convo,
                    system_prompt,
                    model,
                    compact_after_tokens,
                    max_overflow_retries,
                    max_summaries,
                    refocus_note,
                    st,
                )
                if st["action"] == "done":
                    return
                continue
            text, reasoning = st["text"], st["reasoning"]

            tool_calls = collector["tool_calls"]
            # FILET : appel d'outil émis en TEXTE (channel structuré vide) ? On le récupère
            # et on l'exécute, au lieu de s'arrêter sur un appel "raté".
            if not tool_calls:
                salvaged = _salvage_tool_calls(text, reasoning)
                if salvaged:
                    tool_calls = salvaged
                    _debug(
                        "SALVAGE",
                        f"{len(salvaged)} appel(s) d'outil récupéré(s) du texte.",
                    )
            if not tool_calls:
                # Fin de tour SANS outil : boucle dégénérée / continuation length /
                # réponse vide / audit de claim, sinon stop naturel du modèle.
                yield from _dispatch_no_tool_calls(
                    collector,
                    text,
                    convo,
                    strong,
                    max_loop_breaks,
                    max_length_continues,
                    max_empty_retries,
                    max_act_nudges,
                    st,
                    notes_provider=notes_provider,
                    async_events_injector=lambda: _inject_monitor_events(
                        monitor_events_provider, convo
                    ),
                )
                if st["action"] == "done":
                    return
                continue

            yield from _check_no_progress(tool_calls, strong, repeat_limit, st)
            if st["action"] == "done":
                return

            convo.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                # JSON sain dans l'historique : un appel tronqué (JSON
                                # cassé) provoquerait un 500 'parse error' à CHAQUE tour
                                # suivant -> cascade. _safe_args remet {} si invalide.
                                "arguments": _safe_args(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            # PARALLÉLISME (distant uniquement). Règle Loom : local = inline / 1 slot llama-swap
            # -> on sérialise (un sous-agent local = même slot = zéro gain) ; distant = machine du
            # provider -> on EXPLOITE la concurrence. Cas sûr : un tour n'appelant QUE des outils
            # PARALLEL-SAFE (lectures, recherches, dispatch_agent) -> on les lance concurremment
            # (1 thread chacun, activité relayée en direct via une file, résultats recollés DANS
            # L'ORDRE). Dès qu'un outil non-safe est présent (écriture, shell, todo, vérif, image),
            # ou en local, tout reste SÉQUENTIEL (garde-fous P1.1, confirm, ordre... préservés).
            _seq_tool_calls = tool_calls
            _parallel = (
                registry is not None
                and self.is_remote(model)
                and len(tool_calls) >= 2
                and all(tc.get("name") in _PARALLEL_SAFE for tc in tool_calls)
            )
            if _parallel:
                _seq_tool_calls = []  # la boucle séquentielle ne fait rien ce tour
                yield from _run_tools_parallel(registry, tool_calls, convo, st)
            yield from _run_tools_sequential(
                _seq_tool_calls,
                registry,
                permission,
                confirm,
                convo,
                strong,
                st,
            )
            # Forçage debugging (déterministe) : le modèle n'appelle jamais use_skill seul ; à
            # la 2e erreur d'exécution on injecte la méthode systématique, une seule fois par tour.
            if st["fail_count"] >= 2 and not st["debug_forced"]:
                st["debug_forced"] = True
                yield _loom_nudge(convo, "debug", _DEBUG_FORCE)
        yield (
            "content",
            f"\n(arrêt : backstop anti-runaway atteint après {max_iters} tours d'outils — "
            "cas anormal ; relance pour reprendre là où ça s'est arrêté).",
        )
        yield ("done", {"reason": "max_iters"})
