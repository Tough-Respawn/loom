# loom/web/wizard.py
"""Machine à états PURE de /add-model — AUCUN I/O ici (ni réseau ni disque).

Les effets (recherche HF, download, montage) sont exécutés par routes.py : soit via
`deps` (fonctions injectées, appelées pendant une transition), soit via `action`
(effet à exécuter APRÈS la transition finale). L'état est un dict JSON-sérialisable
persisté sur la Conversation -> le parcours survit au refresh. Chaque transition
renvoie un `reply` TOUJOURS affiché dans le chat (exigence spec : étapes visibles),
préfixé d'un indicateur d'étape."""

from __future__ import annotations

from dataclasses import dataclass

CANCEL_WORDS = frozenset({"/cancel", "annuler", "cancel"})


@dataclass
class WizardResult:
    state: dict | None  # None = wizard terminé (ou annulé)
    reply: str  # message assistant, affiché ET persisté par routes.py
    action: dict | None = (
        None  # {"kind": "install", ...} | {"kind": "upsert_remote", ...}
    )
    # Boutons proposés par l'UI pour répondre à CETTE étape (purs raccourcis de
    # frappe : le clic envoie le libellé comme un message normal). None = saisie
    # libre. Non persisté : après un rechargement, on répond au clavier.
    choices: list[str] | None = None


def _valid_id(mid: str) -> bool:
    return bool(mid) and all(c.isalnum() or c in "-_." for c in mid)


def start(arg: str, deps) -> WizardResult:
    a = (arg or "").strip()
    if not a:
        return WizardResult(
            {"step": "kind"},
            "[add-model 1/2] Quel type de modèle ?\n"
            "  1. local — GGUF téléchargé depuis Hugging Face, servi sur la machine\n"
            "  2. distant — API OpenAI-compatible (URL + clé)\n"
            "  3. image — générateur ComfyUI (recette workflow.json)\n"
            "  4. vidéo — générateur ComfyUI (recette workflow.json)\n"
            "(réponds 1-4 — /cancel pour annuler)",
            choices=["local", "distant", "image", "vidéo"],
        )
    low = a.lower()
    # Raccourcis par type : « /add-model image|video » saute le menu, comme « distant ».
    if low in ("image", "video", "vidéo"):
        return _start_image("video" if low.startswith("vid") else "image", deps)
    if low == "local":
        return WizardResult(
            {"step": "l_query"},
            "[add-model 1/4] Que cherches-tu sur Hugging Face ? (ex. « qwen3 30b »)",
        )
    if low.startswith("local "):
        return _search(a.split(None, 1)[1], deps)
    # « /add-model distant [url] » ou une URL brute -> flux DISTANT direct. Une URL
    # n'est JAMAIS une recherche Hugging Face (vécu : URL + clé collées partaient en
    # recherche locale, incompréhensible).
    if low.startswith(("distant", "remote")) or low.startswith(("http://", "https://")):
        tokens = a.split()
        if tokens[0].lower() in ("distant", "remote"):
            tokens = tokens[1:]
        url = (
            tokens[0].rstrip("/")
            if tokens and tokens[0].lower().startswith(("http://", "https://"))
            else None
        )
        # SÉCURITÉ : tout ce qui suit l'URL (une clé API collée ?) est IGNORÉ — la
        # clé se donne à l'étape dédiée. Une clé tapée dans le fil reste dans
        # l'historique de session : mieux vaut la régénérer.
        leaked = len(tokens) > (1 if url else 0)
        warn = (
            "\n⚠️ Le reste de ta commande a été IGNORÉ — la clé se donne à l'étape "
            "dédiée. Si c'était une vraie clé, elle est dans l'historique de la "
            "session : pense à la régénérer chez le provider."
            if leaked
            else ""
        )
        if url:
            return WizardResult(
                {"step": "r_id", "base_url": url},
                f"[add-model — distant 1/5] base_url notée ({url}). "
                "Choisis un id court pour ce modèle (ex. « glm-5-flash ») :" + warn,
            )
        return WizardResult(
            {"step": "r_id"},
            "[add-model — distant 1/5] Choisis un id court pour ce modèle "
            "(ex. « glm-5-flash ») :" + warn,
        )
    return _search(a, deps)  # /add-model <recherche> : direct au flux local


def step(state: dict, text: str, deps) -> WizardResult:
    t = (text or "").strip()
    if t.lower() in CANCEL_WORDS:
        return WizardResult(None, "Ajout de modèle annulé.")
    fn = _STEPS.get(state.get("step"))
    if fn is None:  # état d'une vieille version : on sort proprement
        return WizardResult(
            None, "Assistant dans un état inconnu — annulé. Relance /add-model."
        )
    return fn(state, t, deps)


# ---------- tronc ----------


def _step_kind(state, t, deps):
    low = t.lower()
    if t == "1" or low.startswith("local"):
        return WizardResult(
            {"step": "l_query"},
            "[add-model 1/4] Que cherches-tu sur Hugging Face ? (ex. « qwen3 30b »)",
        )
    if t == "2" or low.startswith("dist"):
        return WizardResult(
            {"step": "r_id"},
            "[add-model — distant 1/5] Choisis un id court pour ce modèle "
            "(ex. « glm-5-flash ») :",
        )
    if t == "3" or low == "image":
        return _start_image("image", deps)
    if t == "4" or low in ("video", "vidéo"):
        return _start_image("video", deps)
    return WizardResult(
        state, "Réponds 1 (local), 2 (distant), 3 (image) ou 4 (vidéo), ou /cancel."
    )


# ---------- flux distant ----------


def _step_r_id(state, t, deps):
    if not _valid_id(t):
        return WizardResult(
            state, f"Id invalide « {t} » (lettres/chiffres/-_. uniquement). Réessaie :"
        )
    if t in deps.existing_ids:
        return WizardResult(state, f"« {t} » existe déjà. Choisis un autre id :")
    if state.get("base_url"):  # URL déjà fournie dans la commande -> étape sautée
        return WizardResult(
            {"step": "r_key", "id": t, "base_url": state["base_url"]},
            _KEY_PROMPT,
        )
    return WizardResult(
        {"step": "r_base_url", "id": t},
        "[add-model — distant 2/5] base_url de l'API "
        "(ex. https://api.z.ai/api/paas/v4) :",
    )


# La clé vient AVANT le choix du modèle : elle permet d'interroger GET /models du
# provider et de CHOISIR dans la liste au lieu de deviner un nom. Masquée dans
# l'historique par routes.py (étape r_key).
_KEY_PROMPT = (
    "[add-model — distant 3/5] Clé API (ou « aucune ») — elle sert aussi à lister "
    "les modèles du provider, et sera MASQUÉE dans l'historique :"
)


def _step_r_base_url(state, t, deps):
    if not t.startswith(("http://", "https://")):
        return WizardResult(state, "L'URL doit commencer par http(s)://. Réessaie :")
    s = dict(state, base_url=t.rstrip("/"), step="r_key")
    return WizardResult(s, _KEY_PROMPT)


def _step_r_key(state, t, deps):
    key = "" if t.lower() in ("aucune", "none", "-") else t
    choices = deps.list_remote_models(state["base_url"], key)
    s = dict(state, api_key=key, step="r_model")
    if choices:
        shown = choices[:30]
        lines = [f"  {i + 1}. {mid}" for i, mid in enumerate(shown)]
        more = (
            f"\n  … ({len(choices)} au total — tape le nom s'il n'est pas listé)"
            if len(choices) > len(shown)
            else ""
        )
        return WizardResult(
            dict(s, choices=shown),
            "[add-model — distant 4/5] Modèles disponibles chez le provider :\n"
            + "\n".join(lines)
            + more
            + "\n(réponds par un numéro, ou tape un nom)",
        )
    return WizardResult(
        s,
        "[add-model — distant 4/5] Impossible de lister les modèles du provider "
        "(URL ? clé ?) — tape le nom du modèle (ex. « glm-4.7 ») :",
    )


def _step_r_model(state, t, deps):
    choices = state.get("choices") or []
    model = choices[int(t) - 1] if t.isdigit() and 1 <= int(t) <= len(choices) else t
    s = {
        "step": "r_adv",
        "id": state["id"],
        "base_url": state["base_url"],
        "api_key": state["api_key"],
        "model": model,
    }
    return WizardResult(
        s,
        f"[add-model — distant 5/5] Modèle « {model} ». Réglages avancés ? "
        "« non » = défauts, sinon par ex. "
        "« contexte=200000 max_tokens=8192 vision=oui » :",
    )


def _step_r_adv(state, t, deps):
    ctx = mt = None
    vision = False
    if t.lower() not in ("non", "no", "n"):
        for tok in t.replace(",", " ").split():
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if k in ("contexte", "context") and v.isdigit():
                ctx = int(v)
            elif k == "max_tokens" and v.isdigit():
                mt = int(v)
            elif k == "vision":
                vision = v.lower() in ("oui", "yes", "true", "1", "on")
    record = {
        "id": state["id"],
        "base_url": state["base_url"],
        "model": state["model"],
        "api_key": state["api_key"],
        "context": ctx,
        "max_tokens": mt,
        "vision": vision,
    }
    return WizardResult(
        None,
        f"Modèle distant « {state['id']} » ajouté — disponible dans le sélecteur.",
        {"kind": "upsert_remote", "record": record},
    )


# ---------- flux suppression (/remove-model) ----------


def start_remove(deps) -> WizardResult:
    """Liste numérotée des modèles supprimables : locaux (dossier sur disque) et
    distants gérés par l'UI (les distants de config/local.toml s'éditent là-bas)."""
    items = deps.removable_models()
    if not items:
        return WizardResult(
            None,
            "Aucun modèle supprimable ici. (Un distant défini dans "
            "config/local.toml se retire en éditant ce fichier.)",
        )
    lines = [f"  {i + 1}. {it['label']}" for i, it in enumerate(items)]
    return WizardResult(
        {"step": "d_pick", "items": items},
        "[remove-model 1/2] Quel modèle supprimer ?\n"
        + "\n".join(lines)
        + "\n(réponds par un numéro — /cancel pour annuler ; un distant de "
        "config/local.toml sera retiré du fichier ; image/vidéo : définition "
        "seule, les poids ComfyUI partagés ne sont pas touchés)",
    )


def _step_d_pick(state, t, deps):
    items = state["items"]
    if not (t.isdigit() and 1 <= int(t) <= len(items)):
        return WizardResult(state, "Réponds par le numéro du modèle (ou /cancel).")
    it = items[int(t) - 1]
    warns = {
        "local": "son DOSSIER et ses fichiers (GGUF compris) seront SUPPRIMÉS du disque",
        "remote": "il sera retiré du store (sa clé avec) et démonté",
        "remote_config": "il sera retiré de config/local.toml (sa clé avec) et démonté",
        "image": "sa définition Loom (model.toml + workflow.json) sera supprimée ; "
        "les poids ComfyUI partagés ne sont PAS touchés",
        "video": "sa définition Loom (model.toml + workflow.json) sera supprimée ; "
        "les poids ComfyUI partagés ne sont PAS touchés",
    }
    warn = warns[it["kind"]]
    if it.get("is_default"):
        warn += (
            " ⚠️ c'est le modèle par défaut de local.toml : au prochain boot, "
            "repli sur le premier modèle installé"
        )
    return WizardResult(
        {"step": "d_confirm", "item": it},
        f"[remove-model 2/2] Supprimer « {it['id']} » ? {warn}.\n"
        "Tape « oui » pour confirmer — toute autre réponse annule.",
        choices=["oui", "annuler"],
    )


def _step_d_confirm(state, t, deps):
    if t.lower() not in ("oui", "o", "yes"):
        return WizardResult(None, "Suppression annulée — rien n'a été touché.")
    it = state["item"]
    return WizardResult(
        None,
        f"Suppression de « {it['id']} »…",
        {"kind": "remove", "id": it["id"], "model_kind": it["kind"]},
    )


# ---------- flux /rebench (recalibration d'un LOCAL TEXTE, tel que configuré) ----------
# La mesure elle-même (topologie + pente + vitesse) vit dans routes.py / loom.setup ;
# ici seulement le dialogue. L'état b_apply est POSÉ PAR ROUTES à la fin du job.


def start_rebench(arg: str, deps) -> WizardResult:
    items = deps.rebenchable_models()
    a = (arg or "").strip()
    if a:
        it = next((i for i in items if i["id"] == a), None)
        if it is None:
            kind = deps.model_kind(a)
            if kind in ("remote", "image", "video"):
                return WizardResult(
                    None,
                    f"« {a} » n'est pas calibrable : seuls les modèles LOCAUX "
                    "texte-à-texte se mesurent (image/vidéo = ComfyUI, distant = "
                    "la fenêtre du provider).",
                )
            return WizardResult(
                None,
                f"Modèle « {a} » inconnu. /rebench sans argument liste les "
                "modèles calibrables.",
            )
        return _rebench_confirm(it)
    if not items:
        return WizardResult(None, "Aucun modèle local texte à calibrer.")
    lines = [f"  {i + 1}. {it['label']}" for i, it in enumerate(items)]
    return WizardResult(
        {"step": "b_pick", "items": items},
        "[rebench 1/2] Quel modèle recalibrer ?\n"
        + "\n".join(lines)
        + "\n(réponds par un numéro — /cancel pour annuler)",
    )


def _rebench_confirm(it) -> WizardResult:
    return WizardResult(
        {"step": "b_confirm", "id": it["id"]},
        f"[rebench 2/2] Recalibrer « {it['id']} » tel qu'il est configuré ? "
        "Durée ~5-20 min : le serveur modèle local sera éteint et les modèles "
        "locaux indisponibles pendant la mesure (les distants restent "
        "utilisables). La progression s'affiche ici.\n"
        "Tape « oui » pour lancer — toute autre réponse annule.",
        choices=["oui", "annuler"],
    )


def _step_b_pick(state, t, deps):
    items = state["items"]
    if not (t.isdigit() and 1 <= int(t) <= len(items)):
        return WizardResult(state, "Réponds par le numéro du modèle (ou /cancel).")
    return _rebench_confirm(items[int(t) - 1])


def _step_b_confirm(state, t, deps):
    if t.lower() not in ("oui", "o", "yes"):
        return WizardResult(None, "Recalibration annulée.")
    return WizardResult(
        None,
        f"Recalibration de « {state['id']} » lancée…",
        {"kind": "rebench", "id": state["id"]},
    )


def _step_b_apply(state, t, deps):
    if t.lower() not in ("oui", "o", "yes"):
        return WizardResult(None, "Config inchangée — rien n'a été touché.")
    return WizardResult(
        None,
        f"Application du contexte {state['context']} à « {state['id']} »…",
        {
            "kind": "rebench_apply",
            "id": state["id"],
            "context": state["context"],
            "mecanisme": state["mecanisme"],
        },
    )


# ---------- flux image/vidéo (ComfyUI) ----------
# Un modèle image/vidéo = dossier local/{image,video}/<id>/ avec model.toml (généré
# ici) + workflow.json (export ComfyUI « format API » fourni par l'utilisateur — le
# wizard ne peut PAS l'inventer). Les poids ComfyUI ne sont jamais gérés par Loom.

_IMG_DEFAULT_DIMS = {"image": (1024, 1024), "video": (832, 480)}


def _start_image(ikind: str, deps) -> WizardResult:
    lab = "image" if ikind == "image" else "vidéo"
    return WizardResult(
        {"step": "i_id", "ikind": ikind},
        f"[add-model — {lab} 1/4] Id du modèle (nom du dossier + sélecteur UI, "
        "ex. « z-image-turbo ») :",
    )


def _step_i_id(state, t, deps):
    ikind = state["ikind"]
    if not _valid_id(t):
        return WizardResult(
            state, f"Id invalide « {t} » (lettres/chiffres/-_.). Réessaie :"
        )
    if t in deps.existing_ids:
        return WizardResult(state, f"« {t} » existe déjà. Choisis un autre id :")
    found = deps.image_dir_state(ikind, t)
    if found == "complete":  # dossier « plus tard » complété -> montage direct
        return WizardResult(
            None,
            f"Le dossier de « {t} » existe déjà avec sa recette — montage direct.",
            {"kind": "mount_image", "id": t, "model_kind": ikind},
        )
    if found == "partial":  # dossier scaffoldé sans recette -> il ne manque qu'elle
        return WizardResult(
            {"step": "i_workflow", "ikind": ikind, "id": t, "resume": True},
            f"Le dossier de « {t} » existe mais il manque workflow.json. "
            "Colle le chemin de ton export ComfyUI (format API), ou « plus tard » :",
        )
    w, h = _IMG_DEFAULT_DIMS[ikind]
    return WizardResult(
        {"step": "i_dims", "ikind": ikind, "id": t},
        f"[add-model — 2/4] Dimensions de génération — « ok » pour {w}x{h}, "
        "ou tape LxH (ex. 1280x720) :",
    )


def _step_i_dims(state, t, deps):
    w, h = _IMG_DEFAULT_DIMS[state["ikind"]]
    if t.lower() not in ("ok", "oui"):
        parts = t.lower().replace("×", "x").split("x")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            return WizardResult(
                state, "Format attendu : LxH (ex. 1024x1024), ou « ok ». Réessaie :"
            )
        w, h = int(parts[0]), int(parts[1])
    return WizardResult(
        dict(state, step="i_desc", width=w, height=h),
        "[add-model — 3/4] Description en une ligne (infobulle du sélecteur) — "
        "ou « non » :",
    )


def _step_i_desc(state, t, deps):
    desc = "" if t.lower() in ("non", "no", "aucune", "-") else t
    return WizardResult(
        dict(state, step="i_workflow", description=desc),
        "[add-model — 4/4] La recette ComfyUI : colle le chemin de ton export "
        "« format API » (ex. C:\\Users\\toi\\Downloads\\workflow_api.json), "
        "ou « plus tard » pour préparer le dossier :",
    )


def _step_i_workflow(state, t, deps):
    ikind = state["ikind"]
    later = t.lower() in ("plus tard", "later", "non")
    path = None if later else t.strip().strip('"').strip("'")
    warn = ""
    if path:
        chk = deps.check_workflow(path)
        if not chk["ok"]:
            return WizardResult(
                state,
                f"Recette illisible : {chk['error']}. "
                "Colle un autre chemin, ou « plus tard » :",
            )
        if chk["warnings"]:
            warn = "\n⚠️ " + " ; ".join(chk["warnings"])
    action = {
        "kind": "install_image",
        "model_id": state["id"],
        "model_kind": ikind,
        "width": state.get("width", _IMG_DEFAULT_DIMS[ikind][0]),
        "height": state.get("height", _IMG_DEFAULT_DIMS[ikind][1]),
        "description": state.get("description", ""),
        "workflow_path": path,
    }
    return WizardResult(None, f"Création de « {state['id']} »…" + warn, action)


# ---------- flux local ----------


def _search(query, deps):
    hits = deps.search_models(query)
    if not hits:
        return WizardResult(
            {"step": "l_query"},
            f"[add-model 1/4] Aucun repo GGUF trouvé pour « {query} ». "
            "Autre recherche ? (ou /cancel)",
        )
    lines = [
        f"  {i + 1}. {h['repo_id']}  ({h['downloads']} téléchargements, {h['likes']} likes)"
        for i, h in enumerate(hits)
    ]
    return WizardResult(
        {"step": "l_repo", "hits": hits},
        "[add-model 2/4] Modèles trouvés :\n"
        + "\n".join(lines)
        + "\n(réponds par un numéro, ou tape une autre recherche)",
    )


def _step_l_query(state, t, deps):
    return _search(t, deps)


def _step_l_repo(state, t, deps):
    hits = state["hits"]
    if not (t.isdigit() and 1 <= int(t) <= len(hits)):
        return _search(t, deps)  # texte libre = nouvelle recherche
    repo = hits[int(t) - 1]["repo_id"]
    files = deps.list_gguf_files(repo)
    # is_aux exclut mmproj ET mtp (repli sur is_mmproj pour un state d'avant la clé)
    weights = [f for f in files if not f.get("is_aux", f["is_mmproj"])]
    if not weights:
        return WizardResult(
            {"step": "l_query"},
            f"[add-model] « {repo} » ne contient aucun GGUF de poids. Autre recherche ?",
        )
    annotated = deps.recommend(weights)
    lines = []
    for i, f in enumerate(annotated):
        tag = (
            "  <- recommandé"
            if f["recommended"]
            else ("  (ne tiendra pas)" if not f["fits"] else "")
        )
        lines.append(f"  {i + 1}. {f['filename']}  {f['size_mb'] / 1024:.1f} Go{tag}")
    mmprojs = sorted(f["filename"] for f in files if f["is_mmproj"])
    s = {
        "step": "l_quant",
        "repo": repo,
        "files": annotated,
        "mmproj": mmprojs[0] if mmprojs else None,
    }
    extra = f"\n(vision : « {s['mmproj']} » sera installé avec)" if s["mmproj"] else ""
    return WizardResult(
        s,
        f"[add-model 3/4] Quants disponibles dans {repo} :\n"
        + "\n".join(lines)
        + extra
        + "\n(réponds par un numéro — le choix reste LIBRE, même « ne tiendra pas »)",
    )


def _step_l_quant(state, t, deps):
    files = state["files"]
    if not (t.isdigit() and 1 <= int(t) <= len(files)):
        return WizardResult(
            state, "Réponds par le numéro du quant choisi (ou /cancel)."
        )
    chosen = files[int(t) - 1]
    proposed = deps.derive_id(state["repo"])
    s = {
        "step": "l_id",
        "repo": state["repo"],
        "chosen": chosen,
        "mmproj": state.get("mmproj"),
        "proposed_id": proposed,
    }
    return WizardResult(
        s,
        f"[add-model 4/4] Id du modèle (nom du dossier + sélecteur UI) — "
        f"« ok » pour « {proposed} », ou tape un autre id :",
    )


def _step_l_id(state, t, deps):
    mid = state["proposed_id"] if t.lower() in ("ok", "oui") else t
    if not _valid_id(mid):
        return WizardResult(
            state, f"Id invalide « {mid} » (lettres/chiffres/-_.). Réessaie :"
        )
    if mid in deps.existing_ids:
        return WizardResult(state, f"« {mid} » existe déjà. Choisis un autre id :")
    chosen = state["chosen"]
    action = {
        "kind": "install",
        "model_id": mid,
        "repo": state["repo"],
        "filename": chosen["filename"],
        "files": list(chosen["part_files"]),
        "size_mb": chosen["size_mb"],
        "mmproj_filename": state.get("mmproj"),
    }
    return WizardResult(
        None,
        f"Installation de « {mid} » : téléchargement de "
        f"{chosen['size_mb'] / 1024:.1f} Go lancé — la progression s'affiche ici. "
        "Si ça coupe, la reprise est automatique au premier lancement du modèle.",
        action,
    )


_STEPS = {
    "kind": _step_kind,
    "b_pick": _step_b_pick,
    "b_confirm": _step_b_confirm,
    "b_apply": _step_b_apply,
    "i_id": _step_i_id,
    "i_dims": _step_i_dims,
    "i_desc": _step_i_desc,
    "i_workflow": _step_i_workflow,
    "l_query": _step_l_query,
    "l_repo": _step_l_repo,
    "l_quant": _step_l_quant,
    "l_id": _step_l_id,
    "r_id": _step_r_id,
    "r_base_url": _step_r_base_url,
    "r_model": _step_r_model,
    "r_key": _step_r_key,
    "r_adv": _step_r_adv,
    "d_pick": _step_d_pick,
    "d_confirm": _step_d_confirm,
}
