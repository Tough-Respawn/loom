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
            "(réponds 1 ou 2 — /cancel pour annuler)",
        )
    low = a.lower()
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
    if t == "1" or t.lower().startswith("local"):
        return WizardResult(
            {"step": "l_query"},
            "[add-model 1/4] Que cherches-tu sur Hugging Face ? (ex. « qwen3 30b »)",
        )
    if t == "2" or t.lower().startswith("dist"):
        return WizardResult(
            {"step": "r_id"},
            "[add-model — distant 1/5] Choisis un id court pour ce modèle "
            "(ex. « glm-5-flash ») :",
        )
    return WizardResult(state, "Réponds 1 (local) ou 2 (distant), ou /cancel.")


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
        + "\n(réponds par un numéro — /cancel pour annuler)",
    )


def _step_d_pick(state, t, deps):
    items = state["items"]
    if not (t.isdigit() and 1 <= int(t) <= len(items)):
        return WizardResult(state, "Réponds par le numéro du modèle (ou /cancel).")
    it = items[int(t) - 1]
    warn = (
        "son DOSSIER et ses fichiers (GGUF compris) seront SUPPRIMÉS du disque"
        if it["kind"] == "local"
        else "il sera retiré du store (sa clé avec) et démonté"
    )
    return WizardResult(
        {"step": "d_confirm", "item": it},
        f"[remove-model 2/2] Supprimer « {it['id']} » ? {warn}.\n"
        "Tape « oui » pour confirmer — toute autre réponse annule.",
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
