# loom/runtime/models_fetch.py
"""Garantit la présence locale d'un fichier GGUF (download si absent).

Toute absence NON récupérable (repo gabarit jamais édité, repo privé/inexistant,
hors-ligne) est ramenée à un `ModelUnavailable` au message ACTIONNABLE — on dit au
user quoi poser et OÙ — plutôt qu'une stacktrace Hugging Face illisible.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


class ModelUnavailable(Exception):
    """GGUF absent en local ET non téléchargeable. Porte un message prêt à montrer."""


# Repos placeholder du gabarit (loom/models/_TEMPLATE non édité). On les reconnaît pour
# éviter un appel réseau voué au 401 et donner d'emblée la bonne consigne.
_PLACEHOLDER_REPOS = frozenset({"org/mon-modele-GGUF", "org/mon-modele-mmproj"})


def _missing_msg(
    filename: str,
    target: Path,
    repo: str,
    *,
    placeholder: bool = False,
    cause: Exception | None = None,
) -> str:
    """Message actionnable : ce qui manque, et le chemin EXACT où poser le fichier."""
    lines = [f"Modèle indisponible : « {filename} » n'est pas présent en local."]
    if placeholder:
        lines.append(
            "Ce dossier utilise encore le repo gabarit (loom/models/_TEMPLATE non "
            "personnalisé) : édite 'repo' / 'filename' dans son model.toml, ou télécharge "
            "le GGUF à la main."
        )
    elif not repo:
        lines.append(
            "Aucun 'repo' n'est renseigné dans le model.toml : impossible de "
            "télécharger. Renseigne-le, ou pose le fichier à la main."
        )
    else:
        lines.append(
            f"Le téléchargement depuis Hugging Face a échoué (repo « {repo} » : privé, "
            "introuvable, ou pas de connexion)."
        )
    lines.append(f"  -> Pose le fichier ici : {target}")
    if repo and not placeholder:
        lines.append(f"     (source attendue : https://huggingface.co/{repo})")
    if cause is not None:
        # str(cause) HF est multi-lignes (aide + URL) : on ne garde que la 1re ligne utile.
        head = str(cause).strip().splitlines()[0] if str(cause).strip() else ""
        lines.append(f"     cause : {type(cause).__name__}: {head}")
    return "\n".join(lines)


def ensure_model(repo: str, filename: str, models_dir: str | Path) -> Path:
    """Renvoie le chemin local du GGUF, en le téléchargeant depuis HF si absent.

    Lève `ModelUnavailable` (message montrable) si le fichier manque et ne peut pas être
    récupéré — JAMAIS l'exception HF brute, qui remonterait en stacktrace pour l'utilisateur.
    """
    models_dir = Path(models_dir)
    target = models_dir / filename
    if target.exists():
        return target
    # Présent mais pas encore vu plus haut : gabarit non édité -> consigne directe, zéro réseau.
    if not repo or repo in _PLACEHOLDER_REPOS:
        raise ModelUnavailable(
            _missing_msg(filename, target, repo, placeholder=repo in _PLACEHOLDER_REPOS)
        )
    try:
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(models_dir))
    # hf_hub_download ne lève jamais ModelUnavailable (exception interne à ce module) :
    # ce re-raise est théoriquement mort, mais on le garde par sécurité défensive au cas
    # où un futur appelant lèverait une exception maison dans un wrapper.
    except ModelUnavailable:
        raise
    except (
        HfHubHTTPError,
        OSError,
    ) as exc:  # 401/404/réseau/HF : tout devient actionnable
        raise ModelUnavailable(_missing_msg(filename, target, repo, cause=exc)) from exc
    return target
