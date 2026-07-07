"""Fonctions partagées entre les harnais d'éval (run_eval.py, run_review_eval.py).

Centralise les utilitaires communs : chemins projet, accès git HEAD, construction
du client modèle et de la callable de permissions. Les deux harnais importent
depuis ici pour éviter la duplication (~150 lignes) identifiée par l'audit P2-11.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loom.agent.client import LoomClient
from loom.config import load_config
from loom.permissions import evaluate

# Racine du projet (parent du dossier evals/) et racine du paquet loom.
_ROOT = Path(__file__).resolve().parent.parent
_RT = _ROOT / "loom"


def git_show(rel: str) -> str:
    """Renvoie le contenu d'un fichier à git HEAD (chemin relatif à la racine du projet).

    Ne strip PAS la sortie : les appelants qui ont besoin d'un strip l'appliquent
    eux-mêmes (run_eval le fait sur les prompts, run_review ne le fait pas).
    """
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel}"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if r.returncode != 0:
        raise RuntimeError(f"git show HEAD:{rel} a échoué : {r.stderr.strip()}")
    return r.stdout


def git_head_sha() -> str:
    """SHA court de git HEAD (baseline persistante : épingler un report par commit).
    '' si git indisponible — l'épinglage est alors simplement sauté."""
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def load_eval_config():
    """Charge la config Loom depuis les fichiers standard.

    La config vit à la racine du repo (`config/defaults.toml` versionné +
    `config/local.toml` surcharge machine, gitignored) depuis le refactor repo-layout ;
    l'ancien chemin `loom/loom.config.toml` n'existe plus.
    """
    _cfg = _ROOT / "config"
    return load_config(_cfg / "defaults.toml", _cfg / "local.toml")


def make_client(cfg, model: str) -> tuple[LoomClient, str]:
    """Construit le LoomClient pointant sur le serveur modèle local.

    Renvoie (client, base_url) pour que l'appelant puisse afficher l'URL.
    """
    base_url = f"http://127.0.0.1:{cfg.port}/v1"
    client = LoomClient(
        base_url=base_url,
        model=model,
        timeout=cfg.chat.request_timeout,
        max_retries=cfg.chat.max_retries,
    )
    return client, base_url


def make_perm(cfg):
    """Renvoie la callable de permission (name, args) -> bool."""
    return lambda name, a: evaluate(name, a, cfg.permissions)  # noqa: E731
