"""Utilitaires partagés : timestamps ISO, estimation de tokens, diagnostic réseau.

Centralise les petites fonctions utilitaires dupliquées auparavant dans plusieurs
modules (agent/session.py, agent/reflect.py, extend/plugins.py, agent/context.py,
memory/identity.py).
"""

from __future__ import annotations

import ssl
from datetime import datetime, timezone

#: Heuristique partagée : ~4 caractères par token (prose moyenne).
#: Utilisée par agent/context.py et memory/identity.py pour borner les budgets.
CHARS_PER_TOKEN = 4


def now_iso() -> str:
    """Horodatage ISO 8601 UTC tronqué à la seconde (ex. ``2025-01-02T15:04:05+00:00``).

    Variante SECONDS : pour les métadonnées persistées (timestamps de session,
    d'installation de plugin...) où la sous-seconde est du bruit.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now() -> str:
    """Horodatage ISO 8601 UTC complet (avec microsecondes).

    Variante par défaut (sans timespec) : pour les marqueurs internes qui
    veulent la résolution maximale (ex. ``created_at``/``updated_at`` des skills
    appris, où deux écritures dans la même seconde doivent rester ordonnables).
    """
    return datetime.now(timezone.utc).isoformat()


# Marqueurs d'un échec de VÉRIFICATION de certificat (pas d'un réseau coupé) :
# l'exception ssl typée peut être enfouie (httpx.ConnectError) ou absente
# (curl_cffi a sa propre pile TLS) -> détection par type ET par texte.
_CERT_MARKERS = (
    "CERTIFICATE_VERIFY_FAILED",
    "certificate verify failed",
    "unable to get local issuer certificate",
    "self signed certificate in certificate chain",
    "self-signed certificate in certificate chain",
)


def explain_network_error(exc: BaseException) -> str | None:
    """Diagnostic actionnable si `exc` cache un échec de vérification TLS.

    Cas visé : proxy d'entreprise à inspection TLS (générique, pas propre à une
    boîte) — le certificat qu'il présente n'est pas reconnu, tout HTTPS sortant
    échoue en CERTIFICATE_VERIFY_FAILED alors que « la connexion marche ».
    Descend la chaîne ``__cause__``/``__context__``. Renvoie None pour toute
    erreur réseau banale (timeout, DNS, connexion refusée) : le message existant
    de l'appelant reste alors seul.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, ssl.SSLCertVerificationError) or any(
            m in str(node) for m in _CERT_MARKERS
        ):
            break
        node = node.__cause__ or node.__context__
    else:
        return None
    import loom

    if not getattr(loom, "TRUSTSTORE_ACTIVE", False):
        return (
            "Vérification TLS échouée : probablement un proxy d'entreprise à "
            "inspection TLS, et le module truststore est absent — réinstalle les "
            "dépendances (`uv sync`) pour que Loom utilise le magasin de "
            "certificats du système."
        )
    return (
        "Vérification TLS échouée : probablement un proxy d'entreprise à "
        "inspection TLS. Loom vérifie déjà via le magasin de certificats du "
        "système ; si l'erreur persiste, installe le certificat racine du proxy "
        "dans le magasin de l'OS (demande à l'IT), ou pointe la variable "
        "d'environnement SSL_CERT_FILE vers le bundle PEM de l'entreprise."
    )


def estimate_tokens(text: str) -> int:
    """Estimation grossière du nombre de tokens : ~1 token pour 4 caractères.

    Renvoie au moins 1 pour un texte non vide. Heuristique partagée avec
    ``memory/identity.py`` (bornage du bloc identité) - garde un seul seuil.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)
