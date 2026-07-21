# Proxy d'entreprise (inspection TLS) : magasin de certificats de l'OS injecté au
# chargement du package (comme pip), et diagnostic actionnable quand une erreur de
# vérification de certificat survient quand même — SANS réseau.
import ssl

import pytest

from loom.utils import explain_network_error


def test_import_loom_injecte_truststore():
    # L'import du package suffit : tout contexte SSL créé ensuite (httpx,
    # huggingface_hub, urllib) vérifie contre le magasin de l'OS.
    import truststore

    import loom

    assert loom.TRUSTSTORE_ACTIVE is True
    assert ssl.SSLContext is truststore.SSLContext


def test_import_loom_desactive_xet():
    # hf-xet (TLS Rust, endpoints CAS) se fige derrière un proxy d'entreprise
    # (gel reproductible, même débit que le CDN classique sinon) -> désactivé
    # par défaut, l'env utilisateur garde la priorité (setdefault).
    import os

    import loom  # noqa: F401

    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"


def test_explain_reconnait_une_cause_ssl_dans_la_chaine():
    # Cas réel : httpx.ConnectError enveloppe ssl.SSLCertVerificationError.
    cause = ssl.SSLCertVerificationError(
        1, "certificate verify failed: unable to get local issuer certificate"
    )
    exc = ConnectionError("All connection attempts failed")
    exc.__cause__ = cause
    msg = explain_network_error(exc)
    assert msg is not None
    assert "proxy" in msg.lower()
    assert "SSL_CERT_FILE" in msg


def test_explain_reconnait_le_marqueur_dans_le_texte():
    # Backend sans exception ssl typée (ex. curl_cffi) : détection par le texte.
    exc = RuntimeError(
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1032)"
    )
    assert explain_network_error(exc) is not None


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("timed out"),
        ConnectionError("connexion refusée"),
        OSError("réseau injoignable"),
    ],
)
def test_explain_ignore_les_erreurs_reseau_banales(exc):
    assert explain_network_error(exc) is None


def test_explain_ne_boucle_pas_sur_une_chaine_cyclique():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert explain_network_error(a) is None


def test_le_diagnostic_arrive_dans_les_erreurs_hf():
    # Le formateur central de hf_catalog appende le diagnostic : le message
    # montré dans le chat / loom-setup devient actionnable.
    from loom.runtime.hf_catalog import search_models

    class _Api:
        def list_models(self, **kw):
            raise ssl.SSLCertVerificationError(
                1, "certificate verify failed: unable to get local issuer certificate"
            )

    with pytest.raises(Exception) as ei:
        search_models("qwen", api=_Api())
    assert "proxy" in str(ei.value).lower()
