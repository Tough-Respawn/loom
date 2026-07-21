# Proxy d'entreprise (inspection TLS) — confiance via magasin OS + diagnostic

**Date** : 2026-07-21 · **Statut** : validé (design approuvé en session)

## Problème

Sur une machine d'entreprise, le proxy fait de l'inspection TLS : il présente son
propre certificat, signé par une racine déployée par l'IT dans le magasin de
certificats de l'OS — mais absente du bundle certifi qu'utilisent httpx /
huggingface_hub. Résultat : tout HTTPS sortant échoue en
`SSL: CERTIFICATE_VERIFY_FAILED` alors que « la connexion marche ». Vécu dans
`loom-setup` : recherche Hugging Face impossible, et pire, un message trompeur
(« famille disparue ? ») parce que l'erreur réseau était avalée en `None`.

Objectif : **générique** (aucune boîte codée en dur) — dans ce type
d'environnement, Loom ne doit pas échouer brutalement mais marcher tout seul,
et si ça casse quand même, expliquer et proposer une solution.

## Décision (option retenue : transparent + diagnostic)

1. **Confiance TLS via le magasin de l'OS** — dépendance `truststore>=0.10`
   (pur Python), injectée au chargement du package dans `loom/__init__.py`
   (`truststore.inject_into_ssl()`, flag `TRUSTSTORE_ACTIVE`). Même mécanisme
   que pip : transparent sur machine perso, récupère la racine du proxy en
   entreprise. Un seul point d'injection couvre `loom-setup`,
   `python -m loom.web` et tout futur entrypoint — les contextes SSL (httpx,
   huggingface_hub, urllib) se créent à l'usage, après l'import.
2. **Diagnostic actionnable** — `loom/utils.py: explain_network_error(exc)` :
   descend la chaîne `__cause__`/`__context__`, reconnaît un échec de
   *vérification* de certificat (type `ssl.SSLCertVerificationError` ou
   marqueurs texte, pour couvrir curl_cffi) et renvoie un message : proxy
   d'entreprise probable, magasin OS déjà utilisé, pistes (cert racine à faire
   installer par l'IT, `SSL_CERT_FILE` vers le bundle PEM). `None` pour toute
   erreur réseau banale (timeout, DNS) — le message existant reste seul.
3. **Branchement aux formateurs d'erreur existants** :
   - `runtime/hf_catalog.py` `_err` (recherche + inventaire HF),
   - `runtime/models_fetch.py` `_missing_msg` (téléchargement GGUF),
   - `setup/llama_release.py` `fetch_latest_release` (binaire llama.cpp),
   - `tools/web.py` `fetch_page` : en mode `raise_status` (fetch_url), un échec
     de cert devient une `ToolError` explicite au lieu d'un snippet vide.
4. **Fin du `None` trompeur** — `setup/catalog.py` : `resolve_entry` et
   `probe_repo` propagent `HfCatalogError` (erreur réseau ≠ « rien de
   jouable ») ; `setup/cli.py` l'attrape et montre le message, diagnostic
   inclus.

## Limite connue

Le chemin `curl_cffi` de `fetch_url` (impersonation anti-bot) a sa propre pile
TLS : truststore ne le couvre pas. Le diagnostic texte, lui, le reconnaît.

## Tests / preuve

- `tests/test_proxy_truststore.py` : injection effective à l'import, détection
  par cause chaînée / par texte, erreurs banales ignorées, chaîne cyclique,
  diagnostic présent dans `HfCatalogError`.
- `tests/test_setup_catalog.py` : nouveau contrat `resolve_entry` (propage).
- Preuve réelle derrière le proxy d'entreprise (2026-07-21) : recherche HF,
  inventaire de quants et release GitHub llama.cpp passent, là où la même
  machine échouait en `CERTIFICATE_VERIFY_FAILED` avant le fix.
