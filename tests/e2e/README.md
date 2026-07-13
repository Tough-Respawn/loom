# Banc E2E Playwright (sans modèle réel)

Rejouable après chaque étape des refactors P2-1/P2-3/P2-4 pour vérifier la chaîne
complète UI → Flask → LoomClient → API (stub) → SSE → rendu, ce que pytest ne voit pas.

## Lancement

```
uv run python tests/e2e/stub_openai.py        # stub OpenAI-compatible, port 18081
uv run python tests/e2e/launch_loom_e2e.py    # loom.web isolé (temp système), port 18090
```

Puis piloter http://127.0.0.1:18090/ avec Playwright (ou à la main). Tuer les deux
processus à la fin — l'instance E2E ne gère AUCUN serveur modèle local.

## Scénario validé le 2026-07-13 (branche test/non-regression-p2)

1. **Chat simple** : envoyer `salut` → bulle streamée « Réponse du stub : bien reçu. »,
   compteurs tokens mis à jour (↑42 ↓21).
2. **Blocage images pendant une génération** : envoyer `explique lentement …`
   (flux ~60 s), joindre une image, taper un texte, Entrée →
   toast « images impossibles pendant une génération — retire-les pour envoyer une
   note texte, ou Stop d'abord » ; RIEN n'est consommé (texte ET image restent dans
   le composer, bouton = Stop).
3. **Note en vol** : retirer l'image, relancer un flux lent, envoyer un texte pendant
   la génération → toast « note transmise — prise en compte au prochain point
   d'arrêt » ; au tour suivant, la réponse du stub contient
   « (note en vol bien visible dans le contexte) » — preuve que la note wrappée a
   bien atteint le payload de l'appel modèle.

Les mécanismes internes (bornes /note 5000c/10 max, payloads tool_result,
chemins parallèle/séquentiel, garde-fous) sont couverts par la suite pytest (`tests/`).
