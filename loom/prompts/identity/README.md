# Identité de Loom (always-on)

Trois fichiers définissent l'identité injectée à CHAQUE tour, en tête du system prompt :

- `SOUL.md`   — qui est Loom : rôle, personnalité, style. Fait foi.
- `USER.md`   — qui tu es, toi : profil, projets, préférences.
- `MEMORY.md` — mémoire durable GLOBALE (vraies constantes : environnement, conventions
  transverses). Le projet-spécifique ne va PAS ici : il va dans la mémoire épisodique
  (`recall`/`remember`), pas dans l'always-on.

Ces fichiers sont **personnels** : ils vivent dans `var/identity/` (gitignored), pas dans
le dépôt. Pour amorcer une machine, copie les gabarits puis édite-les :

```sh
cp loom/prompts/identity/SOUL.example.md   var/identity/SOUL.md
cp loom/prompts/identity/USER.example.md   var/identity/USER.md
cp loom/prompts/identity/MEMORY.example.md var/identity/MEMORY.md
```

Budget : l'ensemble est borné par `chat.identity_max_tokens` (défaut 600). Reste dense.
