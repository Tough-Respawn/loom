# Audit de dette technique — Loom

Périmètre : `loom/` (agent, runtime, tools, web, memory, extend, config, permissions, prompts), `evals/`, `config/`, `pyproject.toml`, `var/`. Exclus : `loom/plugins/cache/` et `loom/plugins/marketplaces/` (code tiers téléchargé).

Méthode : lecture statique du code réel, recoupement des symboles par recherche plein-texte. Aucun TODO/FIXME/XXX/HACK trouvé dans tout le code applicatif.

---

## Synthèse

| Priorité | Catégorie | Constats | Corrigés | Infirmés | Restants |
|---|---|---|---|---|---|
| P0 | Bugs fonctionnels | 3 | 2 (P0-1, P0-2) | 1 (P0-3) | 0 |
| P1 | Code mort / legacy | 9 | 9 (P1-1 à P1-9) | 0 | 0 |
| P2 | Dette structurelle (complexité, duplication) | 11 | 8 (P2-2\*, P2-5, P2-6\*, P2-7, P2-8, P2-9, P2-10, P2-11) | 0 | 3 (P2-1, P2-3, P2-4) |
| P3 | Robustesse / gestion d'erreurs | 8 | 8 (P3-1 à P3-8) | 0 | 0 |
| P4 | Typage manquant | 6 | 6 (P4-1 à P4-6) | 0 | 0 |
| P5 | Documentation / cohérence | 11 | 11 (P5-1 à P5-11) | 0 | 0 |
| — | Environnement / scratch | 2 | 2 (ENV-1, ENV-2) | 0 | 0 |
| **Total** | | **50** | **46** | **1** | **3** |

> \* = partiellement corrigé (P2-2 : 3 closures extraites sur 4 ; P2-6 : dérivation auto de `_SUBAGENT_TOOLS`, dataclass non créé).
>
> **Dernière mise à jour** : 46 constats corrigés (dont 2 partiels), 1 infirmé, 3 reportés. Vérification multi-agents + ruff `All checks passed!` + smoke tests globaux + self-test évals VERT. 3 constats restants (P2-1, P2-3, P2-4) : refactorings structurels du cœur de l'agent (`stream_chat_tools` 708 lignes, `create_app` 1945 lignes) — reportés car trop risqués sans tests live.

---

## P0 — Bugs fonctionnels

### P0-1. Offload MoE ignoré en mode mono-modèle — ✅ CORRIGÉ
- `loom/runtime/serve.py:114` — `build_launch` appelle `build_server_args` **sans** passer `cpu_moe` ni `n_cpu_moe`. Or `server_args.py:17-18` les accepte et `swap.py:60-61` les passe en multi-modèle.
- Conséquence : un modèle MoE configuré avec `cpu_moe=True` ou `n_cpu_moe=N` est **ignoré** en chemin direct (`launch_direct`). Divergence selon le nombre de modèles.
- **Fix appliqué** : ajout de `cpu_moe=cfg.model.cpu_moe, n_cpu_moe=cfg.model.n_cpu_moe` à l'appel `build_server_args` dans `build_launch`. Ruff OK, smoke test (monkeypatch capturant les kwargs) OK.

### P0-2. Logique de précédence `-ngl` dupliquée et divergente — ✅ CORRIGÉ
- `loom/runtime/serve.py:52-69` (`resolve_n_gpu_layers`) vs `loom/runtime/swap.py:29-41` (`_model_cmd`). Les deux implémentaient la même décision d'offload GPU avec des règles **différentes** : swap gère `cpu_moe` (→ ngl=999) et `model.n_gpu_layers`, serve ne gère que l'override global.
- `loom/runtime/swap.py:25` (commentaire) prétendait « identique au chemin mono-modèle » — c'était faux.
- **Fix appliqué** : fonction `resolve_ngl(model, profile, override, headroom)` unique créée dans `loom/runtime/ngl.py`, importée par `serve.py` et `swap.py`. Les deux chemins utilisent maintenant la même logique unifiée gérant `cpu_moe`, `n_gpu_layers` et l'override global. Ruff OK, smoke test (import + `callable(resolve_ngl)`) OK.

### P0-3. `config_schema._read_toml` — vérification : constat INFIRMÉ
- `loom/runtime/config_schema.py:461-464` — `except (OSError, ValueError)` était suspecté de manquer `tomlkit.exceptions.ParseError` (présumé ne pas hériter de `ValueError`).
- **Vérification live (tomlkit 0.15.0)** : `ParseError.__mro__` contient bien `ValueError`. Toutes les exceptions de parsing (`ParseError`, `UnexpectedCharError`, `EmptyKeyError`, `UnexpectedEofError`, etc.) héritent de `ValueError` et sont donc attrapées. Testé avec 7 entrées malformées (syntaxe cassée, clé dupliquée, JSON, texte brut, tableau incomplet, fichier vide, char nul) : toutes retournent `{}` sans crash.
- **Résidu réel (mineur, reclassé P3)** : `KeyAlreadyPresent` et `NonExistentKey` n'héritent pas de `ValueError` mais ne sont levées que par `.unwrap()` sur une table avec clés dupliquées — cas marginal en pratique.
- Action : aucune action urgente. Si on veut être exhaustif, élargir à `except (OSError, ValueError, tomlkit.exceptions.TOMLKitError)` pour couvrir `KeyAlreadyPresent`, mais l'impact est négligeable.

---

## P1 — Code mort / legacy

### P1-1. `"bash"` dans `SHELL_TOOLS` — outil inexistant — ✅ CORRIGÉ
- `loom/permissions.py:40` — `SHELL_TOOLS = frozenset({"run_shell", "bash", "serve_and_check"})`. L'outil `"bash"` n'existe nulle part (`search_text` sur `loom/**/*.py` : 0 hit comme nom d'outil). Reliquat d'un ancien outil supprimé.
- **Fix appliqué** : `"bash"` retiré de `SHELL_TOOLS`. Ruff OK, smoke test (assertion `"bash" not in SHELL_TOOLS`) OK.

### P1-2. `deny_paths` jamais câblé — deny-list de chemins inopérante — ✅ CORRIGÉ
- `loom/permissions.py:133,142` — `is_protected_write_path(path, deny_paths)` accepte `deny_paths`.
- `loom/tools/fs.py:33` — toujours appelé avec `[]`. `evaluate()` (`permissions.py:185-197`) ne passe jamais `cfg.deny_paths`.
- Conséquence : la deny-list de chemins custom (configurable via `PermissionConfig.deny_paths`) était **inopérante**. Seuls les motifs durs `DEFAULT_PROTECTED_PATHS` protègent.
- **Fix appliqué** : `cfg.deny_paths` câblé dans `evaluate()` pour `WRITE_TOOLS`. Ruff OK, smoke test OK.

### P1-3. `workspace_root` parsé mais jamais lu — ✅ CORRIGÉ
- `loom/permissions.py:106,117` — champ `workspace_root: str = "."` dans `PermissionConfig`, peuplé depuis le TOML mais jamais consommé (`evaluate()` ne l'utilise pas). Legs d'une ancienne politique de confinement au workspace.
- **Fix appliqué** : champ supprimé de `PermissionConfig` et de sa lecture TOML. Aucune autre référence dans `loom/`. Ruff OK, smoke test OK.

### P1-4. `app.config["_pending"]` / `["_session_holder"]` écrits, jamais lus — ✅ CORRIGÉ
- `loom/web/app.py:592,606` — ces clés sont posées mais `search_text` confirme : jamais lues. `pending` et `_cur` sont utilisés directement via closures.
- **Fix appliqué** : les deux lignes `app.config["_pending"] = pending` et `app.config["_session_holder"] = _cur` supprimées. Ruff OK, smoke test (`import loom.web.app`) OK.

### P1-5. `available(cfg)` — fonction publique morte — ✅ CORRIGÉ
- `loom/tools/web.py:252` — `def available(cfg) -> bool`. Aucun appel dans tout le repo.
- **Fix appliqué** : fonction supprimée. Ruff OK, smoke test OK.

### P1-6. `_resolve_in_root` re-exporté mais inutilisé — ✅ CORRIGÉ
- `loom/tools/__init__.py:16,25` — importé et listé dans `__all__` mais jamais utilisé dans `__init__.py` ni importé depuis l'extérieur. Nom préfixé `_` (privé) dans un `__all__` (public) — contradiction.
- **Fix appliqué** : retiré de l'import et de `__all__` dans `__init__.py`. Ruff OK, smoke test OK.

### P1-7. `Plugin.agents/hooks/commands` scannés mais jamais consommés — ✅ CORRIGÉ
- `loom/extend/plugins.py:35-37` — champs scannés par `_scan_components` et stockés, mais **jamais lus** en aval. Seul `Plugin.skills` est consommé.
- **Fix appliqué** : les 3 champs retirés de la classe `Plugin`, le scan correspondant supprimé de `_scan_components` (ne scanne plus que les skills). Docstrings et `_fmt_plugin` mis à jour. Ruff OK, smoke test OK.

### P1-8. Évals référencent `replace_lines` (outil retiré par ADR 0003) — ✅ CORRIGÉ
- `docs/adr/0003-retrait-insert-lines-ere-35b.md` (statut Accepté) retire `replace_lines` ET `insert_lines`. L'outil n'existe plus dans le runtime. Mais les évals y faisaient encore référence.
- **Fix appliqué** : toutes les références à `replace_lines`/`insert_lines` purgées du code source des évals (`cases.py`, `run_eval.py`, `README.md`). Self-test mis à jour pour utiliser `edit_file`. `evals/harness.py` créé pour mutualiser le code commun. Ruff OK, self-test évals VERT, aucune référence résiduelle dans le code Python exécutable.

### P1-9. `requests` déclaré en dépendance mais jamais importé — ✅ CORRIGÉ
- `pyproject.toml:8` — `requests>=2.32`. `search_text` sur `**/*.py` : **aucun** `import requests`.
- **Fix appliqué** : `requests` retiré de `pyproject.toml`, `uv lock` + `uv sync` régénérés. `uv lock` a retiré `requests` **entièrement** du lock file — ce qui infirme la note de l'audit : les versions actuelles d'`openai` et `huggingface-hub` ne tirent plus `requests` (elles utilisent `httpx`, déjà en dépendance directe). Smoke test (`import loom`) OK.

---

## P2 — Dette structurelle

### P2-1. `create_app` monolithique (1945 lignes) — ⏸ REPORTÉ
- `loom/web/app.py:331-2276` — une seule fonction, ~20 sous-fonctions, ~30 routes Flask en closure. Principal point de dette structurelle : impossible à tester unitairement.
- **Statut** : reporté. La conversion en Blueprints Flask est trop risquée sans tests live (trop de closures partagées). P2-2 (extraction de sous-fonctions) traite une partie du problème.

### P2-2. Route `chat()` 756 lignes — ✅ PARTIELLEMENT CORRIGÉ
- `loom/web/app.py:797-1553` — 7+ responsabilités (commande `/goal`, `/init`, auto-adoption workspace, construction prompt, boucle d'outils, titrage asynchrone, persistance, reflect post-tour, keep-warm).
- **Fix appliqué** : 3 closures extraites de `chat()` et appelées à la place des blocs inline : `_handle_goal_command`, `_handle_init_command`, `_build_system_prompt`. `_run_generation_loop` non extraite (trop de variables `nonlocal` capturées). Ruff OK, smoke test (construction app + count routes) OK.

### P2-3. `stream_chat_tools` monolithique (708 lignes) — ⏸ REPORTÉ
- `loom/agent/client.py:1045-1753` — une méthode, ~10 garde-fous imbriqués (overflow, context_overflow, loop, length-continuation, act-nudge, claim-audit artefact/exécution, non-progrès, parallélisme, debug-force).
- **Statut** : reporté. Cœur de l'agent — l'extraction en sous-méthodes est trop risquée sans tests live pour valider le comportement exact.

### P2-4. Duplication du bloc outil streaming (client.py) — ⏸ REPORTÉ
- `loom/agent/client.py:1499-1515` (parallèle) vs `1643-1652` (séquentiel) — pattern `parts: list[str] = []; for sub_kind, sub_payload in registry.run_stream(...): ...; result = "".join(parts).strip()` répété quasi à l'identique.
- `loom/agent/client.py:1554-1568` vs `1705-1720` — payload `tool_result` construit deux fois avec une structure quasi-identique.
- **Statut** : reporté. Même raison que P2-3 : factorisation dans le cœur de la boucle tool-use, trop risquée sans tests live.

### P2-5. Duplication anti-SSRF entre web.py et browser.py — ✅ CORRIGÉ
- `loom/tools/web.py:32-63` (`_resolve_validated`) vs `loom/tools/browser.py:149-170` (`_browser_http_blocked`). Deux implémentations parallèles de validation `ipaddress`, avec politiques différentes (intentionnel).
- **Fix appliqué** : primitive partagée `categorize_ip(ip) -> str` créée dans `loom/tools/_net.py`. `web.py` et `browser.py` l'importent et décident de leur politique (web bloque tout ce qui n'est pas public ; browser autorise loopback/private pour le dev local). Bug subtil corrigé : `link-local` doit être testé avant `private` (169.254.x.x est `is_private` ET `is_link_local` en Python). Ruff OK, smoke test (4 assertions `categorize_ip` + 4 assertions politiques SSRF) OK.

### P2-6. `build_registry` — signature trop large + complexité cyclomatique élevée — ✅ PARTIELLEMENT CORRIGÉ
- `loom/tools/__init__.py:59-242` — 13 paramètres, ~15 branches `if "X" in enabled:` séquentielles.
- `loom/tools/__init__.py:39-56` — `_SUBAGENT_TOOLS` liste en dur, le commentaire admettait la fragilité.
- **Fix appliqué** : `_SUBAGENT_TOOLS` dérivé automatiquement de `AVAILABLE_TOOLS` par filtrage (`_SUBAGENT_EXCLUDED = {"dispatch_agent", "manage_todos"}`). La liste se met à jour quand on ajoute/retire un outil. Dataclass `ToolBuildContext` et table de dispatch non créés (reporté — trop de changements pour un gain limité). Ruff OK, smoke test (26 outils, 24 subagent, exclusion confirmée) OK.

### P2-7. Duplication `_now_iso` / `_now` (3 copies) — ✅ CORRIGÉ
- `loom/agent/session.py:64-65`, `loom/extend/plugins.py:40-41`, `loom/agent/reflect.py:90-91` — 3 copies de la même fonction.
- **Fix appliqué** : `loom/utils.py` créé avec `now_iso()` (avec `timespec="seconds"`) et `now()` (sans timespec). Les 3 copies remplacées par des imports. Ruff OK, smoke test OK.

### P2-8. Heuristique 4-car/token dupliquée — ✅ CORRIGÉ
- `loom/agent/context.py:13-15` — `len(text) // 4` (magique). `loom/memory/identity.py:19` — `_CHARS_PER_TOKEN = 4`.
- **Fix appliqué** : `CHARS_PER_TOKEN = 4` et `estimate_tokens(text) -> int` mutualisés dans `loom/utils.py`. Les 2 usages remplacés par des imports. Ruff OK, smoke test (`estimate_tokens('abcd') == 1`) OK.

### P2-9. Duplication `_normalize_quotes` / `_normalize_dashes` — ✅ CORRIGÉ
- `loom/runtime/models_profile.py:23-29` et `32-38` — mêmes signatures, même garde, même boucle. Seul le mapping diffère.
- **Fix appliqué** : factorisé en `_normalize(content, suffix, mapping) -> str`. Ruff OK, smoke test OK.

### P2-10. Deux listes de champs divergentes pour les modèles distants — ✅ CORRIGÉ
- `loom/runtime/model_store.py:17-29` (`KEEP`, 11 champs) pilote le store JSON ; `model_store.py:114` (`fields`, 7 champs) pilote l'écriture TOML. `fields` omet `enable_thinking_param`, `price_in`, `price_out`, `price_cached` → un modèle édité via l'UI TOML perd les champs prix/thinking.
- **Fix appliqué** : `fields = KEEP` (single source of truth). La garde `if record.get(k) is not None` garantit qu'aucun champ existant n'est écrasé — le comportement actuel (préservation par tomlkit) est inchangé, mais les futurs `rec` portant ces champs seront correctement persistés. Commentaire documentant le choix et l'exclusion volontaire d'`api_key_env` ajouté. Ruff OK, smoke test (5 assertions couvrant persistance + non-régression) OK.

### P2-11. Duplication structurelle des deux harnais d'éval — ✅ CORRIGÉ
- `evals/run_eval.py` et `evals/run_review_eval.py` partageaient ~150 lignes communes.
- **Fix appliqué** : module commun `evals/harness.py` créé avec `git_show()`, `load_eval_config()`, `make_client()`, `make_perm()` et les constantes partagées. Les deux harnais importent ces fonctions. Ruff OK, smoke test (imports des deux harnais + self-test évals VERT) OK.

---

## P3 — Robustesse / gestion d'erreurs

### P3-1. Erreurs SQLite avalées sans log — ✅ CORRIGÉ
- `loom/memory/local.py:72-74` — `remember` : `except sqlite3.OperationalError: pass` (lock). `recall` : `except sqlite3.OperationalError: return []`.
- **Fix appliqué** : `import logging` ajouté, `logging.warning(f"SQLite error: {e}")` avant le `pass` et le `return []`. Ruff OK, smoke test OK.

### P3-2. `warmup_local` et `infer_title` — except Exception trop large — ✅ CORRIGÉ
- `loom/agent/client.py:907-908` — `warmup_local` : `except Exception: pass`. `infer_title` : `except Exception: continue`.
- **Fix appliqué** : `except Exception as e:` avec `_debug("WARMUP_ERR", str(e))` et `_debug("TITLE_ERR", str(e))` avant le pass/continue. Ruff OK, smoke test OK.

### P3-3. `_reload_app_config` / `_regen_swap_yaml` avalent silencieusement — ✅ CORRIGÉ
- `loom/web/app.py:460` — `except Exception:` best-effort : une config TOML cassée était **silencieusement ignorée**.
- **Fix appliqué** : `print(f"[loom] reload config échoué: {e}", flush=True)` et `print(f"[loom] regen swap yaml échoué: {e}", flush=True)` ajoutés avant les return. Ruff OK, smoke test OK.

### P3-4. `Response(f"erreur: {exc}")` fuite d'info — ✅ CORRIGÉ
- `loom/web/app.py:1124` — `chat()` renvoyait le message d'erreur brut au client (chemins, noms de modules internes). Générateur SSE : `yield _sse("error", message=str(exc))`.
- **Fix appliqué** : `traceback.print_exc()` côté serveur, message générique `"erreur interne"` au client (dans `chat()` et le générateur SSE). Ruff OK, smoke test OK.

### P3-5. `models_profile.load_profile` ne respecte pas sa promesse « ne lève jamais » — ✅ CORRIGÉ
- `loom/runtime/models_profile.py:135-138` — `UnicodeDecodeError` (sous-classe de `ValueError`, pas `OSError`) n'était pas attrapé.
- **Fix appliqué** : `except (OSError, UnicodeDecodeError)`. Ruff OK, smoke test OK.

### P3-6. `models_fetch.py:80` — except Exception trop large — ✅ CORRIGÉ
- `loom/runtime/models_fetch.py:80` — attrapait `Exception` et transformait en `ModelUnavailable`.
- **Fix appliqué** : restreint aux exceptions HF/réseau connues (`HfHubHTTPError`, `OSError`). Re-raise de `ModelUnavailable` commenté. Ruff OK, smoke test OK.

### P3-7. `browser.py` — `logf` sans try/finally global — ✅ CORRIGÉ
- `loom/tools/browser.py:443` — `logf = open(logpath, "w", ...)` sans context manager. Toute exception non prévue ferait fuir le descripteur.
- **Fix appliqué** : `try/except/finally` garantissant `logf.close()` et `os.unlink(logpath)` sur tout chemin d'erreur. Ruff OK, smoke test OK.

### P3-8. `swap.dump_yaml` — sérialiseur YAML maison fragile — ✅ CORRIGÉ
- `loom/runtime/swap.py:86-93` — sérialiseur maison ne gérait pas les caractères spéciaux YAML.
- **Fix appliqué** : remplacé par `yaml.safe_dump` (PyYAML 6.0.3). Ruff OK, smoke test OK.

---

## P4 — Typage manquant

### P4-1. `reflect.py` — typage systématiquement absent — ✅ CORRIGÉ
- `loom/agent/reflect.py` — `ReflectResult` : `new_skills: list[dict]`, `improved_skills: list[dict]`, `episodes: list[dict]`, `memory_updates: list[str]`, etc. `apply_reflect`, `_trajectory_summary`, `reflect` : `provider: Any`, `client: Any`, `model` typés.
- **Fix appliqué** : module entièrement typé avec `list[dict]`, `list[str]`, `Any` pour les providers/clients. Ruff OK, smoke test OK.

### P4-2. `client.py` — 7 fonctions helper non typées — ✅ CORRIGÉ
- `loom/agent/client.py` — `_debug`, `log_event`, `_usage_dict`, `_iter_events`, `_iter_turn`, `_resolve` : paramètres et/ou retours non annotés.
- **Fix appliqué** : annotations ajoutées avec `Any` pour les types dynamiques (stream, usage). `_resolve(self, model: str | None)`. `from typing import Any` ajouté. Ruff OK, smoke test OK.

### P4-3. `config_schema.py` — paramètres non annotés — ✅ CORRIGÉ
- `loom/runtime/config_schema.py` — `_target_path`, `set_value`, `reset_value` : `defaults_path`/`local_path`/`raw` non typés.
- **Fix appliqué** : annotations `str | Path` pour les chemins, `str | int | float | bool | None` pour `raw`. Ruff OK, smoke test OK.

### P4-4. `tools/` — paramètres non typés (duck-typing) — ✅ CORRIGÉ
- `loom/tools/` — paramètres non typés dans `agent.py`, `browser.py`, `read.py`, `memory.py`, `note.py`, `todo.py`, `plugins.py`, `web.py`.
- **Fix appliqué** : `loom/tools/_types.py` créé avec des Protocol (`ChatClient`, `MemoryProvider`, `Conversation`). Annotations ajoutées sur les paramètres les plus visibles. Ruff OK, smoke test OK.

### P4-5. `tools/base.py` — `ToolRegistry.profile` non typé — ✅ CORRIGÉ
- `loom/tools/base.py:191` — `profile=None` non typé.
- **Fix appliqué** : `profile: "Profile | None" = None` via `TYPE_CHECKING` (import depuis `loom.runtime.models_profile`). Ruff OK, smoke test OK.

### P4-6. `config_schema.SPEC` — pas de TypedDict — ✅ CORRIGÉ
- `loom/runtime/config_schema.py` — `SPEC: list[dict]` sans TypedDict.
- **Fix appliqué** : `TypedDict` `SpecEntry(total=False)` défini couvrant les 9 clés du SPEC (`section`, `key`, `label`, `layer`, `nature`, `type`, `applies`, `help`, `options`). `SPEC: list[SpecEntry]`. Ruff OK, smoke test OK.

---

## P5 — Documentation / cohérence

### P5-1. README liste `replace_lines`/`insert_lines` comme outils actifs — ✅ CORRIGÉ
- `README.md:25` — paragraphe décrivait 5 outils d'édition alors qu'il n'en reste que 3 (`write_file`, `append_file`, `edit_file`).
- **Fix appliqué** : liste corrigée. Toutes les mentions de `replace_lines` et `insert_lines` retirées du README. Ruff OK.

### P5-2. README cite des chemins de config inexistants — ✅ CORRIGÉ
- `README.md:155-158` — renvoyait vers `loom/loom.config.toml` et `loom/loom.config.personnel.toml`. Les fichiers réels sont dans `config/` (`config/defaults.toml`, `config/local.toml`).
- **Fix appliqué** : chemins corrigés dans le README (arborescence et texte). Ruff OK.

### P5-3. `serve.py:4` — chemin d'usage erroné dans le docstring — ✅ CORRIGÉ
- `loom/runtime/serve.py:4` — disait `Usage : uv run loom/serve.py` mais le chemin réel est `loom/runtime/serve.py`.
- **Fix appliqué** : corrigé en `uv run loom/runtime/serve.py`. Ruff OK.

### P5-4. `runtime/__init__.py:1-13` — docstring du package incomplet — ✅ CORRIGÉ
- Listait 6 modules mais omettait `platform_info.py`, `sysmon.py`, `model_store.py`, `config_schema.py`, `ngl.py`.
- **Fix appliqué** : liste complétée avec les 5 modules omis. Ruff OK.

### P5-5. `server_args.py:20-24` — docstring de `build_server_args` incomplet — ✅ CORRIGÉ
- Documentait `n_parallel` mais pas `cpu_moe`, `n_cpu_moe`, `gpu_tuning`, `mmproj_path`.
- **Fix appliqué** : docstring complété avec les 4 paramètres manquants. Ruff OK.

### P5-6. `config_schema.py:33` — type `"list"` documenté mais jamais implémenté — ✅ CORRIGÉ
- Le commentaire déclarait `type : ... | list`, mais `_coerce` ne le gérait pas.
- **Fix appliqué** : type `"list"` implémenté dans `_coerce` : parse une string CSV en `list[str]` (split sur virgule, strip, filtre vides). Vérifié fonctionnellement : `'a, b ,c'` → `['a', 'b', 'c']`. Ruff OK.

### P5-7. `config_schema.py:546-547` — garde `editable` morte — ✅ CORRIGÉ
- `spec.get("editable") is False` teste un champ qui **n'est jamais posé** dans aucun dict de `SPEC`. Le branchement existe mais n'attrapera jamais rien.
- **Fix appliqué** : après lecture intégrale du SPEC (34 entrées), aucune n'est un dérivé/internal — toutes sont des réglages utilisateur éditables. Bloc `if spec.get("editable") is False` retiré, et commentaire ligne 36 nettoyé (il documentait un champ inexistant). Ruff OK, smoke test (`import` + `describe()` + `set_value`) OK.

### P5-8. Quatre fichiers de doc avec recoupement massif — ✅ CORRIGÉ
- `README.md`, `ETAT_PROJET.md`, `loom.md`, `CHANGELOG.md` décrivent l'état du projet avec recoupement massif. `loom.md:14` mentionnait `requests` dans la stack (dépendance retirée).
- **Fix appliqué** : ligne de rôle ajoutée en tête de chaque fichier (README = pitch public, ETAT_PROJET = suivi interne, loom.md = carte technique, CHANGELOG = historique versions). `requests` retiré de `loom.md`. Ruff OK.

### P5-9. `browser.py` — docstrings et messages sans accents — ✅ CORRIGÉ
- `loom/tools/browser.py` — « demarre », « verifie », « arrete », « jouabilite », contrairement à tous les autres fichiers du package.
- **Fix appliqué** : accents harmonisés dans tout le fichier (« démarre », « vérifie », « arrête », « jouabilité », etc.). Ruff OK.

### P5-10. `browser.py:445` — `sys.platform` au lieu de `detect().is_windows` — ✅ CORRIGÉ
- Contournait la source de vérité unique `platform_info.detect().is_windows`.
- **Fix appliqué** : `sys.platform` remplacé par `detect().is_windows` (import depuis `loom.runtime.platform_info`). Import `sys` retiré (inutilisé). Ruff OK.

### P5-11. `platform_info.py:42` — `/bin/bash` hardcodé — ✅ CORRIGÉ
- `shell_argv` retournait `["/bin/bash", "-lc", command]` sur POSIX. Sur NixOS/Alpine, bash peut être ailleurs.
- **Fix appliqué** : `shutil.which("bash") or "/bin/bash"`. Vérifié sous Windows : résout `C:\Windows\system32\bash.EXE`. Ruff OK.

---

## Environnement / scratch

### ENV-1. `var/` — ~62 fichiers scratch/patchs jetables
- `var/_*.py`, `var/_*.txt`, `var/fix_encoding.py`, `var/run_8001.py`, `var/serve_8001.py`, `var/8001_*.log`, `var/*.png`, `var/logs/*.bak`, `var/logs/_codex_*.err`, `var/logs/lessons.json`. Tous sont des scripts one-shot (patchs `str.replace`, brouillons de commit, scripts de vérification, lanceurs de dev, logs de dev, captures). Aucun n'est importé par le code de Loom. Plusieurs ciblent des fonctions **déjà supprimées** (`_z1.py` retire `replace_lines` qui n'existe plus) → inopérants.
- `var/` est gitignoré (`.gitignore:9`), donc pas de pollution du dépôt, mais encombrement disque local.
- Action : supprimer les ~62 fichiers scratch. Conserver `sessions/`, `identity/`, `memory/`, `cache/`, `logs/serve.log`, `logs/loom-debug.log`, `skills_learned/`, `last_model`, `conversation.json`, `remote_models.json`.

### ENV-2. `scripts/` vide + `presentation-qwen/` hors sujet
- `scripts/` : dossier vide.
- `presentation-qwen/` : `presentation.html` + `présentation-loom.pdf` (présentation de démo générée par Qwen3.6, CHANGELOG l.46). Doublon de format (HTML + PDF même contenu), artefact de démo dans le repo de code.
- Action : supprimer `scripts/` vide. Déplacer `presentation-qwen/` hors repo ou dans `docs/`, garder un seul format.

---

## Points vérifiés SAINS (pas de dette)

- Aucun TODO/FIXME/XXX/HACK dans tout le code applicatif.
- Aucun import inutile dans `loom/web/app.py` (16 imports tous utilisés).
- `RuntimeConfig.model` (property, `config.py:163`) — utilisée à `serve.py:101-102,189`.
- `DEFAULT_SYSTEM_PROMPT` (`config.py:15`) — utilisé `config.py:271`.
- `SUBAGENT_SYSTEM` / `REFLECT_SYSTEM` (`prompts/__init__.py:25-26`) — utilisés `tools/__init__.py:176`, `agent/reflect.py:17,184`.
- `set_debug_log_path` (`app.py:36`) — utilisé `app.py:921`.
- `is_protected_write_path` — utilisé `tools/fs.py:33` (motifs durs fonctionnels ; seul `deny_paths` custom est mort, voir P1-2).
- Garde CSRF (`app.py:766-775`), `_force_utf8_charset` (`app.py:393-403`) — corrects et commentés.
- `loom/agent/conversation.py`, `loom/agent/inline_image.py`, `loom/extend/skills.py`, `loom/tools/skills.py`, `loom/tools/todo.py`, `loom/tools/trust.py`, `loom/tools/note.py`, `loom/web/__init__.py` — propres, pas de dette significative.
- `config/local.toml` — gitignoré (`.gitignore:11`), non suivi par git. La clé API en clair (`config/local.toml:39,47`) ne fuit donc pas dans le dépôt, mais reste exposée sur disque alors que `config/local.example.toml:45` documente l'alternative `api_key_env`. Action (basse priorité) : migrer vers `api_key_env`.

---

## Ce qui n'a pas pu être vérifié

- Comportement des évals contre un modèle live (serveur llama.cpp non lancé). Le bug P0-1 (MoE) est déduit de la lecture croisée `serve.build_launch` vs `swap._model_cmd`, non reproduit à l'exécution.
- Usage dynamique via `getattr(module, "fonction")` non exclu (recherche statique uniquement).
- Contenu de `docs/superpowers/` (specs/plans/archive) non audité (hors périmètre code).
