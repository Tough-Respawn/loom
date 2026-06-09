# Moteur unique Loom — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fusionner `run_pipeline` + `run_build` en un seul moteur fan-out durci (mode par fichier create/patch/rewrite, edit ciblé, best-of-N, reviewer sémantique non bloquant), livré en 9 PRs additives sans casser les 214 tests existants.

**Architecture:** On bâtit sur le fan-out existant (`run_build`, `loom/orchestrator.py:172`). Chaque PR ajoute une brique pure et testable isolément (`derive_modes`, `edit_one`, garde-fou rewrite, budget mesuré, `explore`, `review_semantic`, best-of-N, stop anti-divergence), puis une PR finale retire `run_pipeline`. `run_pipeline` reste deprecated derrière `mode=='pipeline'` (`app.py:288`) jusqu'à cette PR finale.

**Tech Stack:** Python 3, `uv`/`ruff`, `pytest`. Pas de dépendance nouvelle. Spec source : [docs/superpowers/specs/2026-06-03-loom-moteur-unique-design.md](../specs/2026-06-03-loom-moteur-unique-design.md).

---

## Pré-requis : dépôt git

Le projet n'est **pas encore sous git** (`ETAT_PROJET.md`). Les étapes « Commit » supposent un dépôt. Avant la PR 1 :

- [ ] **Step 0 : initialiser git (une seule fois)**

Run :
```bash
cd "c:/Users/Amine/Documents/from-claude-to-local-haranessed-llm"
git init && git add -A && git commit -m "chore: snapshot avant moteur unique"
```
Expected : dépôt créé, 1ᵉʳ commit. Si déjà sous git, ignorer.

---

## File Structure

| Fichier | Responsabilité | PRs |
|---|---|---|
| `loom/parallel.py` | Briques pures de génération par fichier : `FileSpec`, **`PlannedFile`** (nouveau), **`derive_modes`** (nouveau), **`cap_rewrites`** (nouveau), `generate_one` (+ clip), **`edit_one`** (nouveau), `fix_one`, `compute_budget` (mesuré), `review_semantic` (nouveau) | 1,2,3,4,6,7 |
| `loom/orchestrator.py` | `run_build` (le moteur) : câble `derive_modes`/`edit_one`/best-of-N/stop anti-divergence ; `run_pipeline` (retiré PR9) | 1,2,3,4,5,7,8,9 |
| `loom/explore.py` | **Nouveau** : `explore()` borné (lecture auto OU boucle outillée) + budget pur sur `list[dict]` | 5 |
| `loom/web/app.py` | Routage `mode=` (build par défaut, pipeline deprecated) ; handler event review (PR6) | 6,9 |
| `loom/web/static/app.js` | Handler de l'event review sémantique (PR6) | 6 |
| `tests/test_parallel.py` | Tests des briques pures | 1,2,3,4,6,7 |
| `tests/test_orchestrator.py` | Tests du moteur ; mapping des 11 `test_run_pipeline_*` (PR9) | 1,2,3,8,9 |
| `tests/test_explore.py` | **Nouveau** : bornes EXPLORE | 5 |
| `tests/test_web.py` | Test d'intégration `/run` mode build | 5/6 |

---

## PR 1 — `PlannedFile` + `derive_modes`

**Objectif :** dériver le mode (`create`/`patch`/`rewrite`) de chaque fichier **déterministiquement** (sans LLM), via `os.path.exists` croisé avec le verify. Aucune modification de `FileSpec` ni de `_parse_plan` (tests verts).

### Task 1.1 : `PlannedFile` + `derive_modes`

**Files:**
- Modify: `loom/parallel.py` (ajouter après `FileSpec`, l.20-24)
- Test: `tests/test_parallel.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

Ajouter à `tests/test_parallel.py` :
```python
def test_derive_modes_create_when_file_absent(tmp_path):
    from loom.parallel import FileSpec, derive_modes

    specs = [FileSpec("new.js", "logique")]
    # verifier ne devrait même pas être appelé pour un fichier absent
    planned = derive_modes(specs, str(tmp_path), verifier=lambda paths: 1 / 0)
    assert [(p.spec.path, p.mode) for p in planned] == [("new.js", "create")]


def test_derive_modes_patch_when_existing_and_verify_ok(tmp_path):
    from loom.parallel import FileSpec, derive_modes
    from loom.verify import VerifyReport

    (tmp_path / "ok.js").write_text("let x = 1;\n", encoding="utf-8")
    specs = [FileSpec("ok.js", "logique")]
    planned = derive_modes(specs, str(tmp_path), verifier=lambda paths: VerifyReport(ok=True))
    assert planned[0].mode == "patch"


def test_derive_modes_rewrite_when_existing_and_verify_fails(tmp_path):
    from loom.parallel import FileSpec, derive_modes
    from loom.verify import Defect, VerifyReport

    (tmp_path / "broken.js").write_text("let x = ;\n", encoding="utf-8")
    specs = [FileSpec("broken.js", "logique")]
    report = VerifyReport(ok=False, defects=[Defect("broken.js:1", "syntax", "Unexpected")])
    planned = derive_modes(specs, str(tmp_path), verifier=lambda paths: report)
    assert planned[0].mode == "rewrite"


def test_derive_modes_patch_when_verifier_returns_none(tmp_path):
    # fichier non-vérifiable (ex. .css) -> verifier renvoie None -> patch (sûr)
    from loom.parallel import FileSpec, derive_modes

    (tmp_path / "style.css").write_text("body{}\n", encoding="utf-8")
    specs = [FileSpec("style.css", "styles")]
    planned = derive_modes(specs, str(tmp_path), verifier=lambda paths: None)
    assert planned[0].mode == "patch"
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run : `uv run pytest tests/test_parallel.py -k derive_modes -v`
Expected : FAIL — `ImportError: cannot import name 'derive_modes'`.

- [ ] **Step 3 : Implémenter `PlannedFile` + `derive_modes`**

Dans `loom/parallel.py`, après la classe `FileSpec` (l.24) :
```python
@dataclass
class PlannedFile:
    """Un fichier planifié + son mode de génération (dérivé, jamais parsé du plan)."""

    spec: FileSpec
    mode: str  # 'create' | 'patch' | 'rewrite'


def derive_modes(specs: list[FileSpec], workspace: str, verifier) -> list[PlannedFile]:
    """Dérive le mode de chaque fichier SANS LLM (cf. spec §5) :
    - absent du disque                 -> create
    - présent et verify échoue déjà    -> rewrite (déclencheur OBJECTIF)
    - présent et verify OK (ou None)    -> patch  (le moins destructeur)

    `verifier(list[str_abspath]) -> VerifyReport | None`. N'est appelé que pour un
    fichier EXISTANT (un fichier absent est create sans vérification).
    """
    root = Path(workspace)
    planned: list[PlannedFile] = []
    for spec in specs:
        abspath = root / spec.path
        if not abspath.exists():
            mode = "create"
        else:
            report = verifier([str(abspath)])
            mode = "rewrite" if (report is not None and not report.ok) else "patch"
        planned.append(PlannedFile(spec=spec, mode=mode))
    return planned
```
(`Path` et `dataclass` sont déjà importés en tête de `parallel.py`.)

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run : `uv run pytest tests/test_parallel.py -k derive_modes -v`
Expected : PASS (4 tests).

- [ ] **Step 5 : Lint + suite complète + commit**

Run : `uv run ruff check loom/parallel.py tests/test_parallel.py && uv run pytest -q`
Expected : ruff clean, 214 + 4 tests verts.
```bash
git add loom/parallel.py tests/test_parallel.py
git commit -m "feat(engine): derive_modes create/patch/rewrite déterministe"
```

---

## PR 2 — `edit_one` (mode patch, 2 temps dans le harness)

**Objectif :** retoucher un fichier existant par un remplacement ciblé (le modèle ne voit jamais un fichier entier non borné ; il forge `old_string`/`new_string` à partir du contenu lu, appliqués via la logique `make_edit_file`). **Fallback** déterministe vers réécriture complète si introuvable/ambigu/JSON invalide. Renvoie le **contenu complet** relu (contrat d'état).

### Task 2.1 : `edit_one`

**Files:**
- Modify: `loom/parallel.py` (ajouter après `generate_one`, l.217 ; imports `json`, `make_edit_file`, `ToolError`)
- Test: `tests/test_parallel.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
def test_edit_one_applies_targeted_edit_and_returns_full_content(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "app.js").write_text("let x = 1;\nlet y = 2;\n", encoding="utf-8")
    client = FakeClient(default='{"old_string": "let x = 1;", "new_string": "let x = 9;"}')
    path, content = edit_one(
        client, "design", FileSpec("app.js", "logique"), str(tmp_path),
        model="m", max_tokens=512, file_char_cap=8192,
    )
    # contrat d'état : on renvoie le fichier ENTIER relu, pas juste le diff
    assert path == "app.js"
    assert content == "let x = 9;\nlet y = 2;\n"
    assert (tmp_path / "app.js").read_text(encoding="utf-8") == "let x = 9;\nlet y = 2;\n"


def test_edit_one_falls_back_to_rewrite_when_old_string_absent(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "app.js").write_text("let x = 1;\n", encoding="utf-8")
    # 1er appel (edit) : old_string introuvable ; 2e appel (generate) : contenu complet
    client = FakeClient(by_keyword={
        "JSON": '{"old_string": "ABSENT", "new_string": "z"}',
        "COMPLET et FINAL": "let x = 1;\nlet z = 3;\n",
    })
    path, content = edit_one(
        client, "design", FileSpec("app.js", "logique"), str(tmp_path),
        model="m", max_tokens=512, file_char_cap=8192,
    )
    assert path == "app.js"
    assert content.strip() == "let x = 1;\nlet z = 3;"  # vient du fallback generate_one


def test_edit_one_falls_back_on_invalid_json(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "app.js").write_text("let x = 1;\n", encoding="utf-8")
    client = FakeClient(by_keyword={
        "JSON": "ceci n'est pas du json",
        "COMPLET et FINAL": "REWRITTEN\n",
    })
    _, content = edit_one(
        client, "design", FileSpec("app.js", "logique"), str(tmp_path),
        model="m", max_tokens=512, file_char_cap=8192,
    )
    assert content.strip() == "REWRITTEN"


def test_edit_one_caps_injected_content_to_half_file_char_cap(tmp_path):
    from loom.parallel import FileSpec, edit_one

    (tmp_path / "big.js").write_text("X" * 50_000, encoding="utf-8")
    client = FakeClient(default='{"old_string": "XX", "new_string": "Y"}')
    edit_one(
        client, "design", FileSpec("big.js", "logique"), str(tmp_path),
        model="m", max_tokens=512, file_char_cap=8192,
    )
    injected = client.calls[0]["prompt"]
    assert "…[tronqué]" in injected
    assert len(injected) < 8192  # contenu borné à file_char_cap/2 + gabarit
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run : `uv run pytest tests/test_parallel.py -k edit_one -v`
Expected : FAIL — `ImportError: cannot import name 'edit_one'`.

- [ ] **Step 3 : Implémenter `edit_one`**

En tête de `loom/parallel.py`, compléter les imports (`json` est déjà importé l.14) :
```python
from loom.tools.base import ToolError
from loom.tools.fs import make_edit_file
```
Après `generate_one` (l.217) :
```python
_EDIT_SYS = (
    "Tu modifies un fichier EXISTANT par UN remplacement ciblé. Tu réponds UNIQUEMENT "
    'un objet JSON {"old_string": "...", "new_string": "..."} où old_string est un '
    "extrait EXACT et UNIQUE du fichier (copié au caractère près, mêmes espaces/retours) "
    "à remplacer par new_string. Pas de markdown, pas d'explication."
)


def _edit_prompt(spec: FileSpec, design: str, content: str, defects: str = "") -> str:
    head = (
        f"Architecture PARTAGÉE :\n{design}\n\n"
        f"Fichier `{spec.path}` ({spec.role}). Contenu ACTUEL (copie old_string À "
        f"L'IDENTIQUE depuis ce texte) :\n-----\n{content}\n-----\n\n"
    )
    if defects:
        head += f"DÉFAUTS à corriger :\n{defects}\n\n"
    return head + "Renvoie le JSON {old_string, new_string} du remplacement ciblé."


def edit_one(
    client,
    design: str,
    spec: FileSpec,
    workspace: str,
    *,
    model: str | None,
    max_tokens: int,
    file_char_cap: int,
    defects: str = "",
) -> tuple[str, str]:
    """PATCH ciblé en 2 temps DANS le harness (cf. spec §5) :
    1) read déterministe du fichier (borné file_char_cap/2, byte-exact pour le match) ;
    2) le modèle renvoie {old_string, new_string} ;
    3) application via make_edit_file (erreurs exploitables) ;
    4) FALLBACK generate_one borné si JSON invalide / introuvable / ambigu.
    Renvoie (path, CONTENU COMPLET relu) — contrat d'état identique à generate_one.
    """
    root = Path(workspace)
    abspath = root / spec.path
    content = abspath.read_bytes().decode("utf-8")  # byte-exact (match edit_file)
    cap = max(256, file_char_cap // 2)
    injected = content if len(content) <= cap else content[:cap] + "\n…[tronqué]"
    raw = client.complete(
        [{"role": "user", "content": _edit_prompt(spec, design, injected, defects)}],
        _EDIT_SYS,
        max_tokens=max_tokens,
        model=model,
        thinking=False,
    )

    def _fallback() -> tuple[str, str]:
        return generate_one(
            client, design, spec, [spec.path], model=model, max_tokens=max_tokens
        )

    try:
        data = _extract_json(raw)
        old_string = data["old_string"]
        new_string = data["new_string"]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return _fallback()
    if not isinstance(old_string, str) or not isinstance(new_string, str) or not old_string:
        return _fallback()
    editor = make_edit_file(str(root))
    try:
        editor.run({"path": spec.path, "old_string": old_string, "new_string": new_string})
    except ToolError:
        return _fallback()  # introuvable / ambigu -> réécriture complète bornée
    return spec.path, abspath.read_bytes().decode("utf-8")
```
Note : le mot-clé `"JSON"` du test de fallback matche `_EDIT_SYS`/`_edit_prompt` ; `"COMPLET et FINAL"` matche `_file_prompt` (`generate_one`).

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run : `uv run pytest tests/test_parallel.py -k edit_one -v`
Expected : PASS (4 tests).

- [ ] **Step 5 : Lint + suite + commit**

Run : `uv run ruff check loom/parallel.py tests/test_parallel.py && uv run pytest -q`
Expected : clean, tout vert.
```bash
git add loom/parallel.py tests/test_parallel.py
git commit -m "feat(engine): edit_one (patch 2 temps + fallback rewrite)"
```

---

## PR 3 — Garde-fou `rewrite` (> 200 lignes → dégrade en patch)

**Objectif :** un fichier existant cassé MAIS volumineux (> 200 lignes) ne doit pas être réécrit en entier (risque de troncature sur 4B) : on **dégrade** son mode `rewrite` → `patch`.

### Task 3.1 : `cap_rewrites`

**Files:**
- Modify: `loom/parallel.py` (après `derive_modes`)
- Test: `tests/test_parallel.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

```python
def test_cap_rewrites_downgrades_large_files_to_patch(tmp_path):
    from loom.parallel import FileSpec, PlannedFile, cap_rewrites

    (tmp_path / "huge.js").write_text("x\n" * 300, encoding="utf-8")  # 300 lignes
    planned = [PlannedFile(FileSpec("huge.js", "logique"), "rewrite")]
    out = cap_rewrites(planned, str(tmp_path), max_lines=200)
    assert out[0].mode == "patch"


def test_cap_rewrites_keeps_small_rewrites(tmp_path):
    from loom.parallel import FileSpec, PlannedFile, cap_rewrites

    (tmp_path / "small.js").write_text("x\n" * 50, encoding="utf-8")
    planned = [PlannedFile(FileSpec("small.js", "logique"), "rewrite")]
    out = cap_rewrites(planned, str(tmp_path), max_lines=200)
    assert out[0].mode == "rewrite"


def test_cap_rewrites_ignores_non_rewrite_modes(tmp_path):
    from loom.parallel import FileSpec, PlannedFile, cap_rewrites

    (tmp_path / "huge.js").write_text("x\n" * 300, encoding="utf-8")
    planned = [
        PlannedFile(FileSpec("huge.js", "logique"), "create"),
        PlannedFile(FileSpec("huge.js", "logique"), "patch"),
    ]
    out = cap_rewrites(planned, str(tmp_path), max_lines=200)
    assert [p.mode for p in out] == ["create", "patch"]
```

- [ ] **Step 2 : Lancer les tests, vérifier l'échec**

Run : `uv run pytest tests/test_parallel.py -k cap_rewrites -v`
Expected : FAIL — `cannot import name 'cap_rewrites'`.

- [ ] **Step 3 : Implémenter `cap_rewrites`**

Après `derive_modes` dans `loom/parallel.py` :
```python
def cap_rewrites(
    planned: list[PlannedFile], workspace: str, *, max_lines: int = 200
) -> list[PlannedFile]:
    """Dégrade rewrite -> patch pour les fichiers existants > max_lines (cf. spec §5) :
    réécrire intégralement un gros fichier sur un 4B risque la troncature."""
    root = Path(workspace)
    out: list[PlannedFile] = []
    for pf in planned:
        if pf.mode == "rewrite":
            abspath = root / pf.spec.path
            try:
                n_lines = abspath.read_text(encoding="utf-8", errors="replace").count("\n") + 1
            except OSError:
                n_lines = 0
            if n_lines > max_lines:
                out.append(PlannedFile(pf.spec, "patch"))
                continue
        out.append(pf)
    return out
```

- [ ] **Step 4 : Lancer les tests, vérifier le succès**

Run : `uv run pytest tests/test_parallel.py -k cap_rewrites -v`
Expected : PASS (3 tests).

- [ ] **Step 5 : Lint + suite + commit**

Run : `uv run ruff check loom/parallel.py tests/test_parallel.py && uv run pytest -q`
```bash
git add loom/parallel.py tests/test_parallel.py
git commit -m "feat(engine): garde-fou rewrite > 200 lignes -> patch"
```

---

## Backlog séquencé — PRs 4 à 9 (à détailler juste-à-temps)

> Chaque PR ci-dessous a un objectif, des fichiers et des tests-clés concrets. On les **développe en TDD au moment de les attaquer** (pas avant : leur détail dépend des signatures figées par les PRs 1–3 et des apprentissages d'exécution). À l'ouverture de chaque PR, ré-invoquer writing-plans pour expanser la PR en tâches `- [ ]` complètes.

### PR 4 — `compute_budget` mesuré + clip `generate_one`
- **Spec :** §6.1, §6.2. **Fichiers :** `loom/parallel.py` (`compute_budget`, `_file_prompt`, `generate_one`), `tests/test_parallel.py`.
- **Changements :** `_file_prompt` clippe le contenu injecté à `file_char_cap` (comme `_fix_prompt`) ; `generate_one`/`fix_one` reçoivent et propagent `file_char_cap` ; `compute_budget` accepte un `reserve_prompt_tokens` **mesuré** (`context.estimate_tokens` du design + plus gros fichier injecté) au lieu de la constante 2048.
- **Tests-clés :** `test_file_prompt_clips_to_file_char_cap` (symétrique de l'existant `test_fix_prompt_clips_*`) ; `test_compute_budget_uses_measured_reserve` (un design volumineux réduit `gen_max_tokens`). **Garde-fou non-régression :** `test_compute_budget_*` existants restent verts (le défaut 2048 inchangé).

### PR 5 — `explore()` borné + budget pur `list[dict]`
- **Spec :** §6.4, §6.5, §6.6. **Fichiers :** **`loom/explore.py`** (nouveau), `loom/context.py` (extraire un budget pur sur `list[dict]`), `loom/orchestrator.py` (câbler EXPLORE avant PLAN), `tests/test_explore.py`.
- **Changements :** routage déterministe (paths explicites sur disque → lecture auto ; sinon workspace non vide → boucle outillée `stream_chat_tools` bornée `max_iters ≤ 3`, `read_file` EXPLORE `≤ 16 KB`, stop dur `conversation_tokens > 0.6·context`) ; sortie = résumé ground-truth (paths + signatures), **jamais** le contenu brut réinjecté ; PLAN brownfield reçoit ce résumé borné `≤ 1500 tok`.
- **Tests-clés :** `test_explore_auto_reads_only_targeted_paths` ; `test_explore_tooled_stops_at_0_6_context` ; `test_explore_caps_read_bytes` ; `test_explore_routing_is_deterministic` (aucun appel « tâche vague ? » au modèle). **+ test d'intégration `/run` mode build** dans `tests/test_web.py` (contrat SSE figé) **avant** la fusion.

### PR 6 — `review_semantic` non bloquant + handler UI
- **Spec :** §7. **Fichiers :** `loom/parallel.py` (`review_semantic`), `loom/orchestrator.py` (l'appeler **seulement si** verify vert ; liste séparée `semantic_defects[]` ; ≤ 1 passe de fix ; **hors** condition de boucle), `loom/web/static/app.js` (handler de l'event), `tests/test_parallel.py` + `tests/test_web.py`.
- **Changements :** signature pure `review_semantic(client, design, current_files, *, model) -> list[Defect]` (`kind='semantic'`) ; l'event émis doit avoir un handler `app.js` (sinon ignoré silencieusement, cf. spec §9 matrice).
- **Tests-clés :** `test_review_semantic_returns_semantic_defects` ; `test_review_semantic_not_called_when_verify_red` ; `test_semantic_defects_do_not_keep_loop_open` (un faux positif sémantique ne relance pas indéfiniment).

### PR 7 — best-of-N (réparation + ancrage passe 1)
- **Spec :** §8. **Fichiers :** `loom/parallel.py` (`generate_best_of` : N candidats **séquentiels** dans le worker, `verify_syntax_file` garde le 1ᵉʳ valide, sinon last-good), `loom/orchestrator.py` (détection des fichiers d'ancrage = référencés par d'autres dans `all_paths` → N=2 ; feuilles → N=1 ; FIX → N=2).
- **Tests-clés :** `test_best_of_keeps_first_valid_candidate` ; `test_best_of_falls_back_to_last_good` ; `test_anchor_files_get_n_gt_1_leaves_get_1` ; **invariant** : `workers <= fit` de `compute_budget` (N n'entre pas dans la concurrence).

### PR 8 — Stop anti-divergence (ensemble des `location`)
- **Spec :** §9. **Fichiers :** `loom/orchestrator.py` (boucle FIX, l.343-369), `tests/test_orchestrator.py`.
- **Changements :** conserver `set(d.location for d in report.defects)` du round précédent ; **arrêter** si l'ensemble courant n'est pas un sous-ensemble strict du précédent (un fix qui résout A et crée B à `len` égal ne doit pas boucler) ; `max_rounds` abaissé à 2-3.
- **Tests-clés :** `test_fix_loop_stops_when_locations_do_not_decrease` ; `test_fix_loop_continues_while_locations_shrink_until_max_rounds`.

### PR 9 (finale, isolée) — Retrait de `run_pipeline`
- **Spec :** §9 (mapping test par test). **Fichiers :** `loom/orchestrator.py` (suppr. `run_pipeline`), `loom/agents.py` (suppr. `is_blocking`/`is_reviewer`/`build_step_messages` si non réutilisés par EXPLORE), `loom/web/app.py` (retrait branche `mode=='pipeline'`), config `[[agents]]`, `tests/test_orchestrator.py` + `tests/test_agents.py`.
- **Pré-condition :** parité prouvée sur 3 tâches de référence (jeu web, script Python, refactor multi-fichiers) — la fusion atteint `VerifyReport.ok` sans intervention, sans fichier tronqué, défauts strictement décroissants.
- **Mapping des 11 `test_run_pipeline_*`** : supprimés (ordre agents, max_tokens, thinking, routage tools, content-pas-reasoning) ; réécrits fan-out (gate verify > texte, boucle bornée) ; supprimés avec tests (`is_blocking`/`is_reviewer`/`build_step_messages`). **Conserver `build_model` indépendant de `[[agents]]`** (ex. `models[0]`) car `selected[0].model` en dépend (`app.py:289`).

---

## Self-Review (effectuée)

- **Couverture spec :** §5→PR1-3, §6.1-6.2→PR4, §6.4-6.6→PR5, §7→PR6, §8→PR7, §9 stop→PR8, §9 retrait→PR9. Critères d'acceptation §11 mappés aux tests-clés ci-dessus. ✅
- **Placeholders :** PRs 1-3 contiennent le code et les tests réels complets ; PRs 4-9 sont un backlog **explicitement** à expanser (pas du faux-détaillé). ✅
- **Cohérence des types :** `PlannedFile(spec, mode)`, `derive_modes(specs, workspace, verifier)`, `cap_rewrites(planned, workspace, *, max_lines)`, `edit_one(..., file_char_cap, defects='')` — signatures stables et réutilisées à l'identique entre tâches. `edit_one` réutilise `make_edit_file` (réel, `fs.py:91`) et `generate_one` (réel, `parallel.py:199`). ✅
```
