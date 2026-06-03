# Loom v3.2 — Skills / Contexte — Design

> Spec de design — 2026-05-31
> Statut : **approuvé verbalement, spec à relire**

## 0. Contexte

L'utilisateur veut injecter sa **propre connaissance** (ex. son architecture Dagster) dans le
contexte du modèle, pour que Loom « connaisse » sa stack et travaille dessus. Inspiré du système
de **skills de Claude Code**, adapté à un petit modèle local (Gemma 4B).

**Décision clé (« le meilleur des deux mondes »)** : on adopte le **format Claude Code**
(`SKILL.md` + frontmatter `name`/`description`), MAIS l'activation est **manuelle** (l'utilisateur
coche le skill dans l'UI) et l'injection se fait par **contenu complet**. Raison : la divulgation
progressive de Claude Code repose sur le jugement du modèle pour charger le bon skill — peu fiable
sur un 4B. L'auto-déclenchement (descriptions en contexte, modèle qui charge) est reporté à la
couche agents (roadmap).

## 1. Objectif v3.2

- Un dossier de **skills** (fichiers markdown de connaissance, format Claude Code).
- **Sélection manuelle** des skills actifs **par conversation** (persistée).
- Le contenu des skills actifs est **injecté dans le system prompt** à chaque message.
- Un **panneau Skills** dans l'UI (cases à cocher).

## 2. Composants

### 2.1 `loom/skills/` 🆕 (dossier de skills)
Format Claude Code : `loom/skills/<nom>/SKILL.md`. Chaque `SKILL.md` :
```markdown
---
name: dagster-archi
description: Architecture Dagster de mon EDP (assets, jobs, schedules…)
---

<contenu markdown de la connaissance>
```
Un **skill d'exemple** est livré (`loom/skills/exemple/SKILL.md`) pour montrer le format.

### 2.2 `loom/skills.py` 🆕
- `@dataclass Skill`: `name: str`, `description: str`, `body: str`.
- `list_skills(skills_dir) -> list[Skill]` : scanne les sous-dossiers, lit chaque `SKILL.md`,
  parse le frontmatter (`name`/`description`) + le corps. Robuste : si pas de frontmatter, `name`
  = nom du dossier, `description` = "". Ignore les dossiers sans `SKILL.md`.
- `load_skill(skills_dir, name) -> Skill | None`.
- `compose_system_prompt(base: str, active: list[Skill]) -> str` : renvoie
  `base` + pour chaque skill actif `\n\n# Skill : <name>\n<body>`.
- Parsing frontmatter : mini-parseur ligne à ligne (`key: value` entre deux `---`), **pas de
  dépendance** (pas de PyYAML) — suffisant pour `name`/`description`.

### 2.3 `loom/conversation.py`
- `Conversation` gagne `active_skills: list[str]` (noms), défaut `[]`. Persisté dans le JSON
  (`{system_prompt, messages, active_skills}`). `load` tolère son absence (anciennes convs).
- Méthode `set_skills(names: list[str])`.

### 2.4 `loom/config.py` + `loom.config.toml`
- `[chat] skills_dir = "loom/skills"`.

### 2.5 `loom/web/app.py`
- `GET /` : passe aussi la liste des skills dispo + l'état actif au template.
- `POST /skills` : reçoit la nouvelle liste de skills actifs (cases cochées), met à jour
  `conversation.set_skills(...)` + `save`, renvoie le panneau (ou la page) à jour.
- `/chat` : construit le **system prompt effectif** =
  `compose_system_prompt(conversation.system_prompt, [load_skill(...) for name in active])`, et le
  passe à `client.stream_chat`. (Le system prompt passé au modèle inclut donc les skills actifs.)

### 2.6 UI (`loom/web/templates/index.html`)
- Un **panneau « Skills »** (repliable) listant les skills dispo (case + nom + description en
  tooltip). Cocher/décocher → `POST /skills` (htmx). Les skills actifs sont visuellement marqués.

## 3. Flux
```
Utilisateur dépose loom/skills/dagster/SKILL.md
   → UI liste "dagster" → il coche la case → POST /skills → conversation.active_skills=["dagster"] + save
   → à chaque /chat : system prompt = base + contenu de dagster/SKILL.md → le modèle raisonne AVEC l'archi
```

## 4. Tests
- `skills.py` : `list_skills` (frontmatter parsé, fallback nom de dossier, dossier sans SKILL.md
  ignoré), `compose_system_prompt` (base + corps des skills actifs).
- `conversation.py` : `active_skills` round-trip save/load ; `load` d'un ancien JSON sans le champ.
- `web` : `GET /` liste les skills ; `POST /skills` met à jour la conversation ; `/chat` (client
  mocké) → le `system_prompt` reçu par le client contient bien le corps du skill actif.

## 5. Robustesse
- Skill introuvable / `SKILL.md` illisible → ignoré proprement (pas de crash).
- Pas de skill actif → comportement identique à avant (base system prompt seul).
- `active_skills` référençant un skill supprimé → ignoré au chargement.

## 6. Limite assumée (→ v3.3)
Un gros skill = beaucoup de tokens ; sur 4096 de contexte ça sature. v3.2 n'optimise pas (injection
complète). Le **budget de contexte / résumé** est traité en **v3.3**. RAG = roadmap ultérieure.

## 7. Dépendances
Aucune nouvelle (mini-parseur frontmatter maison, pas de PyYAML).
