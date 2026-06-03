# Loom v3.3 — Workflow A : Robustesse + Contexte — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Durcir le harness : config enrichie, save atomique, gestion de contexte par résumé auto, client résilient (max_tokens, timeout, retries), concurrence (verrou) et réponses vides.

**Architecture:** On étend l'existant. `context.py` (nouveau) estime les tokens et résume les vieux messages. `client.py` gagne max_tokens + résilience via le SDK openai. `web/app.py` ajoute un verrou et l'intégration contexte.

**Tech Stack:** Python 3.12+, `uv`, `pytest`, openai, flask (threading/os stdlib).

**Spec:** [docs/superpowers/specs/2026-05-31-loom-hardening-design.md](../specs/2026-05-31-loom-hardening-design.md) (§1)

> Projet hors git : sauter les étapes git. Tests sans serveur/modèle réel (tout mocké).

---

## Task 1: Config `[chat]` enrichie

**Files:** `loom/loom.config.toml`, `loom/config.py`, `tests/test_config.py`

- [ ] **Step 1: Ajouter dans `[chat]` de `loom/loom.config.toml`** (sous `skills_dir`)

```toml
max_tokens = 2048
request_timeout = 120
max_retries = 6
context_token_budget = 3000
keep_recent_messages = 6
```

- [ ] **Step 2: Test qui échoue** — ajouter dans `tests/test_config.py`

```python
def test_chat_robustness_defaults(tmp_path):
    cfg_path = _write(tmp_path, "loom.config.toml", BASE)
    cfg = load_config(cfg_path)
    assert cfg.chat.max_tokens == 2048
    assert cfg.chat.request_timeout == 120
    assert cfg.chat.max_retries == 6
    assert cfg.chat.context_token_budget == 3000
    assert cfg.chat.keep_recent_messages == 6
```

- [ ] **Step 3: Vérifier l'échec** — `uv run pytest tests/test_config.py::test_chat_robustness_defaults -v` → FAIL (AttributeError).

- [ ] **Step 4: Modifier `loom/config.py`** — ajouter à `ChatConfig` (après `skills_dir`) :

```python
    max_tokens: int = 2048
    request_timeout: int = 120
    max_retries: int = 6
    context_token_budget: int = 3000
    keep_recent_messages: int = 6
```

Et dans `load_config`, dans la construction `ChatConfig(...)`, ajouter :
```python
        max_tokens=int(ch.get("max_tokens", 2048)),
        request_timeout=int(ch.get("request_timeout", 120)),
        max_retries=int(ch.get("max_retries", 6)),
        context_token_budget=int(ch.get("context_token_budget", 3000)),
        keep_recent_messages=int(ch.get("keep_recent_messages", 6)),
```

- [ ] **Step 5: Vérifier** — `uv run pytest tests/test_config.py -v` → PASS.

- [ ] **Step 6: Commit** — `git add ...; git commit -m "feat(config): params robustesse [chat]"`

---

## Task 2: Save atomique

**Files:** `loom/conversation.py`, `tests/test_conversation.py`

- [ ] **Step 1: Test qui échoue** — ajouter dans `tests/test_conversation.py`

```python
def test_save_is_atomic_no_tmp_residue(tmp_path):
    path = tmp_path / "conv.json"
    conv = Conversation(system_prompt="sys")
    conv.add("user", "x")
    conv.save(path)
    assert path.exists()
    # aucun fichier .tmp résiduel
    assert list(tmp_path.glob("*.tmp")) == []
    loaded = Conversation.load(path, default_system_prompt="d")
    assert loaded.messages == [{"role": "user", "content": "x"}]
```

- [ ] **Step 2: Vérifier** — le test passe peut-être déjà (pas de .tmp aujourd'hui) ; on rend la save atomique quand même. `uv run pytest tests/test_conversation.py::test_save_is_atomic_no_tmp_residue -v`.

- [ ] **Step 3: Rendre `save` atomique dans `loom/conversation.py`**

Ajouter `import os` en haut. Remplacer le corps de `save` par :
```python
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "active_skills": self.active_skills,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
```

- [ ] **Step 4: Vérifier** — `uv run pytest tests/test_conversation.py -v` → PASS (tous).

- [ ] **Step 5: Commit** — `git commit -m "feat(conversation): save atomique (tmp+os.replace)"`

---

## Task 3: `context.py` — résumé automatique

**Files:** `loom/context.py` 🆕, `tests/test_context.py` 🆕

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_context.py
from loom.context import estimate_tokens, conversation_tokens, needs_summary, summarize
from loom.conversation import Conversation


def test_estimate_tokens_roughly_quarter_length():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 1


def test_conversation_tokens_counts_text_and_images():
    msgs = [
        {"role": "user", "content": "a" * 40},
        {"role": "user", "content": [
            {"type": "text", "text": "b" * 40},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]},
    ]
    # 10 (sys) + 10 + 10 + 1000 (image)
    assert conversation_tokens("s" * 40, msgs) == 1030


def test_needs_summary_threshold():
    msgs = [{"role": "user", "content": "a" * 40}]
    assert needs_summary("", msgs, budget=5) is True
    assert needs_summary("", msgs, budget=100) is False


class FakeClient:
    def stream_chat(self, messages, system_prompt):
        yield ("content", "RESUME_COURT")


def test_summarize_replaces_old_keeps_recent():
    conv = Conversation(system_prompt="sys")
    for i in range(10):
        conv.add("user", "x" * 40)  # 10 messages, chacun ~10 tokens
    changed = summarize(conv, FakeClient(), budget=20, keep_recent=3)
    assert changed is True
    # 1 résumé + 3 récents = 4 messages
    assert len(conv.messages) == 4
    assert "RESUME_COURT" in conv.messages[0]["content"]
    assert conv.messages[0]["role"] == "user"


def test_summarize_noop_when_under_budget():
    conv = Conversation(system_prompt="sys")
    conv.add("user", "court")
    assert summarize(conv, FakeClient(), budget=10000, keep_recent=3) is False
    assert len(conv.messages) == 1
```

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_context.py -v` → FAIL (module absent).

- [ ] **Step 3: Implémenter `loom/context.py`**

```python
# loom/context.py
"""Gestion de la fenêtre de contexte : estimation de tokens + résumé automatique."""
from __future__ import annotations

SUMMARY_SYSTEM = "Tu résumes des conversations de façon concise et fidèle, en français."
SUMMARY_INSTRUCTION = (
    "Résume la conversation suivante en quelques phrases, en gardant les faits, décisions "
    "et informations importantes. Voici la conversation :\n\n"
)


def estimate_tokens(text: str) -> int:
    """Estimation grossière : ~1 token pour 4 caractères (min 1)."""
    return max(1, len(text) // 4)


def _content_tokens(content) -> int:
    if isinstance(content, str):
        return estimate_tokens(content)
    total = 0
    for part in content:
        if part.get("type") == "text":
            total += estimate_tokens(part.get("text", ""))
        elif part.get("type") == "image_url":
            total += 1000  # coût visuel approximatif d'une image
    return total


def conversation_tokens(system_prompt: str, messages: list[dict]) -> int:
    total = estimate_tokens(system_prompt)
    for m in messages:
        total += _content_tokens(m["content"])
    return total


def needs_summary(system_prompt: str, messages: list[dict], budget: int) -> bool:
    return conversation_tokens(system_prompt, messages) > budget


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            text = c
        else:
            text = " ".join(
                p.get("text", "[image]") for p in c
            )
        lines.append(f"{m['role']}: {text}")
    return "\n".join(lines)


def summarize(conversation, client, budget: int, keep_recent: int) -> bool:
    """Si au-dessus du budget, remplace les vieux messages par un résumé. Renvoie True si résumé."""
    msgs = conversation.messages
    if not needs_summary(conversation.system_prompt, msgs, budget):
        return False
    if len(msgs) <= keep_recent:
        return False
    old, recent = msgs[:-keep_recent], msgs[-keep_recent:]
    prompt = SUMMARY_INSTRUCTION + _render(old)
    summary = "".join(
        text
        for kind, text in client.stream_chat(
            [{"role": "user", "content": prompt}], SUMMARY_SYSTEM
        )
        if kind == "content"
    )
    conversation.messages = [
        {"role": "user", "content": f"[Résumé de la conversation précédente : {summary}]"},
        *recent,
    ]
    return True
```

- [ ] **Step 4: Vérifier** — `uv run pytest tests/test_context.py -v` → PASS (6 tests).

- [ ] **Step 5: Commit** — `git commit -m "feat(context): resume automatique de la fenetre de contexte"`

---

## Task 4: `client.py` — max_tokens + résilience

**Files:** `loom/client.py`, `tests/test_client.py`

- [ ] **Step 1: Test qui échoue** — ajouter dans `tests/test_client.py`

```python
from loom.client import build_create_kwargs


def test_build_create_kwargs_includes_max_tokens_and_system():
    kw = build_create_kwargs(
        model="local",
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="SYS",
        max_tokens=777,
    )
    assert kw["model"] == "local"
    assert kw["max_tokens"] == 777
    assert kw["stream"] is True
    assert kw["messages"][0] == {"role": "system", "content": "SYS"}
    assert kw["messages"][1] == {"role": "user", "content": "hi"}


def test_loom_client_stores_resilience_params():
    c = LoomClient(base_url="http://x/v1", timeout=99, max_retries=4)
    assert c.timeout == 99
    assert c.max_retries == 4
```

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_client.py::test_build_create_kwargs_includes_max_tokens_and_system -v` → FAIL (import).

- [ ] **Step 3: Modifier `loom/client.py`**

Ajouter la fonction pure (avant la classe) :
```python
def build_create_kwargs(
    model: str, messages: list[dict], system_prompt: str, max_tokens: int
) -> dict:
    return {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": True,
        "max_tokens": max_tokens,
    }
```

Modifier `LoomClient.__init__` pour la résilience :
```python
    def __init__(
        self,
        base_url: str,
        api_key: str = "loom-local",
        model: str = "local",
        timeout: int = 120,
        max_retries: int = 6,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url, api_key=api_key,
            timeout=timeout, max_retries=max_retries,
        )
```

Modifier `stream_chat` pour accepter `max_tokens` :
```python
    def stream_chat(
        self, messages: list[dict], system_prompt: str, max_tokens: int = 2048
    ) -> Iterator[tuple[str, str]]:
        kwargs = build_create_kwargs(self.model, messages, system_prompt, max_tokens)
        stream = self._client.chat.completions.create(**kwargs)
        yield from _iter_events(stream)
```

> Note résilience 503/connexion : on s'appuie sur le retry/backoff **natif du SDK openai**
> (`max_retries=6`) + `timeout` généreux — couvre le « Loading model » sans code custom.

- [ ] **Step 4: Vérifier** — `uv run pytest tests/test_client.py -v` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(client): max_tokens + timeout/max_retries (resilience SDK)"`

---

## Task 5: `web/app.py` — verrou, contexte, réponse vide, max_tokens

**Files:** `loom/web/app.py`, `loom/web/__main__.py`, `tests/test_web.py`

- [ ] **Step 1: MAJ tests** — dans `tests/test_web.py`, adapter `_make` pour passer les nouveaux
  paramètres et exposer le lock, puis ajouter les tests :

```python
def _make(tmp_path, events=(("content", "Hel"), ("content", "lo")), budget=100000):
    conv = Conversation(system_prompt="sys")
    history = tmp_path / "conv.json"
    skills_dir = tmp_path / "skills"
    (skills_dir / "dagster").mkdir(parents=True)
    (skills_dir / "dagster" / "SKILL.md").write_text(
        "---\nname: dagster\ndescription: archi\n---\nARCHI_DAGSTER_XYZ", encoding="utf-8"
    )
    fake = FakeClient(list(events))
    app = create_app(conv, fake, history, skills_dir,
                     max_tokens=2048, context_budget=budget, keep_recent=3)
    app.config["_fake_client"] = fake
    return app, conv, history


def test_chat_busy_returns_429(tmp_path):
    app, _, _ = _make(tmp_path)
    app.config["_chat_lock"].acquire()
    try:
        resp = app.test_client().post("/chat", data={"message": "x"})
        assert resp.status_code == 429
    finally:
        app.config["_chat_lock"].release()


def test_chat_empty_answer_placeholder(tmp_path):
    app, conv, _ = _make(tmp_path, events=[("reasoning", "je pense")])  # aucun 'content'
    resp = app.test_client().post("/chat", data={"message": "x"})
    body = resp.get_data(as_text=True)
    assert "seulement réfléchi" in body
    assert conv.messages[1]["role"] == "assistant"
    assert conv.messages[1]["content"] != ""
```

**IMPORTANT** — mettre à jour `FakeClient.stream_chat` pour accepter `max_tokens` (le `/chat` v3.3
le passe, et `context.summarize` l'appelle sans → param optionnel) :
```python
class FakeClient:
    def __init__(self, events):
        self._events = events
        self.last_system_prompt = None

    def stream_chat(self, messages, system_prompt, max_tokens=2048):
        self.last_system_prompt = system_prompt
        yield from self._events
```
(Garder les autres tests existants.)

- [ ] **Step 2: Vérifier l'échec** — `uv run pytest tests/test_web.py::test_chat_busy_returns_429 -v` → FAIL.

- [ ] **Step 3: Modifier `loom/web/app.py`**

Ajouter les imports :
```python
import threading
from loom import context
```

Signature et lock (début de `create_app`) :
```python
def create_app(
    conversation, client, history_path, skills_dir, *,
    max_tokens=2048, context_budget=3000, keep_recent=6,
) -> Flask:
    app = Flask(__name__)
    history_path = str(history_path)
    skills_dir = str(skills_dir)
    chat_lock = threading.Lock()
    app.config["_chat_lock"] = chat_lock
```

Réécrire `/chat` :
```python
    @app.post("/chat")
    def chat():
        message = (request.form.get("message") or "").strip()
        if not message or len(message) > 5000:
            return Response("message invalide", status=400)
        if not chat_lock.acquire(blocking=False):
            return Response("occupé : un échange est déjà en cours", status=429)

        try:
            image = request.files.get("image")
            if image and image.filename:
                blob = image.read()
                if len(blob) > 10 * 1024 * 1024:
                    chat_lock.release()
                    return Response("image trop grande", status=400)
                mime = image.mimetype or "image/png"
                if not mime.startswith("image/"):
                    chat_lock.release()
                    return Response("fichier non-image", status=400)
                b64 = base64.b64encode(blob).decode("ascii")
                content = [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]
            else:
                content = message

            conversation.add("user", content)
            conversation.save(history_path)

            # Gestion du contexte : résumé auto si trop long
            if context.summarize(conversation, client, context_budget, keep_recent):
                conversation.save(history_path)

            active = [
                s for s in (load_skill(skills_dir, n) for n in conversation.active_skills) if s
            ]
            system_prompt = compose_system_prompt(conversation.system_prompt, active)
        except Exception as exc:  # noqa: BLE001
            chat_lock.release()
            return Response(f"erreur: {exc}", status=500)

        def generate():
            answer = ""
            try:
                for kind, text in client.stream_chat(
                    conversation.to_messages(), system_prompt, max_tokens
                ):
                    if kind == "reasoning":
                        yield f"data: {json.dumps({'type': 'reasoning', 'text': text})}\n\n"
                    else:
                        answer += text
                        yield f"data: {json.dumps({'type': 'text', 'text': text})}\n\n"
                if not answer.strip():
                    answer = "(le modèle a seulement réfléchi — augmente max_tokens)"
                    yield f"data: {json.dumps({'type': 'text', 'text': answer})}\n\n"
                conversation.add("assistant", answer)
                conversation.save(history_path)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            finally:
                chat_lock.release()

        return Response(generate(), mimetype="text/event-stream")
```

- [ ] **Step 4: MAJ `loom/web/__main__.py`** — passer les paramètres robustesse :

```python
    client = LoomClient(
        base_url=base_url, timeout=cfg.chat.request_timeout, max_retries=cfg.chat.max_retries
    )
    app = create_app(
        conversation, client, cfg.chat.history_path, cfg.chat.skills_dir,
        max_tokens=cfg.chat.max_tokens,
        context_budget=cfg.chat.context_token_budget,
        keep_recent=cfg.chat.keep_recent_messages,
    )
```

- [ ] **Step 5: Vérifier** — `uv run pytest tests/test_web.py -v` → PASS, puis suite complète `uv run pytest -q` → PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat(web): verrou 429 + contexte + reponse vide + max_tokens"`

---

## Task 6: Vérification finale

- [ ] **Step 1:** `uv run pytest -q` → tous verts.

## Definition of Done (Workflow A)
- [ ] Config robustesse chargée ; save atomique ; `context.py` testé (résumé) ; client max_tokens +
  timeout/retries ; `/chat` : 429 si occupé, placeholder si réponse vide, résumé auto intégré.
- [ ] `uv run pytest` tout vert.
