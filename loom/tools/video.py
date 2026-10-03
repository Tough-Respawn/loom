"""Outil watch_video : ce qu'une vidéo DIT (parole), jamais ce qu'elle MONTRE.

Choix assumé (2026-10-03) : pas d'images. Le sens d'une conférence, d'un tuto commenté
ou d'un podcast passe par la parole ; les images coûtent cher en contexte local et
exigent un modèle vision. Repris de claude-real-video (crv) : sous-titres existants
d'abord, Whisper seulement en repli, VAD Silero contre les légendes inventées sur la
musique/le silence. Laissé de côté : extraction d'images, dédup de scènes.

Chaîne :
1. yt-dlp lit les métadonnées SANS télécharger la vidéo (titre, durée, chapitres) ;
2. sous-titres existants (manuels > auto dans la langue d'origine > auto traduits) ;
3. sinon : audio seul + faster-whisper sur CPU (la VRAM reste au LLM).
Un fichier local passe directement à l'étape 3 (sauf sous-titre .srt/.vtt voisin).

Le transcript complet est mis en cache sous var/cache/videos : un 2e appel ne refait
pas Whisper, et un transcript trop long pour un résultat d'outil se lit par morceaux
via read_file."""

from __future__ import annotations

import html
import re
import tempfile
import threading
from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.trust import untrusted

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "var" / "cache" / "videos"
# small multilingue : bon compromis FR/EN sur CPU (int8). medium = ~2,5x plus lent.
WHISPER_MODEL = "small"
# Fenêtre de regroupement des cues en paragraphes horodatés (tokens économisés :
# un horodatage par paragraphe, pas par ligne de sous-titre).
PARAGRAPH_SECONDS = 30
DEFAULT_MAX_CHARS = 24000
_SUB_EXTS = ("vtt", "srt")

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]*>")


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_ts(raw: str) -> float | None:
    m = _TS.search(raw)
    if not m:
        return None
    h, mi, s, frac = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000


def parse_subtitles(text: str) -> list[tuple[float, str]]:
    """VTT ou SRT -> [(début_s, ligne)], sans balises ni doublons.

    Les sous-titres auto de YouTube « roulent » : chaque cue répète la ligne
    précédente puis ajoute la suivante. On n'émet une ligne que si elle diffère
    des deux dernières émises."""
    cues: list[tuple[float, str]] = []
    recent: list[str] = []
    for block in re.split(r"\r?\n\s*\r?\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        arrow = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if arrow is None:
            continue  # en-tête WEBVTT, NOTE, STYLE
        start = _parse_ts(lines[arrow].split("-->")[0])
        if start is None:
            continue
        for ln in lines[arrow + 1 :]:
            clean = " ".join(html.unescape(_TAG.sub("", ln)).split())
            if not clean or clean in recent:
                continue
            cues.append((start, clean))
            recent = (recent + [clean])[-2:]
    return cues


def group_paragraphs(
    cues: list[tuple[float, str]], window: int = PARAGRAPH_SECONDS
) -> list[str]:
    """Regroupe les cues en lignes « [mm:ss] texte » d'environ `window` secondes."""
    out: list[str] = []
    start: float | None = None
    buf: list[str] = []
    for t, line in cues:
        if start is None:
            start = t
        elif t - start >= window:
            out.append(f"[{_fmt_ts(start)}] {' '.join(buf)}")
            start, buf = t, []
        buf.append(line)
    if buf and start is not None:
        out.append(f"[{_fmt_ts(start)}] {' '.join(buf)}")
    return out


def _lang_match(key: str, lang: str) -> bool:
    k = key.lower()
    return k == lang or k.startswith(lang + "-")


def pick_track(info: dict, lang: str | None) -> tuple[str, dict, str] | None:
    """Choisit la piste de sous-titres : (clé_langue, format, nature) ou None.

    Ordre : manuels dans `lang` > manuels dans la langue d'origine > auto d'origine
    (plus fidèles qu'une traduction automatique) > manuels quelconques > auto
    traduits dans `lang`. Le modèle traduit mieux lui-même qu'un auto traduit."""
    manual = {
        k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"
    }
    auto = info.get("automatic_captions") or {}
    orig = (info.get("language") or "").lower()
    lang = (lang or "").lower().strip()

    def fmt(entries):
        for ext in _SUB_EXTS:
            for e in entries or []:
                if e.get("ext") == ext and e.get("url"):
                    return e
        return None

    def first(tracks, pred, nature):
        for key, entries in tracks.items():
            if pred(key) and (e := fmt(entries)):
                return key, e, nature
        return None

    candidates = []
    if lang:
        candidates.append((manual, lambda k: _lang_match(k, lang), "manuels"))
    if orig:
        candidates.append((manual, lambda k: _lang_match(k, orig), "manuels"))
    candidates.append((auto, lambda k: k.lower().endswith("-orig"), "automatiques"))
    if orig:
        candidates.append((auto, lambda k: k.lower() == orig, "automatiques"))
    candidates.append((manual, lambda k: True, "manuels"))
    if lang:
        candidates.append(
            (auto, lambda k: _lang_match(k, lang), "automatiques traduits")
        )
    for tracks, pred, nature in candidates:
        if hit := first(tracks, pred, nature):
            return hit
    return None


_whisper_lock = threading.Lock()
_whisper_model = None


def _transcribe(path: Path) -> tuple[list[tuple[float, str]], str]:
    """Transcrit un fichier audio/vidéo en local : (cues, langue détectée)."""
    global _whisper_model
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise ToolError(
            "faster-whisper absent : lance `uv sync` dans le dossier de Loom"
        ) from exc
    # Un seul modèle chargé, une transcription à la fois : le CPU est déjà saturé.
    # Batch 8 + beam 1 mesuré à 6,9x le temps réel (contre 2,9x en beam 5 séquentiel,
    # texte identique) sur 3 min d'audio, 12 threads (2026-10-03). Le VAD Silero est
    # actif par défaut en mode batch : pas de légende inventée sur musique/silence.
    with _whisper_lock:
        if _whisper_model is None:
            _whisper_model = BatchedInferencePipeline(
                model=WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            )
        segments, info = _whisper_model.transcribe(str(path), batch_size=8, beam_size=1)
        cues = [(s.start, s.text.strip()) for s in segments if s.text.strip()]
    return cues, info.language


def _ydl(extra: dict | None = None):
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": 20,
        "noprogress": True,
    }
    opts.update(extra or {})
    return yt_dlp.YoutubeDL(opts)


def _ydl_error(exc: Exception) -> ToolError:
    msg = " ".join(str(exc).replace("ERROR:", "").split())[:400]
    hint = ""
    if "Sign in" in msg or "bot" in msg:
        hint = " (YouTube bloque l'accès anonyme à cette vidéo)"
    elif "403" in msg or "Unsupported" in msg:
        hint = " (yt-dlp est peut-être périmé : `uv lock --upgrade-package yt-dlp`)"
    return ToolError(f"lecture de la vidéo impossible : {msg}{hint}")


def _cache_path(key: str) -> Path:
    return CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:120] + ".txt")


def _render(header: list[str], paragraphs: list[str]) -> str:
    return "\n".join(header + ["--- Transcript ---"] + paragraphs) + "\n"


def _from_url(url: str, lang: str | None) -> tuple[str, str]:
    """(texte_complet, clé_cache) pour une URL de plateforme vidéo."""
    from yt_dlp.utils import DownloadError

    try:
        with _ydl() as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise _ydl_error(exc) from exc
    if info.get("_type") == "playlist":
        raise ToolError("c'est une playlist : passe l'URL d'UNE vidéo")
    # La langue demandée change la piste choisie : elle fait partie de la clé.
    key = f"{info.get('extractor_key', 'video')}-{info.get('id', 'x')}-{lang or 'orig'}"
    cache = _cache_path(key)
    if cache.exists():
        return cache.read_text(encoding="utf-8"), key

    header = [f"Vidéo : {info.get('title') or '?'}"]
    meta = [
        f"Chaîne : {info.get('uploader') or info.get('channel') or '?'}",
        f"Durée : {_fmt_ts(info['duration'])}" if info.get("duration") else "",
        _fmt_date(info.get("upload_date")),
        info.get("webpage_url") or url,
    ]
    header.append(" | ".join(m for m in meta if m))
    if chapters := info.get("chapters"):
        header.append("Chapitres :")
        header += [
            f"  [{_fmt_ts(c.get('start_time') or 0)}] {c.get('title', '')}"
            for c in chapters
        ]
    if desc := (info.get("description") or "").strip():
        short = " ".join(desc.split())
        header.append(
            f"Description : {short[:600]}{' […]' if len(short) > 600 else ''}"
        )

    paragraphs: list[str] = []
    track = pick_track(info, lang)
    if track:
        key_lang, entry, nature = track
        try:
            with _ydl() as ydl:
                raw = ydl.urlopen(entry["url"]).read().decode("utf-8", "replace")
            paragraphs = group_paragraphs(parse_subtitles(raw))
        except Exception:  # noqa: BLE001 - repli Whisper ci-dessous
            paragraphs = []
        if paragraphs:
            header.append(f"Source du transcript : sous-titres {nature} ({key_lang})")
    if not paragraphs:
        cues, detected = _transcribe_remote(url)
        paragraphs = group_paragraphs(cues)
        header.append(
            "Source du transcript : transcription locale Whisper "
            f"(aucun sous-titre ; langue détectée : {detected})"
        )
    text = _render(header, paragraphs or ["(aucune parole détectée)"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text, key


def _transcribe_remote(url: str) -> tuple[list[tuple[float, str]], str]:
    """Télécharge l'AUDIO seul (pas la vidéo, pas de ffmpeg) puis le transcrit."""
    from yt_dlp.utils import DownloadError

    with tempfile.TemporaryDirectory(prefix="loom-video-") as tmp:
        try:
            with _ydl(
                {
                    "skip_download": False,
                    "format": "bestaudio/best",
                    "outtmpl": str(Path(tmp) / "audio.%(ext)s"),
                }
            ) as ydl:
                ydl.extract_info(url, download=True)
        except DownloadError as exc:
            raise _ydl_error(exc) from exc
        files = [p for p in Path(tmp).iterdir() if p.is_file()]
        if not files:
            raise ToolError("téléchargement de l'audio vide")
        return _transcribe(files[0])


def _fmt_date(raw: str | None) -> str:
    if raw and len(raw) == 8 and raw.isdigit():
        return f"Publiée : {raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return ""


def _from_file(path: Path) -> tuple[str, str]:
    """(texte_complet, clé_cache) pour un fichier vidéo/audio local."""
    if not path.is_file():
        raise ToolError(f"fichier introuvable : {path}")
    st = path.stat()
    key = f"file-{path.stem}-{st.st_size}-{int(st.st_mtime)}"
    cache = _cache_path(key)
    if cache.exists():
        return cache.read_text(encoding="utf-8"), key
    header = [f"Fichier : {path}"]
    sidecar = next(
        (
            path.with_suffix(f".{ext}")
            for ext in _SUB_EXTS
            if path.with_suffix(f".{ext}").is_file()
        ),
        None,
    )
    paragraphs: list[str] = []
    if sidecar:
        paragraphs = group_paragraphs(
            parse_subtitles(sidecar.read_text(encoding="utf-8", errors="replace"))
        )
        if paragraphs:
            header.append(f"Source du transcript : sous-titres {sidecar.name}")
    if not paragraphs:
        cues, detected = _transcribe(path)
        paragraphs = group_paragraphs(cues)
        header.append(
            f"Source du transcript : transcription locale Whisper (langue : {detected})"
        )
    text = _render(header, paragraphs or ["(aucune parole détectée)"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text, key


def watch_video(
    source: str,
    workspace_dir: str,
    lang: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    source = (source or "").strip()
    if not source:
        raise ToolError("argument 'url' manquant")
    if re.match(r"^https?://", source, re.IGNORECASE):
        from loom.tools.web import _blocked_host_reason

        if blocked := _blocked_host_reason(source):
            raise ToolError(blocked)
        text, key = _from_url(source, lang)
        origin = f"vidéo {source}"
    else:
        path = _resolve_in_root(Path(workspace_dir), source)
        text, key = _from_file(path)
        origin = f"fichier vidéo {path}"
    if len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        shown = text[: cut if cut > 0 else max_chars]
        line = shown.count("\n") + 1
        text = (
            f"{shown}\n[… transcript tronqué ({len(text)} caractères au total). "
            f'Suite complète : read_file(path="{_cache_path(key).as_posix()}", '
            f"start_line={line}).]"
        )
    return untrusted(text, origin)


def make_watch_video(workspace_dir: str) -> ToolSpec:
    return ToolSpec(
        name="watch_video",
        description=(
            "Reads what a video SAYS: returns title, duration, chapters and a "
            "timestamped transcript of the speech, from a URL (YouTube, Vimeo, "
            "TikTok, X, Twitch, Dailymotion... any yt-dlp site) or a local "
            "video/audio file. Uses the video's own subtitles, else transcribes "
            "the audio locally with Whisper (can take minutes on a long video). "
            "It does NOT see the images: say so if the question is about what is "
            "shown on screen. Use it whenever the user shares a video link."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Video URL, or path to a local video/audio file.",
                },
                "lang": {
                    "type": "string",
                    "description": (
                        "Optional preferred subtitle language code (e.g. 'fr'). "
                        "Default: the video's original language."
                    ),
                },
            },
            "required": ["url"],
        },
        run=lambda args: watch_video(
            args.get("url") or "",
            workspace_dir,
            lang=args.get("lang"),
        ),
    )
