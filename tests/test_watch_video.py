"""watch_video : analyse des sous-titres, choix de piste, troncature, câblage.
Aucun appel réseau ni Whisper : les parties pures sont testées directement."""

from loom.permissions import READ_TOOLS
from loom.tools import build_registry, video

_ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000 align:start position:0%
hello<00:00:00.500><c> world</c>

00:00:02.000 --> 00:00:02.010 align:start position:0%
hello world

00:00:02.010 --> 00:00:04.000 align:start position:0%
hello world
this is<00:00:03.000><c> next</c>

00:00:31.000 --> 00:00:33.000
this is next
after &amp; later
"""

_SRT = """1
00:00:01,500 --> 00:00:03,000
Bonjour <i>à tous</i>

2
01:02:03,000 --> 01:02:05,000
Fin
"""


def test_vtt_roulant_sans_doublons_ni_balises():
    cues = video.parse_subtitles(_ROLLING_VTT)
    assert [c[1] for c in cues] == [
        "hello world",
        "this is next",
        "after & later",
    ]
    assert cues[0][0] == 0.0 and cues[-1][0] == 31.0


def test_srt_heures_et_virgule():
    cues = video.parse_subtitles(_SRT)
    assert cues == [(1.5, "Bonjour à tous"), (3723.0, "Fin")]


def test_paragraphes_horodates():
    paras = video.group_paragraphs(video.parse_subtitles(_ROLLING_VTT))
    assert paras == ["[00:00] hello world this is next", "[00:31] after & later"]
    assert video.group_paragraphs([(3723.0, "x")]) == ["[1:02:03] x"]


def _track(ext="vtt"):
    return [{"ext": "json3", "url": "j"}, {"ext": ext, "url": f"u-{ext}"}]


def test_piste_manuelle_dans_la_langue_demandee_dabord():
    info = {
        "language": "en",
        "subtitles": {"en": _track(), "fr-FR": _track()},
        "automatic_captions": {"en-orig": _track()},
    }
    key, entry, nature = video.pick_track(info, "fr")
    assert (key, entry["url"], nature) == ("fr-FR", "u-vtt", "manuels")


def test_auto_origine_avant_auto_traduit():
    info = {
        "language": "en",
        "subtitles": {"live_chat": _track()},
        "automatic_captions": {"fr": _track(), "en-orig": _track()},
    }
    key, _, nature = video.pick_track(info, "fr")
    assert (key, nature) == ("en-orig", "automatiques")


def test_auto_traduit_en_dernier_recours():
    info = {"automatic_captions": {"de": _track(), "fr": _track("srt")}}
    key, entry, nature = video.pick_track(info, "fr")
    assert (key, entry["url"], nature) == ("fr", "u-srt", "automatiques traduits")


def test_aucune_piste_exploitable():
    info = {"subtitles": {"en": [{"ext": "json3", "url": "j"}]}}
    assert video.pick_track(info, None) is None


def test_transcript_long_tronque_avec_pointeur_read_file(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "CACHE_DIR", tmp_path)
    body = "\n".join(f"[00:{i:02d}] ligne {i}" for i in range(60))
    monkeypatch.setattr(video, "_from_url", lambda url, lang: (body, "YT-abc-orig"))
    monkeypatch.setattr(
        "loom.tools.web._blocked_host_reason", lambda url: None, raising=True
    )
    out = video.watch_video("https://youtu.be/abc", ".", max_chars=200)
    assert "transcript tronqué" in out
    assert (tmp_path / "YT-abc-orig.txt").as_posix() in out
    assert "start_line=" in out
    assert "FRONTIÈRE DE CONFIANCE" in out


def test_hote_interne_refuse():
    out = build_registry(".", 1000, ["watch_video"]).run(
        "watch_video", {"url": "http://127.0.0.1/v.mp4"}
    )
    assert out.startswith("erreur:") and "interne" in out


def test_fichier_local_absent():
    out = build_registry(".", 1000, ["watch_video"]).run(
        "watch_video", {"url": "nexiste_pas.mp4"}
    )
    assert out.startswith("erreur:") and "introuvable" in out


def test_outil_de_lecture_sans_confirmation():
    assert "watch_video" in READ_TOOLS
