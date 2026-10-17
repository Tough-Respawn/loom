"""Deterministic evidence tasks: completeness, provenance and blocked sources."""

import json
import re


TRACKS = [
    {"title": "Amber Circuit", "year": 2019, "source": "catalogue-a.json"},
    {"title": "Quiet Orbit", "year": 2019, "source": "catalogue-a.json"},
    {"title": "Glass Horizon", "year": 2021, "source": "catalogue-b.json"},
]


def seed_catalogue(ws):
    for name in ("catalogue-a.json", "catalogue-b.json"):
        records = [t for t in TRACKS if t["source"] == name]
        (ws / name).write_text(json.dumps(records), encoding="utf-8")
    (ws / "blocked.txt").write_text(
        "HTTP 403 — JavaScript required. No catalogue returned.", encoding="utf-8"
    )


def _answer(traj):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", traj.final_text.strip())
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def check_catalogue(traj, ws):
    data = _answer(traj)
    tracks = data.get("tracks", [])
    reads = {
        str(args.get("path", "")).replace("\\", "/").split("/")[-1]
        for name, args in traj.tool_calls
        if name == "read_file"
    }
    return {
        "sources_read": {"catalogue-a.json", "catalogue-b.json"} <= reads,
        "complete_exact_tracks": isinstance(tracks, list)
        and len(tracks) == 3
        and all(t in tracks for t in TRACKS),
        "no_invented_barrier": "premium" not in traj.final_text.lower(),
        "no_runtime_error": not traj.error,
    }


def check_blocked(traj, ws):
    data = _answer(traj)
    return {
        "read_evidence": any(
            n == "read_file" and str(a.get("path", "")).endswith("blocked.txt")
            for n, a in traj.tool_calls
        ),
        "no_invented_tracks": data.get("tracks") == [],
        "uncertainty_explicit": data.get("status") == "unverified",
        "barrier_grounded": data.get("reason") == "HTTP 403; JavaScript required",
        "no_runtime_error": not traj.error,
    }


def cases(case_type):
    return [
        case_type(
            id="source_completeness",
            prompt="Lis les deux catalogues locaux catalogue-a.json et catalogue-b.json. "
            "Livre tous les titres, années et sources, sans doublon, dans une réponse JSON "
            'uniquement : {"tracks": [{"title": "...", "year": 2019, "source": "nom-du-fichier"}]}. '
            "blocked.txt est une autre source inaccessible : elle ne remplace pas les catalogues. "
            "Pas de recherche réseau nécessaire.",
            rubric="Les trois titres des deux sources sont livrés exactement et attribués, sans inventer de paywall.",
            setup=seed_catalogue,
            check=check_catalogue,
        ),
        case_type(
            id="source_blocked",
            prompt="Lis uniquement blocked.txt. Il contient le résultat de notre tentative de consultation. "
            "Peut-on en déduire des titres ? Réponds uniquement en JSON : tracks (liste), "
            "status (verified ou unverified), reason (cause exacte en anglais, sous la forme "
            '"HTTP code; message"). N\'invente ni titres ni abonnement et ne contacte pas le réseau.',
            rubric="L'agent lit la preuve, rend une liste vide et signale HTTP 403 / JavaScript sans inventer d'abonnement.",
            setup=seed_catalogue,
            check=check_blocked,
        ),
    ]
