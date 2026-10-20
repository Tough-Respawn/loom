"""Images produites par le modèle dans le bac à sable de la session, montrées dans le fil.

Vécu 2026-10-19 (session dc23325088d6) : le modèle a décrit « une fractale avec sa
légende » pour une image qu'il n'avait jamais regardée (bande brune sur fond noir).
Tant que l'utilisateur ne voit pas l'image, il doit croire le modèle sur parole. Ici,
après chaque résultat d'outil, le bac à sable est balayé : tout fichier image nouveau
ou modifié depuis le début du tour devient un événement `artifact` (persisté, rejouable),
que le fil affiche tel quel. Un `.png` qui contient du texte s'affiche cassé : c'est
l'information.
"""

from __future__ import annotations

from pathlib import Path

#: Ce que le fil sait afficher dans une balise <img> (le SVG y est inerte : pas de script).
ARTIFACT_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})


def artifact_url(sid: str, name: str) -> str:
    return f"/session/{sid}/scratch/{name}"


def new_artifacts(scratch_dir: str, since: float, seen: set, sid: str) -> list[dict]:
    """Images du bac à sable écrites ou modifiées depuis `since` (horodatage) et pas
    encore annoncées (`seen` : clés nom@mtime, mis à jour ici). Jamais d'exception :
    un balayage raté ne doit pas casser un tour."""
    out: list[dict] = []
    try:
        root = Path(scratch_dir)
        if not root.is_dir():
            return out
        for p in sorted(root.iterdir(), key=lambda q: q.name):
            if not p.is_file() or p.suffix.lower() not in ARTIFACT_EXTS:
                continue
            st = p.stat()
            if st.st_mtime < since:
                continue
            key = f"{p.name}@{int(st.st_mtime_ns)}"
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {"name": p.name, "url": artifact_url(sid, p.name), "size": st.st_size}
            )
    except OSError:
        return out
    return out
