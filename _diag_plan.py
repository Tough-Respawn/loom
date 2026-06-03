"""Diagnostic : montre la sortie BRUTE du planificateur (pour rendre le parseur robuste)."""

from loom import parallel
from loom.client import LoomClient

MODEL = "gemma-4-E4B-it-uncensored.Q4_K_M.gguf"
client = LoomClient(base_url="http://127.0.0.1:8080/v1", model=MODEL)

raw_holder = {}
_orig = client.complete


def cap(*a, **k):
    r = _orig(*a, **k)
    raw_holder["raw"] = r
    return r


client.complete = cap

TASK = (
    "Cree le jeu Snake jouable dans un navigateur : grille, serpent qui avance seul, "
    "controle clavier (fleches), nourriture qui fait grandir + score, game over si "
    "collision mur/soi, touche Espace pour rejouer."
)
try:
    design, specs = parallel.plan_files(client, TASK, model=MODEL)
    print("PARSE OK -> specs:", [s.path for s in specs])
    print("design (300):", design[:300])
except Exception as e:
    print("PARSE ERROR:", repr(e))

print("\n===================== RAW MODEL OUTPUT =====================")
print(raw_holder.get("raw", "(rien)")[:3500])
