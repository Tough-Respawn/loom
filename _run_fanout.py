"""Run ponctuel : génère le morpion via fan-out parallèle, écrit + vérifie. Jetable."""

import time
from pathlib import Path

from loom.client import LoomClient
from loom.parallel import generate_files, plan_files
from loom.tools.fs import make_write_file
from loom.verify import format_report, verify_files

MODEL = "gemma-4-E4B-it-uncensored.Q4_K_M.gguf"
WS = Path("C:/Users/Amine/Documents/tictactoo-fanout")
WS.mkdir(parents=True, exist_ok=True)

TASK = (
    "Cree un jeu de morpion (tic-tac-toe) jouable dans un navigateur web : index.html "
    "+ style.css + le JavaScript necessaire. Pleinement jouable : 2 joueurs au tour par "
    "tour, detection de victoire (lignes/colonnes/diagonales) et de match nul, affichage "
    "du gagnant, bouton rejouer."
)

client = LoomClient(base_url="http://127.0.0.1:8080/v1", model=MODEL)

t0 = time.monotonic()
design, specs = plan_files(client, TASK, model=MODEL)
t1 = time.monotonic()
print(f"[plan] {t1 - t0:.1f}s -> {len(specs)} fichiers : {[s.path for s in specs]}")
print(f"[design] {design[:200]}...")

files = generate_files(
    client, design, specs, model=MODEL, max_tokens=2048, max_workers=4
)
t2 = time.monotonic()
print(f"[gen parallele] {t2 - t1:.1f}s pour {len(files)} fichiers")

write = make_write_file(str(WS), max_bytes=200000).run


def write_all(pairs):
    paths = []
    for path, content in pairs:
        res = write({"path": path, "content": content})
        print(f"  {path}: {len(content)} car -> {res[:60]}")
        paths.append(str((WS / path).resolve()))
    return paths


written = write_all(files)
report = verify_files(written)
print(f"\n[verify] {format_report(report)}")

# Boucle fermée : tant que le verificateur trouve des defauts, on REGENERE en
# parallele les fichiers en leur donnant tous les fichiers actuels + les defauts.
rounds = 0
while not report.ok and rounds < 3:
    rounds += 1
    print(f"\n[fix round {rounds}] correction parallele sur defauts...")
    current = [(p, (WS / p).read_text(encoding="utf-8")) for p, _ in files]
    files = fix_files(
        client, design, specs, current, format_report(report), model=MODEL
    )
    written = write_all(files)
    report = verify_files(written)
    print(f"[verify] {format_report(report)}")

t3 = time.monotonic()
print(
    f"\n=== TOTAL {t3 - t0:.1f}s (plan {t1 - t0:.1f}s + gen {t2 - t1:.1f}s + {rounds} fix) ==="
)
print(f"OK={report.ok}")
