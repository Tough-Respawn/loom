// loom/verify_web.js
// Vérificateur RUNTIME DOM (offline, via jsdom) : charge une page HTML + ses scripts,
// capture les erreurs d'exécution (TypeError, ReferenceError…) ET vérifie que le
// plateau se rend réellement (conteneur de grille non vide). C'est l'œil "ça tourne"
// qui manque à `node --check` (lequel ne voit que la syntaxe).
//
// Usage  : node verify_web.js <chemin/index.html>
// Sortie : JSON { ok: bool, defects: [{location, kind, evidence}] }  (stdout, 1 ligne)
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");

const htmlPath = process.argv[2];
const defects = [];
let done = false;

function emit() {
  if (done) return;
  done = true;
  process.stdout.write(JSON.stringify({ ok: defects.length === 0, defects }));
  process.exit(0);
}

function addRuntime(evidence) {
  defects.push({
    location: path.basename(htmlPath || "page"),
    kind: "runtime",
    evidence: String(evidence).split("\n").slice(0, 2).join(" ").slice(0, 300),
  });
}

if (!htmlPath || !fs.existsSync(htmlPath)) {
  defects.push({ location: htmlPath || "?", kind: "error", evidence: "index.html introuvable" });
  emit();
}

const html = fs.readFileSync(htmlPath, "utf-8");
const url = "file://" + path.resolve(htmlPath).replace(/\\/g, "/");

const vc = new VirtualConsole();
vc.on("jsdomError", (e) => addRuntime((e && (e.detail && e.detail.stack)) || (e && e.message) || e));

let win;
try {
  const dom = new JSDOM(html, {
    url,
    runScripts: "dangerously",
    resources: "usable",
    pretendToBeVisual: true,
    virtualConsole: vc,
  });
  win = dom.window;
} catch (e) {
  addRuntime(e);
  emit();
}

win.addEventListener("error", (ev) => addRuntime(ev.message || ev.error || ev));

function statusSignature(doc) {
  const st = doc.querySelector("#status, .status, [data-status], #message, .message");
  return st ? (st.textContent || "").trim() : "";
}

function checkRenderAndExit() {
  if (done) return;
  const doc = win.document;
  const selectors = [
    "#board", "#game-board", "#game-board-container", "#grid",
    ".board", ".grid", "[data-board]",
  ];
  let board = null;
  let sel = null;
  for (const s of selectors) {
    const el = doc.querySelector(s);
    if (el) { board = el; sel = s; break; }
  }
  if (!board) {
    defects.push({
      location: path.basename(htmlPath),
      kind: "render",
      evidence: "aucun conteneur de plateau trouvé (#board/#game-board/.board)",
    });
    return emit();
  }
  if (board.children.length === 0) {
    defects.push({
      location: sel,
      kind: "render",
      evidence: `le conteneur du plateau ${sel} est VIDE (0 cellule rendue) — le jeu ne s'affiche pas`,
    });
    return emit();
  }
  // INTERACTION : le rendu ne suffit pas — cliquer une case doit produire un effet
  // (marque posée, statut mis à jour). Sinon le jeu n'est pas JOUABLE (ex: sélecteur
  // de cases incohérent entre HTML et JS -> aucun écouteur attaché).
  const cellSel =
    ".cell, [data-index], [data-cell], " + sel + " > div, " + sel + " > button";
  const getCells = () => doc.querySelectorAll(cellSel);
  if (!getCells().length) {
    // pas de cases identifiables : on ne peut pas tester l'interaction sans risque
    // de faux positif -> on s'abstient (le rendu est OK).
    return emit();
  }
  const sig = () => board.innerHTML + "||" + statusSignature(doc);
  const s0 = sig();
  const pressKey = (key, kc) => {
    for (const tgt of [doc, win]) {
      try {
        tgt.dispatchEvent(
          new win.KeyboardEvent("keydown", {
            key, code: key, keyCode: kc, which: kc, bubbles: true, cancelable: true,
          }),
        );
      } catch (e) {
        /* certains builds jsdom limitent KeyboardEvent — on ignore */
      }
    }
  };
  // Tentative 1 : un CLIC (jeux réactifs type morpion).
  try {
    getCells()[0].click();
  } catch (e) {
    addRuntime("clic case 1 a levé: " + (e && e.message));
  }
  setTimeout(() => {
    if (sig() !== s0) {
      // jeu au clic : un 2e coup (autre case vide, re-query) doit AUSSI prendre —
      // distingue "1er coup OK puis figé" d'un vrai jeu.
      const s1 = sig();
      const cells2 = getCells();
      let target = null;
      for (let i = 1; i < cells2.length; i++) {
        if (((cells2[i].textContent || "").trim()) === "") { target = cells2[i]; break; }
      }
      if (!target) return emit();
      try {
        target.click();
      } catch (e) {
        addRuntime("clic case 2 a levé: " + (e && e.message));
      }
      return setTimeout(() => {
        if (sig() === s1) {
          defects.push({
            location: sel,
            kind: "interaction",
            evidence:
              "le jeu se FIGE après le 1er coup (les clics suivants sont ignorés) — " +
              "ré-attache les écouteurs après chaque re-rendu, OU délègue les événements " +
              "sur le conteneur " + sel,
          });
        }
        emit();
      }, 150);
    }
    // Tentative 2 : CLAVIER + TEMPS (jeux type Snake : la boucle déplace le plateau,
    // les flèches changent la direction). On presse puis on laisse tourner ~0.8s.
    pressKey("ArrowRight", 39);
    pressKey("ArrowDown", 40);
    setTimeout(() => {
      if (sig() === s0) {
        defects.push({
          location: sel,
          kind: "interaction",
          evidence:
            "ni un clic ni une flèche (sur ~0.8s) ne changent le plateau — le jeu n'est " +
            "pas JOUABLE (vérifie les écouteurs clic/keydown ET la boucle setInterval(tick) " +
            "qui doit démarrer au chargement)",
        });
      }
      emit();
    }, 800);
  }, 150);
}

// Laisser le 'load' (scripts chargés + exécutés) se produire, puis vérifier le rendu.
win.addEventListener("load", () => setTimeout(checkRenderAndExit, 300));
setTimeout(checkRenderAndExit, 5000); // garde-fou si 'load' ne se déclenche jamais
