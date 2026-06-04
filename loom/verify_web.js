// loom/verify_web.js
// Vérificateur RUNTIME DOM (offline, via jsdom) : charge une page HTML + ses scripts,
// capture les erreurs d'exécution (TypeError, ReferenceError…) ET vérifie que
// l'interface se rend réellement (conteneur principal non vide). C'est l'œil "ça tourne"
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
    "#app", "#root", "#main", "main",
    "#board", "#game-board", "#game-board-container", "#grid",
    ".board", ".grid", ".container", "[data-board]",
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
      evidence: "aucun conteneur principal trouvé (#app/#root/#board/.container...)",
    });
    return emit();
  }
  if (board.children.length === 0) {
    defects.push({
      location: sel,
      kind: "render",
      evidence: `le conteneur ${sel} est VIDE (0 élément rendu) — l'interface ne s'affiche pas`,
    });
    return emit();
  }
  // INTERACTION : le rendu ne suffit pas — interagir (clic/clavier) doit produire un
  // effet (élément mis à jour, statut changé). Sinon l'interface n'est pas FONCTIONNELLE
  // (ex: sélecteur incohérent entre HTML et JS -> aucun écouteur attaché).
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
  // Tentative 1 : un CLIC (interfaces réactives au clic).
  try {
    getCells()[0].click();
  } catch (e) {
    addRuntime("clic case 1 a levé: " + (e && e.message));
  }
  setTimeout(() => {
    if (sig() !== s0) {
      // réactif au clic : une 2e interaction (autre élément vide, re-query) doit AUSSI
      // prendre — distingue "1re interaction OK puis figé" d'une vraie interface.
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
              "l'interface se FIGE après la 1re interaction (les clics suivants sont " +
              "ignorés) — ré-attache les écouteurs après chaque re-rendu, OU délègue les " +
              "événements sur le conteneur " + sel,
          });
        }
        emit();
      }, 150);
    }
    // Tentative 2 : CLAVIER + TEMPS (interfaces pilotées au clavier et/ou par une boucle
    // temporelle). On presse quelques touches puis on laisse tourner ~0.8s.
    pressKey("ArrowRight", 39);
    pressKey("ArrowDown", 40);
    setTimeout(() => {
      if (sig() === s0) {
        defects.push({
          location: sel,
          kind: "interaction",
          evidence:
            "ni un clic ni une touche (sur ~0.8s) ne changent l'interface — elle ne " +
            "RÉAGIT pas (vérifie les écouteurs clic/keydown ET, si l'app est temporelle, " +
            "la boucle setInterval qui doit démarrer au chargement)",
        });
      }
      emit();
    }, 800);
  }, 150);
}

// Laisser le 'load' (scripts chargés + exécutés) se produire, puis vérifier le rendu.
win.addEventListener("load", () => setTimeout(checkRenderAndExit, 300));
setTimeout(checkRenderAndExit, 5000); // garde-fou si 'load' ne se déclenche jamais
