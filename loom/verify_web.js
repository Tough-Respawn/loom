// loom/verify_web.js
// Vérificateur RUNTIME DOM (offline, via jsdom) : charge une page HTML + ses scripts,
// capture les erreurs d'exécution (TypeError, ReferenceError…), vérifie que l'interface
// rend RÉELLEMENT quelque chose, et l'EXERCE (clique chaque bouton, tape dans les champs,
// presse des touches) pour prouver qu'elle RÉAGIT. Générique : aucune hypothèse de domaine
// (ni #board ni .cell figés). C'est l'œil "ça marche pour de vrai" qui manque à
// `node --check` (lequel ne voit que la syntaxe).
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

function displaySig(doc) {
  // Capture l'état "visible" qu'une interaction est censée changer : la VALEUR des champs
  // (hors innerHTML) et les zones d'affichage/résultat/statut. Complète body.innerHTML.
  let s = "";
  doc
    .querySelectorAll(
      "input, textarea, select, output, #display, .display, #result, .result, #status, .status",
    )
    .forEach((el) => {
      const v = el.value !== undefined && el.value !== null ? el.value : "";
      s += "|" + v + "/" + (el.textContent || "").trim();
    });
  return s;
}

function interactiveEls(doc) {
  // Tout ce qu'un utilisateur peut actionner — AUCUNE hypothèse de domaine (pas de #board).
  return Array.from(
    doc.querySelectorAll(
      "button, [data-action], [data-digit], [data-operator], [data-key], " +
        "[data-index], [data-cell], .cell, [role='button'], a[href]",
    ),
  );
}

function pressKey(doc, key, kc) {
  for (const tgt of [doc, win]) {
    try {
      tgt.dispatchEvent(
        new win.KeyboardEvent("keydown", {
          key, code: key, keyCode: kc, which: kc, bubbles: true, cancelable: true,
        }),
      );
    } catch (e) {
      /* jsdom limite parfois KeyboardEvent — on ignore */
    }
  }
}

function checkRenderAndExit() {
  if (done) return;
  const doc = win.document;
  const body = doc.body;
  const buttons = body ? interactiveEls(doc) : [];
  const inputs = body
    ? Array.from(doc.querySelectorAll("input:not([type='hidden']), textarea, select"))
    : [];
  const bodyText = body ? (body.textContent || "").trim() : "";

  // RENDU (générique) : la page ne rend RIEN si elle n'a ni élément interactif, ni champ,
  // ni texte. On juge le CONTENU réel, plus aucune liste de conteneurs figée (#board…).
  if (!body || (buttons.length === 0 && inputs.length === 0 && bodyText.length < 3)) {
    defects.push({
      location: path.basename(htmlPath),
      kind: "render",
      evidence:
        "la page ne rend rien (aucun élément interactif, aucun champ, aucun texte) — " +
        "l'interface ne s'affiche pas",
    });
    return emit();
  }
  // Page statique (texte/affichage, sans bouton ni champ) : le rendu suffit, rien à exercer.
  if (buttons.length === 0 && inputs.length === 0) return emit();

  // EXERCICE : on actionne TOUT (clique les boutons, tape dans les champs, presse des
  // touches usuelles) et on vérifie qu'AU MOINS une interaction change l'interface. Sinon
  // les écouteurs ne sont pas branchés (sélecteurs incohérents, fonctions non exposées).
  const sig = () => (body.innerHTML || "") + "||" + displaySig(doc);
  const s0 = sig();

  let clicked = 0;
  for (const b of buttons) {
    if (clicked >= 10) break;
    try {
      b.click();
      clicked++;
    } catch (e) {
      addRuntime("un clic a levé: " + (e && e.message));
    }
  }
  for (const inp of inputs.slice(0, 3)) {
    if (inp.readOnly || inp.disabled) continue;
    try {
      if (inp.focus) inp.focus();
      inp.value = (inp.value || "") + "1";
      inp.dispatchEvent(new win.Event("input", { bubbles: true }));
      pressKey(doc, "1", 49);
    } catch (e) {
      /* champ non éditable — on ignore */
    }
  }
  // Apps clavier / temporelles : presse quelques touches usuelles, laisse tourner un peu.
  ["1", "Enter", "ArrowRight", "ArrowDown"].forEach((k, i) => pressKey(doc, k, 49 + i));

  setTimeout(() => {
    if (sig() === s0) {
      defects.push({
        location: path.basename(htmlPath),
        kind: "interaction",
        evidence:
          "aucune interaction (clic sur " + clicked + " bouton(s), saisie, clavier) ne " +
          "change l'interface — les écouteurs ne sont pas branchés (sélecteurs " +
          "incohérents entre HTML et JS, ou fonctions non exposées sur window)",
      });
    }
    emit();
  }, 700);
}

// Laisser le 'load' (scripts chargés + exécutés) se produire, puis vérifier le rendu.
win.addEventListener("load", () => setTimeout(checkRenderAndExit, 300));
setTimeout(checkRenderAndExit, 5000); // garde-fou si 'load' ne se déclenche jamais
