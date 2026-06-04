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

// Bruit d'ENVIRONNEMENT jsdom (pas des bugs de la page) : jsdom n'implémente pas la
// navigation entre documents (un clic sur <a href="autre.html">), ni le chargement réel
// de certaines ressources. Ces messages ne disent RIEN sur la justesse du code -> on les
// ignore pour ne pas faire échouer à tort un site multi-pages qui fonctionne.
function isEnvNoise(s) {
  s = String(s);
  return (
    s.includes("Not implemented:") ||
    s.includes("Could not load") ||
    // jsdom traite file:// comme une origine OPAQUE -> localStorage/sessionStorage y
    // lèvent une SecurityError, alors qu'ils FONCTIONNENT dans un vrai navigateur (double
    // -clic ou http). C'est une limite de jsdom, pas un bug de la page.
    s.includes("opaque origin")
  );
}

function addRuntime(evidence) {
  if (isEnvNoise(evidence)) return;
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

// Polyfill localStorage/sessionStorage : jsdom traite file:// comme une origine OPAQUE et
// fait LEVER ces API, ce qui INTERROMPT les scripts de la page (puis « rien ne se rend »).
// Dans un vrai navigateur elles fonctionnent. On fournit un store mémoire conforme pour que
// la page s'exécute comme en vrai (sans ce faux échec en cascade).
function makeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(String(k)) ? m.get(String(k)) : null),
    setItem: (k, v) => void m.set(String(k), String(v)),
    removeItem: (k) => void m.delete(String(k)),
    clear: () => m.clear(),
    key: (i) => Array.from(m.keys())[i] ?? null,
    get length() {
      return m.size;
    },
  };
}

let win;
try {
  const dom = new JSDOM(html, {
    url,
    runScripts: "dangerously",
    resources: "usable",
    pretendToBeVisual: true,
    virtualConsole: vc,
    beforeParse(window) {
      for (const name of ["localStorage", "sessionStorage"]) {
        try {
          Object.defineProperty(window, name, {
            value: makeStorage(),
            configurable: true,
          });
        } catch (e) {
          /* déjà défini sur certaines versions de jsdom — on garde l'existant */
        }
      }
    },
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

function isNavLink(a) {
  // Lien de NAVIGATION (vers une autre page / externe) : cliquer dessus ne teste pas une
  // interactivité de la page, et jsdom lèverait un faux 'Not implemented: navigation'.
  const h = (a.getAttribute("href") || "").trim();
  if (!h || h === "#") return true;
  return /^https?:|^\/\//.test(h) || /\.html?($|[?#])/i.test(h);
}

function interactiveEls(doc) {
  // Tout ce qu'un utilisateur peut actionner — AUCUNE hypothèse de domaine (pas de #board).
  const els = Array.from(
    doc.querySelectorAll(
      "button, [data-action], [data-digit], [data-operator], [data-key], " +
        "[data-index], [data-cell], .cell, [role='button']",
    ),
  );
  // Les <a> ne comptent QUE s'ils agissent dans la page (href='#...'), pas la navigation.
  doc.querySelectorAll("a[href]").forEach((a) => {
    if (!isNavLink(a)) els.push(a);
  });
  return els;
}

function checkLocalRefs(doc) {
  // Vérif DÉTERMINISTE (pas via le loader jsdom, peu fiable) : tout asset LOCAL référencé
  // (feuille de style, script, image) doit exister sur le disque. Un <link href='style.css'>
  // sans fichier = lien cassé réel (≠ bruit jsdom). C'est le défaut qu'on VEUT remonter.
  const dir = path.dirname(htmlPath);
  const seen = new Set();
  doc.querySelectorAll("link[href], script[src], img[src]").forEach((el) => {
    const u = (el.getAttribute("href") || el.getAttribute("src") || "").trim();
    if (!u || /^(https?:|data:|blob:|\/\/|#|mailto:)/.test(u)) return;
    const rel = u.split(/[?#]/)[0];
    const tag = el.tagName.toLowerCase();
    const ext = (rel.split(".").pop() || "").toLowerCase();
    // MAUVAISE BALISE pour le type de fichier (cause des SyntaxError cryptiques). Un CSS
    // chargé via <script> est EXÉCUTÉ comme du JS -> "Unexpected token ':'". Un JS chargé
    // via <link> n'est jamais exécuté. On donne le défaut PRÉCIS (≠ message jsdom obscur).
    if (tag === "script" && ext === "css") {
      defects.push({
        location: path.basename(htmlPath),
        kind: "asset",
        evidence: `${rel} chargé via <script> : un CSS se charge avec <link rel="stylesheet" href="${rel}">, jamais via <script> (sinon il est exécuté comme du JS -> SyntaxError).`,
      });
    } else if (tag === "link" && (ext === "js" || ext === "mjs")) {
      defects.push({
        location: path.basename(htmlPath),
        kind: "asset",
        evidence: `${rel} chargé via <link> : un script JS se charge avec <script src="${rel}"></script>, pas via <link>.`,
      });
    }
    if (seen.has(rel)) return;
    seen.add(rel);
    if (!fs.existsSync(path.resolve(dir, rel))) {
      defects.push({
        location: path.basename(htmlPath),
        kind: "asset",
        evidence: `référence introuvable: ${rel} (lien/script/image vers un fichier absent)`,
      });
    }
  });
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
  checkLocalRefs(doc); // liens/scripts/images locaux cassés (déterministe, hors jsdom)
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
