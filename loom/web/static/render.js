export function _protectMath(raw) {
  const maths = [];
  const stash = (delim, tex) => {
    // € n'existe pas dans les polices TeX du bundle SVG : le glyphe manquant fait
    // crasher MathJax (« Math input error », vécu sur les montants). En \text{},
    // il retombe sur la police système et affiche un vrai €.
    maths.push(delim + tex.replace(/€/g, "\\text{€}") + delim);
    return `@@MATH${maths.length - 1}@@`;
  };
  let s = raw;
  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => stash("$$", t));
  s = s.replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => stash("$$", t.trim()));
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => stash("$", t.trim()));
  s = s.replace(/(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)/g, (_, t) => stash("$", t));
  return { text: s, maths };
}

export const _escHtml = (s) =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

// Formules composées de façon SYNCHRONE (MathJax.tex2svg) et mises en cache par source :
// pendant un flux, chaque jeton re-rend le markdown, et un typeset asynchrone serait
// effacé par le rendu suivant. Avec le cache, une formule déjà vue ressort composée à
// chaque rendu, dès que son délimiteur de fin est arrivé. MathJax pas encore chargé →
// TeX échappé, composé plus tard par enhance()/_typesetAll (comportement d'origine).
const _mathCache = new Map();

function _mathHtml(src) {
  const MJ = window.MathJax;
  if (!MJ || typeof MJ.tex2svg !== "function") return _escHtml(src);
  const hit = _mathCache.get(src);
  if (hit !== undefined) return hit;
  const display = src.startsWith("$$");
  const tex = display ? src.slice(2, -2) : src.slice(1, -1);
  let out;
  try {
    out = MJ.tex2svg(tex, { display }).outerHTML;
    // Cache de glyphes partagé et feuille de style : à resynchroniser après un tex2svg.
    MJ.startup?.document?.updateDocument?.();
  } catch (e) {
    out = _escHtml(src);
  }
  if (_mathCache.size > 2000) _mathCache.clear();
  _mathCache.set(src, out);
  return out;
}

export const md = (raw) => {
  const { text, maths } = _protectMath(raw || "");
  const out = DOMPurify.sanitize(marked.parse(text));
  return out.replace(/@@MATH(\d+)@@/g, (_, i) => _mathHtml(maths[+i] || ""));
};

export const INIT = JSON.parse(
  document.getElementById("loom-init")?.textContent || "{}",
);
