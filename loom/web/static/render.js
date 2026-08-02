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

export const md = (raw) => {
  const { text, maths } = _protectMath(raw || "");
  const out = DOMPurify.sanitize(marked.parse(text));
  return out.replace(/@@MATH(\d+)@@/g, (_, i) => _escHtml(maths[+i] || ""));
};

export const INIT = JSON.parse(
  document.getElementById("loom-init")?.textContent || "{}",
);
