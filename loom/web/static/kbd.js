// Cheatsheet des raccourcis clavier réels de Loom. Le bouton vit dans la tabbar,
// reconstruite souvent ; la modale, elle, reste montée une seule fois sous <body>.

const OVERLAY_ID = "kbd-cheatsheet";
const BUTTON_ID = "kbd-help";

const GROUPS = [
  {
    title: "Vues",
    rows: [
      ["Scinder la vue", ["Ctrl", "\\"]],
      ["Revenir à une vue", ["Ctrl", "Shift", "\\"]],
      ["Focaliser un panneau", ["Alt", "1…4"]],
    ],
  },
  {
    title: "Composer",
    rows: [
      ["Envoyer", ["Entrée"]],
      ["Nouvelle ligne", ["Shift", "Entrée"]],
      ["Parcourir l’historique", ["↑", "↓"]],
      ["Valider la palette", ["Tab", "Entrée"]],
    ],
  },
  {
    title: "Fermer",
    rows: [["Menu, palette ou fenêtre", ["Échap"]]],
  },
];

let previousFocus = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function closeCheatsheet() {
  const overlay = document.getElementById(OVERLAY_ID);
  if (!overlay || overlay.hidden) return;
  overlay.hidden = true;
  document.getElementById(BUTTON_ID)?.setAttribute("aria-expanded", "false");
  if (previousFocus?.isConnected) previousFocus.focus();
  previousFocus = null;
}

function buildOverlay() {
  const existing = document.getElementById(OVERLAY_ID);
  if (existing) return existing;

  const overlay = element("div", "kb-overlay");
  overlay.id = OVERLAY_ID;
  overlay.hidden = true;

  const modal = element("section", "kb-modal");
  modal.setAttribute("role", "dialog");
  modal.setAttribute("aria-modal", "true");
  modal.setAttribute("aria-labelledby", "kbd-title");

  const head = element("header", "kb-head");
  const title = element("h2", "", "Raccourcis clavier");
  title.id = "kbd-title";
  const close = element("button", "kb-close", "✕");
  close.type = "button";
  close.title = "Fermer";
  close.setAttribute("aria-label", "Fermer les raccourcis clavier");
  close.addEventListener("click", closeCheatsheet);
  head.append(title, close);
  modal.append(head);

  const groups = element("div", "kb-groups");
  for (const group of GROUPS) {
    const section = element("section", "kb-group");
    section.append(element("h3", "", group.title));
    for (const [label, keys] of group.rows) {
      const row = element("div", "kb-row");
      row.append(element("span", "", label));
      const chord = element("span", "kb-chord");
      keys.forEach((key, index) => {
        if (index) chord.append(element("i", "", "+"));
        chord.append(element("kbd", "", key));
      });
      row.append(chord);
      section.append(row);
    }
    groups.append(section);
  }
  modal.append(groups);
  overlay.append(modal);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeCheatsheet();
  });
  document.body.append(overlay);
  return overlay;
}

function openCheatsheet() {
  const overlay = buildOverlay();
  previousFocus = document.activeElement;
  overlay.hidden = false;
  document.getElementById(BUTTON_ID)?.setAttribute("aria-expanded", "true");
  overlay.querySelector(".kb-close")?.focus();
}

export function ensureKbdCheatsheetButton() {
  const tabbar = document.getElementById("tabbar");
  if (!tabbar) return null;
  let button = document.getElementById(BUTTON_ID);
  if (button && button.parentElement === tabbar) return button;
  button = element("button", "tab-help", "?");
  button.id = BUTTON_ID;
  button.type = "button";
  button.title = "Raccourcis clavier";
  button.setAttribute("aria-label", "Afficher les raccourcis clavier");
  button.setAttribute("aria-haspopup", "dialog");
  const overlay = document.getElementById(OVERLAY_ID);
  button.setAttribute("aria-expanded", overlay && !overlay.hidden ? "true" : "false");
  button.addEventListener("click", openCheatsheet);
  tabbar.append(button);
  return button;
}

export function initKbdCheatsheet() {
  buildOverlay();
  ensureKbdCheatsheetButton();
  if (!document.documentElement.dataset.kbdEscapeWired) {
    document.documentElement.dataset.kbdEscapeWired = "1";
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeCheatsheet();
    });
  }
}
