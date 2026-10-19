// Icônes du rail d'étapes du fil (SVG monochromes, stroke courant). Constantes
// locales : jamais du contenu venant du modèle.
const wrap = (body) =>
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + body + "</svg>";

export const STEP_ICONS = {
  // réflexion : ampoule
  think: wrap('<path d="M9 18h6"/><path d="M10 21h4"/><path d="M8 13a5 5 0 1 1 8 0c-1 1-1.5 2-1.5 3h-5c0-1-.5-2-1.5-3z"/>'),
  // terminal : chevron + trait
  terminal: wrap('<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3"/><path d="M12 15h5"/>'),
  // lecture : feuille
  file: wrap('<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M10 13h6"/><path d="M10 17h6"/>'),
  // recherche : loupe
  search: wrap('<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>'),
  // écriture : crayon
  edit: wrap('<path d="M4 20l4-1 11-11-3-3L5 16z"/><path d="M13 7l3 3"/>'),
  // web : globe
  globe: wrap('<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18"/>'),
  // agent : personne
  agent: wrap('<circle cx="12" cy="8" r="3.5"/><path d="M5 20a7 7 0 0 1 14 0"/>'),
  // agents en parallèle : deux personnes
  agents: wrap('<circle cx="9" cy="8" r="3"/><path d="M3 19a6 6 0 0 1 12 0"/><circle cx="17" cy="9" r="2.5"/><path d="M15.5 14.5a5 5 0 0 1 5.5 4.5"/>'),
  // mémoire, notes, todos : marque-page
  note: wrap('<path d="M6 3h12v18l-6-4-6 4z"/>'),
  // skills, plugins : pièce de puzzle
  puzzle: wrap('<path d="M10 3a2 2 0 0 1 4 0v2h4v4h2a2 2 0 0 1 0 4h-2v4h-4v2a2 2 0 0 1-4 0v-2H6v-4H4a2 2 0 0 1 0-4h2V5h4z"/>'),
  // calcul, horloge, moniteur : cadran
  gauge: wrap('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
  // outil générique : clé
  wrench: wrap('<path d="M14 6a4 4 0 0 0 4 6l-8 8-3-3 8-8a4 4 0 0 1-1-3z"/>'),
  // garde-fou Loom : bouclier
  shield: wrap('<path d="M12 3l7 3v6c0 4-3 7-7 9-4-2-7-5-7-9V6z"/>'),
  // réponse du modèle : bulle
  chat: wrap('<path d="M4 5h16v10H9l-5 4z"/>'),
  // erreur : triangle
  alert: wrap('<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17v.01"/>'),
};

// Glyphes d'état et de repli (remplacent les caractères ▸ ▾ ✓ ✕) : même trait.
export const GLYPHS = {
  chevron: wrap('<path d="M9 6l6 6-6 6"/>'),
  check: wrap('<path d="M5 12.5l4.5 4.5L19 7"/>'),
  x: wrap('<path d="M6 6l12 12"/><path d="M18 6L6 18"/>'),
};

const BY_TOOL = {
  run_shell: "terminal",
  read_file: "file",
  read_image: "file",
  list_dir: "file",
  code_outline: "file",
  code_diagnostics: "file",
  find_files: "search",
  search_text: "search",
  tool_search: "search",
  write_file: "edit",
  append_file: "edit",
  edit_file: "edit",
  format_code: "edit",
  web_search: "globe",
  fetch_url: "globe",
  check_page: "globe",
  serve_and_check: "globe",
  watch_video: "globe",
  dispatch_agent: "agent",
  submit_result: "agent",
  run_workflow: "agents",
  remember: "note",
  recall: "note",
  write_note: "note",
  read_note: "note",
  manage_todos: "note",
  use_skill: "puzzle",
  add_marketplace: "puzzle",
  install_plugin: "puzzle",
  list_plugins: "puzzle",
  calculate: "gauge",
  current_date: "gauge",
  monitor: "gauge",
};

/** Icône d'un outil d'après son nom (les outils MCP inconnus tombent sur la clé). */
export function toolIcon(name) {
  return BY_TOOL[name] || "wrench";
}
