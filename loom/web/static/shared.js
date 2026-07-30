// loom/web/static/shared.js — issu du decoupage de app.js (comportement constant).
import { INIT } from "./render.js";
import { workdirPath } from "./panels.js";

export let CMDS = [];

export let _phRaf = false;

export let loomWorkdir =
  localStorage.loomWorkdir ||
  INIT.workspace_dir ||
  (workdirPath && workdirPath.textContent.trim()) ||
  "";

export let machineUnloaded = false;

export let skGenBusy = false;

export function set_loomWorkdir(v){ loomWorkdir = v; }
export function set_CMDS(v){ CMDS = v; }
export function set__phRaf(v){ _phRaf = v; }
export function set_machineUnloaded(v){ machineUnloaded = v; }
export function set_skGenBusy(v){ skGenBusy = v; }
