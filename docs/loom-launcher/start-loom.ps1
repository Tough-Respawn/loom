# Lanceur Loom (raccourci du Bureau, fenêtre PowerShell cachée).
# Copie installée : C:\tools\loom-launcher\start-loom.ps1 (le raccourci pointe dessus).
#
# Démarre loom.web dans une console minimisée avec `--app` : loom.web ouvre lui-même la
# fenêtre d'application (profil navigateur dédié) et la lie à sa vie :
#   - fermer la fenêtre  -> Loom s'arrête proprement (serveur modèle déchargé) ;
#   - fermer la console  -> la fenêtre et le serveur modèle se ferment avec elle.
# Si Loom tourne déjà, on rouvre seulement une fenêtre sur l'instance existante.
$repo = 'C:\Users\Amine\Documents\from-claude-to-local-haranessed-llm'
$url  = 'http://127.0.0.1:8000/'   # [chat] web_port de config/defaults.toml

function Test-Loom {
    try { (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { $false }
}

if (Test-Loom) {
    # Déjà lancé : nouvelle fenêtre dans le MÊME profil dédié, donc dans le processus
    # navigateur déjà surveillé par loom.web.
    $profile = Join-Path $repo 'var\browser-profile'
    $brave = 'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe'
    $edge  = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    $exe = if (Test-Path $brave) { $brave } else { $edge }
    Start-Process -FilePath $exe -ArgumentList "--app=$url", "--user-data-dir=$profile", "--no-first-run"
    exit 0
}

# Console minimisée qui porte les logs. /k : en cas de plantage, la trace reste lisible.
$cmd = '/k call ".venv\Scripts\activate.bat" & python -m loom.web --app'
Start-Process -FilePath 'cmd.exe' -ArgumentList $cmd -WorkingDirectory $repo -WindowStyle Minimized
