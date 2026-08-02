"""Résolution UNIFIÉE du nombre de couches GPU à offloader (-ngl).

Avant, deux implémentations divergentes coexistaient :
  - serve.py (chemin mono-modèle) ne connaissait QUE l'override global et l'auto ;
    il ignorait cpu_moe / model.n_gpu_layers -> un MoE en mono-modèle n'était JAMAIS
    offloadé correctement.
  - swap.py (chemin multi-modèles) gérait cpu_moe + model.n_gpu_layers mais ignorait
    le kv_headroom configuré (toujours le défaut 1024).

Une seule fonction `resolve_ngl` porte désormais TOUTE la précédence, partagée par les
deux chemins -> comportement identique qu'on lance un ou plusieurs modèles.
"""

from __future__ import annotations

from loom.config import ModelConfig
from loom.runtime.hardware import HardwareProfile, recommend_gpu_layers


def resolve_ngl(
    model: ModelConfig,
    profile: HardwareProfile,
    override: int | None,
    headroom: int = 1024,
) -> int:
    """Nombre de couches à offloader sur GPU (-ngl), unifie serve.py et swap.py.

    Précédence (identique sur les deux chemins, mono et multi-modèles) :
      1. MoE offloadé (cpu_moe / n_cpu_moe) -> 999 : toutes les couches DENSES sur GPU,
         les experts routés partent en RAM via --cpu-moe / --n-cpu-moe (build_server_args).
         On ignore l'override global (pensé pour les petits modèles denses).
      2. Champ par modèle (model.n_gpu_layers) -> force explicite, prioritaire sur le global.
      3. Override global ([override] n_gpu_layers, ex. 99 = offload total).
      4. Recommandation auto selon la VRAM libre (`headroom` = marge KV réservée) ;
         0 en CPU-only.

    `headroom` : marge VRAM réservée au KV + buffers (config gpu_kv_headroom_mb).
    """
    if model.cpu_moe or model.n_cpu_moe is not None:
        return 999
    if model.n_gpu_layers is not None:
        return model.n_gpu_layers
    if override is not None:
        return override
    if not profile.has_gpu:
        return 0
    return recommend_gpu_layers(
        profile.vram_free_mb, model.size_mb, model.n_layers, headroom
    )
