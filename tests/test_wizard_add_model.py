# Machine à états PURE de /add-model : aucun réseau, deps entièrement fakés.
from types import SimpleNamespace

from loom.web import wizard


def deps(existing=(), hits=None, files=None):
    return SimpleNamespace(
        existing_ids=set(existing),
        search_models=lambda q: list(hits or []),
        list_gguf_files=lambda repo: list(files or []),
        recommend=lambda fs: [
            dict(f, fits=True, recommended=(i == len(fs) - 1)) for i, f in enumerate(fs)
        ],
        derive_id=lambda repo: "id-propose",
    )


# ---------- tronc commun ----------


def test_start_sans_arg_demande_le_type():
    r = wizard.start("", deps())
    assert r.state == {"step": "kind"}
    assert "1" in r.reply and "local" in r.reply and "distant" in r.reply


def test_cancel_a_toute_etape():
    r = wizard.step(
        {"step": "r_model", "id": "x", "base_url": "https://a"}, "/cancel", deps()
    )
    assert r.state is None and r.action is None
    assert "annulé" in r.reply.lower()


def test_etat_inconnu_annule_proprement():
    r = wizard.step({"step": "n-existe-pas"}, "x", deps())
    assert r.state is None


# ---------- flux distant ----------


def test_distant_parcours_complet():
    d = deps(existing={"deja-la"})
    r = wizard.step({"step": "kind"}, "2", d)
    assert r.state == {"step": "r_id"}

    r = wizard.step(r.state, "deja-la", d)  # id déjà pris -> re-demande
    assert r.state == {"step": "r_id"}

    r = wizard.step(r.state, "glm-5", d)
    assert r.state["step"] == "r_base_url"

    r = wizard.step(r.state, "pas-une-url", d)  # invalide -> re-demande
    assert r.state["step"] == "r_base_url"

    r = wizard.step(r.state, "https://api.z.ai/api/paas/v4/", d)
    assert r.state["step"] == "r_model"
    assert r.state["base_url"] == "https://api.z.ai/api/paas/v4"  # slash retiré

    r = wizard.step(r.state, "glm-5-flash", d)
    assert r.state["step"] == "r_key"

    r = wizard.step(r.state, "sk-secret", d)
    assert r.state["step"] == "r_adv"

    r = wizard.step(r.state, "contexte=200000 vision=oui", d)
    assert r.state is None
    assert r.action == {
        "kind": "upsert_remote",
        "record": {
            "id": "glm-5",
            "base_url": "https://api.z.ai/api/paas/v4",
            "model": "glm-5-flash",
            "api_key": "sk-secret",
            "context": 200000,
            "max_tokens": None,
            "vision": True,
        },
    }


def test_distant_sans_avance_ni_cle():
    d = deps()
    st = {"step": "r_key", "id": "m", "base_url": "https://a", "model": "mm"}
    r = wizard.step(st, "aucune", d)
    r = wizard.step(r.state, "non", d)
    assert r.action["record"]["api_key"] == ""
    assert r.action["record"]["context"] is None
    assert r.action["record"]["vision"] is False


# ---------- flux local ----------

HITS = [
    {"repo_id": "unsloth/Qwen3-30B-GGUF", "downloads": 9000, "likes": 42},
    {"repo_id": "org/autre-GGUF", "downloads": 100, "likes": 1},
]
FILES = [
    {
        "filename": "m.Q4_K_M.gguf",
        "part_files": ["m.Q4_K_M.gguf"],
        "size_mb": 10_000,
        "is_mmproj": False,
    },
    {
        "filename": "m.Q8_0.gguf",
        "part_files": ["m.Q8_0.gguf"],
        "size_mb": 20_000,
        "is_mmproj": False,
    },
    {
        "filename": "mmproj-F16.gguf",
        "part_files": ["mmproj-F16.gguf"],
        "size_mb": 800,
        "is_mmproj": True,
    },
]


def test_local_parcours_complet():
    d = deps(hits=HITS, files=FILES)
    r = wizard.start("qwen3 30b", d)  # /add-model <recherche> : direct à la shortlist
    assert r.state["step"] == "l_repo"
    assert "unsloth/Qwen3-30B-GGUF" in r.reply and "9000" in r.reply

    r = wizard.step(r.state, "1", d)  # choix du repo -> quants annotés
    assert r.state["step"] == "l_quant"
    assert "recommandé" in r.reply
    assert "mmproj-F16.gguf" in r.reply  # vision signalée
    # seuls les POIDS sont proposés (le mmproj n'est pas un choix de quant)
    assert len(r.state["files"]) == 2

    r = wizard.step(r.state, "2", d)  # choix du quant
    assert r.state["step"] == "l_id"
    assert "id-propose" in r.reply

    r = wizard.step(r.state, "ok", d)  # id proposé accepté
    assert r.state is None
    assert r.action == {
        "kind": "install",
        "model_id": "id-propose",
        "repo": "unsloth/Qwen3-30B-GGUF",
        "filename": "m.Q8_0.gguf",
        "files": ["m.Q8_0.gguf"],
        "size_mb": 20_000,
        "mmproj_filename": "mmproj-F16.gguf",
    }


def test_local_recherche_vide_redemande():
    d = deps(hits=[])
    r = wizard.start("nexistepas", d)
    assert r.state == {"step": "l_query"}
    assert "Aucun" in r.reply


def test_local_texte_libre_sur_shortlist_relance_la_recherche():
    d = deps(hits=HITS, files=FILES)
    r = wizard.start("qwen", d)
    r = wizard.step(r.state, "autre recherche", d)
    assert r.state["step"] == "l_repo"  # nouvelle shortlist, pas d'erreur


def test_local_id_deja_pris_redemande():
    d = deps(existing={"id-propose"}, hits=HITS, files=FILES)
    r = wizard.start("qwen", d)
    r = wizard.step(r.state, "1", d)
    r = wizard.step(r.state, "1", d)
    r = wizard.step(r.state, "ok", d)  # id proposé mais déjà pris
    assert r.state["step"] == "l_id"
    r = wizard.step(r.state, "mon-id", d)
    assert r.action["model_id"] == "mon-id"


def test_local_kind_1_va_a_la_recherche():
    r = wizard.step({"step": "kind"}, "1", deps())
    assert r.state == {"step": "l_query"}
