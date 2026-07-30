"""Routes de la web app Loom, eclatees par domaine (helpers = coeur partage)."""
from __future__ import annotations

from loom.web.routes.chat import _boot_prime as _boot_prime, _keepwarm_loop as _keepwarm_loop, _register_chat_routes as _register_chat_routes
from loom.web.routes.config import _register_config_routes as _register_config_routes
from loom.web.routes.misc import _register_misc_routes as _register_misc_routes
from loom.web.routes.models import _register_model_routes as _register_model_routes
from loom.web.routes.sessions import _register_session_routes as _register_session_routes
from loom.web.routes.skills import _register_skill_routes as _register_skill_routes
from loom.web.routes.soul import _register_soul_routes as _register_soul_routes
from loom.web.routes.helpers import _local_busy_notice as _local_busy_notice
from loom.web.routes.models import _REBENCH as _REBENCH, _check_workflow as _check_workflow, _image_dir_state as _image_dir_state, _removable_models as _removable_models, _run_calibration as _run_calibration, _wizard_deps as _wizard_deps
