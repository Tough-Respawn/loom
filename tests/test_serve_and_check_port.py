# serve_and_check attendait « un port qui répond » : un vieux serveur orphelin sur
# ce port suffisait. Le nouveau plantait (EADDRINUSE) et l'outil validait l'ancien.
import socket

import pytest

from loom.tools.base import ToolError
from loom.tools.browser import make_serve_and_check


def test_port_deja_occupe_refuse_sans_lancer(tmp_path):
    squatter = socket.socket()
    squatter.bind(("127.0.0.1", 0))
    squatter.listen()
    port = squatter.getsockname()[1]
    marker = tmp_path / "lance.txt"
    try:
        tool = make_serve_and_check(str(tmp_path))
        with pytest.raises(ToolError) as exc:
            tool.run(
                {
                    "command": f"python -c \"open(r'{marker}', 'w').write('x')\"",
                    "url": f"http://127.0.0.1:{port}",
                    "ready_timeout": 5,
                }
            )
        assert "déjà" in str(exc.value) and str(port) in str(exc.value)
        assert not marker.exists()  # la commande n'a jamais été lancée
    finally:
        squatter.close()
