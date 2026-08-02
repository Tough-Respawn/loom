"""Smoke navigateur autonome : UI réelle -> Flask -> SSE -> stub OpenAI."""

from __future__ import annotations

import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from tests.e2e.launch_loom_e2e import build_e2e_app
from tests.e2e.stub_openai import app as stub_app


def _serve(app, port: int, stack: ExitStack) -> None:
    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stack.callback(thread.join, 5)
    stack.callback(server.shutdown)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="loom-e2e-") as tmp, ExitStack() as stack:
        _serve(stub_app, 18081, stack)
        _serve(build_e2e_app(Path(tmp)), 18090, stack)

        console_errors: list[str] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.on(
                    "console",
                    lambda msg: console_errors.append(msg.text)
                    if msg.type == "error"
                    else None,
                )
                response = page.goto(
                    "http://127.0.0.1:18090/",
                    wait_until="networkidle",
                    timeout=20_000,
                )
                assert response is not None and response.status == 200
                page.locator(".pane textarea").wait_for(
                    state="visible", timeout=10_000
                )

                # Le module clavier et sa modale sont chargés dans un vrai navigateur.
                page.locator("#kbd-help").click()
                page.locator("#kbd-cheatsheet:not([hidden])").wait_for(timeout=5_000)
                page.keyboard.press("Escape")

                # Chaîne principale : composer -> /chat -> client OpenAI -> SSE -> rendu.
                page.locator(".pane textarea").fill("salut")
                page.locator(".pane .send-btn").click()
                answer = page.locator(".pane .msg.assistant").last
                expect(answer).to_contain_text("Réponse du stub", timeout=15_000)
                assert not console_errors, f"erreurs console: {console_errors}"
            finally:
                browser.close()

    print("E2E UI VERT: HTTP 200, raccourcis, chat SSE, rendu, zéro erreur console")


if __name__ == "__main__":
    main()
