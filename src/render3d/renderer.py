from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx

from .models import Model, is_remote
from .views import View

PAGE_DIR = Path(__file__).parent / "page"
THREE_DIR = Path(__file__).resolve().parents[2] / "node_modules" / "three"
HOST = "http://render.dce.invalid"
CHROMIUM_ARGS = ["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
CONTENT_TYPES = {
    ".js": "text/javascript",
    ".wasm": "application/wasm",
    ".html": "text/html",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream",
    ".ktx2": "image/ktx2",
}
READY_TIMEOUT_MS = 30_000
HTTP_TIMEOUT_SECONDS = 60
CHUNK_BYTES = 1 << 20


class RendererUnavailable(RuntimeError):
    pass


class ModelRefused(RuntimeError):
    pass


def content_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return (
        CONTENT_TYPES.get(suffix)
        or mimetypes.guess_type(name)[0]
        or "application/octet-stream"
    )


def read_within(root: Path, relative: str) -> bytes:
    root = root.resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ModelRefused(f"{relative} lies outside {root}")
    return path.read_bytes()


def download(url: str, limit: int) -> bytes:
    received = bytearray()
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=HTTP_TIMEOUT_SECONDS
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(CHUNK_BYTES):
            received.extend(chunk)
            if len(received) > limit:
                raise ModelRefused(
                    f"{url} is larger than the {limit / 1024**2:.0f} MB allowed"
                )
    return bytes(received)


class Renderer:
    def __init__(
        self, size: int, background: str, max_model_bytes: int, timeout_s: float
    ):
        self.size = size
        self.background = background
        self.max_model_bytes = max_model_bytes
        self.timeout_ms = int(timeout_s * 1000)
        self.model: Model | None = None
        self.token = ""
        self.loads = 0
        self.playwright = None
        self.browser = None
        self.page = None

    def __enter__(self) -> Renderer:
        if not (THREE_DIR / "build" / "three.module.js").exists():
            raise RendererUnavailable(
                "three.js is missing. Run npm install in the repository folder first."
            )
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RendererUnavailable(
                "Rendering 3D models needs Playwright and its Chromium:\n"
                "  pip install playwright\n"
                "  python -m playwright install chromium"
            ) from error
        self.playwright = sync_playwright().start()
        try:
            self.browser = self.playwright.chromium.launch(args=CHROMIUM_ARGS)
        except Exception as error:
            self.playwright.stop()
            raise RendererUnavailable(
                "Playwright could not start Chromium. Install it with:\n"
                "  python -m playwright install chromium"
            ) from error
        try:
            self.page = self.browser.new_page()
            self.page.route(f"{HOST}/**", self._serve)
            self.page.goto(f"{HOST}/page/render.html")
            self.page.wait_for_function(
                "window.dceReady === true", timeout=READY_TIMEOUT_MS
            )
        except Exception as error:
            self.__exit__()
            raise RendererUnavailable(
                "Chromium started but could not draw with WebGL on this machine."
            ) from error
        return self

    def __exit__(self, *exc) -> None:
        if self.browser is not None:
            self.browser.close()
        if self.playwright is not None:
            self.playwright.stop()

    def _serve(self, route) -> None:
        path = unquote(urlparse(route.request.url).path)
        section, _, rest = path.lstrip("/").partition("/")
        try:
            if section == "page":
                body = read_within(PAGE_DIR, rest)
            elif section == "three":
                body = read_within(THREE_DIR, rest)
            elif section == "model":
                body = self._model_file(rest)
            else:
                raise ModelRefused(f"nothing is served at {path}")
        except (OSError, httpx.HTTPError, ModelRefused) as error:
            route.fulfill(status=404, body=str(error), content_type="text/plain")
            return
        route.fulfill(status=200, body=body, content_type=content_type(rest))

    def _model_file(self, rest: str) -> bytes:
        token, _, name = rest.partition("/")
        if self.model is None or token != self.token:
            raise ModelRefused("that model is no longer loaded")
        address = self.model.address
        if is_remote(address):
            url = address if name == self.model.file_name else urljoin(address, name)
            return download(url, self.max_model_bytes)
        path = Path(address)
        if name == path.name and path.stat().st_size > self.max_model_bytes:
            raise ModelRefused(
                f"{path} is larger than the "
                f"{self.max_model_bytes / 1024**2:.0f} MB allowed"
            )
        return read_within(path.parent, name)

    def load(self, model: Model) -> dict:
        self.loads += 1
        self.model = model
        self.token = hashlib.sha1(model.address.encode("utf-8")).hexdigest()[:16]
        url = f"{HOST}/model/{self.token}/{quote(model.file_name)}"
        self.page.evaluate(
            """([url, load]) => {
                window.dceResult = null;
                window.dceLoad(url).then(
                    (info) => { window.dceResult = { load, info }; },
                    (error) => {
                        window.dceResult = { load, error: String(error?.message || error) };
                    },
                );
            }""",
            [url, self.loads],
        )
        self.page.wait_for_function(
            f"window.dceResult?.load === {self.loads}",
            timeout=self.timeout_ms,
            polling=200,
        )
        result = self.page.evaluate("window.dceResult")
        if "error" in result:
            raise ModelRefused(result["error"])
        return result["info"]

    def render(self, view: View, mode: str) -> tuple[bytes | None, float]:
        result = self.page.evaluate(
            "options => window.dceRender(options)",
            {
                "theta": view.theta,
                "phi": view.phi,
                "mode": mode,
                "size": self.size,
                "background": self.background,
            },
        )
        image = result.get("image")
        data = base64.b64decode(image.split(",", 1)[1]) if image else None
        return data, float(result["coverage"])
