import logging
import os
import socket
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles

from .api import mcp_server
from .api.routes import browse, embeddings, images, search
from .core.config import built_frontend, settings
from .services import site_pages
from .services.collection_service import DEFAULT_COLLECTION, collection_service
from .services.embedding_service import embedding_service
from .services.index_info import (
    describe_mismatch,
    grouped_model,
    is_grouped,
    read_index_info,
)
from .services.limits import refuse_oversized_upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def index_problem():
    """Return why the loaded index cannot be searched with the configured model"""
    info = read_index_info(embedding_service.embeddings_dir)
    if is_grouped(info) and grouped_model(info, settings.model_name) is None:
        return describe_mismatch(
            embedding_service.embeddings_dir, None, settings.model_name, None
        )
    embedding_service.load_embeddings()
    if not embedding_service.is_loaded:
        return None
    model_dimensions = search.model_service.encode_text(["dimension check"]).shape[-1]
    return describe_mismatch(
        embedding_service.embeddings_dir,
        embedding_service.embeddings.shape[1],
        settings.model_name,
        model_dimensions,
    )


@asynccontextmanager
async def lifespan(app):
    logger.info("Initializing services...")

    problem = index_problem()
    if problem:
        logger.error(problem)
        raise RuntimeError(problem)
    search.hybrid_search.warm()

    logger.info(f"Starting API server on {settings.host}:{settings.port}")
    logger.info(f"Debug mode: {settings.debug}")

    async with AsyncExitStack() as stack:
        if settings.mcp:
            mcp_server.allow_images_from(collection_service.image_hosts())
            await stack.enter_async_context(mcp_server.server.session_manager.run())
            logger.info("Answering the Model Context Protocol at /mcp")
        yield


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    root_path=settings.base_path,
)

cors_origins = [
    "http://0.0.0.0:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://digital-collections-explorer.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def send_the_site_root_to_its_folder(request: Request, call_next):
    """Under a base path, /maps must become /maps/ or the page's assets would not load"""
    if settings.base_path and request.url.path == settings.base_path:
        return RedirectResponse(settings.base_path + "/", status_code=307)
    return await call_next(request)


@app.middleware("http")
async def guard_uploads(request: Request, call_next):
    try:
        refuse_oversized_upload(request)
    except HTTPException as refusal:
        return JSONResponse({"detail": refusal.detail}, status_code=refusal.status_code)
    return await call_next(request)


app.include_router(search.router)
app.include_router(browse.router)
app.include_router(images.router)
app.include_router(embeddings.router)

if settings.mcp:
    app.router.routes.extend(mcp_server.http_routes())


@app.get("/api/health")
async def health_check():
    """Health check endpoint, also identifies which collection is being served"""
    return {
        "status": "healthy",
        "collection": collection_name(),
        "items": embedding_service.get_embedding_count(),
        "embeddings_dir": settings.embeddings_dir,
    }


def collection_name():
    """Name of the collection being served, from DCE_COLLECTION or the data folder"""
    if os.environ.get("DCE_COLLECTION"):
        return os.environ["DCE_COLLECTION"]
    return Path(settings.data_dir).name if settings.data_dir else None


def port_in_use(port: int) -> bool:
    """Check whether something is already listening on the port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def collection_description():
    """What collection.json says about the collection, or the defaults if that fails"""
    try:
        return collection_service.info()
    except Exception as e:
        logger.warning(f"Describing the collection with defaults: {e}")
        return dict(DEFAULT_COLLECTION)


@app.get("/robots.txt", include_in_schema=False)
async def robots():
    """Welcome search engines and AI crawlers, and keep them off the search endpoints"""
    return PlainTextResponse(site_pages.robots_txt(settings.base_path))


@app.get("/llms.txt", include_in_schema=False)
async def llms():
    """Describe the site and its API to AI assistants"""
    return PlainTextResponse(
        site_pages.llms_txt(collection_description(), settings.base_path)
    )


frontend_dir = built_frontend(settings.frontend)

CHECK_EVERY_VISIT = "no-cache"
# A year: the build names every file in assets/ by its content, so an update never reuses a name.
KEEP_FOR_A_YEAR = "public, max-age=31536000, immutable"


class FrontendFiles(StaticFiles):
    """The built frontend: files named by their content are kept a year, the rest are checked on every visit"""

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        named_by_content = Path(self.get_path(scope)).parts[:1] == ("assets",)
        response.headers["Cache-Control"] = (
            KEEP_FOR_A_YEAR if named_by_content else CHECK_EVERY_VISIT
        )
        return response


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def home():
    """The home page, with its title and description filled in from collection.json"""
    if not frontend_dir.exists():
        raise HTTPException(status_code=404, detail="The frontend has not been built")
    index_html = site_pages.read_index(frontend_dir)
    return HTMLResponse(
        site_pages.home_page(index_html, collection_description()),
        headers={"Cache-Control": CHECK_EVERY_VISIT},
    )


if frontend_dir.exists():
    app.mount(
        "/", FrontendFiles(directory=str(frontend_dir), html=True), name="frontend"
    )
    logger.info(f"Serving frontend from {frontend_dir}")
else:
    logger.warning(f"Frontend directory not found at {frontend_dir}")
    logger.warning("The API will run without serving the frontend.")

if __name__ == "__main__":
    if port_in_use(settings.port):
        print(
            f"Port {settings.port} is already in use, possibly by another collection.\n"
            f"Leave it running and choose another port, for example:\n"
            f"  DCE_PORT={settings.port + 1} python -m src.backend.main"
        )
        sys.exit(1)

    problem = index_problem()
    if problem:
        print(problem)
        sys.exit(1)

    uvicorn.run(
        "src.backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
