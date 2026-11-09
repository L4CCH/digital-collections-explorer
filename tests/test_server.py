import asyncio
import socket

import pytest


@pytest.fixture()
def backend(monkeypatch):
    monkeypatch.setattr(
        "src.backend.services.embedding_service_factory.create_embedding_service",
        lambda: object(),
    )
    from src.backend import main

    return main


def test_health_identifies_the_collection_being_served(backend, monkeypatch):
    monkeypatch.setattr(backend.settings, "data_dir", "data/collections/demo")
    monkeypatch.setattr(backend.embedding_service, "item_ids", ["a", "b", "c"])
    monkeypatch.setattr(backend.embedding_service, "is_loaded", True)

    health = asyncio.run(backend.health_check())

    assert health["status"] == "healthy"
    assert health["collection"] == "demo"
    assert health["items"] == 3


def test_port_check_detects_a_listening_server(backend):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        busy_port = server.getsockname()[1]
        assert backend.port_in_use(busy_port)
    assert not backend.port_in_use(busy_port)


def test_container_can_name_the_collection_it_serves(backend, monkeypatch):
    monkeypatch.setattr(backend.settings, "data_dir", "/data")
    monkeypatch.setenv("DCE_COLLECTION", "maps-of-ohio")
    assert backend.collection_name() == "maps-of-ohio"


def test_images_may_be_kept_by_browsers_and_a_cdn(backend, monkeypatch, tmp_path):
    from src.backend.api.routes import images

    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"jpeg")
    document = {"metadata": {"paths": {"processed": str(picture)}}}
    monkeypatch.setattr(
        images.embedding_service, "get_document_by_id", lambda item_id: document
    )
    response = asyncio.run(images.get_image_by_id("a", size="full"))
    assert response.headers["cache-control"] == "public, max-age=86400"


def test_an_image_kept_on_an_iiif_server_is_loaded_from_there(backend, monkeypatch):
    from src.backend.api.routes import images

    service = "https://tile.loc.gov/image-services/iiif/service:gmd:a"
    document = {
        "metadata": {
            "paths": {
                "processed": f"{service}/full/2000,/0/default.jpg",
                "thumbnail": f"{service}/full/400,/0/default.jpg",
            }
        }
    }
    monkeypatch.setattr(
        images.embedding_service, "get_document_by_id", lambda item_id: document
    )
    response = asyncio.run(images.get_image_by_id("a", size="thumbnail"))
    assert response.status_code == 307
    assert response.headers["location"] == f"{service}/full/400,/0/default.jpg"
    assert response.headers["cache-control"] == "public, max-age=86400"


def test_an_image_with_only_an_iiif_service_is_sent_to_it(backend, monkeypatch):
    from src.backend.api.routes import images

    service = "https://tile.loc.gov/image-services/iiif/service:gmd:a"
    document = {"metadata": {"iiif": f"{service}/info.json"}}
    monkeypatch.setattr(
        images.embedding_service, "get_document_by_id", lambda item_id: document
    )
    response = asyncio.run(images.get_image_by_id("a", size="thumbnail"))
    assert response.headers["location"] == f"{service}/full/!400,400/0/default.jpg"
    response = asyncio.run(images.get_image_by_id("a", size="full"))
    assert response.headers["location"] == f"{service}/full/!1920,1920/0/default.jpg"


def test_frontend_files_named_by_content_are_kept_and_the_rest_checked(
    backend, tmp_path
):
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-D3ChR7sy.js").write_text("app")
    (tmp_path / "decoders").mkdir()
    (tmp_path / "decoders" / "draco.wasm").write_bytes(b"wasm")
    (tmp_path / "index.html").write_text("<html></html>")
    site = Starlette()
    site.mount("/", backend.FrontendFiles(directory=str(tmp_path), html=True))
    client = TestClient(site)

    script = client.get("/assets/index-D3ChR7sy.js")
    assert script.headers["cache-control"] == "public, max-age=31536000, immutable"
    decoder = client.get("/decoders/draco.wasm")
    assert decoder.headers["cache-control"] == "no-cache"
    unchanged = client.get(
        "/decoders/draco.wasm", headers={"if-none-match": decoder.headers["etag"]}
    )
    assert unchanged.status_code == 304
    assert client.head("/").headers["cache-control"] == "no-cache"
