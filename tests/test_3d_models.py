import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from PIL import Image

from src.backend.services.catalog_fields import model_url_from, view_orbit_from
from src.backend.services.collection_service import CollectionService
from src.backend.services.embedding_service import EmbeddingService
from src.backend.services.shape_index import ShapeIndex, write_shape_index
from src.ingest.pipeline import Options, make_item_id, run
from src.ingest.state import IngestState
from src.profiling.sources import ParquetManifestSource
from src.render3d.manifest import write_manifest
from src.render3d.models import list_folder, list_manifest

MODEL = "https://example.org/models/stele-150k.glb"
PAGE = "https://example.org/objects/stele"


def test_a_model_is_found_under_common_names_and_must_be_gltf():
    assert model_url_from({"model_url": MODEL}) == MODEL
    assert model_url_from({"glb": "models/vase.GLB"}) == "models/vase.GLB"
    assert model_url_from({"3d_model": "scene.gltf?token=1"}) == "scene.gltf?token=1"
    assert model_url_from({"model_url": "https://example.org/viewer/3d"}) is None
    assert model_url_from({"model": None, "title": "vase.glb"}) is None
    assert view_orbit_from({"view_orbit": " 45deg 75deg auto "}) == "45deg 75deg auto"
    assert view_orbit_from({"view_orbit": ""}) is None


def test_a_folder_lists_its_glb_models_and_the_photos_beside_them(tmp_path):
    (tmp_path / "caves").mkdir()
    (tmp_path / "caves" / "relief.glb").write_bytes(b"glTF")
    (tmp_path / "vase.glb").write_bytes(b"glTF")
    (tmp_path / "scan.obj").write_text("v 0 0 0")
    Image.new("RGB", (4, 4)).save(tmp_path / "vase.jpg")

    listing = list_folder(str(tmp_path))

    assert [m.object_id for m in listing.models] == ["caves/relief", "vase"]
    assert listing.models[0].address == str(tmp_path / "caves" / "relief.glb")
    assert [p["image_url"] for p in listing.photos] == [str(tmp_path / "vase.jpg")]
    assert len(listing.problems) == 1 and "scan.obj" in listing.problems[0]


def test_a_manifest_lists_each_model_once_with_its_record(tmp_path):
    (tmp_path / "catalog.csv").write_text(
        "object_id,title,model_url,image_url\n"
        "stele-1,Stele,models/stele.glb,\n"
        "stele-1,Stele,models/stele.glb,photos/front.jpg\n"
        ",Bowl,,scans/bowl.glb\n"
        "cave-2,Cave,models/cave.gltf,\n"
    )

    listing = list_manifest(str(tmp_path / "catalog.csv"))

    stele, bowl = listing.models
    assert stele.address == str(tmp_path / "models" / "stele.glb")
    assert (stele.object_id, stele.row["title"]) == ("stele-1", "Stele")
    assert "image_url" not in stele.row
    assert (bowl.object_id, bowl.address) == ("bowl", str(tmp_path / "scans/bowl.glb"))
    assert [p["image_url"] for p in listing.photos] == [
        str(tmp_path / "photos" / "front.jpg")
    ]
    assert len(listing.problems) == 1 and "cave.gltf" in listing.problems[0]


class FakeService:
    def encode_image(self, images):
        vectors = torch.ones(len(images), 4)
        return vectors / vectors.norm(dim=-1, keepdim=True)


def test_rendered_views_listed_in_a_manifest_are_indexed_as_views_of_the_model(
    tmp_path,
):
    views = tmp_path / "collection" / "models3d"
    (views / "views").mkdir(parents=True)
    Image.new("RGB", (300, 120), (200, 190, 180)).save(views / "views" / "00.jpg")
    manifest = views / "views.parquet"
    write_manifest(
        manifest,
        [
            {
                "image_url": "views/00.jpg",
                "size_bytes": 1000,
                "object_id": "stele-1",
                "title": "Stele",
                "model_url": MODEL,
                "view_orbit": "0deg 75deg auto",
            }
        ],
    )
    source = ParquetManifestSource(str(manifest))
    (ref,) = list(source.iter_files())
    assert ref.key == "views/00.jpg"
    assert source.read_bytes(ref)[:3] == b"\xff\xd8\xff"

    data_dir = tmp_path / "collection"
    state = IngestState(str(data_dir / "embeddings" / "state.sqlite"), commit_every=1)
    state.seed([(make_item_id(ref.key), ref.key, ref.size)])
    options = Options(
        thumbnails_dir=data_dir / "thumbnails",
        processed_dir=data_dir / "processed",
        download_workers=1,
        decode_workers=1,
        batch_size=1,
    )
    stats, _ = run(source, [ref], FakeService(), state, options)

    assert stats.done == 1
    ((_, _, metadata),) = list(state.done_rows())
    assert metadata["object_id"] == "stele-1"
    assert metadata["model_url"] == MODEL
    assert metadata["view_orbit"] == "0deg 75deg auto"
    assert metadata["paths"]["original"] == str(views / "views" / "00.jpg")


def view(orbit):
    return {
        "object_id": "stele",
        "title": "Stele",
        "model_url": MODEL,
        "view_orbit": orbit,
        "width": 400,
        "height": 160,
    }


@pytest.fixture()
def collection(tmp_path):
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = str(tmp_path)
    embeddings.item_ids = ["stele-0", "stele-90", "stele-photo", "bowl-0", "vase"]
    embeddings.metadata = {
        "stele-0": view("0deg 75deg auto"),
        "stele-90": view("90deg 75deg auto"),
        "stele-photo": {"object_id": "stele", "title": "Stele", "source_url": PAGE},
        "bowl-0": {
            "object_id": "bowl",
            "model_url": "/models/bowl.glb",
            "view_orbit": "0deg 75deg auto",
        },
        "vase": {"object_id": "vase", "title": "Vase"},
    }
    embeddings.embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.5, 0.5]]),
        dim=-1,
    )
    embeddings.is_loaded = True
    return CollectionService(embeddings)


def test_an_object_counts_its_photos_but_not_the_views_of_its_model(collection):
    assert collection.item("stele-0")["image_count"] == 1
    assert collection.item("bowl-0")["image_count"] == 0
    assert collection.item("stele-photo")["metadata"]["model_url"] == MODEL
    assert "model_url" not in collection.item("vase")["metadata"]


def test_the_collection_counts_photos_and_models_apart(collection):
    info = collection.info()
    assert (info["objects"], info["images"], info["models"]) == (3, 2, 2)


def test_an_item_names_its_model_and_where_each_view_was_taken_from(collection):
    item = collection.item("stele-photo")
    assert item["model"] == {
        "id": "stele-0",
        "url": MODEL,
        "file_name": "stele-150k.glb",
        "views": [
            {"id": "stele-0", "orbit": "0deg 75deg auto"},
            {"id": "stele-90", "orbit": "90deg 75deg auto"},
        ],
    }
    assert [photo["id"] for photo in item["photos"]] == [
        "stele-0",
        "stele-90",
        "stele-photo",
    ]
    assert collection.item("bowl-0")["model"]["url"] is None
    assert "model" not in collection.item("vase")


def test_the_wall_shows_a_photo_of_an_object_before_any_view_of_its_model(
    collection,
):
    for seed in range(10):
        shown = {r["object_id"]: r["id"] for r in collection.sample(10, seed)}
        assert shown == {"stele": "stele-photo", "bowl": "bowl-0", "vase": "vase"}


def test_similar_shapes_come_from_other_objects_with_models(collection, tmp_path):
    vectors = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    write_shape_index(tmp_path, ["stele", "bowl", "lost"], vectors, "fake/dino", 8)

    results = collection.similar_in_shape("stele-photo", 10)

    assert [r["id"] for r in results] == ["bowl-0"]
    assert results[0]["score"] == pytest.approx(0.6)
    assert collection.similar_in_shape("vase", 10) == []
    assert collection.similar_in_shape("missing", 10) is None


def written_at(tmp_path, seconds, objects, vectors):
    write_shape_index(tmp_path, objects, np.array(vectors, dtype=np.float32), "m", 4)
    os.utime(tmp_path / "shape" / "vectors.npy", (seconds, seconds))


def test_the_shape_index_is_read_again_when_it_changes(tmp_path):
    shapes = ShapeIndex()
    assert shapes.neighbors(tmp_path, "a") is None

    written_at(tmp_path, 1, ["a", "b"], [[3.0, 0.0], [1.0, 1.0]])
    assert shapes.neighbors(tmp_path, "a") == [("b", pytest.approx(0.7071, 1e-3))]

    written_at(tmp_path, 2, ["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [1.0, 0.1]])
    assert [o for o, _ in shapes.neighbors(tmp_path, "a")] == ["c", "b"]

    (tmp_path / "shape" / "objects.json").write_text(json.dumps(["a"]))
    os.utime(tmp_path / "shape" / "vectors.npy", (3, 3))
    assert shapes.neighbors(tmp_path, "a") is None


@pytest.fixture()
def images_route(monkeypatch):
    monkeypatch.setattr(
        "src.backend.services.embedding_service_factory.create_embedding_service",
        lambda: object(),
    )
    from src.backend.api.routes import images

    return images


def serve_model(images, monkeypatch, address):
    document = {"metadata": {"model_url": address} if address else {}}
    monkeypatch.setattr(
        images.embedding_service, "get_document_by_id", lambda item_id: document
    )
    return asyncio.run(images.get_model("a"))


def test_a_model_kept_on_the_site_is_served_and_a_remote_one_is_linked(
    images_route, monkeypatch, tmp_path
):
    glb = tmp_path / "stele.glb"
    glb.write_bytes(b"glTF")
    response = serve_model(images_route, monkeypatch, str(glb))
    assert response.media_type == "model/gltf-binary"
    assert Path(response.path) == glb

    response = serve_model(images_route, monkeypatch, MODEL)
    assert response.status_code == 307 and response.headers["location"] == MODEL


@pytest.mark.parametrize("address", [None, "scene.gltf", "/no/such/model.glb"])
def test_only_glb_files_that_exist_are_served(images_route, monkeypatch, address):
    with pytest.raises(HTTPException) as refusal:
        serve_model(images_route, monkeypatch, address)
    assert refusal.value.status_code == 404
