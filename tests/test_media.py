import json

import pytest
import torch

from src.backend.core.config import (
    FRONTENDS_DIR,
    REPO_ROOT,
    built_frontend,
    load_config,
)
from src.backend.services.catalog_fields import iiif_info_from, servable_model
from src.backend.services.collection_service import CollectionService
from src.backend.services.collection_store import upgrade_old_pages
from src.backend.services.embedding_service import EmbeddingService
from src.environment import media_shown_by
from src.profiling.models import FieldProfile, ScanStats
from src.profiling.profiler import media_needed

SERVICE = "https://iiif.example.org/iiif/3/map-1"


def configured(tmp_path, values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    return load_config(path)


def test_a_frontend_is_a_folder_in_src_frontend_or_a_path_to_one(tmp_path):
    assert built_frontend("default") == FRONTENDS_DIR / "default" / "dist"
    assert built_frontend("../atlas") == REPO_ROOT / ".." / "atlas" / "dist"
    assert built_frontend(str(tmp_path)) == tmp_path / "dist"


@pytest.mark.parametrize(
    "values, frontend, media",
    [
        ({}, "default", []),
        ({"frontend": "atlas", "media": ["3d", "pdf"]}, "atlas", ["3d", "pdf"]),
        ({"collection_type": "maps"}, "default", ["zoom"]),
        ({"collection_type": "documents"}, "default", ["pdf"]),
        ({"collection_type": "maps", "media": []}, "default", []),
    ],
)
def test_the_config_names_a_frontend_and_what_it_shows(
    tmp_path, values, frontend, media
):
    loaded = configured(tmp_path, values)
    assert (loaded.frontend, loaded.media) == (frontend, media)


def test_a_built_frontend_says_what_it_shows(tmp_path):
    assert media_shown_by(tmp_path) is None
    (tmp_path / "frontend.json").write_text('{"frontend": "default", "media": ["3d"]}')
    assert media_shown_by(tmp_path) == ["3d"]


@pytest.mark.parametrize(
    "value, info",
    [
        (f"{SERVICE}/info.json", f"{SERVICE}/info.json"),
        (SERVICE, f"{SERVICE}/info.json"),
        (f"{SERVICE}/", f"{SERVICE}/info.json"),
        (f"{SERVICE}/full/max/0/default.jpg", f"{SERVICE}/info.json"),
        (f"{SERVICE}/0,0,512,512/!400,400/90/gray.png?x=1", f"{SERVICE}/info.json"),
        ("map-1", None),
        ("", None),
    ],
)
def test_an_iiif_image_service_is_found_from_any_of_its_addresses(value, info):
    assert iiif_info_from({"iiif_url": value}) == info


def test_a_record_that_names_an_iiif_service_can_be_zoomed(tmp_path):
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = str(tmp_path)
    embeddings.item_ids = ["map", "photo"]
    embeddings.metadata = {
        "map": {"catalog": {"title": "Map", "iiif": SERVICE}},
        "photo": {"catalog": {"title": "Photo"}},
    }
    embeddings.embeddings = torch.nn.functional.normalize(torch.rand(2, 4), dim=-1)
    embeddings.is_loaded = True
    collection = CollectionService(embeddings)
    assert collection.item("map")["metadata"]["iiif"] == f"{SERVICE}/info.json"
    assert "iiif" not in collection.item("photo")["metadata"]


def column(name):
    return FieldProfile(name=name)


@pytest.mark.parametrize(
    "stats, fields, media",
    [
        (ScanStats(image_files=10), [column("title")], []),
        (ScanStats(document_files=2), [], ["pdf"]),
        (ScanStats(model_files=1), [], ["3d"]),
        (ScanStats(), [column("Model_URL")], ["3d"]),
        (ScanStats(document_files=1), [column("iiif_url")], ["pdf", "zoom"]),
    ],
)
def test_the_profile_says_which_kinds_of_material_to_build_for(stats, fields, media):
    assert media_needed(stats, fields) == media


@pytest.mark.parametrize(
    "address, servable",
    [
        ("https://example.org/vase.gltf", True),
        ("https://example.org/vase.glb", True),
        ("/data/vase.glb", True),
        ("/data/vase.gltf", False),
        ("s3://bucket/vase.glb", False),
        (None, False),
    ],
)
def test_the_site_shows_only_models_it_can_serve(address, servable):
    assert servable_model(address) == (address if servable else None)


def test_a_model_a_manifest_names_by_path_is_found_beside_it(tmp_path):
    import duckdb

    from src.ingest.pipeline import catalog_metadata
    from src.profiling.sources import ParquetManifestSource

    manifest = tmp_path / "manifest.parquet"
    duckdb.execute(
        "COPY (SELECT * FROM (VALUES ('photos/a.jpg', 'models/a.glb', 'a'), "
        "('photos/b.jpg', 'models/b.gltf', 'b')) AS t(image_url, model_url, object_id)) "
        f"TO '{manifest}' (FORMAT PARQUET)"
    )
    source = ParquetManifestSource(str(manifest))
    first, second = (ref.extra["row"] for ref in source.iter_files())
    assert catalog_metadata(first, source)["model_url"] == str(
        tmp_path / "models" / "a.glb"
    )
    assert "model_url" not in catalog_metadata(second, source)


def collection_of(metadata, tmp_path, groups=None):
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = str(tmp_path)
    embeddings.item_ids = list(metadata)
    embeddings.metadata = metadata
    embeddings.embeddings = torch.nn.functional.normalize(
        torch.rand(len(metadata), 4), dim=-1
    )
    if groups:
        embeddings.groups = [{"id": g, "label": g} for g in sorted(set(groups))]
        embeddings.group_of_row = torch.tensor(
            [sorted(set(groups)).index(g) for g in groups]
        )
    embeddings.is_loaded = True
    return CollectionService(embeddings)


def test_pages_from_an_index_made_before_version_2_are_one_pdf(tmp_path):
    metadata = upgrade_old_pages(
        {
            f"old_{n}": {
                "file_name": "report.pdf",
                "type": "pdf_page",
                "page": n,
                "n_pages": 2,
                "paths": {"original": "data/raw/report.pdf"},
            }
            for n in range(2)
        }
    )
    metadata["photo"] = {"file_name": "a.jpg", "type": "image"}
    collection = collection_of(metadata, tmp_path)
    info = collection.info()
    assert (info["documents"], info["images"], info["objects"]) == (1, 1, 2)
    pages = collection.item("old_1")["photos"]
    assert [p["metadata"]["page"] for p in pages] == [1, 2]
    assert pages[0]["metadata"]["document"] == "data/raw/report.pdf"


def model_object(tmp_path):
    record = {"object_id": "vase", "model_url": "/data/vase.glb"}
    view = {**record, "view_orbit": "0deg 75deg auto"}
    return collection_of(
        {
            "view-1": {"catalog": view, "object_id": "vase", **view},
            "photo": {"catalog": record, "object_id": "vase", **record},
            "view-2": {"catalog": view, "object_id": "vase", **view},
        },
        tmp_path,
        groups=["museum", "museum", "museum"],
    )


def test_views_drawn_from_a_model_are_not_counted_as_images(tmp_path):
    info = model_object(tmp_path).info()
    assert info["images"] == 1
    assert info["groups"][0]["images"] == 1


def test_an_object_found_by_its_record_is_shown_by_its_photo(tmp_path):
    collection = model_object(tmp_path)
    assert collection.embeddings.item_ids[collection.first_image("vase")] == "photo"
