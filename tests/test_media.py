import json

import pytest
import torch

from src.backend.core.config import (
    FRONTENDS_DIR,
    REPO_ROOT,
    built_frontend,
    load_config,
)
from src.backend.services import catalog_fields
from src.backend.services.catalog_fields import iiif_info_from
from src.backend.services.collection_service import CollectionService
from src.backend.services.embedding_service import EmbeddingService
from src.environment import media_shown_by
from src.profiling import sources
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


def test_the_profiler_and_the_site_agree_on_column_names():
    assert sources.MODEL_COLUMNS == catalog_fields.MODEL_URL_KEYS
    assert sources.IIIF_COLUMNS == catalog_fields.IIIF_KEYS


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
