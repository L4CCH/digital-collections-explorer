import json

import duckdb
import numpy as np
import pandas
import pytest
import torch
from PIL import Image

from src.backend.services.collection_service import CollectionService
from src.backend.services.embedding_service import EmbeddingService
from src.backend.services.hybrid_search import HybridSearch
from src.backend.services.site_pages import llms_txt
from src.collection.catalog import (
    GROUP_ID_LEFT_OUT,
    GROUP_ID_NAMES_OBJECTS,
    filed_value,
    group_id_warning,
    kind_of,
)
from src.collection.writer import Group, Item, Model, write_collection
from src.ingest.__main__ import STATE_FILE, foreign_index_exists
from src.ingest.export import group_of, write_from_state
from src.ingest.pipeline import Options, make_item_id, run
from src.ingest.state import IngestState
from src.profiling.models import Provenance
from src.profiling.profiler import profile_collection
from src.profiling.sources import ParquetManifestSource

MODEL = "test-model"
VECTOR_SIZE = 8
QUILT_PAGE = "https://museum.example/art/1985.12"
MAKER = "Mary Smith (American, 1850–1920)"
COLORS = {
    "quilt-front": (200, 30, 30),
    "quilt-back": (180, 40, 40),
    "teapot": (30, 30, 200),
    "unknown": (30, 200, 30),
}
ROWS = [
    {
        "image_url": "files/quilt-front.jpg",
        "object_id": "1985.12",
        "group": "East Wing",
        "title": "Star Quilt",
        "creators": MAKER,
        "creation_date": "c. 1885",
        "place": "[Ohio],",
        "subjects": "Quilts; Stars",
        "classification": "Textile",
        "accession_number": "1985.12",
        "department": "Textiles",
        "source_url": QUILT_PAGE,
    },
    {"image_url": "files/quilt-back.jpg", "object_id": "1985.12", "group": "East Wing"},
    {
        "image_url": "files/teapot.jpg",
        "group": "west",
        "title": "Teapot with roses",
        "creation_date": "1890s",
    },
    {"image_url": "files/unknown.jpg", "group": "west"},
]
CONFIGURED = {
    "title": "Two houses",
    "groups": {
        "west": {
            "label": "West House",
            "institution": "the West House",
            "license": {"name": "CC0 1.0"},
        },
        "East Wing": {
            "institution": "the East Wing Museum",
            "license": {"name": "CC BY 4.0", "note": "credit the museum"},
        },
    },
    "field_labels": {"creation_date": "Date", "creators": "Maker"},
}


class FakeService:
    def encode_image(self, images):
        vectors = torch.rand(len(images), VECTOR_SIZE) + 0.1
        return vectors / vectors.norm(dim=-1, keepdim=True)


def encode(text):
    return torch.nn.functional.normalize(torch.ones(1, VECTOR_SIZE), dim=-1)


def write_manifest(path, rows):
    duckdb.from_df(pandas.DataFrame(rows)).write_parquet(str(path))
    return path


@pytest.fixture()
def manifest(tmp_path):
    root = tmp_path / "source"
    (root / "files").mkdir(parents=True)
    for name, color in COLORS.items():
        Image.new("RGB", (640, 480), color).save(root / "files" / f"{name}.jpg")
    return write_manifest(root / "manifest.parquet", ROWS)


def ingest(manifest, data_dir):
    source = ParquetManifestSource(str(manifest))
    refs = sorted(
        (r for r in source.iter_files() if r.is_image or r.is_document),
        key=lambda r: r.key,
    )
    state = IngestState(str(data_dir / "embeddings" / STATE_FILE), commit_every=2)
    state.seed([(make_item_id(r.key), r.key, r.size) for r in refs])
    options = Options(
        thumbnails_dir=data_dir / "thumbnails",
        processed_dir=data_dir / "processed",
        data_dir=data_dir,
        download_workers=2,
        decode_workers=2,
        batch_size=2,
    )
    run(source, refs, FakeService(), state, options)
    return state


def served(data_dir, monkeypatch):
    from src.backend.services import collection_service, embedding_service

    monkeypatch.setattr(embedding_service.settings, "model_name", MODEL)
    monkeypatch.setattr(collection_service.settings, "data_dir", str(data_dir))
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = data_dir / "embeddings"
    embeddings.load_embeddings()
    return CollectionService(embeddings)


@pytest.fixture()
def ingested(manifest, tmp_path, monkeypatch):
    data_dir = tmp_path / "site"
    state = ingest(manifest, data_dir)
    written = write_from_state(state, data_dir / "embeddings", "fake", MODEL)
    (data_dir / "collection.json").write_text(json.dumps(CONFIGURED))
    return written, served(data_dir, monkeypatch)


def test_ingest_writes_groups_records_and_what_to_filter_by(ingested):
    written, collection = ingested
    assert (written["items"], written["records"], written["groups"]) == (4, 2, 2)
    assert written["uncataloged_items"] == 1
    assert written["filters"] == ["date", "name", "object_type", "place", "topic"]
    info = collection.info()
    assert [(g["id"], g["label"], g["institution"]) for g in info["groups"]] == [
        ("west", "West House", "the West House"),
        ("east-wing", "East Wing", "the East Wing Museum"),
    ]
    assert info["catalog_search"] and info["catalog_filters"]
    assert (info["objects"], info["uncataloged_images"]) == (3, 1)


def test_an_object_shows_its_catalog_under_the_labels_collection_json_gives(
    ingested,
):
    _, collection = ingested
    item = collection.item(make_item_id("files/quilt-front.jpg"))
    assert item["object_id"] == "1985.12" and item["image_count"] == 2
    assert item["metadata"]["title"] == "Star Quilt"
    assert item["metadata"]["source_url"] == QUILT_PAGE
    fields = [(f["label"], f["value"]) for f in item["record"]["fields"]]
    assert fields == [
        ("Maker", MAKER),
        ("Date", "c. 1885"),
        ("Place", "[Ohio],"),
        ("Subjects", "Quilts; Stars"),
        ("Classification", "Textile"),
        ("Accession number", "1985.12"),
        ("Department", "Textiles"),
    ]
    assert item["record"]["facets"] == {
        "date": ["1885"],
        "name": [MAKER],
        "object_type": ["Textile"],
        "place": ["Ohio"],
        "topic": ["Quilts", "Stars"],
    }


def test_the_catalog_is_searched_by_its_words_but_not_by_ids_made_up_for_it(
    ingested,
):
    _, collection = ingested
    hybrid = HybridSearch(collection)

    def found(query, **options):
        answer = hybrid.search(query, encode, 10, mode="keyword", **options)
        return [result["object_id"] for result in answer["results"]]

    teapot = make_item_id("files/teapot.jpg")
    assert found("Mary Smith") == ["1985.12"]
    assert found("1985.12") == ["1985.12"]
    assert found("roses", groups=["west"]) == [teapot]
    assert found("roses", groups=["east-wing"]) == []
    assert found(teapot) == []


def test_a_file_without_any_catalog_is_its_own_object_without_a_record(ingested):
    _, collection = ingested
    collection._ensure_index()
    assert make_item_id("files/unknown.jpg") in collection.uncataloged_objects
    assert make_item_id("files/teapot.jpg") not in collection.uncataloged_objects


def test_llms_txt_names_each_institution_and_its_license(ingested):
    _, collection = ingested
    text = llms_txt(collection.info())
    assert "The images come from the West House and the East Wing Museum." in text
    assert (
        "Licenses: CC0 1.0 for West House; "
        "CC BY 4.0 (credit the museum) for East Wing." in text
    )
    assert "Each image belongs to one group: West House (`west`)" in text


def test_the_profile_says_what_the_site_will_do_with_each_column(manifest):
    profile = profile_collection(
        ParquetManifestSource(str(manifest)), "test", sample=4, progress=False
    )
    roles = {
        f.name: (f.read_as, f.filtered_as)
        for f in profile.fields
        if f.provenance == Provenance.CATALOG
    }
    assert roles["creators"] == ("names", "name")
    assert roles["creation_date"] == ("dates", "date")
    assert roles["department"] == ("notes", None)
    assert roles["source_url"] == ("link", None)
    assert roles["image_url"] == ("file", None)
    assert roles["group"] == ("group", None)
    assert roles["object_id"] == ("object", None)
    assert profile.capabilities.facets == [
        "date",
        "name",
        "object_type",
        "place",
        "topic",
    ]
    assert {"catalog_search", "titles", "links_back", "objects", "group_filter"} <= set(
        profile.capabilities.available
    )
    assert profile.structure.orphan_items == 1
    assert profile.structure.groups == {"East Wing": 2, "west": 2}


def test_a_pdf_with_a_record_is_one_cataloged_object_of_pages(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    from tests.test_documents import write_pdf

    root = tmp_path / "source"
    root.mkdir()
    write_pdf(root / "report.pdf", 3)
    manifest = write_manifest(
        root / "manifest.parquet",
        [{"image_url": "report.pdf", "object_id": "rep-1", "title": "Annual report"}],
    )
    data_dir = tmp_path / "site"
    write_from_state(ingest(manifest, data_dir), data_dir / "embeddings", "fake", MODEL)
    collection = served(data_dir, monkeypatch)
    item = collection.item(make_item_id("report.pdf#page=2"))
    assert item["object_id"] == "rep-1" and item["image_count"] == 3
    page = item["metadata"]
    assert (page["type"], page["page"], page["pages"]) == ("pdf_page", 2, 3)
    assert page["document"] == str(root / "report.pdf")
    assert page["title"] == "Annual report"
    assert collection.info()["documents"] == 1
    assert "rep-1" not in collection.uncataloged_objects


def test_an_image_served_by_iiif_keeps_its_service_in_format_3(tmp_path, monkeypatch):
    service = "https://iiif.example/image/x"
    write_collection(
        tmp_path / "embeddings",
        [Model("fake", "fake", MODEL, 3)],
        [Group("collection", "collection")],
        [
            Item(
                "x",
                "collection",
                thumbnail="thumbnails/x.jpg",
                iiif=f"{service}/info.json",
            )
        ],
        [],
        {"fake": (["x"], np.array([[1.0, 0.0, 0.0]]))},
    )
    collection = served(tmp_path, monkeypatch)
    shown = collection.item("x")["metadata"]
    assert shown["iiif"] == f"{service}/info.json"
    assert shown["image_urls"] == {"full": f"{service}/full/!1920,1920/0/default.jpg"}


def test_a_row_without_a_file_address_is_read_from_its_iiif_service(tmp_path):
    path = write_manifest(
        tmp_path / "manifest.parquet",
        [
            {"image_url": "a.jpg", "iiif": None},
            {"image_url": None, "iiif": "https://iiif.example/image/b/info.json"},
            {"image_url": None, "iiif": None},
        ],
    )
    refs = list(ParquetManifestSource(str(path)).iter_files())
    assert [r.extra["named_by_service"] for r in refs] == [False, True]
    assert refs[1].extra["url"] == "https://iiif.example/image/b/full/max/0/default.jpg"


def test_a_pdf_on_the_web_is_read_like_a_local_one(tmp_path, monkeypatch):
    from src.profiling import sources

    monkeypatch.setattr(
        sources, "fetch_web", lambda url, limit=None: (b"%PDF-1.7", "application/pdf")
    )
    path = write_manifest(
        tmp_path / "manifest.parquet",
        [
            {"image_url": "https://library.example/catalog.pdf"},
            {"image_url": "https://library.example/photo.jpg"},
        ],
    )
    source = ParquetManifestSource(str(path))
    pdf, photo = sorted(source.iter_files(), key=lambda r: r.key)
    assert pdf.is_document and source.read_bytes(pdf) == b"%PDF-1.7"
    assert source.read_bytes(photo) is None


def test_ingest_leaves_a_collection_it_did_not_write_alone(tmp_path):
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    assert not foreign_index_exists(embeddings)
    (embeddings / "index_info.json").write_text(json.dumps({"format": 2}))
    assert foreign_index_exists(embeddings)
    (embeddings / STATE_FILE).write_text("")
    assert not foreign_index_exists(embeddings)


def test_a_group_keeps_its_own_name_and_a_safe_id():
    assert group_of({"group": "Cleveland Museum of Art"}) == (
        "cleveland-museum-of-art",
        "Cleveland Museum of Art",
    )
    assert group_of({"Group": "國立故宮博物院"}) == ("國立故宮博物院", "國立故宮博物院")
    assert group_of({"group": "../x"}) == ("x", "../x")
    assert group_of({"group": "  "}) == ("collection", "collection")
    assert group_of({}) == ("collection", "collection")


def test_a_group_id_column_still_names_objects_and_says_so():
    assert group_id_warning(["image_url", "group_id"]) == GROUP_ID_NAMES_OBJECTS
    assert group_id_warning(["object_id", "group_id"]) == GROUP_ID_LEFT_OUT.format(
        column="object_id"
    )
    assert group_id_warning(["group", "group_id"]) is None
    assert group_id_warning(["image_url"]) is None


def test_filter_values_lose_the_marks_of_cataloging_rules_but_not_abbreviations():
    assert filed_value("[Japan],") == "Japan"
    assert filed_value("[Venezia [Venice]]") == "Venezia [Venice]"
    assert filed_value("[Paris] and [Lyon]") == "[Paris] and [Lyon]"
    assert filed_value(" Ohio ; ") == "Ohio"
    assert filed_value("Richmond, Me.") == "Richmond, Me."
    assert kind_of("creation_date") == "dates"
    assert kind_of("Measurements") == "physical"
