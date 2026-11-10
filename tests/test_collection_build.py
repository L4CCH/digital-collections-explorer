import json
import sqlite3
import sys

import duckdb
import numpy as np
import pandas
import pytest
import torch

from src.backend.services.collection_service import CollectionService
from src.backend.services.embedding_service import EmbeddingService
from src.backend.services.hybrid_search import HybridSearch
from src.collection import __main__ as command
from src.collection.manifest import build, read_manifest

MODEL = "fake/model"
QUILT_PAGE = "https://n2t.net/ark:/65665/east-1"
FILES = [
    {"id": "a1", "group": "east", "object_id": "rec-quilt", "file_name": "a1.jpg"},
    {"id": "a2", "group": "east", "object_id": "rec-quilt", "file_name": "a2.jpg"},
    {
        "id": "v1",
        "group": "east",
        "object_id": "rec-quilt",
        "view_orbit": "0deg 75deg auto",
    },
    {"id": "b1", "group": "west", "object_id": "rec-rifle", "file_name": "b1.jpg"},
    {"id": "b2", "group": "west", "object_id": None, "file_name": "b2.jpg"},
]
RECORDS = [
    {
        "object_id": "rec-quilt",
        "title": "Star Quilt",
        "source_url": QUILT_PAGE,
        "model_url": "models/quilt.glb",
        "guid": "ark:/65665/east-1",
    },
    {"object_id": "rec-rifle", "title": "Percussion Rifle", "source_url": None},
    {"object_id": "rec-unseen", "title": "A record without a file"},
]
FIELDS = [
    {
        "object_id": "rec-quilt",
        "position": 2,
        "field": "date",
        "label": "Date",
        "value": "ca. 1885",
    },
    {
        "object_id": "rec-quilt",
        "position": 1,
        "field": "name",
        "label": "Maker",
        "value": "Mary Smith",
    },
    {
        "object_id": "rec-rifle",
        "position": 1,
        "field": None,
        "label": "Maker",
        "value": "Remington",
    },
]
FACETS = [
    {"object_id": "rec-quilt", "facet": "date", "value": "1880s"},
    {"object_id": "rec-quilt", "facet": "place", "value": "Ohio"},
    {"object_id": "rec-quilt", "facet": "unit_code", "value": "NMAH"},
    {"object_id": "rec-rifle", "facet": "object_type", "value": "rifle"},
    {"object_id": "rec-rifle", "facet": "name", "value": "Remington"},
]
TEXT = [
    {"object_id": "rec-quilt", "field": "title", "value": "Star Quilt"},
    {"object_id": "rec-quilt", "field": "names", "value": "Mary Smith"},
    {
        "object_id": "rec-quilt",
        "field": "notes",
        "value": "A pieced quilt of sixteen stars",
    },
    {"object_id": "rec-quilt", "field": "identifiers", "value": "rec-quilt"},
    {"object_id": "rec-rifle", "field": "identifiers", "value": "rec-rifle"},
]
RELATED = [
    {
        "object_id": "rec-quilt",
        "source_id": "flickr:1",
        "source": "flickr",
        "link_type": "same_record",
        "url": "https://www.flickr.com/photos/1",
        "title": "Quilt on Flickr",
        "thumbnail": "related/flickr/1.jpg",
        "counts": json.dumps({"faves": 3}),
    }
]
TAGS = [
    {"source_id": "flickr:1", "position": 2, "tag": "stars"},
    {"source_id": "flickr:1", "position": 1, "tag": "quilt"},
]
VECTORS = {
    "fake": {
        "east": (
            ["a1", "a2", "v1"],
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        ),
        "west": (["b1", "b2"], [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
    },
    "other": {
        "east": (["a1", "a2", "v1"], [[1.0, 0.0], [0.7, 0.7], [0.6, 0.8]]),
        "west": (["b1", "b2"], [[0.0, 1.0], [1.0, 1.0]]),
    },
}
MODELS = {
    "default_model": "fake",
    "models": [
        {
            "key": "fake",
            "type": "fake",
            "name": MODEL,
            "dimensions": 3,
            "preprocess": "p1",
        },
        {"key": "other", "type": "fake", "name": "other/model", "dimensions": 2},
    ],
}


def write_table(folder, name, rows):
    duckdb.from_df(pandas.DataFrame(rows)).write_parquet(
        str(folder / f"{name}.parquet")
    )


def write_vectors(folder, vectors=VECTORS, models=MODELS):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "models.json").write_text(json.dumps(models))
    for key, groups in vectors.items():
        (folder / key).mkdir()
        for group, (ids, rows) in groups.items():
            np.save(folder / key / f"{group}.npy", np.array(rows, dtype=np.float32))
            np.save(folder / key / f"{group}.ids.npy", np.array(ids))
    return folder


@pytest.fixture()
def manifest(tmp_path):
    folder = tmp_path / "manifest"
    folder.mkdir()
    for name, rows in (
        ("files", FILES),
        ("records", RECORDS),
        ("fields", FIELDS),
        ("facets", FACETS),
        ("text", TEXT),
        ("related_sources", RELATED),
        ("related_tags", TAGS),
        (
            "groups",
            [
                {"group": "east", "label": "East Wing"},
                {"group": "west", "label": "West Wing"},
            ],
        ),
    ):
        write_table(folder, name, rows)
    (folder / "source.json").write_text(json.dumps({"corpus_build": "test"}))
    return folder, write_vectors(tmp_path / "vectors")


def served(data_dir, monkeypatch):
    from src.backend.services import collection_service, embedding_service

    monkeypatch.setattr(embedding_service.settings, "model_name", MODEL)
    monkeypatch.setattr(collection_service.settings, "data_dir", str(data_dir))
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = data_dir / "embeddings"
    embeddings.load_embeddings()
    return CollectionService(embeddings)


@pytest.fixture()
def built(manifest, tmp_path, monkeypatch):
    folder, vectors = manifest
    data_dir = tmp_path / "site"
    written = build(folder, vectors, data_dir)
    (data_dir / "collection.json").write_text(
        json.dumps({"field_labels": {"date": "Year"}})
    )
    return written, data_dir, served(data_dir, monkeypatch)


def test_a_manifest_is_written_as_format_3_with_every_model_and_group(built):
    written, data_dir, _ = built
    assert (written["items"], written["records"], written["groups"]) == (5, 2, 2)
    assert (written["uncataloged_items"], written["records_without_files"]) == (1, 1)
    info = json.loads((data_dir / "embeddings" / "index_info.json").read_text())
    assert info["format"] == 3 and info["default_model"] == "fake"
    assert [m["key"] for m in info["models"]] == ["fake", "other"]
    assert info["models"][0]["preprocess"] == "p1"
    assert info["groups"] == [
        {"id": "east", "label": "East Wing", "images": 3},
        {"id": "west", "label": "West Wing", "images": 2},
    ]
    assert info["source"] == {"corpus_build": "test"}
    other = np.load(data_dir / "embeddings" / "vectors" / "other" / "west.npy")
    assert np.allclose(np.linalg.norm(other, axis=1), 1.0)


def test_an_object_shows_the_fields_filters_and_related_sources_the_pipeline_gave(
    built,
):
    _, _, collection = built
    quilt = collection.item("a1")
    assert quilt["metadata"]["title"] == "Star Quilt"
    assert quilt["metadata"]["source_url"] == QUILT_PAGE
    assert quilt["record"]["fields"] == [
        {"label": "Maker", "value": "Mary Smith"},
        {"label": "Year", "value": "ca. 1885"},
    ]
    assert quilt["record"]["facets"] == {
        "date": ["1880s"],
        "place": ["Ohio"],
        "unit_code": ["NMAH"],
    }
    (flickr,) = quilt["record"]["related"]
    assert (flickr["tags"], flickr["counts"]) == (["quilt", "stars"], {"faves": 3})
    assert quilt["model"]["file_name"] == "quilt.glb"
    assert [view["id"] for view in quilt["model"]["views"]] == ["v1"]
    info = collection.info()
    assert (info["images"], info["objects"], info["uncataloged_images"]) == (4, 3, 1)
    assert info["catalog_search"] and info["catalog_filters"]


def test_catalog_search_matches_the_text_the_pipeline_gave_and_nothing_else(built):
    _, _, collection = built
    hybrid = HybridSearch(collection)

    def found(query):
        encode = lambda text: torch.nn.functional.normalize(torch.ones(1, 3), dim=-1)
        answer = hybrid.search(query, encode, 10, mode="keyword")
        return [result["object_id"] for result in answer["results"]]

    assert found("sixteen stars") == ["rec-quilt"]
    assert found("rec-rifle") == ["rec-rifle"]
    assert found("ark") == []


def test_the_vocabulary_comes_from_the_filters_of_names_places_topics_and_kinds(built):
    _, data_dir, _ = built
    with sqlite3.connect(data_dir / "embeddings" / "metadata.sqlite") as con:
        vocabulary = set(con.execute("SELECT term, field FROM entity_vocab"))
    assert vocabulary == {
        ("Ohio", "place"),
        ("rifle", "object_type"),
        ("Remington", "name"),
    }


def test_without_catalog_tables_the_files_carry_the_catalog(tmp_path, monkeypatch):
    folder = tmp_path / "manifest"
    folder.mkdir()
    write_table(
        folder,
        "files",
        [
            {"id": "a1", "object_id": "rec-quilt", "title": None, "creator": None},
            {
                "id": "a2",
                "object_id": "rec-quilt",
                "title": "Star Quilt",
                "creator": "Mary Smith",
            },
            {"id": "b1", "object_id": None, "title": "A loose photo", "creator": None},
        ],
    )
    vectors = write_vectors(
        tmp_path / "vectors",
        {
            "fake": {
                "east": (["a1", "a2", "b1"], [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
            }
        },
        {"models": [{"key": "fake", "name": MODEL, "dimensions": 3}]},
    )
    build(folder, vectors, tmp_path / "site")
    collection = served(tmp_path / "site", monkeypatch)
    quilt = collection.item("a1")
    assert quilt["metadata"]["title"] == "Star Quilt"
    assert quilt["record"]["fields"] == [{"label": "Creator", "value": "Mary Smith"}]
    loose = collection.item("b1")
    assert loose["object_id"] == "b1" and loose["metadata"]["title"] == "A loose photo"


def test_a_manifest_that_does_not_add_up_is_refused(manifest, tmp_path):
    folder, vectors = manifest
    write_table(
        folder, "files", [*FILES, {"id": "c1", "group": None, "object_id": None}]
    )
    with pytest.raises(ValueError, match="c1 has no group and no vector"):
        read_manifest(folder, vectors)
    write_table(folder, "files", FILES)
    write_table(
        folder, "text", [{"object_id": "rec-quilt", "field": "colors", "value": "red"}]
    )
    with pytest.raises(ValueError, match="names the field colors"):
        read_manifest(folder, vectors)
    with pytest.raises(ValueError, match="models.json"):
        read_manifest(folder, tmp_path)


def test_files_named_on_this_machine_must_be_in_the_collection_folder(
    manifest, tmp_path
):
    folder, vectors = manifest
    write_table(
        folder,
        "files",
        [{**row, "thumbnail": f"thumbnails/{row['id']}.jpg"} for row in FILES],
    )
    data_dir = tmp_path / "site"
    with pytest.raises(ValueError, match="7 files the manifest names are not in"):
        build(folder, vectors, data_dir, check_files=True)
    for name in (
        "thumbnails/a1.jpg",
        "thumbnails/a2.jpg",
        "thumbnails/v1.jpg",
        "thumbnails/b1.jpg",
        "thumbnails/b2.jpg",
        "models/quilt.glb",
        "related/flickr/1.jpg",
    ):
        (data_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / name).write_bytes(b"x")
    assert build(folder, vectors, data_dir, check_files=True)["items"] == 5


def test_the_command_writes_the_collection_and_its_counts(
    manifest, tmp_path, monkeypatch, capsys
):
    folder, vectors = manifest
    counts = tmp_path / "counts.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.collection",
            "build",
            str(folder),
            "--vectors",
            str(vectors),
            "--data-dir",
            str(tmp_path / "site"),
            "--json",
            str(counts),
        ],
    )
    assert command.main() == 0
    assert "Wrote 5 images, as 3 objects in 2 groups" in capsys.readouterr().out
    assert json.loads(counts.read_text())["related_sources"] == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.collection",
            "build",
            str(folder),
            "--vectors",
            str(tmp_path),
            "--data-dir",
            str(tmp_path / "x"),
        ],
    )
    assert command.main() == 1
