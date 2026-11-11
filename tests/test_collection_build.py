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
from src.collection.package import PROFILE, Refused, build, check

MODEL = "fake/model"
QUILT_PAGE = "https://n2t.net/ark:/65665/east-1"
PRODUCER = {
    "name": "dce-source-test",
    "version": "1.2.0",
    "url": "https://example.org/dce-source-test",
}
PROVENANCE = {"corpus_build": "20261026T063627Z", "base_bundle": "20261026T063627Z.2"}
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
TABLES = {
    "files": FILES,
    "records": RECORDS,
    "fields": FIELDS,
    "facets": FACETS,
    "text": TEXT,
    "related_sources": RELATED,
    "related_tags": TAGS,
}
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
MODELS = [
    {"key": "fake", "type": "clip", "name": MODEL, "revision": "abc", "dimensions": 3},
    {
        "key": "other",
        "type": "siglip2",
        "name": "other/model",
        "revision": "def",
        "dimensions": 2,
    },
]
COLLECTION = {
    "title": "Two Wings",
    "field_labels": {"date": "Year"},
    "groups": {"west": {"label": "West Wing"}, "east": {"label": "East Wing"}},
}
TYPES = {
    "VARCHAR": "string",
    "BIGINT": "integer",
    "INTEGER": "integer",
    "DOUBLE": "number",
    "BOOLEAN": "boolean",
}


def table(package, name):
    columns = duckdb.sql(
        f"DESCRIBE SELECT * FROM '{package}/{name}.parquet'"
    ).fetchall()
    return {
        "name": name,
        "type": "table",
        "path": f"{name}.parquet",
        "format": "parquet",
        "mediatype": "application/vnd.apache.parquet",
        "schema": {
            "fields": [{"name": c[0], "type": TYPES.get(c[1], "any")} for c in columns]
        },
    }


def write_table(package, name, rows):
    duckdb.from_df(pandas.DataFrame(rows)).write_parquet(
        str(package / f"{name}.parquet")
    )
    return table(package, name)


def write_vectors(package, key, shards):
    paths = []
    for shard, (ids, rows) in shards.items():
        path = f"vectors/{key}/{shard}.npy"
        (package / path).parent.mkdir(parents=True, exist_ok=True)
        np.save(package / path, np.array(rows, dtype=np.float32))
        np.save(package / f"vectors/{key}/{shard}.ids.npy", np.array(ids))
        paths.append(path)
    return paths


def write_package(
    package, tables=TABLES, models=MODELS, vectors=VECTORS, collection=COLLECTION, **dce
):
    package.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "$schema": PROFILE,
        "name": "two-wings",
        "resources": [
            write_table(package, name, rows) for name, rows in tables.items()
        ],
        "dce": {
            "contract": 1,
            "producer": PRODUCER,
            "provenance": PROVENANCE,
            "default_model": models[0]["key"],
            "models": [
                {**m, "vectors": write_vectors(package, m["key"], vectors[m["key"]])}
                for m in models
            ],
            **dce,
        },
    }
    (package / "datapackage.json").write_text(json.dumps(descriptor, indent=2))
    if collection is not None:
        (package / "collection.json").write_text(json.dumps(collection))
    return package


def problems_with(package, edit):
    path = package / "datapackage.json"
    original = path.read_text()
    descriptor = json.loads(original)
    edit(descriptor)
    path.write_text(json.dumps(descriptor))
    try:
        return check(package).problems
    finally:
        path.write_text(original)


def run(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["src.collection", *map(str, arguments)])
    return command.main()


def served(data_dir, monkeypatch, model=MODEL):
    from src.backend.services import collection_service, embedding_service

    monkeypatch.setattr(embedding_service.settings, "model_name", model)
    monkeypatch.setattr(collection_service.settings, "data_dir", str(data_dir))
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = data_dir / "embeddings"
    embeddings.load_embeddings()
    return CollectionService(embeddings)


def info_of(data_dir):
    return json.loads((data_dir / "embeddings" / "index_info.json").read_text())


@pytest.fixture()
def package(tmp_path):
    return write_package(tmp_path / "package")


@pytest.fixture()
def built(package, tmp_path, monkeypatch):
    data_dir = tmp_path / "site"
    written, _ = build(package, data_dir)
    return written, data_dir, served(data_dir, monkeypatch)


def test_a_package_is_written_as_format_3_with_its_models_groups_and_producer(built):
    written, data_dir, _ = built
    assert (written["items"], written["records"], written["groups"]) == (5, 2, 2)
    assert (written["uncataloged_items"], written["records_without_files"]) == (1, 1)
    info = info_of(data_dir)
    assert info["format"] == 3 and info["default_model"] == "fake"
    assert info["models"] == [
        {
            "key": "fake",
            "type": "clip",
            "name": MODEL,
            "revision": "abc",
            "dimensions": 3,
        },
        {
            "key": "other",
            "type": "siglip2",
            "name": "other/model",
            "revision": "def",
            "dimensions": 2,
        },
    ]
    assert info["groups"] == [
        {"id": "west", "label": "West Wing", "images": 2},
        {"id": "east", "label": "East Wing", "images": 3},
    ]
    assert (info["producer"], info["provenance"]) == (PRODUCER, PROVENANCE)
    assert json.loads((data_dir / "collection.json").read_text()) == COLLECTION
    other = np.load(data_dir / "embeddings" / "vectors" / "other" / "west.npy")
    assert np.allclose(np.linalg.norm(other, axis=1), 1.0)


def test_an_object_shows_the_fields_filters_and_related_sources_the_package_gave(
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
    assert quilt["record"]["facets"] == {"date": ["1880s"], "place": ["Ohio"]}
    (flickr,) = quilt["record"]["related"]
    assert (flickr["tags"], flickr["counts"]) == (["quilt", "stars"], {"faves": 3})
    assert quilt["model"]["file_name"] == "quilt.glb"
    assert [view["id"] for view in quilt["model"]["views"]] == ["v1"]
    info = collection.info()
    assert (info["images"], info["objects"], info["uncataloged_images"]) == (4, 3, 1)
    assert info["catalog_search"] and info["catalog_filters"]


def test_catalog_search_matches_the_text_the_package_gave_and_nothing_else(built):
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
    files = [
        {"id": "a1", "object_id": "rec-quilt", "title": None, "creator": None},
        {
            "id": "a2",
            "object_id": "rec-quilt",
            "title": "Star Quilt",
            "creator": "Mary Smith",
        },
        {"id": "b1", "object_id": None, "title": "A loose photo", "creator": None},
    ]
    package = write_package(
        tmp_path / "package",
        tables={"files": files},
        models=[MODELS[0]],
        vectors={
            "fake": {
                "all": (["a1", "a2", "b1"], [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
            }
        },
        collection=None,
    )
    build(package, tmp_path / "site")
    collection = served(tmp_path / "site", monkeypatch)
    quilt = collection.item("a1")
    assert quilt["metadata"]["title"] == "Star Quilt"
    assert quilt["record"]["fields"] == [{"label": "Creator", "value": "Mary Smith"}]
    loose = collection.item("b1")
    assert loose["object_id"] == "b1" and loose["metadata"]["title"] == "A loose photo"
    assert info_of(tmp_path / "site")["groups"] == [
        {"id": "collection", "label": "collection", "images": 3}
    ]
    assert not (tmp_path / "site" / "collection.json").exists()


def test_a_package_without_a_contract_or_with_a_newer_one_is_refused(
    package, tmp_path, monkeypatch, capsys
):
    assert problems_with(package, lambda d: d["dce"].pop("contract")) == [
        "datapackage.json names no contract. This DCE reads collection package "
        'contract 1: give it "dce": {"contract": 1, ...}.'
    ]
    assert problems_with(package, lambda d: d["dce"].update(contract=2)) == [
        "The package follows collection package contract 2, which is newer than "
        "this DCE reads. This DCE reads collection package contract 1: update DCE, "
        "or write the package with a version of its plugin that writes contract 1."
    ]
    descriptor = json.loads((package / "datapackage.json").read_text())
    descriptor["dce"]["contract"] = 2
    (package / "datapackage.json").write_text(json.dumps(descriptor))
    assert run(monkeypatch, "check", package) == 2
    assert run(monkeypatch, "build", package, "--data-dir", tmp_path / "site") == 2
    assert "contract 2, which is newer" in capsys.readouterr().out
    assert not (tmp_path / "site").exists()


def test_group_ids_are_written_as_python_m_src_ingest_writes_them(tmp_path):
    groups = {"east": "East Wing", "west": "+++"}
    files = [{**row, "group": groups[row["group"]]} for row in FILES]
    package = write_package(tmp_path / "package", tables={**TABLES, "files": files})
    assert check(package).problems == [
        'The files table puts 3 files in the group "East Wing", which is not a group '
        "id: write it as east-wing, in lowercase letters, digits, and hyphens, as "
        "python -m src.ingest does.",
        'The files table puts 2 files in the group "+++", which is not a group id: '
        "write it in lowercase letters, digits, and hyphens, as python -m src.ingest "
        "does.",
    ]


def test_special_columns_are_read_by_their_exact_names_and_aliases_refused(tmp_path):
    files = [
        {
            "id": row["id"],
            "record_id": row["object_id"],
            "Group": row["group"],
            "iiif_url": "https://iiif.example.org/iiif/2/a",
            "thumbnail_url": "https://example.org/a.jpg",
            "group_id": "x",
            "size": 120,
        }
        for row in FILES
    ]
    records = [{**row, "glb": "models/a.glb", "iiif": "x"} for row in RECORDS]
    package = write_package(
        tmp_path / "package", tables={**TABLES, "files": files, "records": records}
    )
    assert check(package).problems == [
        "The files table has a column record_id, which DCE does not read: name it "
        "object_id.",
        "The files table has a column Group, which DCE does not read: name it group.",
        "The files table has a column iiif_url, which DCE does not read: name it iiif.",
        "The files table has a column thumbnail_url, which DCE does not read: name it "
        "thumbnail.",
        "The files table has a column group_id, which DCE does not read: name it "
        "object_id if it names the object each file shows, or give it as group in "
        "the files table if it names a group, such as a museum.",
        "The files table has a column size, which DCE does not read: leave it out, or "
        "name it dimensions if it measures the object.",
        "The records table has a column glb, which DCE does not read: name it "
        "model_url.",
        "The records table has a column iiif, which DCE does not read there: give it "
        "as iiif in the files table.",
    ]


def test_a_source_url_is_an_http_or_https_address(tmp_path):
    records = [
        (
            {**row, "source_url": "ark:/65665/east-1"}
            if row["title"] == "Star Quilt"
            else row
        )
        for row in RECORDS
    ]
    package = write_package(tmp_path / "package", tables={**TABLES, "records": records})
    assert check(package).problems == [
        "The records table gives 1 record a source_url that is not an http:// or "
        "https:// address, such as rec-quilt: give the web address of each record's "
        "page, or leave it empty."
    ]


def test_iiif_is_kept_as_the_info_json_of_its_service(tmp_path):
    files = [
        {
            **row,
            "iiif": f"https://iiif.example.org/iiif/2/{row['id']}/full/max/0/default.jpg",
        }
        for row in FILES
    ]
    package = write_package(tmp_path / "package", tables={**TABLES, "files": files})
    build(package, tmp_path / "site")
    with sqlite3.connect(tmp_path / "site" / "embeddings" / "metadata.sqlite") as con:
        (kept,) = con.execute("SELECT iiif FROM items WHERE id = 'a1'").fetchone()
    assert kept == "https://iiif.example.org/iiif/2/a1/info.json"
    files[0]["iiif"] = "iiif-a1"
    package = write_package(tmp_path / "unserved", tables={**TABLES, "files": files})
    assert check(package).problems == [
        "The files table gives 1 file an iiif value that is not a web address, such as "
        "a1: give the address of each file's IIIF image service."
    ]


def test_a_model_is_one_the_site_can_search_with(package):
    assert problems_with(
        package, lambda d: d["dce"]["models"][1].update(type="fake")
    ) == [
        'The model other has the type "fake", which DCE cannot search with: give '
        "clip, siglip, or siglip2."
    ]


def test_facets_are_the_six_the_site_filters_by(tmp_path):
    facets = [
        *FACETS,
        {"object_id": "rec-quilt", "facet": "unit_code", "value": "NMAH"},
    ]
    package = write_package(tmp_path / "package", tables={**TABLES, "facets": facets})
    assert check(package).problems == [
        "The facets table has the facet unit_code, which the site does not filter by: "
        "give date, name, place, culture, topic, or object_type. A group already "
        "gives a filter by institution."
    ]


def test_text_names_only_the_fields_search_matches(tmp_path):
    text = [*TEXT, {"object_id": "rec-quilt", "field": "colors", "value": "red"}]
    package = write_package(tmp_path / "package", tables={**TABLES, "text": text})
    assert check(package).problems == [
        "The text table names the field colors, which search does not match: give "
        "title, names, dates, places, topics, object_types, notes, physical, or "
        "identifiers."
    ]


def test_every_file_needs_a_vector_from_every_model(tmp_path):
    vectors = {**VECTORS, "other": {**VECTORS["other"], "west": (["b1"], [[0.0, 1.0]])}}
    package = write_package(tmp_path / "package", vectors=vectors)
    assert check(package).problems == [
        "1 file has no vector from other, such as b2: every file needs a vector from "
        "every model."
    ]


def test_records_files_and_vectors_that_do_not_meet_are_reported(
    tmp_path, monkeypatch, capsys
):
    files = [*FILES, {"id": "c1", "group": "west", "object_id": "rec-lost"}]
    vectors = {
        key: {
            **shards,
            "west": (
                [*shards["west"][0], "c1", "x9"],
                [*shards["west"][1], *[shards["west"][1][0]] * 2],
            ),
        }
        for key, shards in VECTORS.items()
    }
    package = write_package(
        tmp_path / "package", tables={**TABLES, "files": files}, vectors=vectors
    )
    report = check(package)
    assert report.problems == []
    assert report.examples == {
        "records_without_files": ["rec-unseen"],
        "files_without_records": ["c1"],
        "vectors_without_files": ["x9"],
    }
    checked = tmp_path / "check.json"
    assert run(monkeypatch, "check", package, "--json", checked) == 0
    out = capsys.readouterr().out
    assert "  1 record without a file is left out, such as rec-unseen.\n" in out
    assert (
        "  1 file whose object has no row in the records table is shown without a "
        "catalog record, such as c1.\n"
    ) in out
    assert "  2 vectors without a file are left out, such as x9.\n" in out
    found = json.loads(checked.read_text())
    assert (
        found["records_without_files"],
        found["files_without_records"],
        found["vectors_without_files"],
    ) == (1, 1, 2)
    counts = tmp_path / "counts.json"
    data_dir = tmp_path / "site"
    assert (
        run(monkeypatch, "build", package, "--data-dir", data_dir, "--json", counts)
        == 0
    )
    out = capsys.readouterr().out
    assert "  2 vectors without a file are left out, such as x9.\n" in out
    written = json.loads(counts.read_text())
    assert (
        written["records_without_files"],
        written["files_without_records"],
        written["vectors_without_files"],
    ) == (1, 1, 2)
    assert served(data_dir, monkeypatch).item("c1")["metadata"]["cataloged"] is False


def test_a_collection_that_ingest_wrote_is_replaced_only_with_overwrite(
    package, tmp_path, monkeypatch, capsys
):
    data_dir = tmp_path / "site"
    (data_dir / "embeddings").mkdir(parents=True)
    for name in ("ingest_state.sqlite", "ingest_state.sqlite-wal"):
        (data_dir / "embeddings" / name).write_bytes(b"state")
    with pytest.raises(Refused, match="that python -m src.ingest wrote"):
        build(package, data_dir)
    assert run(monkeypatch, "build", package, "--data-dir", data_dir) == 2
    assert "--overwrite" in capsys.readouterr().out
    assert not (data_dir / "embeddings" / "index_info.json").exists()
    args = ("build", package, "--data-dir", data_dir, "--overwrite")
    assert run(monkeypatch, *args) == 0
    assert info_of(data_dir)["producer"] == PRODUCER
    assert not list((data_dir / "embeddings").glob("ingest_state.sqlite*"))


def test_build_writes_only_into_a_collection_folder_outside_the_package(package):
    for data_dir in (package, package / "site"):
        with pytest.raises(Refused, match="is inside the package"):
            build(package, data_dir)
    assert not (package / "embeddings").exists()
    assert not (package / "site").exists()


def test_a_shape_index_is_copied_into_the_collection_and_removed_with_it(tmp_path):
    package = write_package(tmp_path / "package", shape="shape")
    shape = package / "shape"
    shape.mkdir()
    np.save(shape / "vectors.npy", np.eye(2, dtype=np.float32))
    (shape / "objects.json").write_text(json.dumps(["rec-quilt", "rec-rifle"]))
    (shape / "info.json").write_text(json.dumps({"model_name": "shape/model"}))
    data_dir = tmp_path / "site"
    build(package, data_dir)
    copied = data_dir / "embeddings" / "shape"
    assert json.loads((copied / "objects.json").read_text()) == [
        "rec-quilt",
        "rec-rifle",
    ]
    assert np.array_equal(np.load(copied / "vectors.npy"), np.eye(2))
    assert (copied / "info.json").exists()
    assert check(package).counts["shape_objects"] == 2
    (shape / "objects.json").write_text(json.dumps(["rec-quilt"]))
    assert check(package).problems == [
        "The shape index in shape has 2 rows in vectors.npy, but 1 object in "
        "objects.json."
    ]
    build(write_package(tmp_path / "without"), data_dir)
    assert not copied.exists()


def test_a_package_that_does_not_follow_the_standard_is_refused(package):
    assert problems_with(package, lambda d: d.pop("$schema")) == [
        f'datapackage.json must give "$schema": "{PROFILE}", the version of the Data '
        "Package standard it follows."
    ]

    def unlisted(descriptor):
        fields = descriptor["resources"][1]["schema"]["fields"]
        fields[:] = [f for f in fields if f["name"] != "title"]

    assert problems_with(package, unlisted) == [
        "records.parquet has the column title, which the Table Schema of records "
        "does not list."
    ]

    def reordered(descriptor):
        descriptor["resources"][1]["schema"]["fields"].reverse()

    assert problems_with(package, reordered) == [
        "The Table Schema of records lists the columns of records.parquet in another "
        "order: list them in the order of the file."
    ]

    def outside(descriptor):
        descriptor["resources"][0]["path"] = "../files.parquet"

    assert problems_with(package, outside) == [
        "The resource files has the path ../files.parquet, which is not a file inside "
        "the package: give a path relative to the package folder, such as "
        "files.parquet."
    ]
    assert problems_with(package, lambda d: d["dce"]["producer"].pop("version")) == [
        "dce.producer must name the plugin that wrote the package and its version, as "
        '{"name": "my-plugin", "version": "1.0.0"}.'
    ]
    assert problems_with(package, lambda d: d["dce"].update(provenance="x")) == [
        'dce.provenance must be a JSON object, such as {"commit": "4f2a9c1"}.'
    ]
    assert problems_with(
        package, lambda d: d["dce"]["models"][0].update(dimensions=4)
    ) == [
        "vectors/fake/east.npy has rows of 3 numbers, but the model fake has 4 "
        "dimensions.",
        "vectors/fake/west.npy has rows of 3 numbers, but the model fake has 4 "
        "dimensions.",
    ]


def test_tables_and_vectors_that_cannot_be_read_are_refused_in_sentences(package):
    (package / "vectors" / "fake" / "west.ids.npy").unlink()
    np.save(
        package / "vectors" / "other" / "west.ids.npy",
        np.array(["b1", None], dtype=object),
        allow_pickle=True,
    )
    (package / "facets.parquet").write_text("not parquet")
    problems = check(package).problems
    assert problems[0].startswith("facets.parquet could not be read as parquet: ")
    assert problems[1:] == [
        "vectors/fake/west.npy has no ids beside it: write the id of the file in each "
        "of its rows to vectors/fake/west.ids.npy.",
        "vectors/other/west.ids.npy could not be read as the ids of "
        "vectors/other/west.npy: Object arrays cannot be loaded when "
        "allow_pickle=False. Write them as a NumPy array of strings.",
    ]


def test_related_sources_are_listed_once_with_counts_as_a_json_object(tmp_path):
    related = [*RELATED, {**RELATED[0], "counts": "faves: 3"}]
    package = write_package(
        tmp_path / "package", tables={**TABLES, "related_sources": related}
    )
    assert check(package).problems == [
        "The related_sources table lists 1 source more than once for the same object, "
        "such as flickr:1 for rec-quilt.",
        "The related_sources table gives 1 source counts that are not a JSON object, "
        "such as flickr:1.",
    ]


def test_check_notes_what_the_site_will_not_use(tmp_path):
    files = [{**row, "caption": "a caption"} for row in FILES]
    collection = {**COLLECTION, "groups": {**COLLECTION["groups"], "north": {}}}
    package = write_package(
        tmp_path / "package",
        tables={**TABLES, "files": files},
        models=[{k: v for k, v in m.items() if k != "revision"} for m in MODELS],
        collection=collection,
    )
    descriptor = json.loads((package / "datapackage.json").read_text())
    descriptor["resources"].append({"name": "provenance", "path": "provenance.csv"})
    (package / "datapackage.json").write_text(json.dumps(descriptor))
    report = check(package)
    assert report.problems == []
    assert report.notes == [
        "DCE does not read the resource provenance; it reads files, records, fields, "
        "facets, text, related_sources, and related_tags.",
        "The model fake records no revision, so the site may load another version of "
        "fake/model than the one that made the vectors.",
        "The model other records no revision, so the site may load another version "
        "of other/model than the one that made the vectors.",
        "DCE does not read the column caption of the files table, since the records "
        "table holds the catalog.",
        "collection.json describes the group north, which no file is in.",
    ]


def test_files_named_on_this_machine_must_be_in_the_collection_folder(tmp_path):
    files = [{**row, "thumbnail": f"thumbnails/{row['id']}.jpg"} for row in FILES]
    package = write_package(tmp_path / "package", tables={**TABLES, "files": files})
    data_dir = tmp_path / "site"
    with pytest.raises(Refused, match="7 files the package names are not in"):
        build(package, data_dir, check_files=True)
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
    assert build(package, data_dir, check_files=True)[0]["items"] == 5


def test_the_command_checks_a_package_then_builds_it(
    package, tmp_path, monkeypatch, capsys
):
    checked = tmp_path / "check.json"
    assert run(monkeypatch, "check", package, "--json", checked) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        f"{package} can be built: it follows collection package contract 1 and was "
        "written by dce-source-test 1.2.0.\n  It holds 5 files in 2 groups, 3 "
        "records, and the vectors of fake and other.\n"
    )
    found = json.loads(checked.read_text())
    assert (found["can_build"], found["producer"], found["provenance"]) == (
        True,
        PRODUCER,
        PROVENANCE,
    )
    counts = tmp_path / "counts.json"
    data_dir = tmp_path / "site"
    assert (
        run(monkeypatch, "build", package, "--data-dir", data_dir, "--json", counts)
        == 0
    )
    assert "Wrote 5 images, as 3 objects in 2 groups" in capsys.readouterr().out
    assert json.loads(counts.read_text())["related_sources"] == 1
    (package / "files.parquet").unlink()
    assert run(monkeypatch, "check", package, "--json", checked) == 2
    assert (
        "  - The resource files names files.parquet, which is not in the package."
        in (capsys.readouterr().out)
    )
    assert json.loads(checked.read_text())["can_build"] is False


HARBOR_RESOURCES = [
    {
        "name": "files",
        "type": "table",
        "path": "files.parquet",
        "format": "parquet",
        "mediatype": "application/vnd.apache.parquet",
        "schema": {
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "group", "type": "string"},
                {"name": "object_id", "type": "string"},
                {"name": "file_name", "type": "string"},
                {"name": "width", "type": "integer"},
                {"name": "height", "type": "integer"},
                {"name": "thumbnail", "type": "string"},
                {"name": "iiif", "type": "string"},
            ]
        },
    },
    {
        "name": "records",
        "type": "table",
        "path": "records.parquet",
        "format": "parquet",
        "mediatype": "application/vnd.apache.parquet",
        "schema": {
            "fields": [
                {"name": "object_id", "type": "string"},
                {"name": "title", "type": "string"},
                {"name": "source_url", "type": "string"},
                {"name": "creator", "type": "string"},
                {"name": "date", "type": "string"},
            ]
        },
    },
]
HARBOR_DCE = {
    "contract": 1,
    "producer": {
        "name": "dce-source-harbor",
        "version": "0.3.0",
        "url": "https://github.com/example/dce-source-harbor",
    },
    "provenance": {"api_export": "2026-11-10", "commit": "4f2a9c1"},
    "default_model": "siglip2",
    "models": [
        {
            "key": "siglip2",
            "type": "siglip2",
            "name": "google/siglip2-base-patch16-224",
            "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            "dimensions": 768,
            "vectors": ["vectors/siglip2/paintings.npy", "vectors/siglip2/maps.npy"],
        }
    ],
}
HARBOR_COLLECTION = {
    "title": "Harbor Museum",
    "institution": "the Harbor Museum",
    "license": {
        "name": "CC0 1.0",
        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
    },
    "group_label": "Department",
    "groups": {
        "paintings": {"label": "Paintings"},
        "maps": {"label": "Maps and Charts"},
    },
}


def test_check_accepts_what_a_plugin_following_the_page_on_plugins_writes(
    tmp_path, monkeypatch
):
    package = tmp_path / "harbor-museum"
    package.mkdir()
    files = [
        {
            "id": "p1",
            "group": "paintings",
            "object_id": "obj-1",
            "file_name": "p1.jpg",
            "width": 1600,
            "height": 1200,
            "thumbnail": "thumbnails/p1.jpg",
            "iiif": None,
        },
        {
            "id": "p2",
            "group": "paintings",
            "object_id": "obj-1",
            "file_name": "p2.jpg",
            "width": 1600,
            "height": 1100,
            "thumbnail": "thumbnails/p2.jpg",
            "iiif": None,
        },
        {
            "id": "m1",
            "group": "maps",
            "object_id": "obj-2",
            "file_name": "m1.tif",
            "width": 6000,
            "height": 4000,
            "thumbnail": None,
            "iiif": "https://iiif.harbor.example.org/iiif/2/m1",
        },
    ]
    records = [
        {
            "object_id": "obj-1",
            "title": "Harbor at Dusk",
            "source_url": "https://collections.harbor.example.org/objects/1",
            "creator": "Ada Lane",
            "date": "1885",
        },
        {
            "object_id": "obj-2",
            "title": "Chart of the Bay",
            "source_url": "https://collections.harbor.example.org/objects/2",
            "creator": None,
            "date": "1790s",
        },
    ]
    resources = [
        write_table(package, "files", files),
        write_table(package, "records", records),
    ]
    assert resources == HARBOR_RESOURCES
    rows = np.random.default_rng(7).normal(size=(3, 768)).astype(np.float32)
    write_vectors(
        package,
        "siglip2",
        {"paintings": (["p1", "p2"], rows[:2]), "maps": (["m1"], rows[2:])},
    )
    descriptor = {
        "$schema": "https://datapackage.org/profiles/2.0/datapackage.json",
        "name": "harbor-museum",
        "title": "Harbor Museum",
        "version": "2026.11.11",
        "created": "2026-11-11T09:30:00Z",
        "resources": resources,
        "dce": HARBOR_DCE,
    }
    (package / "datapackage.json").write_text(json.dumps(descriptor, indent=2))
    (package / "collection.json").write_text(json.dumps(HARBOR_COLLECTION, indent=2))

    report = check(package)
    assert (report.problems, report.notes) == ([], [])
    data_dir = tmp_path / "harbor-site"
    build(package, data_dir)
    info = info_of(data_dir)
    assert info["groups"] == [
        {"id": "paintings", "label": "Paintings", "images": 2},
        {"id": "maps", "label": "Maps and Charts", "images": 1},
    ]
    assert info["provenance"] == HARBOR_DCE["provenance"]
    chart = served(data_dir, monkeypatch, "google/siglip2-base-patch16-224").item("m1")
    assert (
        chart["metadata"]["iiif"]
        == "https://iiif.harbor.example.org/iiif/2/m1/info.json"
    )
    assert chart["metadata"]["title"] == "Chart of the Bay"
