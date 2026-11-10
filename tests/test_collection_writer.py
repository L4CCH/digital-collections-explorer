import json
import sqlite3

import numpy as np
import pytest

from src.backend.services.catalog_filter import matching_records
from src.backend.services.collection_service import CollectionService
from src.backend.services.embedding_service import EmbeddingService
from src.backend.services.hybrid_search import HybridSearch
from src.collection.catalog import filed_dates, kind_of, label_of, read_catalog
from src.collection.writer import (
    Group,
    Item,
    Model,
    Record,
    Related,
    write_collection,
)
from src.environment import index_state
from tests.test_hybrid_search import encode

MODEL = "fake/model"
FAKE = Model(key="fake", type="fake", name=MODEL, dimensions=3)
GROUPS = [Group("east", "East Wing"), Group("west", "West Wing")]
QUILT_PAGE = "https://n2t.net/ark:/65665/east-1"


def item(item_id, group, record):
    return Item(
        item_id,
        group,
        record,
        file_name=f"{item_id}.jpg",
        thumbnail=f"thumbnails/{item_id}.jpg",
        processed=f"processed/{item_id}.jpg",
    )


ITEMS = [
    item("a1", "east", "rec-quilt"),
    item("a2", "east", "rec-quilt"),
    item("a3", "east", "rec-tea"),
    item("b1", "west", "rec-rifle"),
    item("b2", "west", None),
]
VECTORS = {
    "fake": (
        [i.id for i in ITEMS],
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.95, 0.0, 0.05],
                [0.0, 0.0, 2.0],
            ]
        ),
    )
}
QUILT = {
    "title": "Star Quilt",
    "creator": "Smith, Mary; Jones, Ann",
    "date": "ca. 1885",
    "place": "Ohio",
    "culture": "American",
    "subjects": "Quilts | Textiles",
    "medium": "cotton",
    "accession_number": "TE.E388879",
    "description": "A pieced quilt with stars.",
    "source_url": QUILT_PAGE,
    "image_url": "https://example.org/a1.jpg",
    "file_count": 2,
    "Credit Line": "Gift of the maker",
}
RECORDS = [
    Record("rec-quilt", catalog=QUILT),
    Record(
        "rec-tea",
        catalog={"title": "Tea Set", "date": "1880s", "object_type": "tea set"},
    ),
    Record(
        "rec-rifle",
        title="Percussion Rifle",
        fields=[("Maker", "Remington")],
        facets=[("object_type", "rifle"), ("date", "1862")],
    ),
    Record("rec-unseen", catalog={"title": "A record with no file"}),
]


def write(root, **overrides):
    arguments = dict(
        models=[FAKE], groups=GROUPS, items=ITEMS, records=RECORDS, vectors=VECTORS
    )
    arguments.update(overrides)
    return write_collection(root / "embeddings", **arguments)


def loaded(root, monkeypatch):
    from src.backend.services import collection_service, embedding_service

    monkeypatch.setattr(embedding_service.settings, "model_name", MODEL)
    monkeypatch.setattr(collection_service.settings, "data_dir", str(root))
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = root / "embeddings"
    embeddings.load_embeddings()
    return CollectionService(embeddings)


def tables(root):
    with sqlite3.connect(root / "embeddings" / "metadata.sqlite") as con:
        return {name for (name,) in con.execute("SELECT name FROM sqlite_master")}


def test_a_collection_is_written_in_format_3_with_a_shard_per_group(tmp_path):
    written = write(tmp_path)
    info = json.loads((tmp_path / "embeddings" / "index_info.json").read_text())
    assert info["format"] == 3
    assert info["default_model"] == "fake"
    assert [(g["id"], g["images"]) for g in info["groups"]] == [
        ("east", 3),
        ("west", 2),
    ]
    shard = np.load(tmp_path / "embeddings" / "vectors" / "fake" / "west.npy")
    assert np.allclose(np.linalg.norm(shard, axis=1), 1)
    assert written["records"] == 3
    assert written["records_without_files"] == 1
    assert written["uncataloged_items"] == 1
    assert {"record_fields", "record_facets", "records_fts", "entity_vocab"} <= tables(
        tmp_path
    )
    assert "related_sources" not in tables(tmp_path)


def test_the_site_reads_a_written_collection(tmp_path, monkeypatch):
    write(tmp_path)
    collection = loaded(tmp_path, monkeypatch)
    info = collection.info()
    assert info["objects"] == 4
    assert [g["id"] for g in info["groups"]] == ["east", "west"]
    assert info["catalog_search"] and info["catalog_filters"]

    quilt = collection.item("a1")
    assert quilt["metadata"]["title"] == "Star Quilt"
    assert quilt["metadata"]["source_url"] == QUILT_PAGE
    assert [p["id"] for p in quilt["photos"]] == ["a1", "a2"]
    assert [(f["label"], f["value"]) for f in quilt["record"]["fields"]] == [
        ("Creator", "Smith, Mary; Jones, Ann"),
        ("Date", "ca. 1885"),
        ("Place", "Ohio"),
        ("Culture", "American"),
        ("Subjects", "Quilts | Textiles"),
        ("Medium", "cotton"),
        ("Accession number", "TE.E388879"),
        ("Description", "A pieced quilt with stars."),
        ("Credit Line", "Gift of the maker"),
    ]
    assert quilt["record"]["facets"] == {
        "culture": ["American"],
        "date": ["1885"],
        "name": ["Jones, Ann", "Smith, Mary"],
        "place": ["Ohio"],
        "topic": ["Quilts", "Textiles"],
    }


def test_a_pipelines_own_fields_and_filters_are_kept(tmp_path, monkeypatch):
    write(tmp_path)
    rifle = loaded(tmp_path, monkeypatch).item("b1")
    assert rifle["metadata"]["title"] == "Percussion Rifle"
    assert rifle["record"]["fields"] == [{"label": "Maker", "value": "Remington"}]
    assert rifle["record"]["facets"] == {"date": ["1862"], "object_type": ["rifle"]}


def test_search_and_filters_read_the_catalog_of_a_written_collection(
    tmp_path, monkeypatch
):
    write(tmp_path)
    search = HybridSearch(loaded(tmp_path, monkeypatch))
    by_name = search.search("smith", encode, 10, mode="keyword")
    assert [r["object_id"] for r in by_name["results"]] == ["rec-quilt"]
    assert search.search("1885", encode, 10)["search"]["kind"] == "date"

    database = tmp_path / "embeddings" / "metadata.sqlite"
    assert matching_records(database, {"place": ["ohio"]}) == {"rec-quilt"}
    assert matching_records(database, {}, 1880, 1889) == {"rec-quilt", "rec-tea"}


def test_a_collection_without_catalog_text_is_searched_by_its_pictures(
    tmp_path, monkeypatch
):
    plain = [Item(i.id, "all", None, thumbnail=i.thumbnail) for i in ITEMS]
    write(tmp_path, groups=[Group("all", "All")], items=plain, records=[])
    collection = loaded(tmp_path, monkeypatch)
    info = collection.info()
    assert len(info["groups"]) == 1
    assert not info["catalog_search"] and not info["catalog_filters"]
    assert HybridSearch(collection).index is None
    assert "records_fts" not in tables(tmp_path)


def test_writing_again_replaces_the_old_format_and_keeps_indexing_progress(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    for name in ("embeddings.pt", "item_ids.pt", "metadata.json", "catalog.sqlite"):
        (embeddings_dir / name).write_text("old")
    (embeddings_dir / "ingest_state.sqlite").write_text("progress")
    write(tmp_path)
    write(tmp_path, groups=GROUPS[:1], items=ITEMS[:3])
    assert sorted(p.name for p in embeddings_dir.iterdir()) == [
        "index_info.json",
        "ingest_state.sqlite",
        "metadata.sqlite",
        "vectors",
    ]
    shards = sorted(p.name for p in (embeddings_dir / "vectors" / "fake").iterdir())
    assert shards == ["east.ids.npy", "east.npy"]
    assert index_state(embeddings_dir) == "created_by_ingest"


def test_related_sources_are_written_for_the_records_shown(tmp_path, monkeypatch):
    related = [
        Related(
            "rec-quilt",
            "flickr:1",
            "flickr",
            "same_record",
            url="https://www.flickr.com/photos/smithsonian/1/",
            title="Star Quilt on Flickr",
            counts={"faves": 3},
            tags=["quilt", "textiles"],
        ),
        Related("rec-unseen", "flickr:2", "flickr", "same_record"),
    ]
    write(tmp_path, related=related)
    found = loaded(tmp_path, monkeypatch).item("a1")["record"]["related"]
    assert [(r["id"], r["counts"], r["tags"]) for r in found] == [
        ("flickr:1", {"faves": 3}, ["quilt", "textiles"])
    ]


def test_items_without_vectors_or_in_unknown_groups_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no vector"):
        write(
            tmp_path, vectors={"fake": (VECTORS["fake"][0][:4], VECTORS["fake"][1][:4])}
        )
    with pytest.raises(ValueError, match="not listed"):
        write(tmp_path, groups=GROUPS[:1])


def test_catalog_columns_are_read_by_name_whatever_the_source_calls_them():
    catalog = read_catalog(
        {
            "Photographers": "Koch, Augustus",
            "Location": "Seattle, Wash.",
            "Keywords": "Harbors; Ships",
            "Notes": "Drawn from a balloon",
            "Year": 1884,
            "segment_num": 3,
            "mimetype": "image/jp2",
            "group": "west",
            "object_id": "rec-1",
        },
        "rec-1",
    )
    assert [label for _, label, _ in catalog.fields] == [
        "Photographers",
        "Location",
        "Keywords",
        "Notes",
        "Year",
    ]
    assert ("topic", "Ships") in catalog.facets and ("date", "1884") in catalog.facets
    assert catalog.text["dates"] == ["1884", "1880s"]
    assert catalog.text["identifiers"] == ["rec-1"]
    assert kind_of("Date Created") == "dates" and kind_of("keywords") == "subjects"
    assert label_of("date_created") == "Date created"


@pytest.mark.parametrize(
    "date, filed",
    [
        ("1860s", ["1860s"]),
        ("1914-1918", ["1914-1918"]),
        ("ca. 1885", ["1885"]),
        ("printed 1862, issued 1865", ["1862-1865"]),
        ("early 1900s", ["1900s"]),
        ("undated", []),
    ],
)
def test_dates_are_filed_as_the_date_filter_reads_them(date, filed):
    assert filed_dates([date]) == filed


def test_a_smithsonian_bundle_still_reads_as_before(tmp_path, monkeypatch):
    embeddings_dir = tmp_path / "embeddings"
    shard = embeddings_dir / "vectors" / "clip"
    shard.mkdir(parents=True)
    np.save(shard / "nmah.npy", np.eye(3, dtype=np.float32)[:2])
    np.save(shard / "nmah.ids.npy", np.array(["i1", "i2"]))
    with sqlite3.connect(embeddings_dir / "metadata.sqlite") as con:
        con.executescript("""
            CREATE TABLE items (id TEXT PRIMARY KEY, group_id TEXT NOT NULL,
                record_id TEXT, file_name TEXT, width INTEGER, height INTEGER,
                thumbnail TEXT NOT NULL, processed TEXT NOT NULL, original TEXT,
                unit_code TEXT, etag TEXT, guid TEXT);
            CREATE TABLE records (record_id TEXT PRIMARY KEY, title TEXT,
                source_url TEXT, unit_code TEXT, guid TEXT, record_link TEXT,
                rights TEXT, record_type TEXT);
            CREATE TABLE groups (id TEXT PRIMARY KEY, label TEXT NOT NULL,
                images INTEGER NOT NULL);
            CREATE TABLE record_fields (record_id TEXT NOT NULL,
                position INTEGER NOT NULL, field TEXT NOT NULL, label TEXT NOT NULL,
                value TEXT NOT NULL, PRIMARY KEY (record_id, position));
            INSERT INTO items VALUES ('i1', 'nmah', 'edanmdm-1', 'a.jpg', NULL, NULL,
                'thumbnails/a.jpg', 'processed/a.jpg', NULL, 'NMAH', 'e1', 'g1');
            INSERT INTO items VALUES ('i2', 'nmah', NULL, 'b.jpg', NULL, NULL,
                'thumbnails/b.jpg', 'processed/b.jpg', NULL, 'NMAH', 'e2', NULL);
            INSERT INTO records VALUES ('edanmdm-1', 'Saxophone',
                'https://n2t.net/ark:/65665/1', 'NMAH', 'g1', NULL, 'CC0', 'object');
            INSERT INTO groups VALUES ('nmah', 'American History', 2);
            INSERT INTO record_fields VALUES ('edanmdm-1', 0, 'name', 'Maker',
                'Conn');
            """)
    info = {
        "format": 2,
        "models": [{"key": "clip", "type": "clip", "name": MODEL, "dimensions": 3}],
        "default_model": "clip",
        "groups": [{"id": "nmah", "label": "American History", "images": 2}],
    }
    (embeddings_dir / "index_info.json").write_text(json.dumps(info))
    collection = loaded(tmp_path, monkeypatch)
    saxophone = collection.item("i1")
    assert saxophone["metadata"]["title"] == "Saxophone"
    assert saxophone["record"]["fields"] == [{"label": "Maker", "value": "Conn"}]
    assert collection.info()["uncataloged_images"] == 1
