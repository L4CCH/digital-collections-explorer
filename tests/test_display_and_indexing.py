import sqlite3

import duckdb
from PIL import Image

from src.profiling.profiler import profile_collection
from src.profiling.report import render
from src.profiling.sources import ParquetManifestSource
from tests.test_related import ingest

COLUMNS = (
    '"image_url", "object_id", "title", "creator: Sitter", "creator: Artist", '
    '"date: Date made", "creator:", "date:", "credit"'
)
ROWS = (
    "('images/0.png', 'npg-1', 'Abraham Lincoln', 'Abraham Lincoln', "
    "'Mathew Brady, 1823 - 1896', 'c. 1864', ['Lincoln, Abraham', 'Brady, Mathew B.'], "
    "'1860s', NULL), "
    "('images/1.png', 'npg-2', 'A harbor', NULL, NULL, NULL, NULL, NULL, 'Gift')"
)


def manifest(tmp_path):
    (tmp_path / "images").mkdir()
    for n in range(2):
        Image.new("RGB", (64, 48), (n * 90, 60, 30)).save(
            tmp_path / "images" / f"{n}.png"
        )
    path = tmp_path / "manifest.parquet"
    duckdb.execute(
        f"COPY (SELECT * FROM (VALUES {ROWS}) AS t({COLUMNS})) TO '{path}' "
        "(FORMAT PARQUET)"
    )
    return ParquetManifestSource(str(path))


def test_a_collection_shows_display_text_and_filters_by_indexing_values(
    tmp_path, monkeypatch
):
    source = manifest(tmp_path)
    assert all(None not in ref.extra["row"].values() for ref in source.iter_files())
    code, data_dir = ingest(tmp_path, monkeypatch)
    assert code == 0
    con = sqlite3.connect(data_dir / "embeddings" / "metadata.sqlite")
    fields = con.execute(
        "SELECT label, value FROM record_fields WHERE record_id = 'npg-1' "
        "ORDER BY position"
    ).fetchall()
    assert fields == [
        ("Sitter", "Abraham Lincoln"),
        ("Artist", "Mathew Brady, 1823 - 1896"),
        ("Date made", "c. 1864"),
    ]
    facets = con.execute(
        "SELECT facet, value FROM record_facets WHERE record_id = 'npg-1' "
        "ORDER BY facet, value"
    ).fetchall()
    assert facets == [
        ("date", "1860s"),
        ("name", "Brady, Mathew B."),
        ("name", "Lincoln, Abraham"),
    ]


def test_the_profile_says_where_each_kind_of_column_goes(tmp_path):
    report = render(profile_collection(manifest(tmp_path), "npg", progress=False))
    assert "names: on the page as Sitter, searched" in report
    assert "names: searched, filter by name, not on the page" in report
    assert "dates: on the page as Date made, searched" in report
