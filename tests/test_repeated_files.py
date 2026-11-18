import sqlite3

import duckdb
from PIL import Image

from src.profiling.profiler import profile_collection
from src.profiling.report import render
from src.profiling.sources import ParquetManifestSource
from tests.test_related import ingest

ROWS = (
    "('images/0.png', 'obj-a', 'A harbor'), ('images/1.png', 'obj-b', 'A boat'), "
    "('images/0.png', 'obj-b', 'A boat'), ('images/0.png', 'obj-c', 'A chapel')"
)


def collection(tmp_path):
    (tmp_path / "images").mkdir()
    for index in range(2):
        Image.new("RGB", (64, 48), (index * 90, 60, 30)).save(
            tmp_path / "images" / f"{index}.png"
        )
    duckdb.execute(
        f"COPY (SELECT * FROM (VALUES {ROWS}) AS t(image_url, object_id, title)) "
        f"TO '{tmp_path / 'manifest.parquet'}' (FORMAT PARQUET)"
    )
    return ParquetManifestSource(str(tmp_path / "manifest.parquet"))


def test_a_file_several_rows_name_is_indexed_once_for_the_first_row(
    tmp_path, monkeypatch, capsys
):
    collection(tmp_path)
    related = tmp_path / "related.parquet"
    duckdb.execute(
        f"COPY (SELECT 'obj-a' AS object_id, 'flickr' AS source, 'flickr:1' AS "
        f"source_id) TO '{related}' (FORMAT PARQUET)"
    )
    code, data_dir = ingest(tmp_path, monkeypatch, related)
    assert code == 0
    out = capsys.readouterr().out
    assert (
        "The manifest names 1 file in more than one row, 3 rows in all; each file is "
        "indexed once, for the first row that names it. That leaves 1 object without "
        "an image, such as obj-c; it is left out." in out
    )
    assert "Found 2 images." in out and "Indexed 2," in out
    con = sqlite3.connect(data_dir / "embeddings" / "metadata.sqlite")
    shown = con.execute(
        "SELECT items.file_name, records.record_id, records.title FROM items "
        "JOIN records USING (record_id) ORDER BY items.file_name"
    ).fetchall()
    assert shown == [("0.png", "obj-a", "A harbor"), ("1.png", "obj-b", "A boat")]


def test_the_profile_counts_each_file_once_and_names_the_objects_left_out(tmp_path):
    source = collection(tmp_path)
    profile = profile_collection(source, "test", sample=2, progress=False)
    assert profile.scan.image_files == 2
    issues = {issue.kind: issue for issue in profile.scan.issues}
    assert issues["repeated_file"].count == 1
    assert (issues["no_image_left"].count, issues["no_image_left"].examples) == (
        1,
        ["obj-c"],
    )
    report = render(profile)
    assert (
        "files named by more than one row, 3 rows in all; each is indexed once, for "
        "the first row that names it (e.g. images/0.png)" in report
    )
    assert (
        "objects left out, since earlier rows name all their images (e.g. obj-c)"
        in (report)
    )
