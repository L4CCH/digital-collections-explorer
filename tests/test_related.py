import io
import json
import sqlite3
import sys

import duckdb
from PIL import Image

from src.ingest.pipeline import make_item_id
from src.ingest.related import REQUIRED, related_problems
from tests.test_ingest import FakeService

ROWS = {
    "harbor": "('obj-a', 'flickr', 'flickr:1', 'https://flickr.example/1', "
    "'Harbor at dawn', 'thumbs/1.png', '{\"faves\": 3, \"comments\": 1}', "
    "['harbor', 'boats'], 'images/00.png', 'same_image')",
    "boat": "('obj-b', 'flickr', 'flickr:2', NULL, NULL, 'https://example.org/2.jpg', "
    "NULL, NULL::VARCHAR[], NULL, NULL)",
    "elsewhere": "('obj-z', 'flickr', 'flickr:3', NULL, NULL, NULL, NULL, "
    "NULL::VARCHAR[], NULL, NULL)",
}
COLUMNS = "object_id, source, source_id, url, title, thumbnail, counts, tags, image, link_type"


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (20, 80, 160)).save(buffer, "PNG")
    return buffer.getvalue()


def collection(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "thumbs").mkdir()
    for index in range(3):
        Image.new("RGB", (64, 48), (index * 60, 90, 30)).save(
            tmp_path / "images" / f"{index:02}.png"
        )
    (tmp_path / "thumbs" / "1.png").write_bytes(png())
    duckdb.execute(
        "COPY (SELECT * FROM (VALUES ('images/00.png', 'obj-a', 'A harbor'), "
        "('images/01.png', 'obj-a', 'A harbor'), ('images/02.png', 'obj-b', 'A boat')) "
        f"AS t(image_url, object_id, title)) TO '{tmp_path / 'manifest.parquet'}' "
        "(FORMAT PARQUET)"
    )


def related_table(path, rows, columns=COLUMNS):
    duckdb.execute(
        f"COPY (SELECT * FROM (VALUES {', '.join(rows)}) AS t({columns})) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
    return path


def ingest(tmp_path, monkeypatch, related=None, *options):
    from src.ingest import __main__ as command

    class Service(FakeService):
        model_name = "test-model"
        device = "cpu"

    for name in ("data_dir", "embeddings_dir", "thumbnails_dir", "processed_data_dir"):
        monkeypatch.setattr(command.settings, name, getattr(command.settings, name))
    monkeypatch.setattr(command.settings, "device", command.settings.device)
    monkeypatch.setattr(command, "create_embedding_service", Service)
    monkeypatch.setattr(command, "cloud_machine_type", lambda: None)
    data_dir = tmp_path / "site"
    manifest = str(tmp_path / "manifest.parquet")
    arguments = ["ingest", manifest, "--data-dir", str(data_dir), "--device", "cpu"]
    if related:
        arguments += ["--related", str(related)]
    monkeypatch.setattr(sys, "argv", arguments + list(options))
    return command.main(), data_dir


def test_ingest_writes_related_items_with_thumbnails_tags_and_counts(
    tmp_path, monkeypatch, capsys
):
    collection(tmp_path)
    related = related_table(tmp_path / "related.parquet", list(ROWS.values()))
    web = {"https://example.org/2.jpg": (png(), "image/png")}
    monkeypatch.setattr(
        "src.ingest.related.fetch_web", lambda url, limit=None: web.get(url)
    )
    code, data_dir = ingest(tmp_path, monkeypatch, related)
    assert code == 0
    con = sqlite3.connect(data_dir / "embeddings" / "metadata.sqlite")
    rows = {
        row[0]: row
        for row in con.execute(
            "SELECT source_id, record_id, source, link_type, image_id, url, title, "
            "thumbnail, counts FROM related_sources"
        )
    }
    assert set(rows) == {"flickr:1", "flickr:2"}
    first, second = rows["flickr:1"], rows["flickr:2"]
    assert first[1:5] == (
        "obj-a",
        "flickr",
        "same_image",
        make_item_id("images/00.png"),
    )
    assert first[5:7] == ("https://flickr.example/1", "Harbor at dawn")
    assert json.loads(first[8]) == {"faves": 3, "comments": 1}
    assert (second[1], second[3], second[4]) == ("obj-b", "same_record", None)
    for row in (first, second):
        assert row[7].startswith("related/") and (data_dir / row[7]).exists()
    tags = con.execute(
        "SELECT tag FROM related_tags WHERE source_id = 'flickr:1' ORDER BY position"
    ).fetchall()
    assert [tag for (tag,) in tags] == ["harbor", "boats"]
    out = capsys.readouterr().out
    assert (
        "The related table names objects the manifest does not have in 1 row, "
        "such as obj-z; it is left out." in out
    )


def test_a_related_table_without_source_id_is_refused_before_indexing(
    tmp_path, monkeypatch, capsys
):
    collection(tmp_path)
    related = related_table(
        tmp_path / "related.parquet", ["('obj-a', 'flickr')"], "object_id, source"
    )
    code, data_dir = ingest(tmp_path, monkeypatch, related)
    assert code == 2
    out = capsys.readouterr().out
    assert "The related table lacks source_id, which every row needs" in out
    assert not (data_dir / "embeddings" / "index_info.json").exists()


def test_a_repeated_source_id_is_refused(tmp_path, monkeypatch, capsys):
    collection(tmp_path)
    twice = [ROWS["harbor"], ROWS["harbor"].replace("'obj-a'", "'obj-b'")]
    related = related_table(tmp_path / "related.parquet", twice)
    code, _ = ingest(tmp_path, monkeypatch, related)
    assert code == 2
    assert "repeats the source_id flickr:1; each item needs an id" in (
        capsys.readouterr().out
    )


def test_notes_say_what_is_left_out_and_why():
    columns = list(REQUIRED) + ["image", "counts"]
    row = {"object_id": "obj-a", "source": "flickr", "source_id": "flickr:1"}
    rows = [{**row, "image": "images/09.png", "counts": "not json"}]
    refused, notes = related_problems(columns, rows, {"obj-a"}, {"images/00.png"})
    assert refused == []
    assert notes == [
        "The related table names images the manifest does not list in 1 row, such "
        "as images/09.png; it is kept without the image.",
        'The related table gives counts that are not a JSON object, such as {"faves": '
        "3}, in 1 row; those counts are left out.",
    ]


def test_the_profile_checks_a_related_table_as_ingest_would(tmp_path):
    from src.profiling.profiler import profile_collection, related_items
    from src.profiling.report import render
    from src.profiling.sources import ParquetManifestSource

    collection(tmp_path)
    related = related_table(tmp_path / "related.parquet", list(ROWS.values()))
    source = ParquetManifestSource(str(tmp_path / "manifest.parquet"))
    profile = profile_collection(source, "test", sample=3, progress=False)
    profile.related = related_items(source, str(related))
    report = render(profile)
    assert "3 items for 2 objects: 3 from flickr." in report
    assert "names objects the manifest does not have in 1 row, such as obj-z" in report
    twice = related_table(tmp_path / "twice.parquet", [ROWS["harbor"]] * 2)
    profile.related = related_items(source, str(twice))
    assert (
        "Must fix before indexing: The related table repeats the source_id flickr:1"
        in render(profile)
    )
