import sqlite3

import duckdb
from PIL import Image

from tests.test_related import ingest

TITLES = [f"Work {n}" for n in range(12)]


def images(tmp_path):
    (tmp_path / "images").mkdir()
    for n in range(len(TITLES)):
        Image.new("RGB", (64, 48), (n * 20, 90, 30)).save(
            tmp_path / "images" / f"{n:02}.png"
        )


def manifest(tmp_path, rows):
    path = tmp_path / "manifest.parquet"
    path.unlink(missing_ok=True)
    values = ", ".join(
        f"('images/{n:02}.png', 'obj-{n}', '{title}')" for n, title in rows
    )
    duckdb.execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS t(image_url, object_id, title)) "
        f"TO '{path}' (FORMAT PARQUET)"
    )


def shown(data_dir):
    con = sqlite3.connect(data_dir / "embeddings" / "metadata.sqlite")
    return dict(con.execute("SELECT record_id, title FROM records"))


def test_running_again_reads_catalog_text_again_and_leaves_out_what_is_gone(
    tmp_path, monkeypatch, capsys
):
    images(tmp_path)
    manifest(tmp_path, enumerate(TITLES))
    code, data_dir = ingest(tmp_path, monkeypatch)
    assert code == 0 and len(shown(data_dir)) == 12
    renamed = [
        (n, "A new title" if n == 0 else title) for n, title in enumerate(TITLES)
    ]
    manifest(tmp_path, renamed[:11])
    capsys.readouterr()
    assert ingest(tmp_path, monkeypatch)[0] == 0
    out = capsys.readouterr().out
    assert "11 already indexed, 0 to go." in out
    assert "1 image no longer listed by the source, left out but kept indexed" in out
    records = shown(data_dir)
    assert len(records) == 11 and "obj-11" not in records
    assert records["obj-0"] == "A new title"


def test_a_source_that_loses_more_than_a_tenth_at_once_stops_until_allowed(
    tmp_path, monkeypatch, capsys
):
    images(tmp_path)
    manifest(tmp_path, enumerate(TITLES))
    assert ingest(tmp_path, monkeypatch)[0] == 0
    manifest(tmp_path, list(enumerate(TITLES))[:9])
    capsys.readouterr()
    code, data_dir = ingest(tmp_path, monkeypatch)
    assert code == 2
    assert (
        "The source lists 9 files now, but 3 files of the 12 indexed before are no "
        "longer among them, such as images/09.png, images/10.png, and images/11.png."
        in capsys.readouterr().out
    )
    assert len(shown(data_dir)) == 12
    assert ingest(tmp_path, monkeypatch, None, "--allow-removals")[0] == 0
    assert len(shown(data_dir)) == 9
