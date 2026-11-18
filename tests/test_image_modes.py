import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import duckdb
import pytest
from PIL import Image

from src.backend.services.catalog_fields import image_source, image_urls
from src.ingest.pipeline import Options, make_item_id, run
from src.ingest.state import IngestState
from src.profiling.models import FieldProfile
from src.profiling.profiler import image_decision, profile_collection
from src.profiling.report import render
from src.profiling.sources import ParquetManifestSource
from tests.test_ingest import FakeService

ORIGINAL = (6000, 4000)


def jpeg(size):
    buffer = io.BytesIO()
    Image.new("RGB", size, (90, 60, 30)).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture()
def server():
    asked = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            asked.append(self.path)
            if self.path.endswith("/info.json"):
                size = {"width": ORIGINAL[0], "height": ORIGINAL[1]}
                body, kind = json.dumps(size).encode(), "application/json"
            elif "/full/!" in self.path:
                edge = int(self.path.split("/full/!")[1].split(",")[0])
                body, kind = jpeg((edge, edge * 2 // 3)), "image/jpeg"
            else:
                body, kind = jpeg((800, 600)), "image/jpeg"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", asked
    httpd.shutdown()


def manifest(tmp_path, columns, rows):
    path = tmp_path / "manifest.parquet"
    values = ", ".join("(" + ", ".join(f"'{v}'" for v in row) + ")" for row in rows)
    duckdb.execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS t({', '.join(columns)})) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
    return ParquetManifestSource(str(path))


def indexed(source, tmp_path, images):
    refs = list(source.iter_files())
    data_dir = tmp_path / "collection"
    state = IngestState(str(data_dir / "embeddings" / "state.sqlite"), 1)
    state.seed([(make_item_id(ref.key), ref.key, ref.size) for ref in refs])
    options = Options(
        thumbnails_dir=data_dir / "thumbnails",
        processed_dir=data_dir / "processed",
        data_dir=data_dir,
        download_workers=1,
        decode_workers=1,
        batch_size=1,
        images=images,
    )
    run(source, refs, FakeService(), state, options)
    return [metadata for _, _, metadata in state.done_rows()], data_dir


def read_from(data_dir):
    return IngestState(str(data_dir / "embeddings" / "state.sqlite"), 1).bytes_read()


def test_copies_are_cut_from_the_size_the_iiif_service_is_asked_for(server, tmp_path):
    address, asked = server
    source = manifest(tmp_path, ["iiif"], [[f"{address}/iiif/map-1"]])
    (metadata,), data_dir = indexed(source, tmp_path, "copy")
    assert "/iiif/map-1/full/!1920,1920/0/default.jpg" in asked
    assert (metadata["width"], metadata["height"]) == ORIGINAL
    assert "file_name" not in metadata or not metadata["file_name"]
    for size in ("thumbnail", "processed"):
        assert (data_dir / metadata["paths"][size]).exists()
    assert image_urls(metadata) == {}


def test_thumbnails_keep_a_small_preview_and_show_the_rest_from_the_service(
    server, tmp_path
):
    address, asked = server
    service = f"{address}/iiif/map-1"
    source = manifest(tmp_path, ["iiif", "title"], [[service, "A map"]])
    (metadata,), data_dir = indexed(source, tmp_path, "thumbnails")
    assert "/iiif/map-1/full/!448,448/0/default.jpg" in asked
    assert (data_dir / metadata["paths"]["thumbnail"]).exists()
    assert "processed" not in metadata["paths"]
    assert not (data_dir / "processed").exists()
    assert image_urls(metadata) == {"full": f"{service}/full/!1920,1920/0/default.jpg"}
    assert metadata["paths"]["original"] == f"{service}/full/max/0/default.jpg"


def test_link_keeps_nothing_when_the_service_can_show_every_size(server, tmp_path):
    address, _ = server
    service = f"{address}/iiif/map-1"
    source = manifest(tmp_path, ["iiif"], [[service]])
    (metadata,), data_dir = indexed(source, tmp_path, "link")
    assert not (data_dir / "thumbnails").exists()
    assert image_urls(metadata) == {
        "thumbnail": f"{service}/full/!400,400/0/default.jpg",
        "full": f"{service}/full/!1920,1920/0/default.jpg",
    }
    assert read_from(data_dir) == len(jpeg((448, 298)))


def test_link_reads_only_the_size_of_an_image_whose_vector_is_given(server, tmp_path):
    address, asked = server
    service = f"{address}/iiif/map-1"
    path = tmp_path / "manifest.parquet"
    duckdb.execute(
        f"COPY (SELECT '{service}' AS iiif, [0.6, 0.8, 0.0, 0.0]::FLOAT[] AS vector) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
    (metadata,), data_dir = indexed(ParquetManifestSource(str(path)), tmp_path, "link")
    assert asked == ["/iiif/map-1/info.json"]
    assert (metadata["width"], metadata["height"]) == ORIGINAL
    assert not (data_dir / "thumbnails").exists()
    assert read_from(data_dir) == 0


def test_link_uses_the_sources_own_thumbnail_and_image_addresses(server, tmp_path):
    address, _ = server
    source = manifest(
        tmp_path,
        ["image_url", "thumbnail_url"],
        [[f"{address}/files/a.jpg", f"{address}/thumbs/a.jpg"]],
    )
    (metadata,), data_dir = indexed(source, tmp_path, "link")
    assert metadata["file_name"] == "a.jpg"
    assert image_urls(metadata) == {
        "thumbnail": f"{address}/thumbs/a.jpg",
        "full": f"{address}/files/a.jpg",
    }
    assert not (data_dir / "thumbnails").exists()


def test_what_the_source_cannot_show_is_still_copied(server, tmp_path):
    address, _ = server
    source = manifest(
        tmp_path,
        ["image_url"],
        [[f"{address}/files/a.jpg"], [f"{address}/files/b.tif"]],
    )
    shown, kept = indexed(source, tmp_path, "link")
    by_name = {metadata["file_name"]: metadata for metadata in shown}
    assert image_source(by_name["a.jpg"], "full") == f"{address}/files/a.jpg"
    assert (kept / image_source(by_name["a.jpg"], "thumbnail")).exists()
    assert image_urls(by_name["b.tif"]) == {}


def test_the_profile_reads_only_the_size_of_images_an_iiif_service_names(
    server, tmp_path
):
    address, asked = server
    rows = [[f"{address}/iiif/map-{n}"] for n in range(3)]
    profile = profile_collection(manifest(tmp_path, ["iiif"], rows), "maps", sample=3)
    assert asked and all(path.endswith("/info.json") for path in asked)
    width = profile.scan.dimensions["width"]
    assert (profile.scan.sampled, width["p50"]) == (3, ORIGINAL[0])
    assert profile.directory_signals == [] and profile.name_patterns == []
    assert profile.field("iiif").note == ""


def test_the_profile_does_not_call_images_of_unknown_size_broken(server, tmp_path):
    address, _ = server
    rows = [[f"{address}/iiif/map-{n}"] for n in range(3)]
    profile = profile_collection(manifest(tmp_path, ["iiif"], rows), "maps", sample=3)
    assert (profile.scan.images_without_size, profile.scan.size_percentiles) == (3, {})
    assert all(issue.kind != "tiny_image" for issue in profile.scan.issues)
    report = render(profile)
    assert "3 images; the manifest does not give their sizes" in report
    assert "in total" not in report
    assert "Index a few hundred with --limit" in report


def test_the_profile_asks_what_to_keep_only_when_the_source_can_show_images():
    iiif = image_decision([FieldProfile(name="iiif_url")], web_images=False)
    assert iiif.recommended == "thumbnails"
    assert [option["id"] for option in iiif.options] == ["copy", "thumbnails", "link"]
    assert image_decision([], web_images=True).recommended == "copy"
    assert image_decision([FieldProfile(name="title")], web_images=False) is None
