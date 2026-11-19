import socket
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.ingest.pipeline import Options, make_item_id, run
from src.ingest.state import IngestState
from src.profiling import sources
from src.profiling.profiler import profile_collection
from src.profiling.sources import fetch_web, unread_reason
from tests.test_image_modes import jpeg, manifest
from tests.test_ingest import FakeService


@pytest.fixture()
def flaky(monkeypatch):
    monkeypatch.setattr(sources, "RETRY_WAITS", (0, 0, 0))
    planned: dict[str, list[int]] = {}
    asked: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            asked.append(self.path)
            statuses = planned.get(self.path, [])
            status = statuses.pop(0) if statuses else 200
            if self.path.endswith("/info.json"):
                body, kind = b'{"width": 600, "height": 400}', "application/json"
            else:
                body, kind = jpeg((60, 40)), "image/jpeg"
            if status != 200:
                body, kind = b"", "text/plain"
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", planned, asked
    httpd.shutdown()


def closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def indexed(source, tmp_path):
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
        images="link",
    )
    run(source, refs, FakeService(), state, options)
    return dict(state.con.execute("SELECT key, coalesce(error, status) FROM items"))


def test_a_request_that_fails_for_a_moment_is_tried_again(flaky):
    address, planned, asked = flaky
    planned["/a.jpg"] = [503, 429]
    assert fetch_web(f"{address}/a.jpg")[1] == "image/jpeg"
    assert asked == ["/a.jpg"] * 3 and unread_reason() == ""


def test_a_missing_file_is_not_tried_again_and_says_why(flaky):
    address, planned, asked = flaky
    planned["/b.jpg"] = [404] * 5
    assert fetch_web(f"{address}/b.jpg") is None
    assert asked == ["/b.jpg"]
    assert unread_reason() == "the server answered 404 Not Found"


def test_a_server_that_cannot_be_reached_is_tried_again_and_says_so(monkeypatch):
    monkeypatch.setattr(sources, "RETRY_WAITS", (0, 0))
    tries = []
    real = sources.urllib.request.urlopen

    def drops_twice(request, timeout):
        tries.append(request.full_url)
        if len(tries) <= 2:
            raise urllib.error.URLError(OSError(51, "Network is unreachable"))
        return real(request, timeout=timeout)

    monkeypatch.setattr(sources.urllib.request, "urlopen", drops_twice)
    assert fetch_web(f"http://127.0.0.1:{closed_port()}/c.jpg") is None
    assert len(tries) == 3
    assert unread_reason().startswith("the server could not be reached (")


def test_iiif_falls_back_to_the_whole_image_only_when_the_size_is_refused(
    flaky, tmp_path
):
    address, planned, asked = flaky
    planned["/iiif/down/full/!448,448/0/default.jpg"] = [503] * 10
    planned["/iiif/fixed/full/!448,448/0/default.jpg"] = [400]
    rows = [[f"{address}/iiif/down"], [f"{address}/iiif/fixed"]]
    outcome = indexed(manifest(tmp_path, ["iiif"], rows), tmp_path)
    down, fixed = sorted(outcome, key=lambda key: "fixed" in key)
    assert outcome[down].endswith(
        "the file could not be read from the source: the server answered 503 "
        "Service Unavailable"
    )
    assert outcome[fixed] == "done"
    assert "/iiif/down/full/max/0/default.jpg" not in asked
    assert "/iiif/fixed/full/max/0/default.jpg" in asked


def test_the_profile_stops_sampling_after_five_files_cannot_be_read(flaky, tmp_path):
    address, planned, asked = flaky
    rows = [[f"{address}/iiif/gone-{n}"] for n in range(20)]
    for n in range(20):
        planned[f"/iiif/gone-{n}/info.json"] = [404]
    profile = profile_collection(manifest(tmp_path, ["iiif"], rows), "gone", sample=20)
    assert profile.scan.sampled == 0 and len(asked) == 5
