import asyncio
import json
import sys
from pathlib import Path

import pytest
import torch
from fastapi import HTTPException
from PIL import Image, ImageDraw

from src.backend.services.collection_service import CollectionService
from src.backend.services.embedding_service import EmbeddingService
from src.backend.services.site_pages import llms_txt
from src.ingest.export import export_for_backend
from src.ingest.pipeline import Options, make_item_id, run
from src.ingest.state import IngestState
from src.profiling.sources import LocalDirSource

pytest.importorskip("pypdfium2")

VECTOR_SIZE = 8
PAGES = 3


class FakeService:
    def encode_image(self, images):
        vectors = torch.rand(len(images), VECTOR_SIZE) + 0.1
        return vectors / vectors.norm(dim=-1, keepdim=True)


def write_pdf(path: Path, pages: int) -> None:
    drawn = []
    for number in range(1, pages + 1):
        page = Image.new("RGB", (612, 792), "white")
        ImageDraw.Draw(page).text((80, 80), f"page {number}", fill="black")
        drawn.append(page)
    drawn[0].save(path, "PDF", save_all=True, append_images=drawn[1:], resolution=72)


@pytest.fixture()
def folder(tmp_path):
    root = tmp_path / "collection"
    (root / "reports").mkdir(parents=True)
    Image.new("RGB", (800, 600), (40, 60, 80)).save(root / "photo.jpg")
    write_pdf(root / "reports" / "annual.pdf", PAGES)
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 not really")
    return root


def ingest(folder, data_dir):
    source = LocalDirSource(str(folder))
    refs = sorted(
        (r for r in source.iter_files() if r.is_image or r.is_document),
        key=lambda r: r.key,
    )
    state = IngestState(str(data_dir / "embeddings" / "state.sqlite"), commit_every=2)
    ids = [make_item_id(r.key) for r in refs]
    state.seed([(i, r.key, r.size) for i, r in zip(ids, refs)])
    finished = state.finished_ids(ids, retry_failed=False)
    pending = [r for i, r in zip(ids, refs) if i not in finished]
    options = Options(
        thumbnails_dir=data_dir / "thumbnails",
        processed_dir=data_dir / "processed",
        download_workers=2,
        decode_workers=2,
        batch_size=2,
    )
    stats, _ = run(source, pending, FakeService(), state, options)
    return state, stats


def exported(state, data_dir):
    embeddings_dir = data_dir / "embeddings"
    export_for_backend(state, embeddings_dir, "clip", "test-model")
    return json.loads((embeddings_dir / "metadata.json").read_text())


def test_each_page_of_a_pdf_is_indexed_as_an_image_of_one_document(folder, tmp_path):
    state, stats = ingest(folder, tmp_path)
    assert (stats.done, stats.failed) == (2, 1)
    pages = [m for m in exported(state, tmp_path).values() if m["type"] == "pdf_page"]
    assert sorted(m["page"] for m in pages) == [1, 2, 3]
    for page in pages:
        assert page["pages"] == PAGES
        assert page["document_key"] == "reports/annual.pdf"
        assert page["document"] == str(folder / "reports" / "annual.pdf")
        assert page["file_name"] == "annual.pdf"
        assert max(page["width"], page["height"]) == 1920
        assert Path(page["paths"]["thumbnail"]).exists()


def test_a_pdf_that_cannot_be_read_is_recorded_once_as_failed(folder, tmp_path):
    state, _ = ingest(folder, tmp_path)
    assert state.counts() == {"done": 1 + PAGES, "failed": 1}
    ((key, error),) = state.failed_files(10)
    assert key == "broken.pdf" and error.startswith("decode:")


def test_a_finished_pdf_is_not_read_again(folder, tmp_path):
    ingest(folder, tmp_path)
    _, stats = ingest(folder, tmp_path)
    assert stats.total == 0


def test_a_pdf_with_a_page_that_fails_is_tried_again_later(
    folder, tmp_path, monkeypatch
):
    from src.ingest import pipeline

    render = pipeline.Pdf.render

    def second_page_fails(pdf, index, edge):
        if index == 1:
            raise ValueError("damaged page")
        return render(pdf, index, edge)

    monkeypatch.setattr(pipeline.Pdf, "render", second_page_fails)
    state, stats = ingest(folder, tmp_path)
    assert (stats.done, stats.failed) == (1, 2)
    assert ("reports/annual.pdf", "1 of 3 pages could not be indexed") in (
        state.failed_files(10)
    )
    document = make_item_id("reports/annual.pdf")
    assert not state.finished_ids([document], retry_failed=True)


def test_ingest_asks_for_the_pdf_reader_before_starting(
    folder, tmp_path, monkeypatch, capsys
):
    from src.ingest import __main__ as command

    monkeypatch.setattr(command, "can_read_pdfs", lambda: False)
    monkeypatch.setattr(
        sys, "argv", ["ingest", str(folder), "--data-dir", str(tmp_path / "out")]
    )
    assert command.main() == 2
    assert "pip install -r requirements-pdf.txt" in capsys.readouterr().out


@pytest.fixture()
def collection(folder, tmp_path):
    state, _ = ingest(folder, tmp_path)
    metadata = exported(state, tmp_path)
    embeddings = EmbeddingService()
    embeddings.embeddings_dir = str(tmp_path / "embeddings")
    embeddings.item_ids = list(metadata)
    embeddings.metadata = metadata
    embeddings.embeddings = torch.nn.functional.normalize(
        torch.rand(len(metadata), VECTOR_SIZE), dim=-1
    )
    embeddings.is_loaded = True
    return CollectionService(embeddings)


def page_id(number):
    return make_item_id(f"reports/annual.pdf#page={number}")


def test_the_pages_of_a_pdf_are_one_object(collection):
    item = collection.item(page_id(2))
    assert item["object_id"] == "reports/annual.pdf"
    assert item["image_count"] == PAGES
    assert [p["metadata"]["page"] for p in item["photos"]] == [1, 2, 3]


def test_the_collection_counts_pdfs_apart_from_images(collection):
    info = collection.info()
    assert (info["images"], info["documents"], info["objects"]) == (1, 1, 2)
    assert "It also holds 1 PDF, searched page by page." in llms_txt(info)


def test_an_uncataloged_pdf_stays_uncataloged(collection):
    collection._ensure_index()
    assert "reports/annual.pdf" in collection.uncataloged_objects


@pytest.fixture()
def images_route(monkeypatch):
    monkeypatch.setattr(
        "src.backend.services.embedding_service_factory.create_embedding_service",
        lambda: object(),
    )
    from src.backend.api.routes import images

    return images


def serve_document(images, monkeypatch, address):
    document = {"metadata": {"document": address} if address else {}}
    monkeypatch.setattr(
        images.embedding_service, "get_document_by_id", lambda item_id: document
    )
    return asyncio.run(images.get_document("a"))


def test_a_pdf_kept_on_the_site_opens_in_the_browser(images_route, monkeypatch, folder):
    pdf = folder / "reports" / "annual.pdf"
    response = serve_document(images_route, monkeypatch, str(pdf))
    assert response.media_type == "application/pdf"
    assert response.headers["content-disposition"] == "inline"
    assert Path(response.path) == pdf

    remote = "https://example.org/annual.pdf"
    response = serve_document(images_route, monkeypatch, remote)
    assert response.status_code == 307 and response.headers["location"] == remote


@pytest.mark.parametrize("address", [None, "notes.txt", "/no/such/file.pdf"])
def test_only_pdfs_that_exist_are_served(images_route, monkeypatch, address):
    with pytest.raises(HTTPException) as refusal:
        serve_document(images_route, monkeypatch, address)
    assert refusal.value.status_code == 404
