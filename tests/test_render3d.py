import argparse
import io
import json
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from src.backend.core.config import settings
from src.backend.services.shape_index import write_shape_index
from src.render3d import __main__ as render3d
from src.render3d.evaluate import compare, evaluate, read_pairs, render_comparison
from src.render3d.models import Model
from src.render3d.views import views_around


def test_views_go_around_the_model_starting_from_the_front():
    views = views_around(4)
    assert [v.orbit for v in views] == [
        "0deg 75deg auto",
        "90deg 75deg auto",
        "180deg 75deg auto",
        "270deg 75deg auto",
    ]
    assert views[1].file_name("relief") == "01-relief.jpg"


def test_views_that_show_little_of_the_model_are_left_out():
    coverages = {0: 0.27, 1: 0.2, 2: 0.01, 3: 0.021, 4: 0.0}
    assert render3d.kept_views(coverages) == [0, 1]
    assert render3d.kept_views({0: 0.0}) == []


def test_pairs_are_read_after_a_header_row(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("left,right,note\nA,B,same cave\n,C,\nD , E\n")
    assert read_pairs(str(path)) == [("A", "B"), ("D", "E")]


def test_each_pair_is_asked_both_ways():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.6, 0.8]])
    result = evaluate(vectors, ["a", "b", "c", "d"], [("a", "b"), ("a", "c")], [1, 2])
    assert result["queries"] == 4
    assert result["found_within"] == {"1": 0.5, "2": 0.5}
    assert result["median_rank"] == 2.0


def test_shape_and_appearance_are_compared_on_the_same_objects(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    item_ids = ["a-0", "a-1", "b-0", "c-photo", "d-0"]
    metadata = {
        "a-0": {"object_id": "a", "view_orbit": "0deg 75deg auto"},
        "a-1": {"object_id": "a", "view_orbit": "90deg 75deg auto"},
        "b-0": {"object_id": "b", "view_orbit": "0deg 75deg auto"},
        "c-photo": {"object_id": "c"},
        "d-0": {"object_id": "d", "view_orbit": "0deg 75deg auto"},
    }
    vectors = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    torch.save(vectors, embeddings_dir / "embeddings.pt")
    torch.save(item_ids, embeddings_dir / "item_ids.pt")
    (embeddings_dir / "metadata.json").write_text(json.dumps(metadata))
    write_shape_index(
        embeddings_dir,
        ["a", "b", "d"],
        np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]),
        "fake/dino",
        8,
    )

    result = compare(embeddings_dir, "fake/siglip", [("a", "b"), ("a", "c")], [1])

    assert (result["objects_with_models"], result["pairs_usable"]) == (3, 1)
    assert result["pairs_missing"] == [["a", "c"]]
    methods = result["methods"]
    assert methods["Appearance (fake/siglip)"]["found_within"]["1"] == 1.0
    assert methods["Shape (fake/dino)"]["found_within"]["1"] == 0.0
    shown = render_comparison(result, [1])
    assert "1 of 2 pairs" in shown and "100%" in shown


def plate_glb(path, width=2.0, height=1.0, depth=0.05):
    x, y, z = width / 2, height / 2, depth / 2
    corners = [
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
    ]
    faces = [
        (0, 1, 2, 3),
        (5, 4, 7, 6),
        (4, 0, 3, 7),
        (1, 5, 6, 2),
        (3, 2, 6, 7),
        (4, 5, 1, 0),
    ]
    positions, indices = [], []
    for face in faces:
        start = len(positions)
        positions.extend(corners[i] for i in face)
        indices.extend([start, start + 1, start + 2, start, start + 2, start + 3])
    position_bytes = np.array(positions, dtype=np.float32).tobytes()
    index_bytes = np.array(indices, dtype=np.uint16).tobytes()
    binary = position_bytes + index_bytes
    binary += b"\0" * (-len(binary) % 4)
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {
                "buffer": 0,
                "byteOffset": len(position_bytes),
                "byteLength": len(index_bytes),
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": [-x, -y, -z],
                "max": [x, y, z],
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
    }
    text = json.dumps(gltf).encode("utf-8")
    text += b" " * (-len(text) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text
    chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
    return path


@pytest.fixture()
def renderer():
    from src.render3d.renderer import Renderer, RendererUnavailable

    try:
        with Renderer(256, "#ffffff", 10 * 1024**2, 60) as started:
            yield started
    except RendererUnavailable as reason:
        pytest.skip(str(reason))


@pytest.fixture()
def chromium():
    from src.render3d.renderer import Renderer, RendererUnavailable

    try:
        with Renderer(64, "#ffffff", 1, 10):
            pass
    except RendererUnavailable as reason:
        pytest.skip(str(reason))


def test_a_thin_plate_seen_edge_on_is_left_out(renderer, tmp_path):
    glb = plate_glb(tmp_path / "plate.glb")
    info = renderer.load(Model(str(glb), "plate"))
    assert info["triangles"] == 12 and not info["textured"]

    coverages = {}
    for view in views_around(4):
        for mode in ("color", "relief"):
            data, coverage = renderer.render(view, mode)
            if data:
                image = Image.open(io.BytesIO(data))
                assert image.format == "JPEG"
        coverages[view.number] = coverage

    front = Image.open(io.BytesIO(renderer.render(views_around(4)[0], "color")[0]))
    assert front.width > 1.6 * front.height
    assert render3d.kept_views(coverages) == [0, 2]


def test_a_missing_model_file_is_reported_not_rendered(renderer, tmp_path):
    from src.render3d.renderer import ModelRefused

    with pytest.raises(ModelRefused):
        renderer.load(Model(str(tmp_path / "gone.glb"), "gone"))


def test_rendering_a_folder_writes_views_ready_for_ingest_and_can_resume(
    chromium, tmp_path, monkeypatch, capsys
):
    for name in ("data_dir", "embeddings_dir", "thumbnails_dir", "processed_data_dir"):
        monkeypatch.setattr(settings, name, getattr(settings, name))
    source = tmp_path / "models"
    source.mkdir()
    plate_glb(source / "plate.glb")
    data_dir = tmp_path / "collection"
    args = argparse.Namespace(
        uri=str(source),
        data_dir=str(data_dir),
        views=4,
        size=256,
        limit=None,
        max_model_mb=10.0,
        timeout=60.0,
        again=False,
        no_shape=True,
        shape_model="unused",
        device="cpu",
    )

    assert render3d.render_command(args) == 0
    assert "1 models rendered" in capsys.readouterr().out
    from src.profiling.sources import ParquetManifestSource

    manifest = data_dir / "models3d" / "views.parquet"
    refs = list(ParquetManifestSource(str(manifest)).iter_files())
    rows = [ref.extra["row"] for ref in refs]
    assert [row["view_orbit"] for row in rows] == [
        "0deg 75deg auto",
        "180deg 75deg auto",
    ]
    assert {row["object_id"] for row in rows} == {"plate"}
    assert all((manifest.parent / ref.key).exists() for ref in refs)

    assert render3d.render_command(args) == 0
    assert "1 already rendered, 0 to go" in capsys.readouterr().out
