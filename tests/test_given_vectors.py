import sys

import duckdb
import numpy as np
import torch
from PIL import Image

from src.ingest.pipeline import Options, given_vectors_check, make_item_id, run, unit
from src.ingest.state import IngestState
from src.profiling.sources import ParquetManifestSource

COLORS = [((i * 37) % 256, (i * 91) % 256, (i * 53 + 20) % 256) for i in range(30)]


class ColorService:
    model_name = "color-model"
    device = "cpu"

    def __init__(self):
        self.encoded = 0

    def encode_image(self, images):
        self.encoded += len(images)
        means = [np.asarray(image, np.float32).mean(axis=(0, 1)) for image in images]
        vectors = torch.tensor([color_vector(mean) for mean in means])
        return vectors / vectors.norm(dim=-1, keepdim=True)


def color_vector(color):
    return [float(channel) / 255 - 0.5 for channel in color] + [0.25]


def write_images(folder, count):
    folder.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (64, 48), COLORS[index]).save(folder / f"{index:02}.png")


def write_manifest(path, rows):
    def vector(values):
        if values is None:
            return "NULL::FLOAT[]"
        return "[" + ", ".join(f"{value:.6f}" for value in values) + "]::FLOAT[]"

    listed = ", ".join(f"('{name}', {vector(values)})" for name, values in rows)
    duckdb.execute(
        f"COPY (SELECT * FROM (VALUES {listed}) AS t(image_url, vector)) "
        f"TO '{path}' (FORMAT PARQUET)"
    )
    return ParquetManifestSource(str(path))


def colored(count, vector_of=color_vector):
    return [(f"images/{i:02}.png", vector_of(COLORS[i])) for i in range(count)]


def options_for(data_dir):
    return Options(
        thumbnails_dir=data_dir / "thumbnails",
        processed_dir=data_dir / "processed",
        data_dir=data_dir,
        download_workers=2,
        decode_workers=2,
        batch_size=4,
    )


def indexed(source, refs, service, data_dir):
    state = IngestState(str(data_dir / "embeddings" / "state.sqlite"), 1)
    state.seed([(make_item_id(ref.key), ref.key, ref.size) for ref in refs])
    stats, _ = run(source, refs, service, state, options_for(data_dir))
    rows = state.done_rows()
    return stats, {item: np.frombuffer(vector, np.float32) for item, vector, _ in rows}


def test_given_vectors_are_kept_and_only_a_sample_is_embedded_again(tmp_path):
    write_images(tmp_path / "images", 30)
    source = write_manifest(tmp_path / "manifest.parquet", colored(30))
    refs = list(source.iter_files())
    service = ColorService()
    accepted, message = given_vectors_check(
        source, refs, service, options_for(tmp_path / "site")
    )
    assert accepted and service.encoded == 20
    assert "for 30 images" in message and "median similarity of 1.00" in message
    stats, vectors = indexed(source, refs, service, tmp_path / "site")
    assert (stats.done, stats.failed, service.encoded) == (30, 0, 20)
    for ref in refs:
        given = unit(ref.extra["vector"])
        assert np.allclose(vectors[make_item_id(ref.key)], given)
    assert "vector" not in refs[0].extra["row"]


def test_vectors_from_another_model_are_refused(tmp_path):
    write_images(tmp_path / "images", 30)
    rows = colored(30, lambda color: [-value for value in color_vector(color)])
    source = write_manifest(tmp_path / "manifest.parquet", rows)
    accepted, message = given_vectors_check(
        source, list(source.iter_files()), ColorService(), options_for(tmp_path)
    )
    assert not accepted
    assert message.startswith("The manifest's vectors do not match color-model")
    assert "median similarity of -1.00" in message


def test_vectors_of_another_length_are_refused(tmp_path):
    write_images(tmp_path / "images", 3)
    rows = colored(3, lambda color: color_vector(color)[:3])
    source = write_manifest(tmp_path / "manifest.parquet", rows)
    accepted, message = given_vectors_check(
        source, list(source.iter_files()), ColorService(), options_for(tmp_path)
    )
    assert not accepted
    assert "have 3 numbers each, but color-model makes vectors of 4" in message


def test_rows_without_a_vector_are_embedded_and_a_zero_vector_fails(tmp_path):
    write_images(tmp_path / "images", 4)
    rows = colored(4)
    rows[1], rows[2] = (rows[1][0], None), (rows[2][0], None)
    rows[3] = (rows[3][0], [0.0, 0.0, 0.0, 0.0])
    source = write_manifest(tmp_path / "manifest.parquet", rows)
    service = ColorService()
    stats, _ = indexed(source, list(source.iter_files()), service, tmp_path / "site")
    assert (stats.done, stats.failed, service.encoded) == (3, 1, 2)


def test_ingest_stops_before_indexing_when_the_vectors_do_not_match(
    tmp_path, monkeypatch, capsys
):
    from src.ingest import __main__ as command

    write_images(tmp_path / "images", 30)
    rows = colored(30, lambda color: [-value for value in color_vector(color)])
    write_manifest(tmp_path / "manifest.parquet", rows)
    for name in ("data_dir", "embeddings_dir", "thumbnails_dir", "processed_data_dir"):
        monkeypatch.setattr(command.settings, name, getattr(command.settings, name))
    monkeypatch.setattr(command.settings, "device", command.settings.device)
    monkeypatch.setattr(command, "create_embedding_service", ColorService)
    monkeypatch.setattr(command, "cloud_machine_type", lambda: None)
    data_dir = tmp_path / "site"
    manifest = str(tmp_path / "manifest.parquet")
    arguments = ["ingest", manifest, "--data-dir", str(data_dir), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert command.main() == 2
    assert "do not match color-model" in capsys.readouterr().out
    assert not (data_dir / "embeddings" / "index_info.json").exists()


def test_the_profile_counts_given_vectors_and_says_what_to_fix(tmp_path):
    from src.profiling.profiler import profile_collection
    from src.profiling.report import render

    write_images(tmp_path / "images", 4)
    rows = colored(4)
    rows[1] = (rows[1][0], None)
    rows[2] = (rows[2][0], [0.0, 0.0, 0.0, 0.0])
    rows[3] = (rows[3][0], color_vector(COLORS[3])[:3])
    source = write_manifest(tmp_path / "manifest.parquet", rows)
    profile = profile_collection(source, "test", sample=4, progress=False)
    assert profile.vectors.model_dump() == {
        "rows": 4,
        "given": 3,
        "lengths": {4: 2, 3: 1},
        "unusable": 1,
    }
    report = render(profile)
    assert "3 of 4 rows give a vector of 3 or 4 numbers." in report
    assert "after re-embedding 3 of them to check" in report
    assert (
        "Must fix before indexing: every vector needs as many numbers as the model "
        "makes, but these have 3 or 4." in report
    )
    assert "1 vector is zero or not numbers; those images will fail." in report
    assert "1 row gives no vector; indexing computes those." in report
