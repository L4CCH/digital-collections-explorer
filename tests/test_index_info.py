import json

import pytest

from src.backend.services.index_info import describe_mismatch

CLIP = "openai/clip-vit-base-patch32"
SIGLIP = "google/siglip-base-patch16-224"


def write_info(folder, name, dimensions):
    model = {"key": "m", "type": "siglip", "name": name, "dimensions": dimensions}
    info = {"format": 3, "models": [model], "groups": []}
    (folder / "index_info.json").write_text(json.dumps(info))


def test_matching_model_is_accepted(tmp_path):
    write_info(tmp_path, SIGLIP, 768)
    assert describe_mismatch(tmp_path, 768, SIGLIP, 768) is None


def test_index_from_another_model_is_refused_with_a_way_out(tmp_path):
    write_info(tmp_path, CLIP, 512)
    problem = describe_mismatch(tmp_path, 512, SIGLIP, 768)
    assert CLIP in problem and SIGLIP in problem
    assert "Choose one of the models the index holds" in problem


def test_same_width_but_different_model_is_still_refused(tmp_path):
    write_info(tmp_path, "some/other-768-model", 768)
    assert describe_mismatch(tmp_path, 768, SIGLIP, 768) is not None


def test_a_folder_without_a_collection_has_no_model_to_mismatch(tmp_path):
    assert describe_mismatch(tmp_path, 512, SIGLIP, 768) is None


def write_config(tmp_path, model_config):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_config": model_config}))
    return path


def test_default_model_type_and_name_agree(tmp_path):
    from src.backend.core.config import load_config

    loaded = load_config(write_config(tmp_path, {}))
    assert (loaded.model_type.value, loaded.model_name) == ("siglip", SIGLIP)


def test_config_that_only_names_a_legacy_clip_model_stays_on_clip(tmp_path):
    from src.backend.core.config import load_config

    loaded = load_config(write_config(tmp_path, {"clip_model": CLIP}))
    assert (loaded.model_type.value, loaded.model_name) == ("clip", CLIP)


@pytest.mark.parametrize(
    "name, model_type",
    [
        ("openai/clip-vit-large-patch14", "clip"),
        ("google/siglip2-base-patch16-224", "siglip2"),
        ("google/siglip-so400m-patch14-384", "siglip"),
        ("someone/another-model", "siglip"),
    ],
)
def test_a_config_that_names_only_the_model_gets_its_type_from_the_name(
    tmp_path, name, model_type
):
    from src.backend.core.config import load_config

    loaded = load_config(write_config(tmp_path, {"model_name": name}))
    assert (loaded.model_type.value, loaded.model_name) == (model_type, name)


def test_choosing_clip_without_a_name_gets_the_clip_default(tmp_path):
    from src.backend.core.config import load_config

    loaded = load_config(write_config(tmp_path, {"model_type": "clip"}))
    assert loaded.model_name == CLIP


def test_a_revision_in_config_json_is_read(tmp_path):
    from src.backend.core.config import load_config

    loaded = load_config(write_config(tmp_path, {"model_revision": "abc123"}))
    assert loaded.model_revision == "abc123"


def test_choosing_siglip2_without_a_name_gets_the_siglip2_default(tmp_path):
    from src.backend.core.config import load_config

    loaded = load_config(write_config(tmp_path, {"model_type": "siglip2"}))
    assert loaded.model_name == "google/siglip2-base-patch16-224"
