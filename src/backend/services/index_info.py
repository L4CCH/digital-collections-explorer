"""Records which model built an index, so it is never searched with another one."""

import json
from pathlib import Path
from typing import Optional

from ..core.formats import INDEX_INFO_FILE, INGEST_STATE_FILE, READABLE_FORMATS

# Vector width of each documented model, to name the builder of an older index.
OLD_LAYOUT_FILE = "embeddings.pt"

KNOWN_DIMENSIONS = {
    512: "openai/clip-vit-base-patch32",
    768: "google/siglip-base-patch16-224",
}


def is_grouped(info: Optional[dict]) -> bool:
    """Whether index_info.json describes a collection in vectors/ and metadata.sqlite"""
    return bool(info) and info.get("format") in READABLE_FORMATS


def grouped_model(info: dict, model_name: str) -> Optional[dict]:
    """The entry for a model in a grouped index, if the index holds its vectors"""
    return next(
        (m for m in info.get("models", []) if m.get("name") == model_name), None
    )


def read_index_info(embeddings_dir: Path) -> Optional[dict]:
    """Load the model details of an index, if it has any"""
    path = Path(embeddings_dir) / INDEX_INFO_FILE
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def describe_mismatch(
    embeddings_dir: Path,
    index_dimensions: Optional[int],
    model_name: str,
    model_dimensions: Optional[int],
) -> Optional[str]:
    """Explain why the collection cannot be searched with the configured model, if so"""
    info = read_index_info(embeddings_dir)
    if not is_grouped(info):
        return None
    names = [m.get("name") for m in info.get("models", [])]
    if model_name in names and index_dimensions == model_dimensions:
        return None
    held = ", ".join(n for n in names if n) or "no model"
    return (
        f"The index in {embeddings_dir} holds vectors from {held}, but the site is "
        f"set to {model_name}. Searching it would return meaningless results.\n"
        f"Choose one of the models the index holds, or build vectors for {model_name}."
    )


def describe_old_layout(embeddings_dir: Path) -> Optional[str]:
    """Explain what to do with an index in the layout of earlier versions, if this is one"""
    embeddings_dir = Path(embeddings_dir)
    if is_grouped(read_index_info(embeddings_dir)):
        return None
    if not (embeddings_dir / OLD_LAYOUT_FILE).exists():
        return None
    found = (
        f"{embeddings_dir} holds an index in the old layout ({OLD_LAYOUT_FILE}, "
        "item_ids.pt and metadata.json), which this version no longer reads.\n"
    )
    if (embeddings_dir / INGEST_STATE_FILE).exists():
        return found + (
            "Run the python -m src.ingest command that built it again: every file "
            "is indexed already, so it only writes the collection anew."
        )
    return found + (
        "Serve it with v2.0.0-beta.2, the last version that reads it, or index its "
        "images again into a new folder with python -m src.ingest."
    )
