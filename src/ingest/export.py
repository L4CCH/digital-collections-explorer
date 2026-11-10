"""Write what ingest has indexed as a collection the site reads, through the one writer."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.backend.services.catalog_fields import object_id_from
from src.collection.catalog import read_catalog
from src.collection.writer import Group, Item, Model, Record, write_collection

from .state import IngestState

GROUP_COLUMN = "group"
DEFAULT_GROUP = "collection"


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return isinstance(value, str) and not value.strip()


def group_of(catalog: dict) -> tuple[str, str]:
    """The id and the name of the group a file belongs to, such as a museum"""
    name = next(
        (
            str(value).strip()
            for column, value in catalog.items()
            if str(column).strip().lower() == GROUP_COLUMN and not is_blank(value)
        ),
        None,
    )
    if name is None:
        return DEFAULT_GROUP, DEFAULT_GROUP
    return re.sub(r"[\W_]+", "-", name.lower()).strip("-") or DEFAULT_GROUP, name


def record_of(item_id: str, metadata: dict, catalog: dict) -> tuple[str | None, bool]:
    """The record a file shows, and whether its id is the catalog's own"""
    named = metadata.get("object_id") or object_id_from(catalog)
    if named:
        return str(named), True
    if metadata.get("type") == "pdf_page":
        return metadata.get("document_key"), False
    read = read_catalog(catalog)
    if read.title or read.fields or read.source_url:
        return item_id, False
    return None, False


def merged(rows: list[dict]) -> dict:
    """An object's catalog: each column's first value among its files"""
    catalog: dict = {}
    for row in rows:
        for column, value in row.items():
            if is_blank(catalog.get(column)) and not is_blank(value):
                catalog[column] = value
    return catalog


def first(members: list[dict], name: str) -> Any:
    return next((m[name] for m in members if not is_blank(m.get(name))), None)


def write_from_state(
    state: IngestState, embeddings_dir: Path, model_type: str, model_name: str
) -> dict | None:
    """Write the collection from the files ingest has indexed; None when there are none"""
    ids, vectors, items = [], [], []
    groups: dict[str, str] = {}
    members: dict[str, list[dict]] = {}
    named: dict[str, bool] = {}
    for item_id, embedding, metadata in state.done_rows():
        catalog = metadata.get("catalog") or {}
        group, name = group_of(catalog)
        groups.setdefault(group, name)
        record, is_named = record_of(item_id, metadata, catalog)
        if record:
            members.setdefault(record, []).append(metadata)
            named[record] = named.get(record, False) or is_named
        paths = metadata.get("paths") or {}
        items.append(
            Item(
                id=item_id,
                group=group,
                record=record,
                file_name=metadata.get("file_name"),
                width=metadata.get("width"),
                height=metadata.get("height"),
                thumbnail=paths.get("thumbnail"),
                processed=paths.get("processed"),
                original=paths.get("original"),
                iiif=metadata.get("iiif"),
                page=(
                    metadata.get("page") if metadata.get("type") == "pdf_page" else None
                ),
                view_orbit=metadata.get("view_orbit"),
            )
        )
        ids.append(item_id)
        vectors.append(np.frombuffer(embedding, dtype=np.float32))
    if not items:
        return None

    records = []
    for record, files in members.items():
        pages = [m for m in files if m.get("type") == "pdf_page"]
        records.append(
            Record(
                id=record,
                source_url=first(files, "source_url"),
                model_url=first(files, "model_url"),
                document=first(pages, "document"),
                pages=first(pages, "pages"),
                catalog=merged([m.get("catalog") or {} for m in files]),
                searchable_id=named[record],
            )
        )
    model = Model(
        key=model_type, type=model_type, name=model_name, dimensions=len(vectors[0])
    )
    return write_collection(
        embeddings_dir,
        [model],
        [Group(key, groups[key]) for key in sorted(groups, key=groups.get)],
        items,
        records,
        {model.key: (ids, np.stack(vectors))},
    )
