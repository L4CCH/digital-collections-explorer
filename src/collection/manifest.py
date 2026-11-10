"""Read a collection manifest: the tables a pipeline writes for DCE, and the vectors it made."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.backend.services.catalog_fields import is_web_address

from .catalog import SEARCHED, read_catalog
from .writer import Group, Item, Model, Record, Related, write_collection

MODELS_FILE = "models.json"
MODEL_KEYS = ("key", "type", "name", "dimensions", "revision")
SOURCE_FILE = "source.json"
REQUIRED = {
    "files": ("id",),
    "records": ("object_id",),
    "fields": ("object_id", "label", "value"),
    "facets": ("object_id", "facet", "value"),
    "text": ("object_id", "field", "value"),
    "related_sources": ("object_id", "source_id", "source", "link_type"),
    "related_tags": ("source_id", "tag"),
    "groups": ("group", "label"),
}
ITEM_COLUMNS = (
    "file_name",
    "width",
    "height",
    "thumbnail",
    "processed",
    "original",
    "iiif",
    "page",
    "view_orbit",
)
RECORD_COLUMNS = ("title", "source_url", "model_url", "document", "pages")
RELATED_COLUMNS = (
    "image_id",
    "url",
    "title",
    "thumbnail",
    "rights",
    "rights_url",
    "counts",
)


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return isinstance(value, str) and not value.strip()


def present(value: Any) -> Any:
    return None if is_blank(value) else value


def merged(rows: list[dict]) -> dict:
    """An object's catalog: each column's first value among its files"""
    catalog: dict = {}
    for row in rows:
        for column, value in row.items():
            if is_blank(catalog.get(column)) and not is_blank(value):
                catalog[column] = value
    return catalog


def read_table(folder: Path, name: str) -> list[dict] | None:
    """The rows of one table of the manifest, or None when the manifest has no such table"""
    path = Path(folder) / f"{name}.parquet"
    if not path.exists():
        return None
    import duckdb

    con = duckdb.connect()
    try:
        cursor = con.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = [column[0] for column in cursor.description]
        missing = [c for c in REQUIRED.get(name, ()) if c not in columns]
        if missing:
            raise ValueError(f"{path.name} has no column {', '.join(missing)}")
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        con.close()


def read_vectors(folder: Path) -> tuple[list[Model], str | None, dict, dict[str, str]]:
    """The models, the default one, each model's ids and vectors, and the group of each id"""
    folder = Path(folder)
    path = folder / MODELS_FILE
    if not path.exists():
        raise ValueError(f"{folder} has no {MODELS_FILE} naming the models")
    described = json.loads(path.read_text())
    models = [
        Model(
            key=m["key"],
            type=m.get("type") or m["key"],
            name=m["name"],
            dimensions=int(m["dimensions"]),
            revision=m.get("revision"),
            details={k: v for k, v in m.items() if k not in MODEL_KEYS},
        )
        for m in described.get("models") or []
    ]
    vectors, group_of = {}, {}
    for model in models:
        ids, parts = [], []
        for listed in sorted((folder / model.key).glob("*.ids.npy")):
            group = listed.name.removesuffix(".ids.npy")
            names = [str(i) for i in np.load(listed).tolist()]
            matrix = np.load(listed.with_name(f"{group}.npy"))
            if matrix.shape[0] != len(names):
                raise ValueError(
                    f"{model.key}/{group}.npy has {matrix.shape[0]} rows but "
                    f"{len(names)} ids"
                )
            ids += names
            parts.append(np.asarray(matrix, dtype=np.float32))
            for name in names:
                group_of.setdefault(name, group)
        matrix = (
            np.concatenate(parts) if parts else np.empty((0, model.dimensions), "f4")
        )
        vectors[model.key] = (ids, matrix)
    return models, described.get("default_model"), vectors, group_of


def grouped(rows: list[dict] | None, key: str = "object_id") -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = defaultdict(list)
    for row in rows or []:
        if not is_blank(row.get(key)):
            found[str(row[key])].append(row)
    return found


def explicit_fields(rows: list[dict]) -> list[tuple]:
    ordered = sorted(
        enumerate(rows),
        key=lambda pair: (
            pair[1]["position"] if not is_blank(pair[1].get("position")) else pair[0],
            pair[0],
        ),
    )
    return [
        (present(row.get("field")), str(row["label"]), str(row["value"]))
        for _, row in ordered
        if not is_blank(row["label"]) and not is_blank(row["value"])
    ]


def explicit_text(object_id: str, rows: list[dict]) -> dict[str, list[str]]:
    text: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        field = str(row["field"])
        if field not in SEARCHED:
            raise ValueError(
                f"text of {object_id} names the field {field}, not one of "
                f"{', '.join(SEARCHED)}"
            )
        if not is_blank(row["value"]):
            text[field].append(str(row["value"]))
    return dict(text)


def counts_of(value: Any) -> dict | None:
    if is_blank(value):
        return None
    return json.loads(value) if isinstance(value, str) else dict(value)


def read_manifest(folder: Path, vectors_dir: Path) -> dict[str, Any]:
    """Everything write_collection takes, read from a manifest folder and its vectors"""
    folder = Path(folder)
    files = read_table(folder, "files")
    if files is None:
        raise ValueError(f"{folder} has no files.parquet")
    models, default_model, vectors, group_of = read_vectors(vectors_dir)
    tables = {name: read_table(folder, name) for name in REQUIRED if name != "files"}

    items, files_of = [], defaultdict(list)
    has_records = tables["records"] is not None
    for row in files:
        item_id = str(row["id"])
        catalog = {k: v for k, v in row.items() if k not in ("id", *ITEM_COLUMNS)}
        group = present(row.get("group")) or group_of.get(item_id)
        if group is None:
            raise ValueError(f"The file {item_id} has no group and no vector")
        record = present(row.get("object_id"))
        if record is None and not has_records:
            read = read_catalog(catalog)
            record = item_id if (read.title or read.fields or read.source_url) else None
        if record is not None:
            files_of[str(record)].append(catalog)
        items.append(
            Item(
                id=item_id,
                group=str(group),
                record=None if record is None else str(record),
                **{column: present(row.get(column)) for column in ITEM_COLUMNS},
            )
        )

    given = {str(row["object_id"]): row for row in tables["records"] or []}
    by_table = {name: grouped(tables[name]) for name in ("fields", "facets", "text")}
    records = []
    for record_id in dict.fromkeys([*given, *files_of]):
        rows = files_of.get(record_id, [])
        catalog = given.get(record_id) if has_records else merged(rows)
        catalog = catalog or {}
        named = has_records or any(not is_blank(r.get("object_id")) for r in rows)
        records.append(
            Record(
                id=record_id,
                catalog=catalog,
                searchable_id=named,
                fields=(
                    explicit_fields(by_table["fields"].get(record_id, []))
                    if tables["fields"] is not None
                    else None
                ),
                facets=(
                    [
                        (str(row["facet"]), str(row["value"]))
                        for row in by_table["facets"].get(record_id, [])
                        if not is_blank(row["value"])
                    ]
                    if tables["facets"] is not None
                    else None
                ),
                text=(
                    explicit_text(record_id, by_table["text"].get(record_id, []))
                    if tables["text"] is not None
                    else None
                ),
                **{column: present(catalog.get(column)) for column in RECORD_COLUMNS},
            )
        )

    tags = grouped(tables["related_tags"], "source_id")
    related = [
        Related(
            record=str(row["object_id"]),
            source_id=str(row["source_id"]),
            source=str(row["source"]),
            link_type=str(row["link_type"]),
            tags=[
                str(tag["tag"])
                for _, tag in sorted(
                    enumerate(tags.get(str(row["source_id"]), [])),
                    key=lambda pair: (
                        (
                            pair[1]["position"]
                            if not is_blank(pair[1].get("position"))
                            else pair[0]
                        ),
                        pair[0],
                    ),
                )
            ],
            **{
                column: (
                    counts_of(row.get(column))
                    if column == "counts"
                    else present(row.get(column))
                )
                for column in RELATED_COLUMNS
            },
        )
        for row in tables["related_sources"] or []
    ]

    labels = {str(row["group"]): str(row["label"]) for row in tables["groups"] or []}
    order = list(
        dict.fromkeys(
            [*labels, *sorted(set(group_of.values())), *(i.group for i in items)]
        )
    )
    source_path = folder / SOURCE_FILE
    return {
        "models": models,
        "default_model": default_model,
        "groups": [Group(group, labels.get(group, group)) for group in order],
        "items": items,
        "records": records,
        "vectors": vectors,
        "related": related,
        "source": json.loads(source_path.read_text()) if source_path.exists() else None,
    }


def missing_files(data_dir: Path, read: dict[str, Any]) -> list[str]:
    """The files the collection names on this machine that are not under its folder"""
    named = [
        getattr(item, column)
        for item in read["items"]
        for column in ("thumbnail", "processed", "original")
    ]
    named += [
        getattr(record, column)
        for record in read["records"]
        for column in ("model_url", "document")
    ]
    named += [related.thumbnail for related in read["related"]]
    local = {path for path in named if path and not is_web_address(str(path))}
    root = Path(data_dir)
    return sorted(p for p in local if not (root / p).exists() and not Path(p).is_file())


def build(
    manifest: Path, vectors_dir: Path, data_dir: Path, check_files: bool = False
) -> dict[str, Any]:
    """Write the collection a manifest describes into a collection folder"""
    read = read_manifest(manifest, vectors_dir)
    missing = missing_files(data_dir, read) if check_files else []
    if missing:
        raise ValueError(
            f"{len(missing):,} files the manifest names are not in {data_dir}, "
            f"such as {', '.join(missing[:3])}"
        )
    return write_collection(
        Path(data_dir) / "embeddings",
        read["models"],
        read["groups"],
        read["items"],
        read["records"],
        read["vectors"],
        default_model=read["default_model"],
        related=read["related"],
        source=read["source"],
    )
