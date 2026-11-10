"""Write a collection in the one format the site reads, whatever indexed or prepared it."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.backend.core.formats import COLLECTION_FORMAT, INDEX_INFO_FILE
from src.backend.services.collection_store import METADATA_DB, VECTORS_DIR

from .catalog import RECOGNIZED, SEARCHED, read_catalog

OLD_FORMAT_FILES = ("embeddings.pt", "item_ids.pt", "metadata.json", "catalog.sqlite")

TABLES = {
    "items": """
        CREATE TABLE items (
            id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            record_id TEXT,
            file_name TEXT,
            width INTEGER,
            height INTEGER,
            thumbnail TEXT,
            processed TEXT,
            original TEXT,
            iiif TEXT,
            page INTEGER,
            view_orbit TEXT
        );
        CREATE INDEX items_group ON items(group_id);
        CREATE INDEX items_record ON items(record_id);
    """,
    "records": """
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            title TEXT,
            source_url TEXT,
            model_url TEXT,
            document TEXT,
            pages INTEGER
        );
    """,
    "record_fields": """
        CREATE TABLE record_fields (
            record_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            field TEXT,
            label TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (record_id, position)
        );
    """,
    "record_facets": """
        CREATE TABLE record_facets (
            record_id TEXT NOT NULL,
            facet TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (record_id, facet, value)
        );
        CREATE INDEX record_facets_value ON record_facets(facet, value);
    """,
    "records_fts": f"""
        CREATE VIRTUAL TABLE records_fts USING fts5(
            record_id UNINDEXED, {", ".join(SEARCHED)},
            tokenize = 'unicode61 remove_diacritics 2'
        );
        CREATE TABLE entity_vocab (
            term TEXT NOT NULL,
            field TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            PRIMARY KEY (field, term)
        );
    """,
    "related_sources": """
        CREATE TABLE related_sources (
            record_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source TEXT NOT NULL,
            link_type TEXT NOT NULL,
            image_id TEXT,
            url TEXT,
            title TEXT,
            thumbnail TEXT,
            rights TEXT,
            rights_url TEXT,
            counts TEXT,
            PRIMARY KEY (record_id, source_id)
        );
        CREATE TABLE related_tags (
            source_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (source_id, position)
        );
    """,
}


@dataclass
class Model:
    key: str
    type: str
    name: str
    dimensions: int
    revision: str | None = None


@dataclass
class Group:
    id: str
    label: str


@dataclass
class Item:
    """One image, page of a PDF or view of a 3D model, as the site searches it"""

    id: str
    group: str
    record: str | None = None
    file_name: str | None = None
    width: int | None = None
    height: int | None = None
    thumbnail: str | None = None
    processed: str | None = None
    original: str | None = None
    iiif: str | None = None
    page: int | None = None
    view_orbit: str | None = None


@dataclass
class Record:
    """One object's catalog record, read from its catalog columns unless given its fields"""

    id: str
    title: str | None = None
    source_url: str | None = None
    model_url: str | None = None
    document: str | None = None
    pages: int | None = None
    catalog: dict[str, Any] = field(default_factory=dict)
    fields: list[tuple[str, str]] | None = None
    facets: list[tuple[str, str]] | None = None


@dataclass
class Related:
    """A photo or page elsewhere, such as on Flickr Commons, that shows a record"""

    record: str
    source_id: str
    source: str
    link_type: str
    image_id: str | None = None
    url: str | None = None
    title: str | None = None
    thumbnail: str | None = None
    rights: str | None = None
    rights_url: str | None = None
    counts: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def checked(models, groups, items, vectors, default_model) -> str:
    if not models:
        raise ValueError("A collection needs the vectors of at least one model")
    keys = [model.key for model in models]
    default = default_model or keys[0]
    if default not in keys:
        raise ValueError(f"The default model {default} is not among {keys}")
    known_groups = [group.id for group in groups]
    if len(set(known_groups)) != len(known_groups):
        raise ValueError("Two groups share an id")
    strays = {item.group for item in items} - set(known_groups)
    if strays:
        raise ValueError(
            f"Items belong to groups that are not listed: {sorted(strays)}"
        )
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Two items share an id")
    for model in models:
        if model.key not in vectors:
            raise ValueError(f"No vectors for the model {model.key}")
        model_ids, matrix = vectors[model.key]
        if matrix.shape != (len(model_ids), model.dimensions):
            raise ValueError(
                f"The vectors of {model.key} are {matrix.shape}, not "
                f"{len(model_ids)} rows of {model.dimensions}"
            )
        missing = set(ids) - set(model_ids)
        if missing:
            raise ValueError(
                f"{len(missing):,} items have no vector from {model.key}, "
                f"such as {sorted(missing)[:3]}"
            )
    return default


def write_vectors(folder: Path, models, groups, items, vectors) -> dict[str, int]:
    members = {
        group.id: [item.id for item in items if item.group == group.id]
        for group in groups
    }
    members = {group_id: ids for group_id, ids in members.items() if ids}
    for model in models:
        model_ids, matrix = vectors[model.key]
        matrix = np.asarray(matrix, dtype=np.float32)
        row_of = {item_id: row for row, item_id in enumerate(model_ids)}
        shard_dir = folder / model.key
        shard_dir.mkdir(parents=True)
        for group_id, ids in members.items():
            rows = matrix[[row_of[i] for i in ids]]
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            rows = rows / np.where(norms == 0, 1, norms)
            np.save(shard_dir / f"{group_id}.npy", rows.astype(np.float32))
            np.save(shard_dir / f"{group_id}.ids.npy", np.array(ids))
    return {group_id: len(ids) for group_id, ids in members.items()}


def described(record: Record) -> tuple[tuple, list, list, dict[str, list[str]]]:
    """A record's row, its page's fields, its filters and its search text"""
    catalog = read_catalog(record.catalog, record.id)
    fields = (
        [(None, label, value) for label, value in record.fields]
        if record.fields is not None
        else catalog.fields
    )
    facets = record.facets if record.facets is not None else catalog.facets
    title = record.title or catalog.title
    text = dict(catalog.text)
    if title and title not in text.get("title", []):
        text["title"] = [title, *text.get("title", [])]
    row = (
        record.id,
        title,
        record.source_url or catalog.source_url,
        record.model_url,
        record.document,
        record.pages,
    )
    return row, fields, facets, text


def write_metadata(path: Path, items, records, related) -> dict[str, int]:
    rows, fields, facets, texts = [], [], [], []
    vocabulary: Counter = Counter()
    for record in records:
        row, shown, filters, text = described(record)
        rows.append(row)
        fields += [
            (record.id, n, f, label, value) for n, (f, label, value) in enumerate(shown)
        ]
        facets += [(record.id, facet, value) for facet, value in filters]
        vocabulary.update(
            {(value, facet) for facet, value in filters if facet in RECOGNIZED}
        )
        if any(text.get(column) for column in SEARCHED if column != "identifiers"):
            texts.append((record.id, *("\n".join(text.get(c, [])) for c in SEARCHED)))

    with sqlite3.connect(path) as con:
        con.executescript(TABLES["items"] + TABLES["records"])
        con.executemany(
            "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    i.id,
                    i.group,
                    i.record,
                    i.file_name,
                    i.width,
                    i.height,
                    i.thumbnail,
                    i.processed,
                    i.original,
                    i.iiif,
                    i.page,
                    i.view_orbit,
                )
                for i in items
            ),
        )
        con.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)", rows)
        if fields:
            con.executescript(TABLES["record_fields"])
            con.executemany("INSERT INTO record_fields VALUES (?, ?, ?, ?, ?)", fields)
        if facets:
            con.executescript(TABLES["record_facets"])
            con.executemany(
                "INSERT OR IGNORE INTO record_facets VALUES (?, ?, ?)", facets
            )
        if texts:
            con.executescript(TABLES["records_fts"])
            marks = ", ".join("?" * (len(SEARCHED) + 1))
            con.executemany(f"INSERT INTO records_fts VALUES ({marks})", texts)
            con.executemany(
                "INSERT INTO entity_vocab VALUES (?, ?, ?)",
                ((term, kind, count) for (term, kind), count in vocabulary.items()),
            )
            con.execute("INSERT INTO records_fts(records_fts) VALUES ('optimize')")
        if related:
            con.executescript(TABLES["related_sources"])
            con.executemany(
                "INSERT INTO related_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        r.record,
                        r.source_id,
                        r.source,
                        r.link_type,
                        r.image_id,
                        r.url,
                        r.title,
                        r.thumbnail,
                        r.rights,
                        r.rights_url,
                        json.dumps(r.counts) if r.counts else None,
                    )
                    for r in related
                ),
            )
            con.executemany(
                "INSERT OR IGNORE INTO related_tags VALUES (?, ?, ?)",
                (
                    (r.source_id, n, tag)
                    for r in related
                    for n, tag in enumerate(r.tags)
                ),
            )
    con.close()
    return {
        "records": len(rows),
        "record_fields": len(fields),
        "record_facets": len(facets),
        "searchable_records": len(texts),
        "vocabulary": len(vocabulary),
        "related_sources": len(related),
    }


def replace_folder(building: Path, target: Path) -> None:
    old = target.with_name(f".{target.name}.old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    building.rename(target)
    shutil.rmtree(old, ignore_errors=True)


def write_collection(
    embeddings_dir: Path,
    models: Sequence[Model],
    groups: Sequence[Group],
    items: Sequence[Item],
    records: Sequence[Record],
    vectors: dict[str, tuple[Sequence[str], np.ndarray]],
    default_model: str | None = None,
    related: Sequence[Related] = (),
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a collection's index, metadata and vectors, replacing what was there before"""
    embeddings_dir = Path(embeddings_dir)
    default = checked(models, groups, items, vectors, default_model)
    shown_records = {item.record for item in items if item.record}
    given = {record.id: record for record in records}
    kept = [given[r] for r in given if r in shown_records]
    kept += [Record(r) for r in sorted(shown_records - set(given))]
    known = {record.id for record in kept}
    kept_related = [r for r in related if r.record in known]

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    building = embeddings_dir / f".{VECTORS_DIR}.building"
    shutil.rmtree(building, ignore_errors=True)
    images = write_vectors(building, models, groups, items, vectors)
    database = embeddings_dir / f"{METADATA_DB}.building"
    database.unlink(missing_ok=True)
    counts = write_metadata(database, items, kept, kept_related)

    os.replace(database, embeddings_dir / METADATA_DB)
    replace_folder(building, embeddings_dir / VECTORS_DIR)
    info = {
        "format": COLLECTION_FORMAT,
        "built_at": now(),
        "default_model": default,
        "models": [
            {
                "key": m.key,
                "type": m.type,
                "name": m.name,
                "revision": m.revision,
                "dimensions": m.dimensions,
            }
            for m in models
        ],
        "groups": [
            {"id": g.id, "label": g.label, "images": images[g.id]}
            for g in groups
            if g.id in images
        ],
        **({"source": source} if source else {}),
    }
    written = embeddings_dir / f"{INDEX_INFO_FILE}.building"
    written.write_text(json.dumps(info, indent=2))
    os.replace(written, embeddings_dir / INDEX_INFO_FILE)
    for name in OLD_FORMAT_FILES:
        (embeddings_dir / name).unlink(missing_ok=True)

    return {
        "items": len(items),
        "uncataloged_items": sum(1 for item in items if not item.record),
        "records_without_files": len(given) - len(given.keys() & shown_records),
        "groups": len(images),
        **counts,
    }
