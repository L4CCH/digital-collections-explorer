"""Reads a collection's index from disk, whether it is one index or split into groups."""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .index_info import grouped_model, read_index_info

logger = logging.getLogger(__name__)

METADATA_DB = "metadata.sqlite"
VECTORS_DIR = "vectors"

ITEMS_QUERY = """
    SELECT i.id, i.group_id, i.record_id, i.file_name, i.width, i.height,
           i.thumbnail, i.processed, i.original, r.title, r.source_url, {optional}
    FROM items i LEFT JOIN records r ON r.record_id = i.record_id
"""
OPTIONAL_COLUMNS = {
    "items": ("view_orbit", "iiif", "page"),
    "records": ("model_url", "document", "pages"),
}
FIELDS_QUERY = (
    "SELECT {field}, label, value FROM record_fields WHERE record_id = ? "
    "ORDER BY position"
)
FACETS_QUERY = (
    "SELECT facet, value FROM record_facets WHERE record_id = ? ORDER BY facet, value"
)
RELATED_QUERY = """
    SELECT source_id, source, link_type, image_id, url, title, thumbnail, rights,
           rights_url, counts
    FROM related_sources WHERE record_id = ?
    ORDER BY source, image_id IS NULL, source_id
"""
TAGS_QUERY = "SELECT tag FROM related_tags WHERE source_id = ? ORDER BY position"
THUMBNAIL_QUERY = (
    "SELECT thumbnail FROM related_sources "
    "WHERE source_id = ? AND thumbnail IS NOT NULL"
)


@dataclass
class LoadedIndex:
    """Vectors, the id of the item behind each row, metadata, and group membership"""

    embeddings: torch.Tensor
    item_ids: List[str]
    metadata: Dict[str, Dict[str, Any]]
    groups: List[Dict[str, Any]] = field(default_factory=list)
    group_of_row: Optional[torch.Tensor] = None


def load_single(embeddings_dir: Path) -> Optional[LoadedIndex]:
    """Load embeddings.pt, item_ids.pt and metadata.json, the layout ingest writes"""
    embeddings_path = embeddings_dir / "embeddings.pt"
    item_ids_path = embeddings_dir / "item_ids.pt"
    metadata_path = embeddings_dir / "metadata.json"

    logger.info(f"Looking for embeddings at {embeddings_path}")
    logger.info(f"Looking for item IDs at {item_ids_path}")
    logger.info(f"Looking for metadata at {metadata_path}")

    if not (embeddings_path.exists() and item_ids_path.exists()):
        if not embeddings_path.exists():
            logger.error(f"Embeddings file not found at {embeddings_path}")
        if not item_ids_path.exists():
            logger.error(f"Item IDs file not found at {item_ids_path}")
        return None

    embeddings = torch.load(embeddings_path, map_location="cpu")
    item_ids = torch.load(item_ids_path)
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = upgrade_old_pages(json.load(f))
    else:
        metadata = {}
        logger.warning("Metadata file not found, proceeding without metadata.")
    return LoadedIndex(embeddings, item_ids, metadata)


def upgrade_old_pages(metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Give PDF pages from indexes made before version 2 the fields pages have now"""
    for item in metadata.values():
        if item.get("type") != "pdf_page" or "document_key" in item:
            continue
        if "n_pages" not in item:
            continue
        original = (item.get("paths") or {}).get("original")
        item["document_key"] = original or item.get("file_name")
        item["pages"] = item["n_pages"]
        item["page"] = int(item.get("page") or 0) + 1
        if original:
            item.setdefault("document", original)
    return metadata


def load_grouped(embeddings_dir: Path, model_name: str) -> LoadedIndex:
    """Load one vector shard per group for the given model, and metadata.sqlite"""
    info = read_index_info(embeddings_dir) or {}
    model = grouped_model(info, model_name)
    if model is None:
        raise ValueError(f"{embeddings_dir} holds no vectors from {model_name}")

    shards, item_ids, rows_per_group = [], [], []
    for group in info.get("groups", []):
        base = embeddings_dir / VECTORS_DIR / model["key"] / group["id"]
        vectors = np.load(f"{base}.npy")
        ids = [str(i) for i in np.load(f"{base}.ids.npy").tolist()]
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"{base}.npy has {vectors.shape[0]} rows but {len(ids)} item ids"
            )
        shards.append(np.ascontiguousarray(vectors, dtype=np.float32))
        item_ids.extend(ids)
        rows_per_group.append(len(ids))

    if shards:
        embeddings = torch.from_numpy(np.concatenate(shards))
    else:
        embeddings = torch.empty(0, model.get("dimensions") or 0)
    group_of_row = torch.repeat_interleave(
        torch.arange(len(rows_per_group)),
        torch.tensor(rows_per_group, dtype=torch.long),
    )
    groups = [
        {"id": g["id"], "label": g.get("label") or g["id"]}
        for g in info.get("groups", [])
    ]
    metadata = read_grouped_metadata(embeddings_dir / METADATA_DB)
    missing = sum(1 for item_id in item_ids if item_id not in metadata)
    if missing:
        logger.warning(f"{missing} indexed items have no row in {METADATA_DB}")
    return LoadedIndex(embeddings, item_ids, metadata, groups, group_of_row)


def read_grouped_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """Describe every image, page and view from the items and records of metadata.sqlite"""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        optional = [
            f"{table[0]}.{name}" if name in columns(con, table) else "NULL"
            for table, names in OPTIONAL_COLUMNS.items()
            for name in names
        ]
        rows = con.execute(ITEMS_QUERY.format(optional=", ".join(optional))).fetchall()
        with_fields = (
            {r for (r,) in con.execute("SELECT DISTINCT record_id FROM record_fields")}
            if "record_id" in columns(con, "record_fields")
            else set()
        )
    finally:
        con.close()
    metadata = {}
    for (
        item_id,
        group_id,
        record_id,
        file_name,
        width,
        height,
        thumbnail,
        processed,
        original,
        title,
        source_url,
        view_orbit,
        iiif,
        page,
        model_url,
        document,
        pages,
    ) in rows:
        paths = {"thumbnail": thumbnail, "processed": processed, "original": original}
        is_page = page is not None
        described = {
            "file_name": file_name,
            "type": "pdf_page" if is_page else "image",
            "width": width,
            "height": height,
            "title": title,
            "source_url": source_url,
            "object_id": record_id,
            "cataloged": bool(
                record_id and (title or source_url or record_id in with_fields)
            ),
            "group": group_id,
            "paths": {k: v for k, v in paths.items() if v},
            "model_url": model_url,
            "view_orbit": view_orbit,
            "iiif": iiif,
            "page": page,
            "pages": pages if is_page else None,
            "document": document if is_page else None,
            "document_key": record_id if is_page else None,
        }
        metadata[item_id] = {k: v for k, v in described.items() if v is not None}
    return metadata


def columns(con: sqlite3.Connection, table: str) -> set:
    """The column names of a table in metadata.sqlite"""
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def has_table(path: Path, name: str) -> bool:
    """Whether metadata.sqlite exists and holds a table of this name"""
    if not path.exists():
        return False
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        found = con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,))
        return found.fetchone() is not None
    finally:
        con.close()


def describe_related(
    con: sqlite3.Connection, row: tuple, has_tags: bool
) -> Dict[str, Any]:
    """One photo or page from another source, such as Flickr, that shows this record"""
    (
        source_id,
        source,
        link_type,
        image_id,
        url,
        title,
        thumbnail,
        rights,
        rights_url,
        counts,
    ) = row
    described = {
        "id": source_id,
        "source": source,
        "link_type": link_type,
        "image_id": image_id,
        "url": url,
        "title": title,
        "has_thumbnail": bool(thumbnail),
        "rights": {"name": rights, "url": rights_url} if rights else None,
        "counts": json.loads(counts) if counts else {},
        "tags": (
            [tag for (tag,) in con.execute(TAGS_QUERY, (source_id,))]
            if has_tags
            else []
        ),
    }
    return {k: v for k, v in described.items() if v is not None}


def read_record_details(path: Path, record_id: str) -> Dict[str, Any]:
    """The catalog fields, facets and related sources of one record, when kept"""
    fields: List[Dict[str, str]] = []
    facets: Dict[str, List[str]] = {}
    related: Optional[List[Dict[str, Any]]] = None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            name
            for (name,) in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "record_fields" in tables:
            named = "field" if "field" in columns(con, "record_fields") else "NULL"
            fields = [
                {"field": field, "label": label, "value": value}
                for field, label, value in con.execute(
                    FIELDS_QUERY.format(field=named), (record_id,)
                )
            ]
        if "record_facets" in tables:
            for facet, value in con.execute(FACETS_QUERY, (record_id,)):
                facets.setdefault(facet, []).append(value)
        if "related_sources" in tables:
            related = [
                describe_related(con, row, "related_tags" in tables)
                for row in con.execute(RELATED_QUERY, (record_id,)).fetchall()
            ]
    finally:
        con.close()
    details: Dict[str, Any] = {"fields": fields, "facets": facets}
    if related is not None:
        details["related"] = related
    return details


def related_thumbnail(path: Path, source_id: str) -> Optional[str]:
    """Where the thumbnail of a related photo is kept, relative to the collection"""
    if not has_table(path, "related_sources"):
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        found = con.execute(THUMBNAIL_QUERY, (source_id,)).fetchone()
    finally:
        con.close()
    return found[0] if found else None
