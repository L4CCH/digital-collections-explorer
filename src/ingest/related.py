from __future__ import annotations

import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from src.backend.services.catalog_fields import is_web_address, object_id_from
from src.collection.catalog import is_blank
from src.collection.writer import Related
from src.profiling.sources import fetch_web

from .pipeline import (
    THUMBNAIL_EDGE,
    THUMBNAIL_QUALITY,
    save_resized,
    sharded_path,
    stored_path,
)

REQUIRED = ("object_id", "source", "source_id")
DEFAULT_LINK_TYPE = "same_record"
EXAMPLES = 3


def manifest_keys(refs) -> tuple[set[str], set[str]]:
    object_ids = {
        str(found).strip()
        for found in (object_id_from(ref.extra.get("row") or {}) for ref in refs)
        if found
    }
    addresses = {ref.extra["url"] for ref in refs if ref.extra.get("url")}
    return object_ids, addresses


def read_related(path: str) -> tuple[list[str], list[dict[str, Any]]]:
    import duckdb

    cursor = duckdb.connect().execute(f"SELECT * FROM read_parquet('{path}')")
    columns = [described[0] for described in cursor.description]
    return columns, [dict(zip(columns, values)) for values in cursor.fetchall()]


def text(value: Any) -> str | None:
    return None if is_blank(value) else str(value).strip()


def counts_of(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if is_blank(value):
        return None
    try:
        counts = json.loads(str(value))
    except ValueError:
        return None
    return counts if isinstance(counts, dict) else None


def tags_of(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(tag).strip() for tag in values if not is_blank(tag)]


def counted(count: int, one: str, many: str) -> str:
    return f"{count:,} {one if count == 1 else many}"


def they(count: int) -> str:
    return "it is" if count == 1 else "they are"


def listed(values) -> str:
    values = [str(value) for value in values]
    if len(values) < 3:
        return " and ".join(values)
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def such_as(values) -> str:
    shown = sorted({str(value) for value in values})[:EXAMPLES]
    return listed(shown)


def related_problems(
    columns: list[str],
    rows: list[dict[str, Any]],
    object_ids: set[str],
    addresses: set[str],
) -> tuple[list[str], list[str]]:
    missing = [column for column in REQUIRED if column not in columns]
    if missing:
        return [
            f"The related table lacks {listed(missing)}, which every row needs: "
            "object_id names the manifest's object, source names where the item is, "
            "such as flickr, and source_id gives the item an id of its own, such as "
            f"flickr:2535935153. Its columns are {listed(columns)}."
        ], []
    refused, notes = [], []
    blank = [row for row in rows if any(is_blank(row.get(c)) for c in REQUIRED)]
    if blank:
        refused.append(
            "The related table leaves object_id, source, or source_id empty in "
            f"{counted(len(blank), 'row', 'rows')}."
        )
    ids = Counter(text(row.get("source_id")) for row in rows if row not in blank)
    twice = [source_id for source_id, count in ids.items() if count > 1]
    if twice:
        repeated = (
            f"the source_id {such_as(twice)}"
            if len(twice) <= EXAMPLES
            else f"{len(twice):,} source_id values, such as {such_as(twice)}"
        )
        refused.append(
            f"The related table repeats {repeated}; each item needs an id of its own."
        )
    unknown = [
        row
        for row in rows
        if row not in blank and text(row.get("object_id")) not in object_ids
    ]
    if unknown:
        notes.append(
            "The related table names objects the manifest does not have in "
            f"{counted(len(unknown), 'row', 'rows')}, such as "
            f"{such_as(text(r['object_id']) for r in unknown)}; "
            f"{they(len(unknown))} left out."
        )
    elsewhere = [
        row
        for row in rows
        if text(row.get("image")) and text(row.get("image")) not in addresses
    ]
    if elsewhere:
        notes.append(
            "The related table names images the manifest does not list in "
            f"{counted(len(elsewhere), 'row', 'rows')}, such as "
            f"{such_as(text(r['image']) for r in elsewhere)}; "
            f"{they(len(elsewhere))} kept without the image."
        )
    uncounted = [
        row
        for row in rows
        if not is_blank(row.get("counts")) and counts_of(row.get("counts")) is None
    ]
    if uncounted:
        notes.append(
            "The related table gives counts that are not a JSON object, such as "
            f'{{"faves": 3}}, in {counted(len(uncounted), "row", "rows")}; those '
            "counts are left out."
        )
    return refused, notes


def read_file(address: str, base_dir: Path) -> bytes | None:
    if is_web_address(address):
        found = fetch_web(address)
        return found[0] if found else None
    path = Path(address).expanduser()
    try:
        return (path if path.is_absolute() else base_dir / path).read_bytes()
    except OSError:
        return None


def kept_related(
    rows: list[dict[str, Any]],
    image_ids: dict[str, str],
    root: Path,
    data_dir: Path | None,
    base_dir: Path,
    workers: int,
) -> tuple[list[Related], int]:
    def thumbnail(row: dict[str, Any]) -> str | None:
        address = text(row.get("thumbnail"))
        if address is None:
            return None
        path = sharded_path(root, text(row["source_id"]))
        if not path.exists():
            data = read_file(address, base_dir)
            if data is None:
                return None
            try:
                image = Image.open(io.BytesIO(data)).convert("RGB")
                save_resized(image, THUMBNAIL_EDGE, THUMBNAIL_QUALITY, path)
            except Exception:
                return None
        return stored_path(path, data_dir)

    with ThreadPoolExecutor(max(1, workers)) as pool:
        thumbnails = list(pool.map(thumbnail, rows))
    related = [
        Related(
            record=text(row["object_id"]),
            source_id=text(row["source_id"]),
            source=text(row["source"]),
            link_type=text(row.get("link_type")) or DEFAULT_LINK_TYPE,
            image_id=image_ids.get(text(row.get("image"))),
            url=text(row.get("url")),
            title=text(row.get("title")),
            thumbnail=kept,
            rights=text(row.get("rights")),
            rights_url=text(row.get("rights_url")),
            counts=counts_of(row.get("counts")),
            tags=tags_of(row.get("tags")),
        )
        for row, kept in zip(rows, thumbnails)
    ]
    unread = sum(
        1
        for row, kept in zip(rows, thumbnails)
        if text(row.get("thumbnail")) and kept is None
    )
    return related, unread
