from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.backend.services.catalog_fields import (
    MODEL_EXTENSIONS,
    model_url_from,
    object_id_from,
)
from src.profiling.sources import URL_COLUMN_GUESSES, LocalDirSource, is_parquet

IMAGE_COLUMN = "image_url"
SIZE_COLUMN = "size_bytes"
OTHER_3D_EXTS = {"obj", "ply", "stl", "fbx", "dae", "3ds", "usdz", "usd", "e57"}


def is_remote(address: str) -> bool:
    return address.startswith(("http://", "https://"))


def is_model_address(value: str) -> bool:
    return urlparse(value).path.lower().endswith(MODEL_EXTENSIONS)


def absolute(value: str, base: Path) -> str:
    if "://" in value:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else Path(os.path.normpath(base / path)))


@dataclass
class Model:
    address: str
    object_id: str
    row: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return hashlib.sha1(self.address.encode("utf-8")).hexdigest()

    @property
    def file_name(self) -> str:
        path = urlparse(self.address).path if is_remote(self.address) else self.address
        return Path(path).name

    @property
    def stem(self) -> str:
        return self.file_name.rsplit(".", 1)[0]


@dataclass
class Listing:
    models: list[Model] = field(default_factory=list)
    photos: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def conversion_hint(name: str) -> str:
    return (
        f"{name} must become a .glb file before it can be shown, for example "
        f"with Blender's glTF 2.0 export"
    )


def list_folder(root: str) -> Listing:
    source = LocalDirSource(root)
    listing = Listing()
    for ref in sorted(source.iter_files(), key=lambda ref: ref.key):
        if ref.is_junk or ref.extra.get("unreadable"):
            continue
        full = os.path.join(source.root, ref.key)
        if ref.ext == "glb":
            listing.models.append(Model(full, ref.key.rsplit(".", 1)[0]))
        elif ref.ext == "gltf" or ref.ext in OTHER_3D_EXTS:
            listing.problems.append(conversion_hint(ref.key))
        elif ref.is_image:
            listing.photos.append({IMAGE_COLUMN: full, SIZE_COLUMN: ref.size})
    return listing


def read_rows(path: str) -> tuple[list[str], list[dict]]:
    import duckdb

    reader = "read_parquet" if is_parquet(path) else "read_csv_auto"
    quoted = path.replace("'", "''")
    relation = duckdb.connect().execute(f"SELECT * FROM {reader}('{quoted}')")
    columns = [column[0] for column in relation.description]
    return columns, [dict(zip(columns, row)) for row in relation.fetchall()]


def image_column_of(columns: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    return next((lowered[c] for c in URL_COLUMN_GUESSES if c in lowered), None)


def list_manifest(path: str) -> Listing:
    base = Path(path).expanduser().resolve().parent
    columns, rows = read_rows(path)
    image_column = image_column_of(columns)
    listing = Listing()
    seen: set[str] = set()
    for row in rows:
        image = str(row.get(image_column) or "").strip() if image_column else ""
        address = model_url_from(row) or (image if is_model_address(image) else None)
        if address:
            address = absolute(address, base)
            local_gltf = not is_remote(address) and address.lower().endswith(".gltf")
            if local_gltf:
                listing.problems.append(conversion_hint(address))
            elif address not in seen:
                seen.add(address)
                carried = {k: v for k, v in row.items() if k != image_column}
                stem = Path(urlparse(address).path).stem
                listing.models.append(
                    Model(address, object_id_from(row) or stem, carried)
                )
        if image and not is_model_address(image):
            photo = {**row, IMAGE_COLUMN: absolute(image, base)}
            if address:
                photo["model_url"] = address
            listing.photos.append(photo)
    return listing


def list_source(uri: str) -> Listing:
    if is_parquet(uri) or uri.lower().endswith(".csv"):
        return list_manifest(uri)
    return list_folder(uri)
