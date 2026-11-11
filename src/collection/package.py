"""Check a collection package, the Data Package a source plugin writes, and build from it."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import numpy as np

from src.backend.core.catalog_keys import IIIF_KEYS, MODEL_URL_KEYS, THUMBNAIL_KEYS
from src.backend.core.config import ModelType
from src.backend.core.formats import INGEST_STATE_FILE
from src.backend.services.catalog_fields import (
    OBJECT_ID_KEYS,
    SOURCE_URL_KEYS,
    iiif_info_from,
    is_web_address,
)
from src.backend.services.shape_index import (
    INFO_FILE,
    OBJECTS_FILE,
    SHAPE_DIR,
    VECTORS_FILE,
)

from .catalog import (
    DEFAULT_GROUP,
    SEARCHED,
    group_id_of,
    is_blank,
    merged,
    read_catalog,
)
from .writer import (
    Group,
    Item,
    Model,
    Record,
    Related,
    replace_folder,
    write_collection,
)

CONTRACT = 1
DESCRIPTOR = "datapackage.json"
PROFILE = "https://datapackage.org/profiles/2.0/datapackage.json"
COLLECTION_FILE = "collection.json"
REQUIRED = {
    "files": ("id",),
    "records": ("object_id",),
    "fields": ("object_id", "label", "value"),
    "facets": ("object_id", "facet", "value"),
    "text": ("object_id", "field", "value"),
    "related_sources": ("object_id", "source_id", "source", "link_type"),
    "related_tags": ("source_id", "tag"),
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
FACETS = ("date", "name", "place", "culture", "topic", "object_type")
MODEL_TYPES = tuple(model_type.value for model_type in ModelType)
FIELD_TYPES = (
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
    "list",
    "datetime",
    "date",
    "time",
    "year",
    "yearmonth",
    "duration",
    "geopoint",
    "geojson",
    "any",
)
FILE_SIZES = ("size_bytes", "size", "bytes", "filesize")
NAMED = {
    **dict.fromkeys(OBJECT_ID_KEYS, "object_id"),
    **dict.fromkeys(SOURCE_URL_KEYS, "source_url"),
    **dict.fromkeys(IIIF_KEYS, "iiif"),
    **dict.fromkeys(MODEL_URL_KEYS, "model_url"),
    **dict.fromkeys(THUMBNAIL_KEYS, "thumbnail"),
    **dict.fromkeys(("image_url", "url", "path", "file", "uri"), "original"),
    **{name: name for name in ("group", "file_name", "width", "height", "view_orbit")},
}
STATE_FILES = tuple(f"{INGEST_STATE_FILE}{end}" for end in ("", "-wal", "-shm"))
EXAMPLES = 3


@dataclass
class Report:
    """What check found: what refuses a package, what is worth knowing, what it holds"""

    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    contract: Any = None
    producer: dict[str, str] | None = None
    provenance: dict[str, Any] | None = None
    counts: dict[str, Any] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Shard:
    """One file of a model's vectors, with the id of the file in each row"""

    path: str
    ids: list[str]
    matrix: np.ndarray


@dataclass
class Package:
    """The tables, vectors and description a package holds"""

    folder: Path
    producer: dict[str, str]
    provenance: dict[str, Any] | None
    models: list[Model]
    default_model: str | None
    tables: dict[str, list[dict] | None]
    columns: dict[str, list[str]]
    shards: dict[str, list[Shard]]
    shape: str | None
    collection: dict | None


class Refused(Exception):
    """A package, or a collection folder, that build will not write from or into"""

    def __init__(self, problems: list[str]):
        super().__init__(" ".join(problems))
        self.problems = problems


def plural(count: int, noun: str, nouns: str | None = None) -> str:
    return f"{count:,} {noun if count == 1 else nouns or noun + 's'}"


def listed(values, conjunction: str = "and") -> str:
    """Values as a sentence lists them, with the serial comma"""
    values = [str(value) for value in values]
    if len(values) < 3:
        return f" {conjunction} ".join(values)
    return f"{', '.join(values[:-1])}, {conjunction} {values[-1]}"


def the(noun: str, values) -> str:
    """Values after their noun: the facet date, or the facets date and name"""
    values = list(values)
    return f"the {noun if len(values) == 1 else noun + 's'} {listed(values)}"


def such_as(values) -> str:
    values = list(values)
    return f", such as {listed(values[:EXAMPLES])}" if values else ""


def present(value: Any) -> Any:
    return None if is_blank(value) else value


def inside(path: Any) -> bool:
    """Whether a path names a file inside the package, as the standard writes one"""
    if not isinstance(path, str) or not path or "\\" in path or "://" in path:
        return False
    return (
        not path.startswith((".", "/", "~")) and ".." not in PurePosixPath(path).parts
    )


def read_descriptor(folder: Path) -> tuple[dict | None, str | None]:
    path = folder / DESCRIPTOR
    if not path.is_file():
        return None, f"{folder} has no {DESCRIPTOR}, so it is not a collection package."
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        return None, f"{DESCRIPTOR} is not valid JSON: {error}."
    if not isinstance(descriptor, dict):
        return None, f"{DESCRIPTOR} must hold one JSON object."
    return descriptor, None


def contract_problem(contract: Any) -> str | None:
    """Why this DCE cannot read a package of this contract, if it cannot"""
    reads = f"This DCE reads collection package contract {CONTRACT}"
    if type(contract) is int and contract == CONTRACT:
        return None
    if contract is None:
        return (
            f"{DESCRIPTOR} names no contract. {reads}: give it "
            f'"dce": {{"contract": {CONTRACT}, ...}}.'
        )
    if type(contract) is int and contract > CONTRACT:
        return (
            f"The package follows collection package contract {contract}, which is "
            f"newer than this DCE reads. {reads}: update DCE, or write the package "
            f"with a version of its plugin that writes contract {CONTRACT}."
        )
    return f"The package names the contract {json.dumps(contract)}. {reads}."


def producer_of(producer: Any, report: Report) -> dict[str, str] | None:
    """The plugin that wrote the package, and its version and address"""
    named = isinstance(producer, dict) and all(
        isinstance(producer.get(key), str) and producer[key].strip()
        for key in ("name", "version")
    )
    if not named:
        report.problems.append(
            "dce.producer must name the plugin that wrote the package and its "
            'version, as {"name": "my-plugin", "version": "1.0.0"}.'
        )
        return None
    if producer.get("url") is not None and not is_web_address(producer["url"]):
        report.problems.append(
            "dce.producer.url must be an http:// or https:// address."
        )
        return None
    return {
        key: producer[key]
        for key in ("name", "version", "url")
        if producer.get(key) is not None
    }


def resource_problems(name: str, resource: dict) -> list[str]:
    problems = []
    path = resource.get("path")
    if not isinstance(path, str):
        problems.append(
            f"The resource {name} must give the path of one parquet file, such as "
            f"{name}.parquet."
        )
    elif not inside(path):
        problems.append(
            f"The resource {name} has the path {path}, which is not a file inside the "
            f"package: give a path relative to the package folder, such as "
            f"{name}.parquet."
        )
    schema = resource.get("schema")
    fields = schema.get("fields") if isinstance(schema, dict) else None
    if not (
        isinstance(fields, list)
        and fields
        and all(isinstance(f, dict) and isinstance(f.get("name"), str) for f in fields)
    ):
        problems.append(
            f"The resource {name} must give its Table Schema under schema, with each "
            f'of its columns under fields, as {{"name": "id", "type": "string"}}.'
        )
        return problems
    odd = [str(f["type"]) for f in fields if f.get("type", "any") not in FIELD_TYPES]
    if odd:
        problems.append(
            f"The Table Schema of {name} gives {the('type', odd)}, which Table "
            f"Schema does not define: use string, integer, number, boolean, or "
            f"another of its types."
        )
    return problems


def read_resources(descriptor: dict, report: Report) -> dict[str, dict]:
    """The tables a package lists, by name"""
    resources = descriptor.get("resources")
    if not isinstance(resources, list) or not resources:
        report.problems.append(
            f"{DESCRIPTOR} lists no resources; it needs at least the files table."
        )
        return {}
    named: dict[str, dict] = {}
    for resource in resources:
        name = resource.get("name") if isinstance(resource, dict) else None
        if not isinstance(name, str):
            report.problems.append(
                "Each resource must be an object with a name, such as files."
            )
        elif name in named:
            report.problems.append(f"Two resources are named {name}.")
        else:
            named[name] = resource
    unknown = [name for name in named if name not in REQUIRED]
    if unknown:
        report.notes.append(
            f"DCE does not read {the('resource', unknown)}; it reads "
            f"{listed(REQUIRED)}."
        )
    if named and "files" not in named:
        report.problems.append(
            "The package has no resource named files, the table of its files."
        )
    tables = {name: named[name] for name in REQUIRED if name in named}
    for name, resource in tables.items():
        report.problems += resource_problems(name, resource)
    return tables


def model_problems(key: str, entry: dict) -> list[str]:
    problems = []
    if entry.get("type") not in MODEL_TYPES:
        problems.append(
            f"The model {key} has the type {json.dumps(entry.get('type'))}, which DCE "
            f"cannot search with: give clip, siglip, or siglip2."
        )
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        problems.append(
            f"The model {key} must give its name on Hugging Face, such as "
            f"google/siglip2-base-patch16-224."
        )
    revision = entry.get("revision")
    if revision is not None and not isinstance(revision, str):
        problems.append(
            f"The model {key} must give its revision as a string: the commit of the "
            f"model on Hugging Face."
        )
    dimensions = entry.get("dimensions")
    if type(dimensions) is not int or dimensions < 1:
        problems.append(
            f"The model {key} must give the number of its dimensions, such as 768."
        )
    vectors = entry.get("vectors")
    if not (
        isinstance(vectors, list)
        and vectors
        and all(
            inside(path) and path.endswith(".npy") and not path.endswith(".ids.npy")
            for path in vectors
        )
    ):
        problems.append(
            f"The model {key} must list its vector files under vectors, as .npy files "
            f"inside the package, such as vectors/{key}/collection.npy."
        )
    return problems


def read_models(dce: dict, report: Report) -> tuple[list[Model], dict[str, list[str]]]:
    """The models a package holds vectors from, and the files of each one's vectors"""
    entries = dce.get("models")
    if not isinstance(entries, list) or not entries:
        report.problems.append(
            "dce.models must list each model whose vectors the package holds."
        )
        return [], {}
    models, paths = [], {}
    seen: set[str] = set()
    for entry in entries:
        key = entry.get("key") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not key or group_id_of(key) != key:
            fixed = group_id_of(key) if isinstance(key, str) else ""
            report.problems.append(
                f"The model key {json.dumps(key)} must be in lowercase letters, digits, "
                f"and hyphens, such as {fixed or 'siglip2'}."
            )
            continue
        if key in seen:
            report.problems.append(f"Two models have the key {key}.")
            continue
        seen.add(key)
        problems = model_problems(key, entry)
        report.problems += problems
        if problems:
            continue
        if entry.get("revision") is None:
            report.notes.append(
                f"The model {key} records no revision, so the site may load another "
                f"version of {entry['name']} than the one that made the vectors."
            )
        models.append(
            Model(
                key=key,
                type=entry["type"],
                name=entry["name"],
                dimensions=entry["dimensions"],
                revision=entry.get("revision"),
            )
        )
        paths[key] = list(dict.fromkeys(entry["vectors"]))
    default = dce.get("default_model")
    if default is not None and default not in seen:
        report.problems.append(
            f"dce.default_model is {json.dumps(default)}, which is not the key of one "
            f"of the models."
        )
    return models, paths


def read_table(path: Path) -> tuple[list[str], list[dict]]:
    con = duckdb.connect()
    try:
        cursor = con.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = [column[0] for column in cursor.description]
        return columns, [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        con.close()


def schema_problems(name: str, path: str, fields: list, columns: list) -> list[str]:
    """How a table's Table Schema differs from the columns of its file"""
    named = [f["name"] for f in fields]
    if named == columns:
        return []
    missing = [column for column in named if column not in columns]
    unlisted = [column for column in columns if column not in named]
    problems = []
    if missing:
        problems.append(
            f"The Table Schema of {name} lists {listed(missing)}, which {path} has no "
            f"column for."
        )
    if unlisted:
        problems.append(
            f"{path} has {the('column', unlisted)}, which "
            f"the Table Schema of {name} does not list."
        )
    return problems or [
        f"The Table Schema of {name} lists the columns of {path} in another order: "
        f"list them in the order of the file."
    ]


def read_tables(folder: Path, resources: dict[str, dict], report: Report):
    """Each table's columns and rows, with None for a table the package does not give"""
    tables: dict[str, list[dict] | None] = dict.fromkeys(REQUIRED)
    columns: dict[str, list[str]] = {}
    for name, resource in resources.items():
        path = resource["path"]
        if not (folder / path).is_file():
            report.problems.append(
                f"The resource {name} names {path}, which is not in the package."
            )
            continue
        try:
            columns[name], tables[name] = read_table(folder / path)
        except duckdb.Error as error:
            report.problems.append(
                f"{path} could not be read as parquet: {str(error).splitlines()[0]}"
            )
            continue
        report.problems += schema_problems(
            name, path, resource["schema"]["fields"], columns[name]
        )
        missing = [c for c in REQUIRED[name] if c not in columns[name]]
        if missing:
            report.problems.append(
                f"{path} has no {'column' if len(missing) == 1 else 'columns'} "
                f"{listed(missing)}, which the {name} table needs."
            )
    return tables, columns


def read_shard(folder: Path, model: Model, path: str) -> Shard | str:
    """One file of a model's vectors and its ids, or why it cannot be read"""
    ids_path = f"{path.removesuffix('.npy')}.ids.npy"
    if not (folder / path).is_file():
        return f"The model {model.key} lists {path}, which is not in the package."
    if not (folder / ids_path).is_file():
        return (
            f"{path} has no ids beside it: write the id of the file in each of its "
            f"rows to {ids_path}."
        )
    try:
        matrix = np.load(folder / path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        return f"{path} could not be read as a NumPy array: {error}"
    if matrix.ndim != 2 or matrix.dtype.kind != "f":
        return (
            f"{path} must hold one row of floating-point numbers per file, but it "
            f"holds {matrix.dtype} in the shape {matrix.shape}."
        )
    if matrix.shape[1] != model.dimensions:
        return (
            f"{path} has rows of {matrix.shape[1]:,} numbers, but the model "
            f"{model.key} has {model.dimensions:,} dimensions."
        )
    try:
        ids = np.load(folder / ids_path, allow_pickle=False)
        if ids.ndim != 1 or ids.dtype.kind not in "USiu":
            raise ValueError(f"it holds {ids.dtype} in the shape {ids.shape}")
        names = [i.decode() if isinstance(i, bytes) else str(i) for i in ids.tolist()]
    except (OSError, ValueError) as error:
        return (
            f"{ids_path} could not be read as the ids of {path}: {error}. Write them "
            f"as a NumPy array of strings."
        )
    if len(names) != matrix.shape[0]:
        return (
            f"{path} has {plural(matrix.shape[0], 'row')}, but {ids_path} has "
            f"{plural(len(names), 'id')}."
        )
    return Shard(path, names, matrix)


def read_shape(folder: Path, path: str) -> int | str:
    """How many objects the shape index of a package holds, or why it cannot be served"""
    shape = folder / path
    if not shape.is_dir():
        return f"dce.shape names {path}, which is not a folder in the package."
    missing = [n for n in (VECTORS_FILE, OBJECTS_FILE) if not (shape / n).is_file()]
    if missing:
        return (
            f"The shape index in {path} has no {listed(missing)}: write it as "
            f"python -m src.render3d does, with {VECTORS_FILE}, {OBJECTS_FILE}, and "
            f"{INFO_FILE}."
        )
    try:
        vectors = np.load(shape / VECTORS_FILE, mmap_mode="r", allow_pickle=False)
        objects = json.loads((shape / OBJECTS_FILE).read_text(encoding="utf-8"))
        info = shape / INFO_FILE
        described = (
            json.loads(info.read_text(encoding="utf-8")) if info.exists() else {}
        )
    except (OSError, ValueError) as error:
        return f"The shape index in {path} could not be read: {error}."
    if not (
        vectors.ndim == 2
        and vectors.dtype.kind == "f"
        and isinstance(objects, list)
        and isinstance(described, dict)
    ):
        return (
            f"The shape index in {path} must hold a row of numbers per object in "
            f"{VECTORS_FILE}, the list of their ids in {OBJECTS_FILE}, and what made "
            f"them in {INFO_FILE}, as python -m src.render3d writes them."
        )
    if len(objects) != vectors.shape[0]:
        return (
            f"The shape index in {path} has {plural(vectors.shape[0], 'row')} in "
            f"{VECTORS_FILE}, but {plural(len(objects), 'object')} in {OBJECTS_FILE}."
        )
    return len(objects)


def read_collection_file(folder: Path, report: Report) -> dict | None:
    path = folder / COLLECTION_FILE
    if not path.exists():
        return None
    try:
        described = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        report.problems.append(f"{COLLECTION_FILE} is not valid JSON: {error}.")
        return None
    if not isinstance(described, dict):
        report.problems.append(f"{COLLECTION_FILE} must hold one JSON object.")
        return None
    if not isinstance(described.get("groups", {}), dict):
        report.problems.append(
            f"The groups of {COLLECTION_FILE} must be an object that names each group "
            f"by its id, with its label, institution, and license."
        )
    return described


def read_columns(table: str, has_records: bool) -> set[str]:
    """The columns of a table that build reads by their names"""
    if table == "records":
        return {"object_id", *RECORD_COLUMNS}
    return {
        "id",
        "group",
        "object_id",
        *ITEM_COLUMNS,
        *(() if has_records else RECORD_COLUMNS),
    }


def column_problem(table: str, column: str, has_records: bool) -> str | None:
    """Why a column's name would hide it from build, which reads exact names"""
    read = read_columns(table, has_records)
    if column in read:
        return None
    lowered = str(column).strip().lower()
    found = f"The {table} table has a column {column}, which DCE does not read"
    if lowered in FILE_SIZES:
        return (
            f"{found}: leave it out, or name it dimensions if it measures the object."
        )
    instead = NAMED.get(lowered)
    if instead is None:
        return None
    if lowered == "group_id":
        return (
            f"{found}: name it object_id if it names the object each file shows, or "
            f"give it as group in the files table if it names a group, such as a "
            f"museum."
        )
    if instead in read:
        return f"{found}: name it {instead}."
    other = "files" if table == "records" else "records"
    return f"{found} there: give it as {instead} in the {other} table."


def file_problems(files: list[dict]) -> list[str]:
    problems = []
    blank = sum(1 for row in files if is_blank(row["id"]))
    if blank:
        problems.append(f"The files table has {plural(blank, 'row')} without an id.")
    twice = [i for i, n in Counter(str(row["id"]) for row in files).items() if n > 1]
    if twice:
        problems.append(
            f"The files table lists {plural(len(twice), 'file')} more than once"
            f"{such_as(twice)}."
        )
    groups = Counter(
        str(row["group"]) for row in files if not is_blank(row.get("group"))
    )
    for group, count in groups.items():
        fixed = group_id_of(group)
        if fixed != group:
            written = f"as {fixed}, " if fixed else ""
            problems.append(
                f'The files table puts {plural(count, "file")} in the group "{group}", '
                f"which is not a group id: write it {written}in lowercase letters, "
                f"digits, and hyphens, as python -m src.ingest does."
            )
    unserved = [
        str(row["id"])
        for row in files
        if not is_blank(row.get("iiif"))
        and iiif_info_from({"iiif": str(row["iiif"])}) is None
    ]
    if unserved:
        problems.append(
            f"The files table gives {plural(len(unserved), 'file')} an iiif value that "
            f"is not a web address{such_as(unserved)}: give the address of each "
            f"file's IIIF image service."
        )
    return problems


def record_problems(tables: dict[str, list[dict] | None]) -> list[str]:
    problems = []
    records = tables["records"]
    if records is not None:
        blank = sum(1 for row in records if is_blank(row["object_id"]))
        if blank:
            problems.append(
                f"The records table has {plural(blank, 'row')} without an object_id."
            )
        counted = Counter(str(row["object_id"]) for row in records)
        twice = [record for record, n in counted.items() if n > 1]
        if twice:
            problems.append(
                f"The records table lists {plural(len(twice), 'object')} more than once"
                f"{such_as(twice)}."
            )
    rows, key, noun = (
        (records, "object_id", "record")
        if records is not None
        else (tables["files"], "id", "file")
    )
    unlinked = [
        str(row[key])
        for row in rows
        if not is_blank(row.get("source_url"))
        and not is_web_address(str(row["source_url"]))
    ]
    if unlinked:
        problems.append(
            f"The {noun}s table gives {plural(len(unlinked), noun)} a source_url that is "
            f"not an http:// or https:// address{such_as(unlinked)}: give the web "
            f"address of each record's page, or leave it empty."
        )
    return problems


def counts_of(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    try:
        counts = json.loads(value)
    except (TypeError, ValueError):
        return None
    return counts if isinstance(counts, dict) else None


def catalog_problems(tables: dict[str, list[dict] | None]) -> list[str]:
    """What the facets, text and related sources give that the site cannot use"""
    problems = []
    facets = {str(row["facet"]) for row in tables["facets"] or []}
    unfiltered = sorted(facets - set(FACETS))
    if unfiltered:
        problems.append(
            f"The facets table has {the('facet', unfiltered)}, which the site does "
            f"not filter by: give {listed(FACETS, 'or')}. A group already gives a "
            f"filter by institution."
        )
    fields = {str(row["field"]) for row in tables["text"] or []}
    unsearched = sorted(fields - set(SEARCHED))
    if unsearched:
        problems.append(
            f"The text table names {the('field', unsearched)}, which search does not "
            f"match: give {listed(SEARCHED, 'or')}."
        )
    related = tables["related_sources"] or []
    pairs = Counter((str(r["object_id"]), str(r["source_id"])) for r in related)
    twice = [f"{source} for {record}" for (record, source), n in pairs.items() if n > 1]
    if twice:
        problems.append(
            f"The related_sources table lists {plural(len(twice), 'source')} more than "
            f"once for the same object{such_as(twice)}."
        )
    uncounted = [
        str(row["source_id"])
        for row in related
        if not is_blank(row.get("counts")) and counts_of(row["counts"]) is None
    ]
    if uncounted:
        problems.append(
            f"The related_sources table gives {plural(len(uncounted), 'source')} counts "
            f"that are not a JSON object{such_as(uncounted)}."
        )
    return problems


def vector_problems(package: Package) -> tuple[list[str], list[str]]:
    """Files without a vector from every model, and the ids of vectors without a file"""
    problems, without_files = [], []
    ids = [str(row["id"]) for row in package.tables["files"]]
    known = set(ids)
    for model in package.models:
        rows = Counter(i for shard in package.shards[model.key] for i in shard.ids)
        twice = [i for i, n in rows.items() if n > 1]
        if twice:
            problems.append(
                f"The vectors of {model.key} give {plural(len(twice), 'file')} more "
                f"than one row{such_as(twice)}."
            )
        missing = [i for i in ids if i not in rows]
        if missing:
            have = "has" if len(missing) == 1 else "have"
            problems.append(
                f"{plural(len(missing), 'file')} {have} no vector from {model.key}"
                f"{such_as(missing)}: every file needs a vector from every model."
            )
        without_files += [
            i for i, n in rows.items() if i not in known for _ in range(n)
        ]
    return problems, without_files


def unmatched(tables: dict[str, list[dict] | None]) -> tuple[list[str], list[str]]:
    """The records that no file shows, and the files whose object has no record"""
    files, records = tables["files"], tables["records"]
    if records is None:
        return [], []
    objects = {str(row["id"]): present(row.get("object_id")) for row in files}
    shown = {str(record) for record in objects.values() if record is not None}
    given = list(dict.fromkeys(str(row["object_id"]) for row in records))
    recorded = set(given)
    unrecorded = [
        item
        for item, record in objects.items()
        if record is not None and str(record) not in recorded
    ]
    return [record for record in given if record not in shown], unrecorded


def group_of(row: dict) -> str:
    group = present(row.get("group"))
    return DEFAULT_GROUP if group is None else str(group)


def examine(folder: Path) -> tuple[Report, Package | None]:
    """Check a package, and read it if it can be built"""
    folder = Path(folder)
    report = Report()
    descriptor, problem = read_descriptor(folder)
    if problem:
        report.problems.append(problem)
        return report, None
    dce = descriptor.get("dce") if isinstance(descriptor.get("dce"), dict) else {}
    report.contract = dce.get("contract")
    problem = contract_problem(report.contract)
    if problem:
        report.problems.append(problem)
        return report, None
    if descriptor.get("$schema") != PROFILE:
        report.problems.append(
            f'{DESCRIPTOR} must give "$schema": "{PROFILE}", the version of the Data '
            f"Package standard it follows."
        )
    report.producer = producer_of(dce.get("producer"), report)
    provenance = dce.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        report.problems.append(
            'dce.provenance must be a JSON object, such as {"commit": "4f2a9c1"}.'
        )
    report.provenance = provenance or None
    shape = dce.get("shape")
    if shape is not None and not inside(shape):
        report.problems.append(
            "dce.shape must give the path of a folder inside the package, such as "
            "shape."
        )
    resources = read_resources(descriptor, report)
    models, paths = read_models(dce, report)
    if report.problems:
        return report, None

    tables, columns = read_tables(folder, resources, report)
    shards: dict[str, list[Shard]] = {model.key: [] for model in models}
    for model in models:
        for path in paths[model.key]:
            shard = read_shard(folder, model, path)
            if isinstance(shard, Shard):
                shards[model.key].append(shard)
            else:
                report.problems.append(shard)
    shape_objects = None if shape is None else read_shape(folder, shape)
    if isinstance(shape_objects, str):
        report.problems.append(shape_objects)
    collection = read_collection_file(folder, report)
    if report.problems:
        return report, None

    package = Package(
        folder,
        report.producer,
        report.provenance,
        models,
        dce.get("default_model"),
        tables,
        columns,
        shards,
        shape,
        collection,
    )
    has_records = tables["records"] is not None
    for table in ("files", "records"):
        for column in columns.get(table, []):
            problem = column_problem(table, column, has_records)
            if problem:
                report.problems.append(problem)
    report.problems += file_problems(tables["files"])
    report.problems += record_problems(tables)
    report.problems += catalog_problems(tables)
    problems, without_files = vector_problems(package)
    report.problems += problems

    if has_records:
        unread = [
            column
            for column in columns["files"]
            if column not in read_columns("files", True)
            and column_problem("files", column, True) is None
        ]
        if unread:
            report.notes.append(
                f"DCE does not read {the('column', unread)} of the files table, since "
                f"the records table holds the catalog."
            )
    groups = list(dict.fromkeys(group_of(row) for row in tables["files"]))
    described = (collection or {}).get("groups") or {}
    undescribed = [group for group in described if group not in groups]
    if undescribed:
        report.notes.append(
            f"{COLLECTION_FILE} describes {the('group', undescribed)}, which no "
            f"file is in."
        )

    unshown, unrecorded = unmatched(tables)
    report.counts = {
        "files": len(tables["files"]),
        "records": None if tables["records"] is None else len(tables["records"]),
        "groups": len(groups),
        "models": [model.key for model in models],
        "records_without_files": len(unshown),
        "files_without_records": len(unrecorded),
        "vectors_without_files": len(without_files),
        "shape_objects": shape_objects,
    }
    report.examples = {
        "records_without_files": unshown[:EXAMPLES],
        "files_without_records": unrecorded[:EXAMPLES],
        "vectors_without_files": list(dict.fromkeys(without_files))[:EXAMPLES],
    }
    return report, None if report.problems else package


def check(folder: Path) -> Report:
    """What a package holds, and whatever keeps it from being built"""
    return examine(folder)[0]


def grouped(rows: list[dict] | None, key: str = "object_id") -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = defaultdict(list)
    for row in rows or []:
        if not is_blank(row.get(key)):
            found[str(row[key])].append(row)
    return found


def in_order(rows: list[dict]) -> list[dict]:
    """Rows in the order of their position, then of the table"""
    return [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda pair: (
                (
                    pair[1]["position"]
                    if not is_blank(pair[1].get("position"))
                    else pair[0]
                ),
                pair[0],
            ),
        )
    ]


def explicit_fields(rows: list[dict]) -> list[tuple]:
    return [
        (present(row.get("field")), str(row["label"]), str(row["value"]))
        for row in in_order(rows)
        if not is_blank(row["label"]) and not is_blank(row["value"])
    ]


def explicit_text(rows: list[dict]) -> dict[str, list[str]]:
    text: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if not is_blank(row["value"]):
            text[str(row["field"])].append(str(row["value"]))
    return dict(text)


def assemble(package: Package) -> dict[str, Any]:
    """Everything write_collection takes, from a package that check accepts"""
    tables = package.tables
    has_records = tables["records"] is not None
    items, files_of = [], defaultdict(list)
    for row in tables["files"]:
        item_id = str(row["id"])
        catalog = {k: v for k, v in row.items() if k not in ("id", *ITEM_COLUMNS)}
        record = present(row.get("object_id"))
        if record is None and not has_records:
            read = read_catalog(catalog)
            record = item_id if (read.title or read.fields or read.source_url) else None
        if record is not None:
            files_of[str(record)].append(catalog)
        shown = {column: present(row.get(column)) for column in ITEM_COLUMNS}
        if shown["iiif"] is not None:
            shown["iiif"] = iiif_info_from({"iiif": str(shown["iiif"])})
        items.append(
            Item(
                id=item_id,
                group=group_of(row),
                record=None if record is None else str(record),
                **shown,
            )
        )

    given = {str(row["object_id"]): row for row in tables["records"] or []}
    by_table = {name: grouped(tables[name]) for name in ("fields", "facets", "text")}
    records = []
    for record_id in dict.fromkeys([*given, *files_of]):
        rows = files_of.get(record_id, [])
        catalog = (given.get(record_id) if has_records else merged(rows)) or {}
        records.append(
            Record(
                id=record_id,
                catalog=catalog,
                searchable_id=has_records
                or any(not is_blank(r.get("object_id")) for r in rows),
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
                    explicit_text(by_table["text"].get(record_id, []))
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
                str(tag["tag"]) for tag in in_order(tags.get(str(row["source_id"]), []))
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

    described = (package.collection or {}).get("groups") or {}
    present_groups = {item.group for item in items}
    order = [g for g in described if g in present_groups]
    order += sorted(present_groups - set(order))

    def label_of(group: str) -> str:
        entry = described.get(group)
        label = entry.get("label") if isinstance(entry, dict) else None
        return label if isinstance(label, str) and label.strip() else group

    vectors = {
        model.key: (
            [i for shard in package.shards[model.key] for i in shard.ids],
            np.concatenate(
                [
                    np.asarray(shard.matrix, dtype=np.float32)
                    for shard in package.shards[model.key]
                ]
            ),
        )
        for model in package.models
    }
    return {
        "models": package.models,
        "default_model": package.default_model,
        "groups": [Group(group, label_of(group)) for group in order],
        "items": items,
        "records": records,
        "vectors": vectors,
        "related": related,
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
    folder: Path,
    data_dir: Path,
    check_files: bool = False,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Report]:
    """Check a package, then write the collection it describes into a collection folder"""
    folder, data_dir = Path(folder), Path(data_dir)
    embeddings_dir = data_dir / "embeddings"
    root = folder.resolve()
    if data_dir.resolve() == root or root in data_dir.resolve().parents:
        raise Refused(
            [
                f"{data_dir} is inside the package {folder}. Build writes the "
                f"collection into a folder of its own and never changes the package: "
                f"give --data-dir a folder outside it."
            ]
        )
    if (embeddings_dir / INGEST_STATE_FILE).exists() and not overwrite:
        raise Refused(
            [
                f"{data_dir} holds a collection that python -m src.ingest wrote, which "
                f"ingest would write anew from {INGEST_STATE_FILE} when run there "
                f"again. Build into another folder, or pass --overwrite to replace "
                f"the collection and remove {INGEST_STATE_FILE}."
            ]
        )
    report, package = examine(folder)
    if package is None:
        raise Refused(report.problems)
    read = assemble(package)
    missing = missing_files(data_dir, read) if check_files else []
    if missing:
        raise Refused(
            [
                f"{plural(len(missing), 'file')} the package names "
                f"{'is' if len(missing) == 1 else 'are'} not in {data_dir}"
                f"{such_as(missing)}."
            ]
        )
    written = write_collection(
        embeddings_dir,
        read["models"],
        read["groups"],
        read["items"],
        read["records"],
        read["vectors"],
        default_model=read["default_model"],
        related=read["related"],
        producer=package.producer,
        provenance=package.provenance,
    )
    shape = embeddings_dir / SHAPE_DIR
    if package.shape is None:
        shutil.rmtree(shape, ignore_errors=True)
    else:
        building = embeddings_dir / f".{SHAPE_DIR}.building"
        shutil.rmtree(building, ignore_errors=True)
        building.mkdir()
        for name in (VECTORS_FILE, OBJECTS_FILE, INFO_FILE):
            if (folder / package.shape / name).is_file():
                shutil.copyfile(folder / package.shape / name, building / name)
        replace_folder(building, shape)
    if package.collection is not None:
        building = data_dir / f".{COLLECTION_FILE}.building"
        shutil.copyfile(folder / COLLECTION_FILE, building)
        os.replace(building, data_dir / COLLECTION_FILE)
    for name in STATE_FILES:
        (embeddings_dir / name).unlink(missing_ok=True)
    return written, report
