"""Read one record's catalog as what an object's page shows, its filters and its search text."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.backend.core.catalog_keys import IIIF_KEYS, MODEL_URL_KEYS, THUMBNAIL_KEYS
from src.backend.services.catalog_fields import (
    OBJECT_ID_KEYS,
    SOURCE_URL_KEYS,
    is_web_address,
)

SEARCHED = (
    "title",
    "names",
    "dates",
    "places",
    "topics",
    "object_types",
    "notes",
    "physical",
    "identifiers",
)
KINDS = {
    "title": ("title", "label"),
    "names": (
        "creator",
        "author",
        "artist",
        "maker",
        "photographer",
        "contributor",
        "publisher",
        "names",
        "people",
        "person",
        "sitter",
    ),
    "dates": ("date", "dates", "year", "date_created", "created", "period"),
    "places": (
        "place",
        "location",
        "geography",
        "spatial",
        "coverage",
        "country",
        "region",
        "state",
        "county",
        "city",
    ),
    "cultures": ("culture", "nationality", "people_group"),
    "subjects": ("subject", "topic", "keyword", "tag", "theme"),
    "object_types": ("object_type", "genre", "classification", "kind", "work_type"),
    "physical": (
        "medium",
        "material",
        "technique",
        "dimensions",
        "extent",
        "physical_description",
    ),
    "identifiers": (
        "identifier",
        "accession_number",
        "accession",
        "object_number",
        "call_number",
        "catalog_number",
        "inventory_number",
    ),
    "notes": (
        "description",
        "notes",
        "note",
        "summary",
        "abstract",
        "caption",
        "inscription",
        "transcription",
    ),
}
KIND_OF_COLUMN = {column: kind for kind, columns in KINDS.items() for column in columns}
SEARCHED_AS = {
    "title": "title",
    "names": "names",
    "dates": "dates",
    "places": "places",
    "cultures": "topics",
    "subjects": "topics",
    "object_types": "object_types",
    "physical": "physical",
    "identifiers": "identifiers",
    "notes": "notes",
}
FILTERED_AS = {
    "names": "name",
    "dates": "date",
    "places": "place",
    "cultures": "culture",
    "subjects": "topic",
    "object_types": "object_type",
}
RECOGNIZED = ("name", "place", "topic", "object_type")
NOT_CATALOG = {
    "image_url",
    "url",
    "path",
    "file",
    "uri",
    "file_name",
    "size_bytes",
    "size",
    "bytes",
    "filesize",
    "width",
    "height",
    "view_orbit",
    "group",
    *IIIF_KEYS,
    *MODEL_URL_KEYS,
    *THUMBNAIL_KEYS,
    *OBJECT_ID_KEYS,
}
LINKS = set(SOURCE_URL_KEYS)
LIST_SEPARATOR = re.compile(r"\s*[;|]\s*")
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
DECADE_IN_TEXT = re.compile(r"\b(1[0-9]{2}0|20[0-9]0)s\b")
FILED_DATE = re.compile(r"\d{1,4}s|\d{1,4}\s*[-–]\s*\d{1,4}|\d{1,4}")
NUMBER = re.compile(r"[\d.,\s-]+")
FILE_OR_MEDIA_TYPE = re.compile(
    r"[\w.+-]+/[\w.+-]+|\S+\.(jpe?g|png|tiff?|gif|webp|jp2|pdf|glb|gltf|parquet)",
    re.IGNORECASE,
)


@dataclass
class Catalog:
    """A record's catalog, sorted into what the site does with each part"""

    title: str | None = None
    source_url: str | None = None
    fields: list[tuple[str, str, str]] = field(default_factory=list)
    facets: list[tuple[str, str]] = field(default_factory=list)
    text: dict[str, list[str]] = field(default_factory=dict)


def values_of(value: Any) -> list[str]:
    """The text a catalog cell holds, as separate values, leaving out addresses and file names"""
    items = value if isinstance(value, (list, tuple)) else [value]
    found = []
    for item in items:
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, float) and math.isnan(item):
            continue
        text = str(item).strip()
        if text and not is_web_address(text) and not FILE_OR_MEDIA_TYPE.fullmatch(text):
            found.append(text)
    return found


def kind_of(column: str) -> str | None:
    """What a column holds, judged by its name, singular or plural"""
    name = column.strip().lower().replace(" ", "_")
    return KIND_OF_COLUMN.get(name) or KIND_OF_COLUMN.get(name.removesuffix("s"))


def label_of(column: str) -> str:
    """A column's name as a page shows it: date_created becomes Date created"""
    text = column.strip().replace("_", " ")
    return text[:1].upper() + text[1:] if text == text.lower() else text


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def split(values: list[str]) -> list[str]:
    return unique(
        [part for value in values for part in LIST_SEPARATOR.split(value) if part]
    )


def with_decades(dates: list[str]) -> list[str]:
    """Dates, followed by each year they name and its decade, as date queries look for"""
    years = [year for date in dates for year in YEAR.findall(date)]
    return unique([*dates, *years, *(f"{year[:3]}0s" for year in years)])


def filed_dates(dates: list[str]) -> list[str]:
    """Dates as the date filter reads them: a year, a span of years or a decade"""
    filed = []
    for date in dates:
        if FILED_DATE.fullmatch(date.strip()):
            filed.append(date.strip())
            continue
        years = sorted(set(YEAR.findall(date)))
        if len(years) == 1:
            filed.append(years[0])
        elif years:
            filed.append(f"{years[0]}-{years[-1]}")
        else:
            filed += [f"{decade}s" for decade in DECADE_IN_TEXT.findall(date)]
    return unique(filed)


def read_catalog(columns: dict[str, Any], record_id: str | None = None) -> Catalog:
    """What the site shows, filters by and searches of one record's catalog columns"""
    by_kind: dict[str, list[str]] = defaultdict(list)
    shown: list[tuple[str, str, str]] = []
    source_url = None
    for column, value in columns.items():
        name = str(column)
        lowered = name.strip().lower()
        if lowered in NOT_CATALOG:
            continue
        if lowered in LINKS:
            source_url = source_url or next(
                (
                    v
                    for v in map(str, value if isinstance(value, list) else [value])
                    if is_web_address(v)
                ),
                None,
            )
            continue
        values = values_of(value)
        if not values:
            continue
        kind = kind_of(name)
        if kind is None:
            if all(NUMBER.fullmatch(v) for v in values):
                continue
            kind = "notes"
        by_kind[kind].extend(values)
        if kind != "title":
            shown += [(name, label_of(name), v) for v in values]

    text: dict[str, list[str]] = defaultdict(list)
    for kind, values in by_kind.items():
        text[SEARCHED_AS[kind]].extend(values)
    if record_id:
        text["identifiers"].insert(0, record_id)
    if text.get("dates"):
        text["dates"] = with_decades(text["dates"])

    facets: list[tuple[str, str]] = []
    for kind, facet in FILTERED_AS.items():
        values = by_kind.get(kind, [])
        filed = filed_dates(values) if facet == "date" else split(values)
        facets += [(facet, value) for value in filed]

    return Catalog(
        title=next(iter(by_kind.get("title", [])), None),
        source_url=source_url,
        fields=shown,
        facets=unique(facets),
        text={column: unique(values) for column, values in text.items() if values},
    )
