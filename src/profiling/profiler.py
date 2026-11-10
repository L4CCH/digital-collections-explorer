from __future__ import annotations

import datetime as dt
import io
import re
import statistics
from collections import Counter, defaultdict

from PIL import Image, ImageFile

from src.backend.core.catalog_keys import IIIF_KEYS, MODEL_URL_KEYS, THUMBNAIL_KEYS
from src.backend.services.catalog_fields import OBJECT_ID_KEYS
from src.collection.catalog import (
    SEARCHED_AS,
    column_role,
    group_id_warning,
    kind_of,
    read_catalog,
)

from .models import (
    Capabilities,
    Capability,
    CatalogCandidate,
    CollectionProfile,
    Decision,
    DirectorySignal,
    FieldProfile,
    Issue,
    NamePattern,
    Provenance,
    ScanStats,
    Sizing,
    Structure,
)
from .sources import MODEL_EXTS, FileRef, Source

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Largest image seen in the Smithsonian NMAH collection was 123 megapixels.
Image.MAX_IMAGE_PIXELS = 400_000_000

# Enough of a JPEG to read its dimensions and EXIF block without a full download.
HEAD_BYTES = 128 * 1024

# EXIF tag ids.
EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306
EXIF_GPS = 34853
GPS_LATITUDE = 2
GPS_LONGITUDE = 4

# Longest edge of the two copies src/ingest keeps of every image.
THUMBNAIL_EDGE = 400
PROCESSED_EDGE = 1920
# Measured on a full run of 20,481 Smithsonian images: 7.9 GB of copies holding
# 2.68 megapixels per image, and 5.7 KB of state and metadata per image.
STORED_BYTES_PER_PIXEL = 0.145
DB_BYTES_PER_ITEM = 5_700
# Used when no image could be sampled. The same run averaged 384 KB per image.
IMAGE_BYTES_PER_ITEM = 390_000

# Embedding width per model type, stored as float32.
VECTOR_DIMS = {"clip": 512, "siglip": 768, "siglip2": 768, "imagebind": 1024}
FLOAT32_BYTES = 4

# Images per second at batch 32. mps and cpu were measured on an Apple M4;
# cuda was not measured and is a conservative figure. SigLIP 2 and ImageBind were
# not measured and borrow the slower SigLIP figures.
ENCODE_RATE = {
    "clip": {"cuda": 300.0, "mps": 168.0, "cpu": 87.0},
    "siglip": {"cuda": 100.0, "mps": 54.0, "cpu": 22.0},
}
ENCODE_RATE["siglip2"] = ENCODE_RATE["siglip"]
ENCODE_RATE["imagebind"] = ENCODE_RATE["siglip"]

# JPEG decoding with PIL draft mode, measured on an Apple M4, spread over the
# ingest pipeline's 4 decode workers.
DECODE_SECONDS_PER_MEGAPIXEL = 0.005
DECODE_WORKERS = 4

# Model weights and Python runtime held in memory while serving.
SERVING_BASE_RAM_BYTES = 400 * 1024 * 1024

# Typical S3 throughput to an EC2 instance in the same region.
IN_REGION_BYTES_PER_SECOND = 100 * 1024 * 1024

# Files that may describe the images. Below these sizes a file cannot hold real
# content: an empty .xlsx workbook is already about 4 KB, and a text table
# needs at least a header row.
CATALOG_EXTS = {"csv", "tsv", "xlsx", "xls", "json", "xml", "parquet"}
MIN_CATALOG_BYTES = {"xlsx": 2048, "xls": 2048, "parquet": 256}
MIN_TEXT_CATALOG_BYTES = 64
# Rows checked for an image address browsers can load: a manifest uses one kind throughout.
WEB_IMAGE_SAMPLE = 50
MAX_CATALOG_CANDIDATES = 10

TINY_IMAGE_BYTES = 10_240
LARGE_SOURCE_BYTES = 20 * 1024**3
PROGRESS_EVERY_FILES = 20_000
TYPE_SAMPLE_ROWS = 500
DIR_SAMPLE_NAMES = 400
MAX_FACET_VALUES = 200
ROLE_SAMPLE_VALUES = 5000
MAX_LISTED_GROUPS = 20
MIN_EXIF_COVERAGE = 0.05
MIN_DEPTH_COVERAGE = 0.5
COLLAPSE_MEAN_ITEMS = 1.3
COLLAPSE_MAX_ITEMS = 10

GROUP_COLUMN = "group"
CAMERA_PREFIXES = re.compile(r"^(DSC|IMG|DSCN|PICT|CIMG)", re.I)
UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
LONG_HEX = re.compile(r"^[0-9a-f]{16,}$", re.I)
YEAR = re.compile(r"^(1[6-9]\d{2}|20[0-4]\d)s?$")
YEAR_INSIDE = re.compile(r"(1[6-9]\d{2}|20[0-4]\d)")
DIGITS = re.compile(r"^\d+$")
LETTERS_THEN_DIGITS = re.compile(r"^([A-Za-z]+)\d+$")
IDENTIFIER = re.compile(r"^[a-z]+[_-]?\d{4,}$", re.I)
PLACEHOLDER_FOLDER = re.compile(r"^(new folder|untitled|copy of|temp|tmp)", re.I)
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]|$)")

DIR_ROLE_LABEL = {
    "year": "years",
    "year_mostly": "mostly years",
    "category": "categories",
    "identifier": "identifiers",
    "uuid": "UUIDs",
    "noise": "placeholder folder names",
    "constant": "a fixed prefix",
    "unclear": "no clear pattern",
}


def token_class(token: str) -> str:
    if UUID.match(token):
        return "<uuid>"
    if LONG_HEX.match(token):
        return f"<hex{len(token)}>"
    if YEAR.match(token):
        return "<year>"
    if DIGITS.match(token):
        return f"<n{len(token)}>"
    letters = LETTERS_THEN_DIGITS.match(token)
    if letters:
        return f"{letters.group(1)}<n>"
    return token


def name_template(name: str) -> str:
    stem, _, ext = name.rpartition(".")
    stem = stem or name
    tokens = re.split(r"([_\-. ])", stem)
    body = "".join(t if t in "_-. " else token_class(t) for t in tokens)
    return body + ("." + ext.lower() if ext else "")


def infer_type(values: list) -> str:
    sample = values[:TYPE_SAMPLE_ROWS]
    if not sample:
        return "string"
    if all(isinstance(v, bool) for v in sample):
        return "bool"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample):
        return "number"
    if all(isinstance(v, (dt.date, dt.datetime)) for v in sample):
        return "date"
    if all(ISO_DATE.match(str(v)) for v in sample):
        return "date"
    return "string"


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction):
        return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]

    return {
        "min": ordered[0],
        "p50": at(0.5),
        "mean": int(statistics.fmean(ordered)),
        "p90": at(0.9),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def share(names: list[str], pattern: re.Pattern) -> float:
    return sum(bool(pattern.match(n)) for n in names) / len(names)


def guess_dir_role(names: list[str]) -> tuple[bool, str]:
    if len(set(names)) < 2:
        return False, "constant"
    sample = names[:DIR_SAMPLE_NAMES]
    year_share = share(sample, YEAR)
    if year_share > 0.6:
        return True, "year" if year_share > 0.95 else "year_mostly"
    if share(sample, UUID) > 0.8:
        return False, "uuid"
    if share(sample, PLACEHOLDER_FOLDER) > 0.5:
        return False, "noise"
    if share(sample, IDENTIFIER) > 0.8:
        return False, "identifier"
    wordy = sum(1 for n in sample if len(n) > 2 and not n.isdigit())
    if wordy / len(sample) > 0.7:
        return True, "category"
    return False, "unclear"


def scan(
    source: Source, sample: int, max_files: int | None, progress: bool
) -> tuple[ScanStats, list[FileRef], dict]:
    stats = ScanStats()
    images: list[FileRef] = []
    junk = Counter()
    other_exts = Counter()
    unreadable: list[str] = []
    catalogs: list[FileRef] = []

    for ref in source.iter_files():
        stats.total_files += 1
        if ref.extra.get("unreadable"):
            stats.unreadable_files += 1
            unreadable.append(ref.key)
        elif ref.is_junk:
            stats.non_image_files += 1
            junk[ref.basename.lower()] += 1
        elif ref.is_document:
            stats.document_files += 1
            stats.document_bytes += ref.size
        elif ref.ext in MODEL_EXTS:
            stats.model_files += 1
        elif ref.is_image:
            stats.image_files += 1
            stats.total_bytes += ref.size
            stats.formats[ref.ext] = stats.formats.get(ref.ext, 0) + 1
            stats.format_bytes[ref.ext] = stats.format_bytes.get(ref.ext, 0) + ref.size
            images.append(ref)
        else:
            stats.non_image_files += 1
            other_exts[ref.ext or "no extension"] += 1
            if ref.ext in CATALOG_EXTS:
                catalogs.append(ref)
        if progress and stats.total_files % PROGRESS_EVERY_FILES == 0:
            print(f"    scanned {stats.total_files:,} files...", flush=True)
        if max_files and stats.total_files >= max_files:
            break

    sizes = [ref.size for ref in images]
    stats.size_percentiles = percentiles(sizes)
    stats.catalog_candidates = [
        catalog_candidate(ref) for ref in catalogs[:MAX_CATALOG_CANDIDATES]
    ]
    for name, count in junk.most_common(5):
        stats.issues.append(
            Issue(kind="junk_file", count=count, note=f"system file {name}")
        )
    if unreadable:
        stats.issues.append(
            Issue(
                kind="unreadable",
                count=len(unreadable),
                note="could not be read (permissions or damage)",
                examples=unreadable[:5],
            )
        )
    for ext, count in other_exts.most_common(6):
        stats.issues.append(
            Issue(kind="non_image", count=count, note=f".{ext} files are not images")
        )
    tiny = sum(1 for size in sizes if size < TINY_IMAGE_BYTES)
    if tiny:
        stats.issues.append(
            Issue(
                kind="tiny_image",
                count=tiny,
                note="under 10 KB, possibly broken or placeholder images",
            )
        )

    deep = sample_images(source, images, sample)
    stats.sampled = deep["sampled"]
    stats.dimensions = deep["dimensions"]
    if images and deep["sampled"] == 0:
        note = (
            source.fetch_note or "The image files could not be read from this source."
        )
        stats.issues.append(Issue(kind="unfetchable", count=0, note=note))
    elif deep["broken"]:
        stats.issues.append(
            Issue(
                kind="corrupt",
                count=deep["broken"],
                note=f"could not be decoded (out of {deep['sampled']} sampled)",
                examples=deep["broken_examples"],
            )
        )
    deep["bandwidth_bps"] = source.probe_bandwidth(images) if deep["sampled"] else 0.0
    return stats, images, deep


def catalog_candidate(ref: FileRef) -> CatalogCandidate:
    minimum = MIN_CATALOG_BYTES.get(ref.ext, MIN_TEXT_CATALOG_BYTES)
    if ref.size == 0:
        note = "The file is empty."
    elif ref.size < minimum:
        note = f"Only {ref.size} bytes, too small to hold any real content."
    else:
        note = "May describe the images. It is not used yet."
    return CatalogCandidate(
        key=ref.key, size=ref.size, looks_empty=ref.size < minimum, note=note
    )


def stored_pixels(width: int, height: int) -> float:
    longest = max(width, height, 1)
    return sum(
        width * height * min(1.0, edge / longest) ** 2
        for edge in (THUMBNAIL_EDGE, PROCESSED_EDGE)
    )


def sample_images(source: Source, images: list[FileRef], count: int) -> dict:
    result = {
        "sampled": 0,
        "dimensions": {},
        "broken": 0,
        "broken_examples": [],
        "exif_dates": [],
        "exif_gps": 0,
    }
    if not images:
        return result
    step = max(1, len(images) // count)
    widths, heights, modes = [], [], Counter()
    for ref in images[::step][:count]:
        head = source.read_bytes(ref, HEAD_BYTES)
        if head is None:
            continue
        result["sampled"] += 1
        try:
            image = Image.open(io.BytesIO(head))
        except Exception:
            result["broken"] += 1
            result["broken_examples"] = (result["broken_examples"] + [ref.key])[:5]
            continue
        widths.append(image.size[0])
        heights.append(image.size[1])
        modes[image.mode] += 1
        try:
            exif = image.getexif()
        except Exception:
            continue
        taken = exif.get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
        if taken:
            result["exif_dates"].append(str(taken))
        try:
            gps = exif.get_ifd(EXIF_GPS)
        except Exception:
            gps = {}
        if GPS_LATITUDE in gps and GPS_LONGITUDE in gps:
            result["exif_gps"] += 1
    if widths:
        width, height = percentiles(widths), percentiles(heights)
        result["dimensions"] = {
            "width": width,
            "height": height,
            "megapixels_p50": round(width["p50"] * height["p50"] / 1e6, 1),
            "stored_pixels_mean": round(
                sum(map(stored_pixels, widths, heights)) / len(widths)
            ),
            "modes": dict(modes),
        }
    return result


def describe_template(template: str, example: str) -> str:
    stem = template.rsplit(".", 1)[0]
    literal = re.sub(r"<[^>]*>", "", stem).strip("_-. ")
    if CAMERA_PREFIXES.match(literal):
        return "camera default name, carries no information"
    if "<hex" in stem or "<uuid>" in stem:
        return "hash or UUID name, carries no readable information"
    if "<year>" in stem or YEAR_INSIDE.search(example):
        return "may contain a year that can be extracted"
    if "<" in stem and len(literal) >= 3:
        return f"fixed prefix '{literal[:16]}' plus a serial number, possibly a catalog number"
    if not literal:
        return "carries no usable information"
    return ""


def name_patterns(images: list[FileRef], top: int = 6) -> list[NamePattern]:
    counts = Counter()
    examples: dict[str, str] = {}
    for ref in images:
        template = name_template(ref.basename)
        counts[template] += 1
        examples.setdefault(template, ref.basename)
    total = max(1, len(images))
    return [
        NamePattern(
            template=template,
            count=count,
            share=round(count / total, 4),
            example=examples[template],
            meaning=describe_template(template, examples[template]),
        )
        for template, count in counts.most_common(top)
    ]


def directory_signals(
    images: list[FileRef], max_depth: int = 3
) -> list[DirectorySignal]:
    by_depth: dict[int, Counter] = defaultdict(Counter)
    for ref in images:
        for depth, part in enumerate(ref.parts[:max_depth]):
            by_depth[depth][part] += 1
    total = max(1, len(images))
    signals = []
    for depth in sorted(by_depth):
        if sum(by_depth[depth].values()) / total < MIN_DEPTH_COVERAGE:
            continue
        names = list(by_depth[depth])
        meaningful, guess = guess_dir_role(names)
        signals.append(
            DirectorySignal(
                depth=depth,
                distinct=len(names),
                looks_meaningful=meaningful,
                guess=guess,
                examples=[name for name, _ in by_depth[depth].most_common(4)],
            )
        )
    return signals


def catalog_field(column: str, values: list, object_column: str | None) -> FieldProfile:
    present = [v for v in values if v is not None and v != ""]
    distinct = len(set(map(str, present)))
    read_as, filtered_as = column_role(column, present[:ROLE_SAMPLE_VALUES])
    profile = FieldProfile(
        name=column,
        type=infer_type(present),
        provenance=Provenance.CATALOG,
        coverage=round(len(present) / max(1, len(values)), 4),
        cardinality=distinct,
        sample_values=[str(v)[:60] for v in present[:3]],
        searchable=read_as in SEARCHED_AS,
        facetable=filtered_as is not None,
        read_as=read_as,
        filtered_as=filtered_as,
    )
    if read_as == "object":
        profile.note = (
            "The object each file shows. Files with the same value are shown together."
            if column == object_column
            else f"Left out, since {object_column} names each file's object."
        )
    elif read_as == "group":
        profile.note = (
            f"Splits the collection into {distinct} groups, each a choice of the group "
            "filter."
        )
    elif read_as is None:
        profile.note = "Holds no catalog text, only numbers or addresses."
    elif kind_of(column) is None:
        profile.note = "Read as notes, since the site does not know this column name."
    elif filtered_as and distinct == 1:
        profile.note = f"Only one value ({str(present[0])[:40]}) to filter by."
    return profile


def catalog_structure(
    columns: list[str], rows: list[tuple], object_column: str
) -> Structure:
    index = columns.index(object_column)
    group_sizes = list(Counter(str(row[index]) for row in rows if row[index]).values())
    if not group_sizes:
        return Structure(collapse_reason="No object level. Each image is one result.")
    mean, largest = statistics.fmean(group_sizes), max(group_sizes)
    collapse = mean > COLLAPSE_MEAN_ITEMS or largest > COLLAPSE_MAX_ITEMS
    reason = (
        (
            f"{mean:.2f} images per object on average and up to {largest}. "
            f"Without grouping, one object can flood the results."
        )
        if collapse
        else ""
    )
    return Structure(
        group_unit="record",
        group_field=object_column,
        items_per_group={
            "mean": round(mean, 2),
            "median": float(statistics.median(group_sizes)),
            "max": float(largest),
        },
        suggested_collapse=collapse,
        collapse_reason=reason,
    )


def without_record(
    columns: list[str], rows: list[tuple], object_column: str | None
) -> int:
    """How many files have neither an object id nor any catalog text, as the writer reads them"""
    index = columns.index(object_column) if object_column else None
    count = 0
    for row in rows:
        if index is not None and row[index] not in (None, ""):
            continue
        read = read_catalog(dict(zip(columns, row)))
        if not (read.title or read.fields or read.source_url):
            count += 1
    return count


def files_per_group(columns: list[str], rows: list[tuple]) -> tuple[str | None, dict]:
    """The column that splits the collection into groups, and how many files each holds"""
    column = next((c for c in columns if c.strip().lower() == GROUP_COLUMN), None)
    if column is None:
        return None, {}
    index = columns.index(column)
    counts = Counter(
        str(row[index]).strip() for row in rows if row[index] not in (None, "")
    )
    return column, dict(counts.most_common(MAX_LISTED_GROUPS))


def file_derived_fields(
    image_count: int, deep: dict, signals: list[DirectorySignal]
) -> list[FieldProfile]:
    fields = []
    sampled = max(1, deep["sampled"])
    dates = deep["exif_dates"]

    if len(dates) / sampled > MIN_EXIF_COVERAGE:
        years = sorted({d[:4] for d in dates if d[:4].isdigit()})
        span = f" spanning {years[0]} to {years[-1]}" if years else ""
        fields.append(
            FieldProfile(
                name="capture_date",
                type="date",
                provenance=Provenance.FILE_DERIVED,
                coverage=round(len(dates) / sampled, 3),
                sample_values=dates[:3],
                note=f"When each picture was taken, read from EXIF in "
                f"{deep['sampled']} sampled images{span}. For scanned or photographed "
                f"objects this is the digitization date, not the date of the object. "
                f"The site does not use it.",
            )
        )
    else:
        fields.append(
            FieldProfile.unavailable(
                "capture_date",
                type="date",
                note=f"Only {len(dates)} of {deep['sampled']} sampled images carry an EXIF date.",
                remedy={
                    "action": "infer dates from file or folder names",
                    "confidence": "low",
                },
            )
        )

    if deep["exif_gps"] / sampled > MIN_EXIF_COVERAGE:
        fields.append(
            FieldProfile(
                name="location",
                type="geo",
                provenance=Provenance.FILE_DERIVED,
                coverage=round(deep["exif_gps"] / sampled, 3),
                note="Where each picture was taken, read from EXIF GPS. For scanned "
                "or photographed objects this is the studio, not the object's origin. "
                "The site does not use it.",
            )
        )
    else:
        fields.append(
            FieldProfile.unavailable(
                "location",
                type="geo",
                note="No location field, and the images carry no GPS data.",
            )
        )

    for signal in signals:
        if signal.looks_meaningful:
            role = DIR_ROLE_LABEL.get(signal.guess, signal.guess)
            fields.append(
                FieldProfile(
                    name=f"folder_level_{signal.depth + 1}",
                    provenance=Provenance.FILE_DERIVED,
                    coverage=1.0,
                    cardinality=signal.distinct,
                    sample_values=signal.examples,
                    note=f"Folder level {signal.depth + 1} looks like {role}. The site "
                    f"does not use folder names; a manifest column could carry them.",
                )
            )
    return fields


def infer_fields(
    source: Source, images: list[FileRef], deep: dict, signals: list[DirectorySignal]
) -> tuple[list[FieldProfile], Structure]:
    fields: list[FieldProfile] = []
    structure = Structure(collapse_reason="No object level. Each image is one result.")

    catalog = source.catalog_rows()
    if catalog:
        columns, rows = catalog
        lowered = {c.strip().lower(): c for c in columns}
        object_column = next((lowered[k] for k in OBJECT_ID_KEYS if k in lowered), None)
        for index, column in enumerate(columns):
            fields.append(
                catalog_field(column, [row[index] for row in rows], object_column)
            )
        if object_column:
            structure = catalog_structure(columns, rows, object_column)
        structure.orphan_items = without_record(columns, rows, object_column)
        structure.group_column, structure.groups = files_per_group(columns, rows)
        warning = group_id_warning(columns)
        for f in fields:
            if warning and f.name.strip().lower() == "group_id":
                f.note = warning

    fields.extend(file_derived_fields(len(images), deep, signals))
    return fields, structure


def build_capabilities(
    fields: list[FieldProfile], structure: Structure
) -> Capabilities:
    capability_names = {"capture_date": "date_range", "location": "geo_map"}
    roles = {f.read_as for f in fields if f.provenance == Provenance.CATALOG}
    searched = roles & set(SEARCHED_AS) - {"identifiers"}
    capabilities = Capabilities(
        available=["text_search", "image_search", "visual_similarity"],
        facets=sorted({f.filtered_as for f in fields if f.filtered_as}),
    )
    for capability, present in (
        ("catalog_search", bool(searched)),
        ("titles", "title" in roles),
        ("links_back", "link" in roles),
        ("objects", "object" in roles),
        ("group_filter", len(structure.groups) > 1),
    ):
        if present:
            capabilities.available.append(capability)
    if not searched:
        capabilities.unavailable.append(
            Capability(
                capability="catalog_search",
                reason="There is no catalog text, so search works by appearance alone.",
                remedy="Give a parquet manifest catalog columns such as title, "
                "creator, date or description.",
            )
        )
    for f in fields:
        if not f.available:
            capabilities.unavailable.append(
                Capability(
                    capability=capability_names.get(f.name, f.name),
                    reason=f.note,
                    remedy=(f.remedy or {}).get("action"),
                    cost=f.remedy,
                )
            )
    return capabilities


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def estimate_sizing(stats: ScanStats, bandwidth_bps: float, model_type: str) -> Sizing:
    count = stats.image_files
    vectors = count * VECTOR_DIMS.get(model_type, VECTOR_DIMS["siglip"]) * FLOAT32_BYTES
    device = detect_device()
    megapixels = stats.dimensions.get("megapixels_p50", 0)
    decode = count * megapixels * DECODE_SECONDS_PER_MEGAPIXEL / DECODE_WORKERS
    rates = ENCODE_RATE.get(model_type, ENCODE_RATE["siglip"])
    compute = count / rates[device] + decode
    stored = stats.dimensions.get("stored_pixels_mean")
    image_bytes = stored * STORED_BYTES_PER_PIXEL if stored else IMAGE_BYTES_PER_ITEM
    seconds = {
        "compute": round(compute, 1),
        "in_region_aws": round(
            max(stats.total_bytes / IN_REGION_BYTES_PER_SECOND, compute), 1
        ),
    }
    if bandwidth_bps > 0:
        transfer = stats.total_bytes / bandwidth_bps
        seconds["transfer"] = round(transfer, 1)
        seconds["throughput_mbs"] = round(bandwidth_bps / 1024**2, 1)
        seconds["estimated"] = round(max(transfer, compute), 1)
    return Sizing(
        n_items=count,
        vectors_bytes=vectors,
        index_bytes=vectors,
        device=device,
        thumbnails_bytes=round(count * image_bytes),
        db_bytes=count * DB_BYTES_PER_ITEM,
        serving_ram_bytes=vectors + SERVING_BASE_RAM_BYTES,
        source_bytes=stats.total_bytes,
        embed_seconds=seconds,
    )


def image_decision(fields: list[FieldProfile], web_images: bool) -> Decision | None:
    """Ask where the site shows images from, when their source can show them too"""
    columns = {field.name.lower() for field in fields}
    has_service = bool(columns & set(IIIF_KEYS))
    if not (has_service or web_images or columns & set(THUMBNAIL_KEYS)):
        return None
    return Decision(
        id="images",
        question="The images can also be shown from where they are kept. "
        "What should the site keep of them?",
        options=[
            {"id": "copy", "label": "Its own copies of every image"},
            {
                "id": "thumbnails",
                "label": "Small previews only, with larger views from the source",
            },
            {"id": "link", "label": "Nothing, every image shown from the source"},
        ],
        recommended="thumbnails" if has_service else "copy",
        why=(
            "An IIIF image server sends any size it is asked for. Small previews keep "
            "the search page quick and take little space, and the full view and zoom "
            "come from the server."
            if has_service
            else "Copies keep the site working if the source is slow, moves its files "
            "or does not let other sites show them."
        ),
    )


def build_decisions(
    stats: ScanStats,
    structure: Structure,
    fields: list[FieldProfile] | None = None,
    web_images: bool = False,
) -> list[Decision]:
    decisions = []
    if structure.orphan_items:
        decisions.append(
            Decision(
                id="include_orphans",
                question=f"{structure.orphan_items:,} images have no catalog record. Index them too?",
                options=[{"id": "yes", "label": "Yes"}, {"id": "no", "label": "No"}],
                recommended="yes",
                why="Visual search does not need catalog data. These images stay findable, "
                "they just have no title to show.",
            )
        )
    excluded = sum(
        i.count for i in stats.issues if i.kind in ("junk_file", "non_image")
    )
    if excluded:
        decisions.append(
            Decision(
                id="exclude_junk",
                question=f"Found {excluded:,} files that are not images. Leave them out?",
                options=[
                    {"id": "yes", "label": "Leave them out"},
                    {"id": "no", "label": "Keep them"},
                ],
                recommended="yes",
                why="They cannot be searched visually and only slow down indexing.",
            )
        )
    decisions.append(
        Decision(
            id="data_residency",
            question="May this collection be stored in the cloud?",
            options=[
                {"id": "cloud", "label": "Yes"},
                {"id": "local", "label": "No, it must stay on our own machines"},
                {"id": "unknown", "label": "Not sure, I need to ask"},
            ],
            recommended="local",
            why="Many institutions restrict where collection data may live. "
            "Starting locally is safe and can be changed later.",
        )
    )
    images = image_decision(fields or [], web_images)
    if images:
        decisions.append(images)
    return decisions


def media_needed(stats: ScanStats, fields: list[FieldProfile]) -> list[str]:
    columns = {field.name.lower() for field in fields}
    needed = {
        "3d": bool(stats.model_files or columns & set(MODEL_URL_KEYS)),
        "pdf": bool(stats.document_files),
        "zoom": bool(columns & set(IIIF_KEYS)),
    }
    return [medium for medium, is_needed in needed.items() if is_needed]


def profile_collection(
    source: Source,
    collection_id: str,
    sample: int = 300,
    max_files: int | None = None,
    progress: bool = True,
    model_type: str = "siglip",
) -> CollectionProfile:
    stats, images, deep = scan(source, sample, max_files, progress)
    signals = directory_signals(images)
    fields, structure = infer_fields(source, images, deep, signals)
    return CollectionProfile(
        collection_id=collection_id,
        source=source.describe(),
        scan=stats,
        name_patterns=name_patterns(images),
        directory_signals=signals,
        fields=fields,
        structure=structure,
        capabilities=build_capabilities(fields, structure),
        sizing=estimate_sizing(stats, deep["bandwidth_bps"], model_type),
        decisions=build_decisions(
            stats,
            structure,
            fields,
            any(source.web_image_address(ref) for ref in images[:WEB_IMAGE_SAMPLE]),
        ),
        media=media_needed(stats, fields),
    )
