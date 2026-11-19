from __future__ import annotations

from src.collection.catalog import labeled

from .models import CollectionProfile, Provenance
from .profiler import DIR_ROLE_LABEL, LARGE_SOURCE_BYTES

WIDTH = 74
SOURCE_LABEL = {
    Provenance.CATALOG: "catalog",
    Provenance.FILE_DERIVED: "from files",
    Provenance.INFERRED: "machine guess",
}


def human_bytes(value: float) -> str:
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024


def human_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


def size_line(scan) -> str:
    images = f"  {scan.image_files:,} images"
    sized = scan.image_files - scan.images_without_size
    if not scan.images_without_size:
        return f"{images}, {human_bytes(scan.total_bytes)} in total"
    if not sized:
        return f"{images}; the manifest does not give their sizes"
    return (
        f"{images}; {human_bytes(scan.total_bytes)} in total for the {sized:,} "
        "whose sizes the manifest gives"
    )


def overview_lines(p: CollectionProfile) -> list[str]:
    scan = p.scan
    lines = ["", size_line(scan)]
    if scan.size_percentiles:
        sizes = scan.size_percentiles
        lines.append(
            f"  File size: median {human_bytes(sizes['p50'])}, "
            f"largest {human_bytes(sizes['max'])}"
        )
    if scan.dimensions:
        dims = scan.dimensions
        lines.append(
            f"  Typical dimensions: {dims['width']['p50']} x {dims['height']['p50']} px "
            f"({dims['megapixels_p50']} megapixels)"
        )
    formats = sorted(scan.formats.items(), key=lambda item: -item[1])[:5]
    lines.append(
        "  Formats: " + ", ".join(f"{ext} {count:,}" for ext, count in formats)
    )
    if scan.document_files:
        noun = "PDF" if scan.document_files == 1 else "PDFs"
        size = f", {human_bytes(scan.document_bytes)}" if scan.document_bytes else ""
        lines.append(
            f"  Also {scan.document_files:,} {noun}{size}, indexed page by page"
        )
    if scan.model_files:
        noun = "3D model" if scan.model_files == 1 else "3D models"
        lines.append(
            f"  Also {scan.model_files:,} {noun}, searched through views drawn from them"
        )
    if p.media:
        lines.append(
            f"  Build the site with: npm run setup -- --with={','.join(p.media)}"
        )
    return lines


def problem_lines(p: CollectionProfile) -> list[str]:
    scan = p.scan
    lines = []
    blocked = [i for i in scan.issues if i.kind == "unfetchable"]
    if blocked:
        lines += [
            "",
            "  WARNING: the image files could not be read, so estimates are incomplete.",
        ]
        lines += [f"      {issue.note}" for issue in blocked]
    if scan.catalog_candidates:
        lines += ["", "  Files that may be a catalog of these images:"]
        for candidate in scan.catalog_candidates:
            flag = "WARNING: " if candidate.looks_empty else ""
            lines.append(
                f"      {candidate.key} ({human_bytes(candidate.size)}). "
                f"{flag}{candidate.note}"
            )
    listed = [i for i in scan.issues if i.kind != "unfetchable"]
    if listed:
        lines += ["", "  Things to be aware of:"]
        for issue in listed[:8]:
            example = f" (e.g. {issue.examples[0]})" if issue.examples else ""
            lines.append(f"      {issue.count:>7,}  {issue.note}{example}")
    return lines


def orphan_lines(p: CollectionProfile) -> list[str]:
    orphans = p.structure.orphan_items
    if not orphans or orphans >= p.scan.image_files:
        return []
    percent = 100 * orphans / max(1, p.scan.image_files)
    return [
        "",
        f"  WARNING: {orphans:,} images ({percent:.1f}%) have no catalog record",
    ]


def structure_lines(p: CollectionProfile) -> list[str]:
    structure = p.structure
    if not structure.group_unit:
        return orphan_lines(p)
    per_group = structure.items_per_group
    lines = [
        "",
        f"  Objects: {per_group['mean']:.2f} images per object on average, "
        f"up to {int(per_group['max'])}",
    ]
    lines += orphan_lines(p)[1:]
    if structure.suggested_collapse:
        lines.append(
            f"  Suggestion: group results by object. {structure.collapse_reason}"
        )
    return lines


def group_lines(p: CollectionProfile) -> list[str]:
    groups = p.structure.groups
    if len(groups) < 2:
        return []
    listed = ", ".join(f"{name} {count:,}" for name, count in groups.items())
    return ["", f"  Groups ({p.structure.group_column}): {listed}"]


def naming_lines(p: CollectionProfile) -> list[str]:
    lines = []
    if p.name_patterns:
        lines += ["", "  File naming:"]
        for pattern in p.name_patterns[:4]:
            meaning = f"  <- {pattern.meaning}" if pattern.meaning else ""
            lines.append(
                f"      {pattern.share:>5.1%}  {pattern.template:<26} "
                f"e.g. {pattern.example[:34]}{meaning}"
            )
    if p.directory_signals:
        lines += ["", "  Folders:"]
        fixed = [s for s in p.directory_signals if s.guess == "constant"]
        if fixed:
            prefix = "/".join(s.examples[0] for s in fixed)
            lines.append(
                f"      The first {len(fixed)} level(s) are a fixed prefix "
                f"({prefix}) and carry no meaning"
            )
        for signal in p.directory_signals:
            if signal.guess == "constant":
                continue
            verdict = (
                "meaningful, but not used by the site"
                if signal.looks_meaningful
                else "no clear meaning"
            )
            role = DIR_ROLE_LABEL.get(signal.guess, signal.guess)
            lines.append(
                f"      Level {signal.depth + 1}: {signal.distinct:,} values, "
                f"{role}, {verdict}"
            )
            lines.append(f"               e.g. {', '.join(signal.examples[:3])}")
    return lines


KIND_WORDS = {
    "names": "names",
    "dates": "dates",
    "places": "places",
    "cultures": "cultures",
    "subjects": "subjects",
    "object_types": "kinds of object",
    "physical": "materials",
    "identifiers": "catalog numbers",
    "notes": "notes",
    "rights": "rights and credit",
}
ROLE_WORDS = {
    "title": "the title, searched",
    "link": "the link back to the record",
    "object": "the object each file shows",
    "group": "the group filter",
    "file": "the file, not catalog text",
}
CAPABILITY_WORDS = {
    "text_search": "search by description",
    "image_search": "search by an image",
    "visual_similarity": "find similar images",
    "catalog_search": "search the catalog text as well",
    "titles": "show titles",
    "links_back": "link to each record at the institution",
    "objects": "show an object's images together",
    "group_filter": "filter by group",
}


def use_of(f) -> str:
    """What the site does with a column, in a few words"""
    if f.provenance != Provenance.CATALOG:
        return "not used by the site"
    if f.read_as in ROLE_WORDS:
        return ROLE_WORDS[f.read_as]
    if f.read_as not in KIND_WORDS:
        return "not shown or searched"
    form = labeled(f.name)
    if form and not form[1]:
        place = ""
    elif form:
        place = f"on the page as {form[1]}, "
    else:
        place = "on the page, "
    if not f.searchable:
        where = place.removesuffix(", ") or "not on the page"
        return f"{KIND_WORDS[f.read_as]}: {where}, not searched"
    use = f"{KIND_WORDS[f.read_as]}: {place}searched"
    if f.filtered_as:
        use += f", filter by {f.filtered_as.replace('_', ' ')}"
    if form and not form[1]:
        use += ", not on the page"
    return use


def field_lines(p: CollectionProfile) -> list[str]:
    capabilities = p.capabilities
    able = [CAPABILITY_WORDS.get(c, c) for c in capabilities.available]
    lines = ["", "  The site will:"] + [f"      - {words}" for words in able]
    if capabilities.facets:
        filters = ", ".join(f.replace("_", " ") for f in capabilities.facets)
        lines.append(f"      - filter by {filters}")
    lines += ["", "  What the site does with each column:"]
    for f in p.fields:
        lines.append(
            f"      [{SOURCE_LABEL.get(f.provenance, '?')}] {f.name:<20} "
            f"{round(f.coverage * 10_000) // 100:>3}% filled   {use_of(f)}"
        )
        if f.note:
            lines.append(f"               {f.note}")
    if capabilities.unavailable:
        lines += ["", "  What is not possible:"]
        for c in capabilities.unavailable:
            lines.append(f"      x {CAPABILITY_WORDS[c.capability]}")
            lines.append(f"         Why: {c.reason}")
            lines.append(f"         Possible fix: {c.remedy}")
    return lines


def sizing_lines(p: CollectionProfile) -> list[str]:
    sizing = p.sizing
    seconds = sizing.embed_seconds
    disk = (
        sizing.vectors_bytes
        + sizing.thumbnails_bytes
        + sizing.documents_bytes
        + sizing.db_bytes
        + sizing.index_bytes
    )
    lines = [
        "",
        "  Size and time:",
        f"      Disk needed after indexing: about {human_bytes(disk)}",
        f"      Memory needed to serve searches: about {human_bytes(sizing.serving_ram_bytes)}",
    ]
    if "estimated" in seconds:
        lines.append(
            f"      Indexing will take about {human_time(seconds['estimated'])}"
        )
        lines.append(
            f"         Reading the files: {human_time(seconds['transfer'])} "
            f"(measured {seconds['throughput_mbs']} MB/s)"
        )
        lines.append(
            f"         Analysing the images: {human_time(seconds['compute'])} "
            f"(on {sizing.device})"
        )
        lines.append(
            "         These run side by side, so the slower one sets the total."
        )
    else:
        lines.append(
            f"      Analysing the images alone takes about "
            f"{human_time(seconds['compute'])} (on {sizing.device})"
        )
        lines += [
            "      WARNING: this leaves out downloading the images, which could not be",
            "      timed here and may take far longer. Index a few hundred with --limit",
            "      to time it.",
        ]
    if sizing.source_bytes > LARGE_SOURCE_BYTES:
        lines.append(
            f"      On a cloud machine next to the data: about "
            f"{human_time(seconds['in_region_aws'])}"
        )
    return lines


def decision_lines(p: CollectionProfile) -> list[str]:
    if not p.decisions:
        return []
    lines = ["", "  Decisions for you:"]
    for number, decision in enumerate(p.decisions, 1):
        lines.append(f"      {number}. {decision.question}")
        for option in decision.options:
            marker = ">" if option["id"] == decision.recommended else " "
            suffix = "  (recommended)" if option["id"] == decision.recommended else ""
            lines.append(f"           {marker} {option['label']}{suffix}")
        lines.append(f"           Why: {decision.why}")
    return lines


def plural(count: int, one: str, many: str) -> str:
    return f"{count:,} {one if count == 1 else many}"


def given_lines(p: CollectionProfile) -> list[str]:
    from src.ingest.pipeline import SPOT_CHECK_IMAGES

    lines = []
    vectors = p.vectors
    if vectors and vectors.given:
        sizes = " or ".join(f"{n:,}" for n in sorted(vectors.lengths))
        lines += [
            "",
            "  Vectors given in the manifest:",
            f"      {vectors.given:,} of {vectors.rows:,} rows give a vector of {sizes} "
            "numbers. Indexing uses them instead of computing vectors, after "
            f"re-embedding {min(SPOT_CHECK_IMAGES, vectors.given)} of them to check that they come from the "
            "model in config.json.",
        ]
        if len(vectors.lengths) > 1:
            lines.append(
                "      Must fix before indexing: every vector needs as many numbers "
                f"as the model makes, but these have {sizes}."
            )
        if vectors.unusable:
            lines.append(
                f"      {plural(vectors.unusable, 'vector is', 'vectors are')} zero or "
                "not numbers; those images will fail."
            )
        if vectors.given < vectors.rows:
            left = vectors.rows - vectors.given
            lines.append(
                f"      {plural(left, 'row gives', 'rows give')} no vector; indexing "
                "computes those."
            )
    related = p.related
    if related:
        lines += ["", f"  Related items ({related.path}):"]
        if related.rows and not related.refused:
            by_source = ", ".join(
                f"{count:,} from {source}"
                for source, count in sorted(related.sources.items())
            )
            lines.append(
                f"      {plural(related.rows, 'item', 'items')} for "
                f"{plural(related.objects, 'object', 'objects')}: {by_source}."
            )
        lines += [f"      Must fix before indexing: {line}" for line in related.refused]
        lines += [f"      {line}" for line in related.notes]
    return lines


def render(p: CollectionProfile) -> str:
    lines = [
        "=" * WIDTH,
        f"Collection report: {p.collection_id}",
        f"Source: {p.source.get('uri')}",
        "=" * WIDTH,
    ]
    for section in (
        overview_lines,
        problem_lines,
        given_lines,
        structure_lines,
        group_lines,
        naming_lines,
        field_lines,
        sizing_lines,
        decision_lines,
    ):
        lines += section(p)
    lines += ["", "=" * WIDTH]
    return "\n".join(lines)
