from __future__ import annotations

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


def overview_lines(p: CollectionProfile) -> list[str]:
    scan = p.scan
    lines = [
        "",
        f"  {scan.image_files:,} images, {human_bytes(scan.total_bytes)} in total",
    ]
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
    use = f"{KIND_WORDS[f.read_as]}: on the page, searched"
    if f.filtered_as:
        use += f", filter by {f.filtered_as.replace('_', ' ')}"
    return use


def field_lines(p: CollectionProfile) -> list[str]:
    capabilities = p.capabilities
    able = [CAPABILITY_WORDS.get(c, c) for c in capabilities.available]
    lines = ["", "  The site will:"] + [f"      - {words}" for words in able]
    if capabilities.facets:
        filters = ", ".join(f.replace("_", " ") for f in capabilities.facets)
        lines.append(f"      - filter by {filters}")
    lines += ["", "  What the site does with each column:"]
    for f in (f for f in p.fields if f.available):
        lines.append(
            f"      [{SOURCE_LABEL.get(f.provenance, '?')}] {f.name:<20} "
            f"{f.coverage:>4.0%} filled   {use_of(f)}"
        )
        if f.note:
            lines.append(f"               {f.note}")
    missing = [f for f in p.fields if not f.available]
    unable = [c for c in capabilities.unavailable if c.capability == "catalog_search"]
    if missing or unable:
        lines += ["", "  What is not possible:"]
        for c in unable:
            lines.append(f"      x {CAPABILITY_WORDS[c.capability]}")
            lines.append(f"         Why: {c.reason}")
            lines.append(f"         Possible fix: {c.remedy}")
        for f in missing:
            lines.append(f"      x {f.name}")
            lines.append(f"         Why: {f.note}")
            if f.remedy:
                details = ", ".join(f"{k}: {v}" for k, v in f.remedy.items())
                lines.append(f"         Possible fix: {details}")
    return lines


def sizing_lines(p: CollectionProfile) -> list[str]:
    sizing = p.sizing
    seconds = sizing.embed_seconds
    disk = (
        sizing.vectors_bytes
        + sizing.thumbnails_bytes
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
        lines.append(
            "      WARNING: download speed could not be measured. "
            "The real time may be far longer."
        )
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
