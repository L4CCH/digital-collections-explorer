import argparse
import json
import sys
from pathlib import Path

from .package import CONTRACT, Refused, Report, build, check, listed, plural, such_as

PACKAGE_HELP = (
    "the folder of a collection package: its datapackage.json, tables, and vectors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.collection",
        description="Check a collection package that a source plugin wrote, and "
        "build the collection the site reads from it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    checked = commands.add_parser(
        "check", help="say whether a package can be built, and what it holds"
    )
    checked.add_argument("package", help=PACKAGE_HELP)
    checked.add_argument(
        "--json", dest="json_path", help="also write the report as JSON"
    )
    built = commands.add_parser(
        "build", help="check a package, then write the collection it describes"
    )
    built.add_argument("package", help=PACKAGE_HELP)
    built.add_argument(
        "--data-dir",
        required=True,
        help="the collection folder; the collection is written to its embeddings",
    )
    built.add_argument(
        "--check-files",
        action="store_true",
        help="refuse when a file the package names on this machine is not in the "
        "collection folder",
    )
    built.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a collection that python -m src.ingest wrote, and remove what "
        "ingest kept to resume",
    )
    built.add_argument("--json", dest="json_path", help="also write the counts as JSON")
    return parser.parse_args()


def be(count: int) -> str:
    return "is" if count == 1 else "are"


def unmatched(report: Report) -> list[str]:
    counts, examples = report.counts, report.examples
    lines = []
    count = counts["records_without_files"]
    if count:
        lines.append(
            f"{plural(count, 'record')} without a file {be(count)} left out"
            f"{such_as(examples['records_without_files'])}."
        )
    count = counts["files_without_records"]
    if count:
        lines.append(
            f"{plural(count, 'file')} whose object has no row in the records table "
            f"{be(count)} shown without a catalog record"
            f"{such_as(examples['files_without_records'])}."
        )
    count = counts["vectors_without_files"]
    if count:
        lines.append(
            f"{plural(count, 'vector')} without a file {be(count)} left out"
            f"{such_as(examples['vectors_without_files'])}."
        )
    return lines


def described(report: Report, package: Path) -> list[str]:
    notes = [f"Note: {note}" for note in report.notes]
    if report.problems:
        return [
            f"{package} cannot be built:",
            *(f"  - {problem}" for problem in report.problems),
            *notes,
        ]
    counts, producer = report.counts, report.producer
    shape = counts["shape_objects"]
    holds = [
        f"{plural(counts['files'], 'file')} in {plural(counts['groups'], 'group')}",
        *([] if counts["records"] is None else [plural(counts["records"], "record")]),
        f"the vectors of {listed(counts['models'])}",
        *([] if shape is None else [f"a shape index of {plural(shape, 'object')}"]),
    ]
    return [
        f"{package} can be built: it follows collection package contract {CONTRACT} "
        f"and was written by {producer['name']} {producer['version']}.",
        f"  It holds {listed(holds)}.",
        *(f"  {line}" for line in unmatched(report)),
        *notes,
    ]


def report_json(report: Report, package: Path) -> dict:
    return {
        "package": str(package.resolve()),
        "contract": report.contract,
        "producer": report.producer,
        "provenance": report.provenance,
        "can_build": not report.problems,
        "problems": report.problems,
        "notes": report.notes,
        **report.counts,
        "examples": report.examples,
    }


def write_json(path: str | None, found: dict) -> None:
    if path:
        Path(path).write_text(json.dumps(found, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    package = Path(args.package)
    if args.command == "check":
        report = check(package)
        print("\n".join(described(report, package)))
        write_json(args.json_path, report_json(report, package))
        return 2 if report.problems else 0
    try:
        written, report = build(
            package, Path(args.data_dir), args.check_files, args.overwrite
        )
    except Refused as refused:
        print(f"{package} was not built:")
        print("\n".join(f"  - {problem}" for problem in refused.problems))
        return 2
    except ValueError as problem:
        print(f"ERROR: {problem}")
        return 1
    objects = written["records"] + written["uncataloged_items"]
    print(
        f"Wrote {written['items']:,} images, as {objects:,} objects in "
        f"{written['groups']:,} groups, to {Path(args.data_dir) / 'embeddings'}"
    )
    print(
        f"  {written['record_fields']:,} fields, {written['record_facets']:,} "
        f"filter values, catalog search over {written['searchable_records']:,} records, "
        f"{plural(written['related_sources'], 'related source')}"
    )
    for line in unmatched(report):
        print(f"  {line}")
    for note in report.notes:
        print(f"Note: {note}")
    write_json(
        args.json_path,
        {
            **written,
            "files_without_records": report.counts["files_without_records"],
            "vectors_without_files": report.counts["vectors_without_files"],
            "shape_objects": report.counts["shape_objects"],
            "examples": report.examples,
            "producer": report.producer,
            "provenance": report.provenance,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
