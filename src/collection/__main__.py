"""python -m src.collection build MANIFEST --vectors VECTORS --data-dir DIR"""

import argparse
import json
import sys
from pathlib import Path

from .manifest import build


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.collection",
        description="Write a collection in the format the site reads.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    built = commands.add_parser(
        "build",
        help="write a collection from a manifest and vectors a pipeline made",
    )
    built.add_argument(
        "manifest", help="the folder of files.parquet and any other manifest tables"
    )
    built.add_argument(
        "--vectors",
        required=True,
        help="the folder of models.json and <model>/<group>.npy with .ids.npy",
    )
    built.add_argument(
        "--data-dir",
        required=True,
        help="the collection folder; the collection is written to its embeddings",
    )
    built.add_argument(
        "--check-files",
        action="store_true",
        help="refuse when a file the manifest names on this machine is not there",
    )
    built.add_argument("--json", dest="json_path", help="also write the counts as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = build(
            Path(args.manifest),
            Path(args.vectors),
            Path(args.data_dir),
            args.check_files,
        )
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
        f"{written['related_sources']:,} related sources"
    )
    if written["records_without_files"]:
        print(f"  {written['records_without_files']:,} records without a file left out")
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(written, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
