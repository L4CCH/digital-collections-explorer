from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.backend.core.config import apply_data_dir, settings
from src.backend.services.shape_index import write_shape_index
from src.environment import INGEST_STATE_FILE, detect_device

from .evaluate import compare, read_pairs, render_comparison
from .manifest import write_manifest
from .models import IMAGE_COLUMN, SIZE_COLUMN, Model, list_source
from .renderer import ModelRefused, Renderer, RendererUnavailable
from .shape import DEFAULT_SHAPE_MODEL, ShapeEncoder, cache_name, model_vector
from .views import COLOR, MODES, RELIEF, View, views_around

MODELS_DIR = "models3d"
VIEWS_DIR = "views"
MANIFEST_FILE = "views.parquet"
RECORD_FILE = "models.json"
BACKGROUND = "#e6e4e0"
# A view that shows under a quarter of what the fullest view shows is mostly an edge or an open back.
MIN_SHARE_OF_FULLEST_VIEW = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.render3d",
        description="Make 3D models searchable. Renders views of every .glb model for "
        "the search index, and describes each model's form so that similar shapes "
        "can be found. Safe to stop and rerun.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser(
        "render", help="render views of every model in a folder or a manifest"
    )
    render.add_argument(
        "uri",
        help="a folder of .glb files, or a .parquet or .csv manifest with a "
        "model_url column",
    )
    render.add_argument(
        "--data-dir", help="the collection folder that python -m src.ingest will use"
    )
    render.add_argument(
        "--views", type=int, default=8, help="how many views around each model"
    )
    render.add_argument(
        "--size", type=int, default=1024, help="pixels across a view before cropping"
    )
    render.add_argument("--limit", type=int, help="only render the first N models")
    render.add_argument(
        "--max-model-mb",
        type=float,
        default=500.0,
        help="skip model files larger than this",
    )
    render.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds to wait for one model to load",
    )
    render.add_argument(
        "--again",
        action="store_true",
        help="render every model again, even those already rendered",
    )
    render.add_argument(
        "--no-shape", action="store_true", help="do not build the shape index"
    )
    render.add_argument(
        "--shape-model",
        default=DEFAULT_SHAPE_MODEL,
        help="the image model that describes each model's form",
    )
    render.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )

    evaluate = commands.add_parser(
        "evaluate",
        help="see how well shape and appearance find pairs of objects that "
        "curators know belong together",
    )
    evaluate.add_argument(
        "pairs", help="a CSV file with a header row and two columns of object ids"
    )
    evaluate.add_argument("--data-dir", help="the collection folder to evaluate")
    evaluate.add_argument(
        "--within",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="how far down the list of similar objects counts as found",
    )
    evaluate.add_argument("--json", dest="json_path", help="also write the result")
    return parser.parse_args()


def collection_root() -> Path:
    if settings.data_dir:
        return Path(settings.data_dir)
    return Path(settings.embeddings_dir).parent


def views_folder(root: Path, model: Model) -> Path:
    return root / MODELS_DIR / VIEWS_DIR / model.key[:2] / model.key


def kept_views(coverages: dict[int, float]) -> list[int]:
    fullest = max(coverages.values(), default=0.0)
    if not fullest:
        return []
    return [
        number
        for number, coverage in sorted(coverages.items())
        if coverage >= fullest * MIN_SHARE_OF_FULLEST_VIEW
    ]


def is_current(record: dict | None, folder: Path, views: list[View], size: int) -> bool:
    if not record or record.get("error") or not record.get("kept"):
        return False
    if record.get("views") != len(views) or record.get("size") != size:
        return False
    return all(
        (folder / views[n].file_name(mode)).exists()
        for n in record["kept"]
        for mode in MODES
    )


def render_model(
    renderer: Renderer, model: Model, views: list[View], folder: Path, size: int
) -> dict:
    info = renderer.load(model)
    images: dict[int, dict[str, bytes | None]] = {}
    coverages: dict[int, float] = {}
    for view in views:
        images[view.number] = {}
        for mode in MODES:
            data, coverage = renderer.render(view, mode)
            images[view.number][mode] = data
        coverages[view.number] = coverage
    kept = kept_views(coverages)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in [*folder.glob("*.jpg"), *folder.glob("*.npy")]:
        stale.unlink()
    for number in kept:
        for mode, data in images[number].items():
            (folder / views[number].file_name(mode)).write_bytes(data)
    return {
        "object_id": model.object_id,
        "file_name": model.file_name,
        "views": len(views),
        "size": size,
        "kept": kept,
        "coverage": {str(n): round(c, 4) for n, c in coverages.items()},
        "triangles": info.get("triangles"),
        "textured": info.get("textured"),
        "error": None,
    }


def read_records(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_records(path: Path, records: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(records, indent=2))
    partial.replace(path)


def view_rows(
    root: Path, manifest_dir: Path, model: Model, record: dict, views: list[View]
) -> list[dict]:
    folder = views_folder(root, model)
    rows = []
    for number in record["kept"]:
        path = folder / views[number].file_name(COLOR)
        rows.append(
            {
                **model.row,
                IMAGE_COLUMN: str(path.relative_to(manifest_dir)),
                SIZE_COLUMN: path.stat().st_size,
                "object_id": model.object_id,
                "model_url": model.address,
                "view_orbit": views[number].orbit,
            }
        )
    return rows


def build_shape_index(
    root: Path,
    rendered: list[tuple[Model, dict]],
    views: list[View],
    model_name: str,
    device: str,
) -> int:
    encoder = None
    by_object: dict[str, list[np.ndarray]] = {}
    for model, record in rendered:
        folder = views_folder(root, model)
        cache = folder / cache_name(model_name)
        if cache.exists():
            vector = np.load(cache)
        else:
            if encoder is None:
                print(f"Describing forms with {model_name} on {device} ...")
                encoder = ShapeEncoder(model_name, device)
            paths = [folder / views[n].file_name(RELIEF) for n in record["kept"]]
            vector = model_vector(encoder, paths)
            np.save(cache, vector)
        by_object.setdefault(model.object_id, []).append(vector)
    if not by_object:
        return 0
    write_shape_index(
        Path(settings.embeddings_dir),
        list(by_object),
        np.stack([np.mean(vectors, axis=0) for vectors in by_object.values()]),
        model_name,
        len(views),
    )
    return len(by_object)


def render_command(args: argparse.Namespace) -> int:
    if args.data_dir:
        apply_data_dir(settings, args.data_dir)
    root = collection_root()
    manifest_dir = root / MODELS_DIR
    records_path = manifest_dir / RECORD_FILE

    print(f"Looking for 3D models in {args.uri} ...")
    listing = list_source(args.uri)
    for problem in listing.problems:
        print(f"  Left out: {problem}")
    models = listing.models[: args.limit] if args.limit else listing.models
    if not models:
        print("No .glb models found.")
        return 2
    print(f"Found {len(listing.models):,} models and {len(listing.photos):,} photos.")

    views = views_around(args.views)
    records = read_records(records_path)
    todo = [
        m
        for m in models
        if args.again
        or not is_current(
            records.get(m.address), views_folder(root, m), views, args.size
        )
    ]
    print(f"{len(models) - len(todo):,} already rendered, {len(todo):,} to go.\n")
    redrawn = [m for m in todo if (records.get(m.address) or {}).get("kept")]

    if todo:
        try:
            renderer = Renderer(
                args.size, BACKGROUND, int(args.max_model_mb * 1024**2), args.timeout
            )
            with renderer:
                for number, model in enumerate(todo, 1):
                    started = time.time()
                    try:
                        record = render_model(
                            renderer, model, views, views_folder(root, model), args.size
                        )
                        outcome = (
                            f"{len(record['kept'])} of {len(views)} views, "
                            f"{record['triangles'] or 0:,} triangles"
                        )
                    except (ModelRefused, OSError) as error:
                        record = {"object_id": model.object_id, "error": str(error)}
                        outcome = f"could not be rendered: {error}"
                    except Exception as error:
                        message = str(error).splitlines()[0] if str(error) else ""
                        record = {
                            "object_id": model.object_id,
                            "error": f"{type(error).__name__}: {message}",
                        }
                        outcome = f"could not be rendered: {record['error']}"
                    records[model.address] = record
                    write_records(records_path, records)
                    print(
                        f"  [{number}/{len(todo)}] {model.file_name}: {outcome} "
                        f"({time.time() - started:.1f}s)"
                    )
        except RendererUnavailable as error:
            print(error)
            return 2
        except KeyboardInterrupt:
            print("\n\nStopped. Finished models are kept. Run the same command again.")
            return 130

    rendered = [
        (m, records[m.address])
        for m in models
        if is_current(records.get(m.address), views_folder(root, m), views, args.size)
    ]
    failed = len(models) - len(rendered)
    rows = list(listing.photos)
    for model, record in rendered:
        rows.extend(view_rows(root, manifest_dir, model, record, views))
    manifest = manifest_dir / MANIFEST_FILE
    write_manifest(manifest, rows)
    view_count = len(rows) - len(listing.photos)
    print(
        f"\n{len(rendered):,} models rendered"
        + (f", {failed:,} could not be" if failed else "")
        + f". Wrote {view_count:,} views and {len(listing.photos):,} photos to "
        f"{manifest}"
    )

    if not args.no_shape and rendered:
        device = detect_device() if args.device == "auto" else args.device
        shaped = build_shape_index(root, rendered, views, args.shape_model, device)
        print(f"Shape index of {shaped:,} objects in {settings.embeddings_dir}/shape")

    indexed = (Path(settings.embeddings_dir) / INGEST_STATE_FILE).exists()
    if redrawn and indexed:
        print(
            f"\n{len(redrawn):,} models were drawn again, but the search index still "
            f"holds their earlier views. To search the new views, render and index "
            f"into a new folder with --data-dir."
        )
    data_dir = f" --data-dir {args.data_dir}" if args.data_dir else ""
    print(f"\nNext, index the views with:\n  python -m src.ingest {manifest}{data_dir}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    if args.data_dir:
        apply_data_dir(settings, args.data_dir)
    pairs = read_pairs(args.pairs)
    if not pairs:
        print(f"No pairs found in {args.pairs}. It needs a header row and two columns.")
        return 2
    result = compare(
        Path(settings.embeddings_dir), settings.model_name, pairs, args.within
    )
    print(render_comparison(result, args.within))
    if args.json_path:
        with open(args.json_path, "w") as handle:
            json.dump(result, handle, indent=2)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "render":
        return render_command(args)
    return evaluate_command(args)


if __name__ == "__main__":
    sys.exit(main())
