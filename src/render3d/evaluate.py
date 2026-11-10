from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from src.backend.services.collection_store import load_grouped
from src.backend.services.shape_index import (
    OBJECTS_FILE,
    SHAPE_DIR,
    VECTORS_FILE,
    read_shape_info,
)


def read_pairs(path: str) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    return [
        (row[0].strip(), row[1].strip())
        for row in rows[1:]
        if len(row) >= 2 and row[0].strip() and row[1].strip()
    ]


def unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def shape_vectors(embeddings_dir: Path) -> dict[str, np.ndarray]:
    folder = embeddings_dir / SHAPE_DIR
    if not (folder / VECTORS_FILE).exists():
        return {}
    vectors = np.load(folder / VECTORS_FILE)
    objects = json.loads((folder / OBJECTS_FILE).read_text())
    return dict(zip(objects, vectors))


def appearance_vectors(embeddings_dir: Path, model_name: str) -> dict[str, np.ndarray]:
    index = load_grouped(embeddings_dir, model_name)
    rows: dict[str, list[int]] = {}
    views: dict[str, list[int]] = {}
    for number, item_id in enumerate(index.item_ids):
        metadata = index.metadata.get(item_id, {})
        object_id = metadata.get("object_id") or item_id
        rows.setdefault(object_id, []).append(number)
        if metadata.get("view_orbit"):
            views.setdefault(object_id, []).append(number)
    vectors = index.embeddings.numpy()
    return {
        object_id: vectors[views.get(object_id) or members].mean(axis=0)
        for object_id, members in rows.items()
    }


def rank_of(scores: np.ndarray, query: int, target: int) -> int:
    better = scores > scores[target]
    better[query] = False
    return int(better.sum()) + 1


def evaluate(
    vectors: np.ndarray, objects: list[str], pairs: list[tuple[str, str]], ks
) -> dict:
    row = {o: n for n, o in enumerate(objects)}
    vectors = torch.from_numpy(unit(np.asarray(vectors, dtype=np.float32)))
    ranks = []
    for a, b in pairs:
        if a in row and b in row and a != b:
            for query, target in ((a, b), (b, a)):
                scores = (vectors @ vectors[row[query]]).numpy()
                ranks.append(rank_of(scores, row[query], row[target]))
    return {
        "queries": len(ranks),
        "found_within": {
            str(k): (sum(r <= k for r in ranks) / len(ranks) if ranks else None)
            for k in ks
        },
        "median_rank": statistics.median(ranks) if ranks else None,
    }


def compare(embeddings_dir: Path, model_name: str, pairs, ks) -> dict:
    shapes = shape_vectors(embeddings_dir)
    looks = appearance_vectors(embeddings_dir, model_name)
    objects = [o for o in shapes if o in looks]
    usable = [(a, b) for a, b in pairs if a in objects and b in objects and a != b]
    shape = np.stack([shapes[o] for o in objects]) if objects else np.empty((0, 0))
    look = np.stack([looks[o] for o in objects]) if objects else np.empty((0, 0))
    shape_info = read_shape_info(embeddings_dir) or {}
    methods = {}
    if objects:
        both = np.hstack([unit(shape), unit(look)])
        methods = {
            f"Shape ({shape_info.get('model_name', 'shape model')})": shape,
            f"Appearance ({model_name})": look,
            "Shape and appearance": both,
        }
    return {
        "objects_with_models": len(objects),
        "pairs": len(pairs),
        "pairs_usable": len(usable),
        "pairs_missing": [
            [a, b] for a, b in pairs if a not in objects or b not in objects
        ],
        "methods": {
            name: evaluate(vectors, objects, usable, ks)
            for name, vectors in methods.items()
        },
    }


def percent(value) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def render_comparison(result: dict, ks) -> str:
    lines = [
        f"{result['pairs_usable']} of {result['pairs']} pairs have a 3D model "
        f"on both sides, among {result['objects_with_models']} objects with models.",
    ]
    if result["pairs_missing"]:
        shown = ", ".join(" / ".join(pair) for pair in result["pairs_missing"][:5])
        lines.append(f"Left out because an object has no model here: {shown}")
    if not result["methods"]:
        return "\n".join(lines)
    width = max(len(name) for name in result["methods"])
    header = "".join(f"{'within ' + str(k):>11}" for k in ks)
    lines += ["", f"{'':<{width}}{header}{'median rank':>13}"]
    for name, scores in result["methods"].items():
        found = "".join(f"{percent(scores['found_within'][str(k)]):>11}" for k in ks)
        median = scores["median_rank"]
        lines.append(
            f"{name:<{width}}{found}{('-' if median is None else f'{median:g}'):>13}"
        )
    lines += [
        "",
        "Each pair is asked both ways: how far down the list of similar objects "
        "the other one appears.",
    ]
    return "\n".join(lines)
