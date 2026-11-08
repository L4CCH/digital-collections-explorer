"""One vector per 3D model, from views that show only its form, to find similar shapes."""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

SHAPE_DIR = "shape"
VECTORS_FILE = "vectors.npy"
OBJECTS_FILE = "objects.json"
INFO_FILE = "info.json"


def write_shape_index(
    embeddings_dir: Path,
    object_ids: Sequence[str],
    vectors: np.ndarray,
    model_name: str,
    views: int,
) -> Path:
    """Save one unit-length vector per object, with the model that made them"""
    folder = Path(embeddings_dir) / SHAPE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(folder / VECTORS_FILE, vectors / np.where(norms == 0, 1, norms))
    (folder / OBJECTS_FILE).write_text(json.dumps(list(object_ids)))
    info = {
        "model_name": model_name,
        "views_per_model": views,
        "objects": len(object_ids),
        "dimensions": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (folder / INFO_FILE).write_text(json.dumps(info, indent=2))
    return folder


def read_shape_info(embeddings_dir: Path) -> Optional[dict]:
    """Which model built the shape index, if the collection has one"""
    path = Path(embeddings_dir) / SHAPE_DIR / INFO_FILE
    return json.loads(path.read_text()) if path.exists() else None


@dataclass(frozen=True)
class Shapes:
    stamp: Tuple[str, int]
    vectors: torch.Tensor
    objects: List[str]
    row: Dict[str, int]


class ShapeIndex:
    """The shape index of the collection being served, read again when it changes"""

    def __init__(self):
        self.lock = threading.Lock()
        self.shapes: Optional[Shapes] = None

    def load(self, embeddings_dir: Path) -> Optional[Shapes]:
        """Read the index from disk unless the copy in memory is current"""
        folder = Path(embeddings_dir) / SHAPE_DIR
        path = folder / VECTORS_FILE
        with self.lock:
            if not path.exists() or not (folder / OBJECTS_FILE).exists():
                self.shapes = None
                return None
            stamp = (str(path), path.stat().st_mtime_ns)
            if self.shapes is not None and self.shapes.stamp == stamp:
                return self.shapes
            vectors = np.load(path)
            objects = [str(o) for o in json.loads((folder / OBJECTS_FILE).read_text())]
            if len(objects) != vectors.shape[0]:
                logger.warning(
                    f"{path} has {vectors.shape[0]} rows but {len(objects)} objects"
                )
                self.shapes = None
                return None
            self.shapes = Shapes(
                stamp=stamp,
                vectors=torch.from_numpy(
                    np.ascontiguousarray(vectors, dtype=np.float32)
                ),
                objects=objects,
                row={o: n for n, o in enumerate(objects)},
            )
            return self.shapes

    def neighbors(
        self, embeddings_dir: Path, object_id: str
    ) -> Optional[List[Tuple[str, float]]]:
        """Every other object with a model, the most alike in shape first"""
        shapes = self.load(embeddings_dir)
        if shapes is None or object_id not in shapes.row:
            return None
        scores = (shapes.vectors @ shapes.vectors[shapes.row[object_id]]).numpy()
        return [
            (shapes.objects[n], float(scores[n]))
            for n in np.argsort(-scores, kind="stable").tolist()
            if shapes.objects[n] != object_id
        ]
