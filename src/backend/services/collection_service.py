"""Browsing a collection: samples, single items, similar items, grouped results."""

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import torch

from ..core.config import settings
from .catalog_fields import (
    iiif_info_from,
    image_hosts,
    image_urls,
    object_id_from,
    servable_model,
    source_url_from,
)
from .catalog_filter import can_filter
from .collection_store import METADATA_DB, has_table, read_record_details
from .embedding_service import EmbeddingService, embedding_service
from .index_info import is_grouped, read_index_info
from .shape_index import ShapeIndex

logger = logging.getLogger(__name__)

COLLECTION_FILE = "collection.json"
DEFAULT_COLLECTION = {
    "title": "Digital Collection Explorer",
    "description": "Explore this collection by describing what you are looking for, "
    "or simply start wandering.",
    "example_queries": [],
    "institution": None,
    "source_name": None,
    "license": None,
    "links": [],
}


class CollectionService:
    """Groups images by object and answers browsing questions about them"""

    def __init__(self, embeddings: EmbeddingService):
        self.embeddings = embeddings
        self.indexed_count = -1
        self.position: Dict[str, int] = {}
        self.object_ids: List[str] = []
        self.members: Dict[str, List[int]] = {}
        self.group_index: Dict[str, int] = {}
        self.row_group: List[int] = []
        self.object_groups: Dict[str, Set[int]] = {}
        self.groups: List[Dict[str, Any]] = []
        self.uncataloged_objects: Set[str] = set()
        self.cataloged_rows: Optional[torch.Tensor] = None
        self.object_number: Dict[str, int] = {}
        self.row_object: Optional[torch.Tensor] = None
        self.object_model: Dict[str, int] = {}
        self.view_rows: Set[int] = set()
        self.page_rows: Set[int] = set()
        self.documents: Set[str] = set()
        self.photo_count: Dict[str, int] = {}
        self.shapes = ShapeIndex()

    def _ensure_index(self) -> None:
        if not self.embeddings.is_loaded:
            self.embeddings.load_embeddings()
        item_ids = self.embeddings.item_ids or []
        if len(item_ids) == self.indexed_count:
            return
        self.position, self.object_ids, self.members = {}, [], {}
        self.object_model, self.view_rows = {}, set()
        self.page_rows, self.documents = set(), set()
        cataloged: List[bool] = []
        for index, item_id in enumerate(item_ids):
            metadata = (self.embeddings.metadata or {}).get(item_id, {})
            record = metadata.get("object_id") or object_id_from(
                metadata.get("catalog")
            )
            object_id = record or metadata.get("document_key") or item_id
            self.position[item_id] = index
            self.object_ids.append(object_id)
            self.members.setdefault(object_id, []).append(index)
            cataloged.append(metadata.get("cataloged", bool(record)))
            if servable_model(metadata.get("model_url")):
                self.object_model.setdefault(object_id, index)
            if metadata.get("view_orbit"):
                self.view_rows.add(index)
            if metadata.get("type") == "pdf_page":
                self.page_rows.add(index)
                self.documents.add(metadata.get("document_key") or object_id)
        self.photo_count = {
            o: sum(1 for m in members if m not in self.view_rows)
            for o, members in self.members.items()
        }
        self.cataloged_rows = torch.tensor(cataloged, dtype=torch.bool)
        self.object_number = {o: n for n, o in enumerate(self.members)}
        self.row_object = torch.tensor(
            [self.object_number[o] for o in self.object_ids], dtype=torch.long
        )
        self.uncataloged_objects = {
            self.object_ids[i] for i, known in enumerate(cataloged) if not known
        }
        self._index_groups()
        self.indexed_count = len(item_ids)

    def _index_groups(self) -> None:
        groups = self.embeddings.groups or []
        rows = self.embeddings.group_of_row
        self.group_index = {group["id"]: number for number, group in enumerate(groups)}
        self.row_group = rows.tolist() if groups and rows is not None else []
        self.object_groups = {
            o: {self.row_group[m] for m in members}
            for o, members in self.members.items()
            if self.row_group
        }
        images = Counter(
            group
            for row, group in enumerate(self.row_group)
            if row not in self.view_rows and row not in self.page_rows
        )
        objects = Counter(n for found in self.object_groups.values() for n in found)
        self.groups = [
            {
                "id": group["id"],
                "label": group["label"],
                "images": images[number],
                "objects": objects[number],
            }
            for number, group in enumerate(groups)
        ]

    def wanted_groups(self, value: Optional[str]) -> Optional[List[str]]:
        """Read a comma-separated group filter such as "nmah,saam"; None means all"""
        names = (name.strip() for name in (value or "").split(","))
        wanted = [name for name in dict.fromkeys(names) if name]
        if not wanted:
            return None
        self._ensure_index()
        unknown = [g for g in wanted if g not in self.group_index]
        if unknown:
            raise ValueError(f"This collection has no group named {', '.join(unknown)}")
        return wanted

    def _numbers(self, groups: Optional[List[str]]) -> Optional[Set[int]]:
        if not groups:
            return None
        self._ensure_index()
        return {self.group_index[g] for g in groups}

    def group_mask(
        self,
        groups: Optional[List[str]],
        cataloged_only: bool = False,
        records: Optional[Set[str]] = None,
    ) -> Optional[torch.Tensor]:
        """Which rows belong to the given groups and records, or None when every row counts"""
        numbers = self._numbers(groups)
        mask = None
        if numbers is not None:
            wanted = torch.tensor(sorted(numbers), dtype=torch.long)
            mask = torch.isin(self.embeddings.group_of_row, wanted)
        if cataloged_only:
            self._ensure_index()
            if self.uncataloged_objects:
                known = self.cataloged_rows
                mask = known if mask is None else mask & known
        if records is not None:
            self._ensure_index()
            numbers = [
                self.object_number[o] for o in records if o in self.object_number
            ]
            chosen = torch.isin(
                self.row_object, torch.tensor(numbers, dtype=torch.long)
            )
            mask = chosen if mask is None else mask & chosen
        return mask

    def _objects_in(
        self,
        numbers: Optional[Set[int]],
        cataloged_only: bool = False,
        records: Optional[Set[str]] = None,
    ) -> List[str]:
        objects = (
            list(self.members)
            if numbers is None
            else [o for o in self.members if self.object_groups[o] & numbers]
        )
        if cataloged_only:
            objects = [o for o in objects if o not in self.uncataloged_objects]
        if records is not None:
            objects = [o for o in objects if o in records]
        return objects

    def objects_among(
        self, object_ids: List[str], groups: Optional[List[str]] = None
    ) -> List[str]:
        """The given objects that this collection shows in the given groups, in order"""
        self._ensure_index()
        numbers = self._numbers(groups)
        return [
            o
            for o in object_ids
            if o in self.members
            and (numbers is None or self.object_groups.get(o, set()) & numbers)
        ]

    def group_counts(self, object_ids: List[str]) -> Dict[str, int]:
        """How many of these objects each group holds, leaving out empty groups"""
        self._ensure_index()
        counts = Counter(n for o in object_ids for n in self.object_groups.get(o, ()))
        return {
            group["id"]: counts[number]
            for number, group in enumerate(self.groups)
            if counts[number]
        }

    def first_image(self, object_id: str, groups: Optional[List[str]] = None) -> int:
        """The row of an object's first image in the given groups"""
        self._ensure_index()
        numbers = self._numbers(groups) if self.row_group else None
        return self.shown_rows(object_id, numbers)[0]

    def ranked_objects(
        self,
        scores: torch.Tensor,
        limit: int,
        groups: Optional[List[str]] = None,
        cataloged_only: bool = False,
        records: Optional[Set[str]] = None,
    ) -> List[Tuple[str, int, float]]:
        """Objects by their best image's raw similarity, with that image and score"""
        self._ensure_index()
        flat = scores.flatten()
        mask = self.group_mask(groups, cataloged_only, records)
        if mask is not None:
            flat = flat.masked_fill(~mask, float("-inf"))
        found: List[Tuple[str, int, float]] = []
        seen: Set[str] = set()
        for index in torch.argsort(flat, descending=True).tolist():
            if mask is not None and not bool(mask[index]):
                break
            object_id = self.object_ids[index]
            if object_id in seen:
                continue
            seen.add(object_id)
            found.append((object_id, index, float(flat[index])))
            if len(found) >= limit:
                break
        return found

    def image_hosts(self) -> List[str]:
        """The servers other than this site that show the collection's images"""
        self._ensure_index()
        hosts: Set[str] = set()
        for metadata in (self.embeddings.metadata or {}).values():
            hosts |= image_hosts(metadata)
        return sorted(hosts)

    def group_summary(self, configured: Any = None) -> List[Dict[str, Any]]:
        """Each group's size, with its label, institution and license from collection.json"""
        self._ensure_index()
        named = configured if isinstance(configured, dict) else {}
        position = {key: n for n, key in enumerate(named)}
        summary = []
        for group in self.groups:
            key = group["id"] if group["id"] in named else group["label"]
            own = named.get(key)
            own = own if isinstance(own, dict) else {}
            described = dict(group)
            described["label"] = own.get("label") or group["label"]
            institution = own.get("institution") or own.get("source_name")
            if institution:
                described["institution"] = institution
            if own.get("license"):
                described["license"] = own["license"]
            summary.append((position.get(key, len(position)), described))
        return [described for _, described in sorted(summary, key=lambda g: g[0])]

    def configured(self) -> Dict[str, Any]:
        """What collection.json says of the collection, or nothing if it is missing"""
        path = Path(settings.data_dir or ".") / COLLECTION_FILE
        if not (settings.data_dir and path.exists()):
            return {}
        try:
            found = json.loads(path.read_text())
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"Ignoring unreadable {path}: {e}")
            return {}
        return found if isinstance(found, dict) else {}

    def _metadata(self, index: int) -> Dict[str, Any]:
        item_id = self.embeddings.item_ids[index]
        return (self.embeddings.metadata or {}).get(item_id, {})

    def present(self, index: int, score: Optional[float] = None) -> Dict[str, Any]:
        """Describe one image the way the site shows it"""
        item_id = self.embeddings.item_ids[index]
        metadata = dict(self._metadata(index))
        catalog = metadata.get("catalog")
        metadata.setdefault("title", (catalog or {}).get("title"))
        metadata.setdefault("source_url", source_url_from(catalog))
        if "iiif" not in metadata:
            iiif = iiif_info_from(catalog)
            if iiif:
                metadata["iiif"] = iiif
        urls = image_urls(metadata)
        if urls:
            metadata["image_urls"] = urls
        object_id = self.object_ids[index]
        if object_id in self.object_model:
            metadata.setdefault(
                "model_url", self._metadata(self.object_model[object_id])["model_url"]
            )
        else:
            metadata.pop("model_url", None)
        return {
            "id": item_id,
            "score": score if score is not None else 0.0,
            "metadata": metadata,
            "object_id": object_id,
            "image_count": self.photo_count[object_id],
        }

    def model_of(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Where an object's 3D model is kept, and the views rendered from it"""
        self._ensure_index()
        row = self.object_model.get(object_id)
        if row is None:
            return None
        item_ids = self.embeddings.item_ids
        address = self._metadata(row)["model_url"]
        return {
            "id": item_ids[row],
            "url": address if address.startswith(("http://", "https://")) else None,
            "file_name": Path(urlparse(address).path).name,
            "views": [
                {"id": item_ids[m], "orbit": self._metadata(m)["view_orbit"]}
                for m in self.members[object_id]
                if m in self.view_rows
            ],
        }

    def shown_rows(
        self, object_id: str, numbers: Optional[Set[int]] = None
    ) -> List[int]:
        """The rows that may stand for an object: its photos, or else its first view"""
        members = [
            m
            for m in self.members[object_id]
            if numbers is None or self.row_group[m] in numbers
        ] or self.members[object_id]
        return [m for m in members if m not in self.view_rows] or members[:1]

    def info(self) -> Dict[str, Any]:
        """Describe the collection: its name, size, and suggested searches"""
        self._ensure_index()
        described = dict(DEFAULT_COLLECTION)
        configured = self.configured()
        described.update(configured)
        institution = configured.get("institution") or configured.get("source_name")
        described["institution"] = institution
        described["source_name"] = institution
        described["images"] = (
            self.indexed_count - len(self.view_rows) - len(self.page_rows)
        )
        described["documents"] = len(self.documents)
        described["models"] = len(self.object_model)
        described["objects"] = len(self.members)
        described["uncataloged_images"] = sum(
            1
            for row in torch.nonzero(~self.cataloged_rows).flatten().tolist()
            if row not in self.view_rows and row not in self.page_rows
        )
        index = read_index_info(self.embeddings.embeddings_dir) or {}
        described["groups"] = self.group_summary(configured.get("groups"))
        described["catalog_search"] = is_grouped(index) and has_table(
            Path(self.embeddings.embeddings_dir) / METADATA_DB, "records_fts"
        )
        described["catalog_filters"] = is_grouped(index) and can_filter(
            Path(self.embeddings.embeddings_dir) / METADATA_DB
        )
        return described

    def sample(
        self,
        limit: int,
        seed: Optional[int] = None,
        offset: int = 0,
        groups: Optional[List[str]] = None,
        cataloged_only: bool = False,
        records: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return a varied handful of images, at most one per object, without repeats"""
        self._ensure_index()
        numbers = self._numbers(groups)
        objects = self._objects_in(numbers, cataloged_only, records)
        random.Random(seed).shuffle(objects)
        chosen = objects[offset : offset + limit]
        return [
            self.present(
                random.Random(f"{seed}:{o}").choice(self.shown_rows(o, numbers))
            )
            for o in chosen
        ]

    def object_count(
        self,
        groups: Optional[List[str]] = None,
        cataloged_only: bool = False,
        records: Optional[Set[str]] = None,
    ) -> int:
        """How many distinct objects the collection, or the given groups, hold"""
        self._ensure_index()
        return len(self._objects_in(self._numbers(groups), cataloged_only, records))

    def item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Return one image with every photo of its object and, if kept, its record"""
        self._ensure_index()
        index = self.position.get(item_id)
        if index is None:
            return None
        described = self.present(index)
        described["photos"] = [
            self.present(member) for member in self.members[self.object_ids[index]]
        ]
        model = self.model_of(described["object_id"])
        if model:
            described["model"] = model
        if self.groups:
            record = read_record_details(
                Path(self.embeddings.embeddings_dir) / METADATA_DB,
                described["object_id"],
            )
            labels = self.configured().get("field_labels")
            labels = labels if isinstance(labels, dict) else {}
            for shown in record["fields"]:
                column = shown.pop("field", None)
                shown["label"] = (
                    labels.get(column) or labels.get(shown["label"]) or shown["label"]
                )
            described["record"] = record
        return described

    def similar(
        self,
        item_id: str,
        limit: int,
        groups: Optional[List[str]] = None,
        records: Optional[Set[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return images of other objects that look most like this one"""
        self._ensure_index()
        index = self.position.get(item_id)
        if index is None:
            return None
        vectors = self.embeddings.embeddings
        scores = torch.matmul(vectors, vectors[index])
        return self.ranked(
            scores,
            limit,
            exclude_object=self.object_ids[index],
            groups=groups,
            records=records,
        )

    def similar_in_shape(
        self,
        item_id: str,
        limit: int,
        groups: Optional[List[str]] = None,
        records: Optional[Set[str]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return other objects whose 3D models are most like this one's in form"""
        self._ensure_index()
        index = self.position.get(item_id)
        if index is None:
            return None
        neighbors = self.shapes.neighbors(
            Path(self.embeddings.embeddings_dir), self.object_ids[index]
        )
        numbers = self._numbers(groups)
        results: List[Dict[str, Any]] = []
        for object_id, score in neighbors or []:
            if object_id not in self.members:
                continue
            groups_match = numbers is None or self.object_groups[object_id] & numbers
            if not groups_match or (records is not None and object_id not in records):
                continue
            views = [m for m in self.members[object_id] if m in self.view_rows]
            shown = views[0] if views else self.shown_rows(object_id, numbers)[0]
            results.append(self.present(shown, score))
            if len(results) >= limit:
                break
        return results

    def ranked(
        self,
        scores: torch.Tensor,
        limit: int,
        offset: int = 0,
        group_by_object: bool = True,
        exclude_object: Optional[str] = None,
        score_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        groups: Optional[List[str]] = None,
        cataloged_only: bool = False,
        records: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Rank by raw similarity, showing each object once; transform only reported scores"""
        self._ensure_index()
        results: List[Dict[str, Any]] = []
        seen = {exclude_object} if exclude_object else set()
        skipped = 0
        flat = scores.flatten()
        mask = self.group_mask(groups, cataloged_only, records)
        if mask is not None:
            flat = flat.masked_fill(~mask, float("-inf"))
        order = torch.argsort(flat, descending=True).tolist()
        for index in order:
            if mask is not None and not bool(mask[index]):
                break
            object_id = self.object_ids[index]
            if object_id in seen:
                continue
            if group_by_object:
                seen.add(object_id)
            if skipped < offset:
                skipped += 1
                continue
            results.append(self.present(index, float(flat[index])))
            if len(results) >= limit:
                break
        if score_transform is not None and results:
            reported = score_transform(torch.tensor([r["score"] for r in results]))
            for result, score in zip(results, reported.tolist()):
                result["score"] = score
        return results


collection_service = CollectionService(embedding_service)
