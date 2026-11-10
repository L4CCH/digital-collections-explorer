"""Pick out the catalog fields the site relies on, whatever the collection calls them."""

import math
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ..core.catalog_keys import IIIF_KEYS, MODEL_URL_KEYS

OBJECT_ID_KEYS = ("object_id", "record_id", "item_id", "group_id")

SOURCE_URL_KEYS = ("source_url", "guid", "record_url", "landing_page", "permalink")

MODEL_EXTENSIONS = (".glb", ".gltf")

VIEW_ORBIT_KEY = "view_orbit"

IIIF_IMAGE_REQUEST = re.compile(
    r"/(full|square|pct:[^/]+|\d+,\d+,\d+,\d+)/[^/]+/!?\d+(\.\d+)?"
    r"/(default|color|colour|gray|grey|bitonal|native)\.\w+$",
    re.IGNORECASE,
)


def object_id_from(catalog: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the id of the object this image belongs to, if the catalog has one"""
    named = {str(key).strip().lower(): value for key, value in (catalog or {}).items()}
    for key in OBJECT_ID_KEYS:
        value = named.get(key)
        if isinstance(value, float) and math.isnan(value):
            continue
        if value not in (None, "") and str(value).strip():
            return str(value).strip()
    return None


def source_url_from(catalog: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return a link back to the institution's page for this object, if any"""
    for key in SOURCE_URL_KEYS:
        value = str((catalog or {}).get(key) or "")
        if value.startswith(("http://", "https://")):
            return value
    return None


def model_url_from(catalog: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the address or path of the object's 3D model, a .glb or .gltf file"""
    for key in MODEL_URL_KEYS:
        value = str((catalog or {}).get(key) or "").strip()
        if urlparse(value).path.lower().endswith(MODEL_EXTENSIONS):
            return value
    return None


def servable_model(address: Optional[str]) -> Optional[str]:
    """The model's address if the site can show it: on the web, or a .glb kept here"""
    if not address:
        return None
    if address.startswith(("http://", "https://")):
        return address
    if "://" in address or not address.lower().endswith(".glb"):
        return None
    return address


def view_orbit_from(catalog: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return where the camera stood for an image rendered from a 3D model"""
    value = str((catalog or {}).get(VIEW_ORBIT_KEY) or "").strip()
    return value or None


def iiif_info_from(catalog: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the info.json of the IIIF image service that serves this image, if any"""
    for key in IIIF_KEYS:
        value = str((catalog or {}).get(key) or "").strip()
        if not value.startswith(("http://", "https://")):
            continue
        if urlparse(value).path.endswith("/info.json"):
            return value
        service = IIIF_IMAGE_REQUEST.sub("", value.split("?", 1)[0].rstrip("/"))
        return f"{service}/info.json"
    return None


# The longest edges ingest gives the copies it keeps, so a source's images show at the same sizes.
IIIF_SIZES = {"thumbnail": "!400,400", "full": "!1920,1920"}


def is_web_address(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def image_source(metadata: Dict[str, Any], size: str) -> Optional[str]:
    """Where an image is kept at a size, on this site or at a web address"""
    paths = metadata.get("paths") or {}
    info = metadata.get("iiif")
    service = info.removesuffix("/info.json") if is_web_address(info) else None

    def from_service(name: str) -> Optional[str]:
        return f"{service}/full/{IIIF_SIZES[name]}/0/default.jpg" if service else None

    if size == "thumbnail":
        kept = (
            paths.get("thumbnail"),
            from_service("thumbnail"),
            paths.get("processed"),
        )
    else:
        kept = (paths.get("processed"), from_service("full"))
    return next((place for place in kept if place), None)


def origin_of(address: str) -> str:
    return "/".join(address.split("/", 3)[:3])


def image_hosts(metadata: Dict[str, Any]) -> set:
    """The servers other than this site that a browser loads this image from"""
    paths = metadata.get("paths") or {}
    kept = [paths.get("thumbnail"), paths.get("processed"), metadata.get("iiif")]
    return {origin_of(place) for place in kept if is_web_address(place)}


def image_urls(metadata: Dict[str, Any]) -> Dict[str, str]:
    """The web addresses a browser can load an image from without asking this site"""
    found = {size: image_source(metadata, size) for size in IIIF_SIZES}
    return {size: place for size, place in found.items() if is_web_address(place)}
