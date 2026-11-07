"""Pick out the catalog fields the site relies on, whatever the collection calls them."""

from typing import Any, Dict, Optional
from urllib.parse import urlparse

OBJECT_ID_KEYS = ("object_id", "record_id", "item_id", "group_id")

SOURCE_URL_KEYS = ("source_url", "guid", "record_url", "landing_page", "permalink")

MODEL_URL_KEYS = ("model_url", "model", "glb", "gltf", "3d_model")

MODEL_EXTENSIONS = (".glb", ".gltf")

VIEW_ORBIT_KEY = "view_orbit"


def object_id_from(catalog: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the id of the object this image belongs to, if the catalog has one"""
    for key in OBJECT_ID_KEYS:
        value = (catalog or {}).get(key)
        if value not in (None, ""):
            return str(value)
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


def view_orbit_from(catalog: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return where the camera stood for an image rendered from a 3D model"""
    value = str((catalog or {}).get(VIEW_ORBIT_KEY) or "").strip()
    return value or None
