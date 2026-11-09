from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

from src.backend.core.catalog_keys import IIIF_KEYS
from src.backend.services.catalog_fields import iiif_info_from

IMAGE_EXTS = {
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tif",
    "tiff",
    "webp",
    "heic",
    "heif",
    "avif",
}
BROWSER_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "avif"}
DOCUMENT_EXTS = {"pdf"}
MODEL_EXTS = {"glb", "gltf"}
JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini", ".picasa.ini"}
JUNK_DIRS = {".git", "__pycache__", ".cache", "node_modules", ".thumbnails"}
URL_COLUMN_GUESSES = ["image_url", "url", "path", "file", "uri"]
SIZE_COLUMN_GUESSES = ["size_bytes", "size", "bytes", "filesize"]
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PN"

# Bandwidth probe: whole files between 200 KB and 20 MB, since small reads measure latency.
PROBE_MIN_BYTES = 200_000
PROBE_MAX_BYTES = 20_000_000
PROBE_FILES = 6
PROBE_CONCURRENCY = 8
S3_POOL_SIZE = 32
HTTP_TIMEOUT_SECONDS = 30
USER_AGENT = "digital-collections-explorer (https://github.com/L4CCH/digital-collections-explorer)"


@dataclass
class FileRef:
    key: str
    size: int
    ext: str
    is_image: bool
    is_junk: bool = False
    is_document: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def basename(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def parts(self) -> list[str]:
        return self.key.split("/")[:-1]


def classify(key: str, size: int, extra: dict | None = None) -> FileRef:
    base = key.rsplit("/", 1)[-1]
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    junk = base.lower() in JUNK_NAMES or base.startswith("._")
    return FileRef(
        key=key,
        size=size,
        ext=ext,
        is_image=ext in IMAGE_EXTS,
        is_junk=junk,
        is_document=ext in DOCUMENT_EXTS,
        extra=extra or {},
    )


def is_parquet(uri: str) -> bool:
    path = urlparse(uri).path if "://" in uri else uri
    return path.lower().endswith(".parquet")


def fetch_web(url: str, limit: int | None = None) -> tuple[bytes, str] | None:
    """The body and content type found at a web address, or None if it cannot be read"""
    import urllib.request

    headers = {"User-Agent": USER_AGENT}
    if limit:
        headers["Range"] = f"bytes=0-{limit - 1}"
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return response.read(), (response.headers.get("Content-Type") or "").lower()
    except Exception:
        return None


def s3_client(anonymous: bool, region: str):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    options = {
        "max_pool_connections": S3_POOL_SIZE,
        "retries": {"max_attempts": 4, "mode": "standard"},
    }
    if anonymous:
        options["signature_version"] = UNSIGNED
    return boto3.client("s3", region_name=region, config=Config(**options))


class Source:
    kind = "abstract"
    fetch_note = ""

    def describe(self) -> dict[str, Any]:
        raise NotImplementedError

    def iter_files(self) -> Iterator[FileRef]:
        raise NotImplementedError

    def read_bytes(self, ref: FileRef, limit: int | None = None) -> bytes | None:
        raise NotImplementedError

    def original_uri(self, ref: FileRef) -> str:
        raise NotImplementedError

    def local_address(self, value: str) -> str:
        """Where a file the catalog names by path is on this machine"""
        return value

    def web_image_address(self, ref: FileRef) -> str | None:
        """The web address of the image itself, when browsers can show it from there"""
        return None

    def catalog_rows(self) -> tuple[list[str], list[tuple]] | None:
        return None

    def probe_bandwidth(self, refs: list[FileRef]) -> float:
        picks = [r for r in refs if PROBE_MIN_BYTES < r.size < PROBE_MAX_BYTES][
            :PROBE_FILES
        ]
        picks = picks or refs[:PROBE_FILES]
        if not picks:
            return 0.0
        started = time.time()
        with ThreadPoolExecutor(min(PROBE_CONCURRENCY, len(picks))) as pool:
            received = sum(
                len(data or b"") for data in pool.map(self.read_bytes, picks)
            )
        elapsed = time.time() - started
        return received / elapsed if elapsed > 0 and received else 0.0


class LocalDirSource(Source):
    kind = "local_dir"

    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))

    def describe(self):
        return {"kind": self.kind, "uri": self.root, "credentials_required": False}

    def iter_files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in dirnames if d not in JUNK_DIRS and not d.startswith(".")
            ]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, self.root).replace(os.sep, "/")
                try:
                    size = os.path.getsize(full)
                except OSError:
                    yield FileRef(
                        key=rel,
                        size=0,
                        ext="",
                        is_image=False,
                        extra={"unreadable": True},
                    )
                    continue
                yield classify(rel, size)

    def read_bytes(self, ref, limit=None):
        try:
            with open(os.path.join(self.root, ref.key), "rb") as fh:
                return fh.read(limit) if limit else fh.read()
        except OSError:
            return None

    def original_uri(self, ref):
        return os.path.join(self.root, ref.key)


class S3Source(Source):
    kind = "s3"

    def __init__(self, uri: str, anonymous: bool = False, region: str = "us-west-2"):
        parsed = urlparse(uri)
        self.bucket = parsed.netloc
        self.prefix = parsed.path.lstrip("/")
        self.anonymous = anonymous
        self.client = s3_client(anonymous, region)

    def describe(self):
        return {
            "kind": self.kind,
            "uri": f"s3://{self.bucket}/{self.prefix}",
            "credentials_required": not self.anonymous,
        }

    def iter_files(self):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(self.prefix) :].lstrip("/") if self.prefix else key
                if rel:
                    yield classify(rel, obj["Size"], {"abs_key": key})

    def read_bytes(self, ref, limit=None):
        request = {"Bucket": self.bucket, "Key": ref.extra["abs_key"]}
        if limit:
            request["Range"] = f"bytes=0-{limit - 1}"
        try:
            return self.client.get_object(**request)["Body"].read()
        except Exception:
            return None

    def original_uri(self, ref):
        return f"s3://{self.bucket}/{ref.extra['abs_key']}"


class ParquetManifestSource(Source):
    kind = "parquet_manifest"

    def __init__(
        self,
        path: str,
        url_column: str | None = None,
        size_column: str | None = None,
        fetch_via: str | None = None,
        anonymous: bool = False,
        region: str = "us-west-2",
    ):
        import duckdb

        self.path = path
        self.local_path = self._local_copy(path, anonymous, region)
        self.base_dir = (
            None if path.startswith("s3://") else os.path.dirname(os.path.abspath(path))
        )
        self.con = duckdb.connect()
        self.con.execute("INSTALL httpfs; LOAD httpfs;")
        self.columns = [
            row[0]
            for row in self.con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{self.local_path}')"
            ).fetchall()
        ]
        self.url_column = url_column or self._guess(URL_COLUMN_GUESSES)
        self.iiif_column = self._guess(IIIF_KEYS)
        self.size_column = size_column or self._guess(SIZE_COLUMN_GUESSES)
        if not self.url_column and not self.iiif_column:
            raise ValueError(
                f"No URL or IIIF column found in {path}. Columns: {self.columns}"
            )
        self.s3 = None
        self.s3_bucket = None
        if fetch_via and fetch_via.startswith("s3://"):
            self.s3_bucket = urlparse(fetch_via).netloc
            self.s3 = s3_client(anonymous, region)
        self._rows: list[tuple] | None = None

    @staticmethod
    def _local_copy(path: str, anonymous: bool, region: str) -> str:
        if not path.startswith("s3://"):
            return path
        parsed = urlparse(path)
        handle = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        handle.close()
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        for attempt_anonymous in (anonymous, not anonymous):
            try:
                s3_client(attempt_anonymous, region).download_file(
                    bucket, key, handle.name
                )
                return handle.name
            except Exception as error:
                last_error = error
        raise PermissionError(
            f"Could not read {path}, with or without AWS credentials. "
            f"Check that the file exists and that this machine may read the bucket."
        ) from last_error

    def _guess(self, candidates):
        lowered = {c.lower(): c for c in self.columns}
        return next((lowered[c] for c in candidates if c in lowered), None)

    def _load(self) -> list[tuple]:
        if self._rows is None:
            self._rows = self.con.execute(
                f"SELECT * FROM read_parquet('{self.local_path}')"
            ).fetchall()
        return self._rows

    def describe(self):
        return {
            "kind": self.kind,
            "uri": self.path,
            "credentials_required": False,
            "url_column": self.url_column,
            "iiif_column": self.iiif_column,
            "size_column": self.size_column,
            "fetch_via": f"s3://{self.s3_bucket}" if self.s3_bucket else None,
        }

    def _address(self, row: dict) -> str:
        if self.url_column:
            return str(row[self.url_column])
        info = iiif_info_from({self.iiif_column: row[self.iiif_column]}) or ""
        return f"{info.removesuffix('/info.json')}/full/max/0/default.jpg"

    def iter_files(self):
        size_index = self.columns.index(self.size_column) if self.size_column else None
        for values in self._load():
            row = dict(zip(self.columns, values))
            url = self._address(row)
            key = unquote(urlparse(url).path).lstrip("/") if "://" in url else url
            size = int(values[size_index] or 0) if size_index is not None else 0
            extra = {"url": url, "row": row, "named_by_service": not self.url_column}
            yield classify(key, size, extra)

    def read_bytes(self, ref, limit=None):
        if self.s3 is not None:
            request = {"Bucket": self.s3_bucket, "Key": ref.key}
            if limit:
                request["Range"] = f"bytes=0-{limit - 1}"
            try:
                return self.s3.get_object(**request)["Body"].read()
            except Exception:
                return None
        url = ref.extra.get("url", "")
        if "://" not in url:
            return self._read_local(url, limit)
        return self._read_http(url, limit)

    def _local_file(self, value: str) -> str | None:
        if "://" in value or not self.base_dir:
            return None
        return os.path.join(self.base_dir, os.path.expanduser(value))

    def _read_local(self, value: str, limit: int | None) -> bytes | None:
        path = self._local_file(value)
        if path is None:
            return None
        try:
            with open(path, "rb") as fh:
                return fh.read(limit) if limit else fh.read()
        except OSError:
            return None

    def _read_http(self, url: str, limit: int | None) -> bytes | None:
        if "://" not in url:
            return None
        fetched = fetch_web(url, limit)
        if fetched is None:
            return None
        data, content_type = fetched
        is_image = content_type.startswith("image/") or data[:3] in (
            JPEG_MAGIC,
            PNG_MAGIC,
        )
        if not is_image:
            returned = content_type.split(";")[0] or "an unknown type"
            self.fetch_note = (
                f"The '{self.url_column or self.iiif_column}' column returns {returned}, "
                f"not image data. "
                f"It points to web pages rather than files. "
                f"Use --fetch-via to tell us where the files live."
            )
            return None
        return data

    def original_uri(self, ref):
        if self.s3_bucket:
            return f"s3://{self.s3_bucket}/{ref.key}"
        url = ref.extra.get("url", ref.key)
        return self._local_file(url) or url

    def local_address(self, value):
        return self._local_file(value) or value

    def web_image_address(self, ref):
        url = ref.extra.get("url", "")
        if self.s3 is not None or not url.startswith(("http://", "https://")):
            return None
        return url if ref.ext in BROWSER_IMAGE_EXTS else None

    def catalog_rows(self):
        return self.columns, self._load()


def open_source(
    uri: str, anonymous: bool = False, fetch_via: str | None = None
) -> Source:
    if is_parquet(uri):
        return ParquetManifestSource(uri, fetch_via=fetch_via, anonymous=anonymous)
    if uri.startswith("s3://"):
        return S3Source(uri, anonymous=anonymous)
    return LocalDirSource(uri)
