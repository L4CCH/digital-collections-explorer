from __future__ import annotations

import base64
import hashlib
import io
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from src.backend.services.catalog_fields import (
    iiif_info_from,
    model_url_from,
    object_id_from,
    source_url_from,
    view_orbit_from,
)
from src.profiling.sources import FileRef, Source

from .documents import Pdf, page_key
from .state import IngestState, Result

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Largest image seen in the Smithsonian NMAH collection was 123 megapixels.
Image.MAX_IMAGE_PIXELS = 400_000_000

THUMBNAIL_EDGE = 400
THUMBNAIL_QUALITY = 80
PROCESSED_EDGE = 1920
PROCESSED_QUALITY = 90

# The models resize to 224 px anyway; a small copy keeps queued batches light.
MODEL_INPUT_EDGE = 448

# Headroom reserved per download on top of the reported file size.
RESERVE_PADDING_BYTES = 1 << 20
QUEUE_DEPTH_PER_WORKER = 2
QUEUE_DEPTH_PER_BATCH = 4
IDLE_POLL_SECONDS = 0.5
PROGRESS_INTERVAL_SECONDS = 2.0
WRITER_SHUTDOWN_SECONDS = 10
MAX_ERROR_LENGTH = 500
SHARD_PREFIX_LENGTH = 2


@dataclass
class Options:
    thumbnails_dir: Path
    processed_dir: Path
    data_dir: Path | None = None
    download_workers: int = 16
    decode_workers: int = 4
    batch_size: int = 32
    max_file_bytes: int = 200 * 1024 * 1024
    max_inflight_bytes: int = 2 * 1024**3


@dataclass
class Item:
    ref: FileRef
    item_id: str
    started: float = 0.0
    reserved: int = 0
    blob: bytes | None = None
    image: Image.Image | None = None
    metadata: dict = field(default_factory=dict)
    page_key: str | None = None
    document: Document | None = None


@dataclass
class Document:
    item: Item
    results: queue.Queue
    pages: int = 0
    unfinished: int = 0
    failed: int = 0
    error: str | None = None
    listed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def page_started(self) -> None:
        with self.lock:
            self.unfinished += 1

    def page_finished(self, ok: bool) -> None:
        with self.lock:
            self.unfinished -= 1
            self.failed += 0 if ok else 1
            is_last = self.listed and not self.unfinished
        if is_last:
            self.finish()

    def all_pages_started(self, error: str | None = None) -> None:
        with self.lock:
            self.listed = True
            self.error = error
            is_last = not self.unfinished
        if is_last:
            self.finish()

    def finish(self) -> None:
        error = self.error
        if not error and self.failed:
            error = f"{self.failed} of {self.pages} pages could not be indexed"
        if not error and not self.pages:
            error = "the PDF has no pages"
        self.results.put(
            Result(
                self.item.item_id,
                "failed" if error else "done",
                error,
                elapsed_ms(self.item),
                {
                    "type": "pdf",
                    "file_name": self.item.ref.basename,
                    "pages": self.pages,
                },
            )
        )


@dataclass
class Stats:
    total: int = 0
    done: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_in: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def finished(self) -> int:
        return self.done + self.skipped + self.failed


class ByteBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self.peak = 0
        self.changed = threading.Condition()

    def acquire(self, amount: int) -> int:
        amount = min(amount, self.limit)
        with self.changed:
            while self.used > 0 and self.used + amount > self.limit:
                self.changed.wait()
            self.used += amount
            self.peak = max(self.peak, self.used)
        return amount

    def release(self, amount: int) -> None:
        with self.changed:
            self.used -= amount
            self.changed.notify_all()


def make_item_id(key: str) -> str:
    return base64.urlsafe_b64encode(key.encode("utf-8")).decode("utf-8").rstrip("=")


def sharded_path(root: Path, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return root / digest[:SHARD_PREFIX_LENGTH] / f"{digest}.jpg"


def stored_path(path: Path, data_dir: Path | None) -> str:
    return str(path.relative_to(data_dir)) if data_dir else str(path)


def elapsed_ms(item: Item) -> int:
    return int((time.time() - item.started) * 1000)


def describe_error(stage: str, error: Exception) -> str:
    return f"{stage}: {type(error).__name__}: {error}"[:MAX_ERROR_LENGTH]


def failure(item: Item, stage: str, error: Exception) -> Result:
    return Result(
        item.item_id,
        "failed",
        describe_error(stage, error),
        elapsed_ms(item),
        page_key=item.page_key,
    )


def save_resized(image: Image.Image, edge: int, quality: int, path: Path) -> None:
    if max(image.size) > edge:
        image = image.copy()
        image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=quality)


def download_worker(
    source: Source,
    inbox: queue.Queue,
    outbox: queue.Queue,
    results: queue.Queue,
    budget: ByteBudget,
    stats: Stats,
) -> None:
    while True:
        item: Item | None = inbox.get()
        if item is None:
            return
        item.started = time.time()
        item.reserved = budget.acquire(item.ref.size + RESERVE_PADDING_BYTES)
        try:
            item.blob = source.read_bytes(item.ref)
            if item.blob is None:
                raise IOError("the file could not be read from the source")
            with stats.lock:
                stats.bytes_in += len(item.blob)
            outbox.put(item)
        except Exception as error:
            budget.release(item.reserved)
            results.put(failure(item, "download", error))


def catalog_metadata(catalog: dict | None) -> dict:
    if not catalog:
        return {}
    optional = {
        "model_url": model_url_from(catalog),
        "view_orbit": view_orbit_from(catalog),
        "iiif": iiif_info_from(catalog),
        **{
            name: str(catalog[name])
            for name in ("title", "description")
            if catalog.get(name)
        },
    }
    return {
        "catalog": catalog,
        "object_id": object_id_from(catalog),
        "source_url": source_url_from(catalog),
        **{name: value for name, value in optional.items() if value},
    }


def keep_copies(image: Image.Image, key: str, options: Options) -> dict:
    processed_path = sharded_path(options.processed_dir, key)
    thumbnail_path = sharded_path(options.thumbnails_dir, key)
    save_resized(image, PROCESSED_EDGE, PROCESSED_QUALITY, processed_path)
    save_resized(image, THUMBNAIL_EDGE, THUMBNAIL_QUALITY, thumbnail_path)
    return {
        "processed": stored_path(processed_path, options.data_dir),
        "thumbnail": stored_path(thumbnail_path, options.data_dir),
    }


def model_input(image: Image.Image) -> Image.Image:
    image.thumbnail((MODEL_INPUT_EDGE, MODEL_INPUT_EDGE), Image.Resampling.LANCZOS)
    return image


def decode_image(source: Source, options: Options, item: Item) -> None:
    image = Image.open(io.BytesIO(item.blob))
    width, height = image.size
    image.draft("RGB", (PROCESSED_EDGE, PROCESSED_EDGE))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.load()

    if max(image.size) > PROCESSED_EDGE:
        image.thumbnail((PROCESSED_EDGE, PROCESSED_EDGE), Image.Resampling.LANCZOS)
    copies = keep_copies(image, item.ref.key, options)

    original = source.original_uri(item.ref)
    item.metadata = {
        "file_name": item.ref.basename,
        "type": "image",
        "width": width,
        "height": height,
        "source_uri": original,
        "paths": {
            "original": original if os.path.exists(original) else copies["processed"],
            **copies,
        },
        **catalog_metadata(item.ref.extra.get("row")),
    }
    item.image = model_input(image)


def can_be_opened(address: str) -> bool:
    return address.startswith(("http://", "https://")) or os.path.exists(address)


def split_document(
    source: Source,
    options: Options,
    item: Item,
    outbox: queue.Queue,
    results: queue.Queue,
) -> None:
    document = Document(item, results)
    try:
        pdf = Pdf(item.blob)
    except Exception as error:
        document.all_pages_started(describe_error("decode", error))
        return
    document.pages = pdf.pages
    original = source.original_uri(item.ref)
    catalog = item.ref.extra.get("row")
    shared = {
        "file_name": item.ref.basename,
        "type": "pdf_page",
        "pages": pdf.pages,
        "source_uri": original,
        **({"document": original} if can_be_opened(original) else {}),
        "document_key": item.ref.key,
        **catalog_metadata(catalog),
    }
    error = None
    try:
        for index in range(pdf.pages):
            key = page_key(item.ref.key, index + 1)
            page = Item(
                ref=item.ref,
                item_id=make_item_id(key),
                started=item.started,
                page_key=key,
                document=document,
            )
            document.page_started()
            try:
                image = pdf.render(index, PROCESSED_EDGE)
                copies = keep_copies(image, key, options)
            except Exception as page_error:
                results.put(failure(page, "decode", page_error))
                document.page_finished(ok=False)
                continue
            page.metadata = {
                **shared,
                "page": index + 1,
                "width": image.width,
                "height": image.height,
                "paths": {"original": copies["processed"], **copies},
            }
            page.image = model_input(image)
            outbox.put(page)
    except Exception as stopped:
        error = describe_error("decode", stopped)
    finally:
        pdf.close()
        document.all_pages_started(error)


def decode_worker(
    source: Source,
    options: Options,
    inbox: queue.Queue,
    outbox: queue.Queue,
    results: queue.Queue,
    budget: ByteBudget,
) -> None:
    while True:
        item: Item | None = inbox.get()
        if item is None:
            return
        try:
            if item.ref.is_document:
                split_document(source, options, item, outbox, results)
            else:
                decode_image(source, options, item)
                outbox.put(item)
        except Exception as error:
            results.put(failure(item, "decode", error))
        finally:
            item.blob = None
            budget.release(item.reserved)


def encode_worker(
    service,
    options: Options,
    inbox: queue.Queue,
    results: queue.Queue,
    upstream_workers: int,
) -> None:
    batch: list[Item] = []
    finished_upstream = 0

    def flush():
        if not batch:
            return
        try:
            vectors = service.encode_image([item.image for item in batch]).numpy()
        except Exception as error:
            outcomes = [(item, failure(item, "encode", error)) for item in batch]
        else:
            outcomes = [
                (
                    item,
                    Result(
                        item.item_id,
                        "done",
                        None,
                        elapsed_ms(item),
                        item.metadata,
                        vector.astype(np.float32).tobytes(),
                        item.page_key,
                    ),
                )
                for item, vector in zip(batch, vectors)
            ]
        for item, result in outcomes:
            results.put(result)
            if item.document is not None:
                item.document.page_finished(result.status == "done")
        batch.clear()

    while finished_upstream < upstream_workers:
        try:
            item = inbox.get(timeout=IDLE_POLL_SECONDS)
        except queue.Empty:
            flush()
            continue
        if item is None:
            finished_upstream += 1
            continue
        batch.append(item)
        if len(batch) >= options.batch_size:
            flush()
    flush()


def write_worker(
    state: IngestState, results: queue.Queue, stats: Stats, stop: threading.Event
) -> None:
    while stats.finished < stats.total:
        try:
            result: Result = results.get(timeout=IDLE_POLL_SECONDS)
        except queue.Empty:
            if stop.is_set():
                break
            continue
        state.record(result)
        if result.page_key is not None:
            continue
        with stats.lock:
            if result.status == "done":
                stats.done += 1
            elif result.status == "skipped":
                stats.skipped += 1
            else:
                stats.failed += 1
    state.commit()


def clock(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def progress_worker(stats: Stats, budget: ByteBudget, stop: threading.Event) -> None:
    started = time.time()
    last_count, last_bytes, last_time = 0, 0, started
    while not stop.wait(PROGRESS_INTERVAL_SECONDS):
        now = time.time()
        count, received = stats.finished, stats.bytes_in
        window = max(now - last_time, 1e-6)
        elapsed = max(now - started, 1e-6)
        average_rate = count / elapsed
        remaining = (
            (stats.total - count) / average_rate if average_rate else float("inf")
        )
        sys.stdout.write(
            f"\r  {count:,}/{stats.total:,} ({100 * count / max(stats.total, 1):.1f}%) | "
            f"now {(count - last_count) / window:.2f} img/s "
            f"{(received - last_bytes) / 1024**2 / window:.1f} MB/s | "
            f"avg {average_rate:.2f} img/s {received / 1024**2 / elapsed:.1f} MB/s | "
            f"skipped {stats.skipped} failed {stats.failed} | "
            f"memory {budget.used / 1024**2:.0f} MB | "
            f"elapsed {clock(elapsed)} remaining {clock(remaining)}   "
        )
        sys.stdout.flush()
        last_count, last_bytes, last_time = count, received, now


def start(target, *args) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def run(
    source: Source, refs: list[FileRef], service, state: IngestState, options: Options
) -> tuple[Stats, ByteBudget]:
    stats = Stats(total=len(refs))
    budget = ByteBudget(options.max_inflight_bytes)
    to_download: queue.Queue = queue.Queue(
        options.download_workers * QUEUE_DEPTH_PER_WORKER
    )
    to_decode: queue.Queue = queue.Queue(
        options.download_workers * QUEUE_DEPTH_PER_WORKER
    )
    to_encode: queue.Queue = queue.Queue(options.batch_size * QUEUE_DEPTH_PER_BATCH)
    results: queue.Queue = queue.Queue()
    stop = threading.Event()

    writer = start(write_worker, state, results, stats, stop)
    downloaders = [
        start(download_worker, source, to_download, to_decode, results, budget, stats)
        for _ in range(options.download_workers)
    ]
    decoders = [
        start(decode_worker, source, options, to_decode, to_encode, results, budget)
        for _ in range(options.decode_workers)
    ]
    encoder = start(encode_worker, service, options, to_encode, results, len(decoders))
    reporter = start(progress_worker, stats, budget, stop)

    try:
        for ref in refs:
            item = Item(ref=ref, item_id=make_item_id(ref.key))
            if ref.size > options.max_file_bytes:
                limit_mb = options.max_file_bytes / 1024**2
                results.put(
                    Result(
                        item.item_id,
                        "skipped",
                        f"{ref.size / 1024**2:.1f} MB is over the {limit_mb:.0f} MB limit",
                    )
                )
            else:
                to_download.put(item)
        for _ in downloaders:
            to_download.put(None)
        for thread in downloaders:
            thread.join()
        for _ in decoders:
            to_decode.put(None)
        for thread in decoders:
            thread.join()
        for _ in decoders:
            to_encode.put(None)
        encoder.join()
        writer.join()
    except KeyboardInterrupt:
        stop.set()
        writer.join(timeout=WRITER_SHUTDOWN_SECONDS)
        raise
    finally:
        stop.set()
        reporter.join(timeout=PROGRESS_INTERVAL_SECONDS + 1)
    return stats, budget
