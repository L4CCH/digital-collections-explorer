from __future__ import annotations

import importlib.util
import threading

from PIL import Image

READER_PACKAGE = "pypdfium2"
INSTALL_HINT = "pip install -r requirements-pdf.txt"

ONE_PDFIUM_CALL_AT_A_TIME = threading.Lock()


def can_read_pdfs() -> bool:
    return importlib.util.find_spec(READER_PACKAGE) is not None


def page_key(document_key: str, number: int) -> str:
    return f"{document_key}#page={number}"


class Pdf:
    def __init__(self, blob: bytes):
        import pypdfium2

        with ONE_PDFIUM_CALL_AT_A_TIME:
            self.document = pypdfium2.PdfDocument(blob)
            self.pages = len(self.document)

    def render(self, index: int, edge: int) -> Image.Image:
        with ONE_PDFIUM_CALL_AT_A_TIME:
            page = self.document[index]
            try:
                width, height = page.get_size()
                scale = edge / max(width, height, 1.0)
                return page.render(scale=scale).to_pil().convert("RGB")
            finally:
                page.close()

    def close(self) -> None:
        with ONE_PDFIUM_CALL_AT_A_TIME:
            self.document.close()
