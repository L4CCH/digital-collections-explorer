from __future__ import annotations

import json
import math
from pathlib import Path

from .models import IMAGE_COLUMN, SIZE_COLUMN


def text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def size(value) -> int | None:
    try:
        return None if text(value) is None else int(float(value))
    except (TypeError, ValueError):
        return None


def type_of(rows: list[dict], column: str) -> str:
    if column == SIZE_COLUMN:
        return "BIGINT"
    listed = any(isinstance(row.get(column), (list, tuple)) for row in rows)
    return "VARCHAR[]" if listed else "VARCHAR"


def cell(value, kind: str):
    if kind == "BIGINT":
        return size(value)
    if kind == "VARCHAR":
        return text(value)
    values = value if isinstance(value, (list, tuple)) else [value]
    kept = [text(v) for v in values if text(v) is not None]
    return kept or None


def quoted(name: str, mark: str) -> str:
    return mark + name.replace(mark, mark * 2) + mark


def write_manifest(path: Path, rows: list[dict]) -> int:
    import duckdb

    columns = list(dict.fromkeys([IMAGE_COLUMN, *(k for row in rows for k in row)]))
    kinds = {column: type_of(rows, column) for column in columns}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    lines = path.with_name(path.stem + ".partial.jsonl")
    with lines.open("w") as handle:
        for row in rows:
            cells = {c: cell(v, kinds[c]) for c, v in row.items()}
            kept = {c: v for c, v in cells.items() if v is not None}
            handle.write(json.dumps(kept, ensure_ascii=False) + "\n")
    typed = ", ".join(f"{quoted(c, chr(39))}: '{kind}'" for c, kind in kinds.items())
    chosen = ", ".join(quoted(c, '"') for c in columns)
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT {chosen} FROM read_json({quoted(str(lines), chr(39))}, "
        f"columns = {{{typed}}}, format = 'newline_delimited')) "
        f"TO {quoted(str(partial), chr(39))} (FORMAT PARQUET)"
    )
    con.close()
    lines.unlink()
    partial.replace(path)
    return len(rows)
