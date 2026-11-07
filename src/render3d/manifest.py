from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .models import IMAGE_COLUMN, SIZE_COLUMN


def text(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def column_of(rows: list[dict], column: str) -> pd.Series:
    values = [row.get(column) for row in rows]
    if column == SIZE_COLUMN:
        return pd.to_numeric(pd.Series(values), errors="coerce").astype("Int64")
    return pd.Series([text(value) for value in values], dtype="string")


def write_manifest(path: Path, rows: list[dict]) -> int:
    import duckdb

    columns = list(dict.fromkeys([IMAGE_COLUMN, *(k for row in rows for k in row)]))
    frame = pd.DataFrame({column: column_of(rows, column) for column in columns})
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    con = duckdb.connect()
    con.register("manifest", frame)
    quoted = str(partial).replace("'", "''")
    con.execute(f"COPY manifest TO '{quoted}' (FORMAT PARQUET)")
    con.close()
    partial.replace(path)
    return len(frame)
