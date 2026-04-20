from io import StringIO
from typing import Any
import csv


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        rows: dict[str, Any] = {}
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            rows.update(_flatten(value, next_prefix))
        return rows
    if isinstance(data, list):
        return {prefix: "; ".join(str(item) for item in data)}
    return {prefix: data}


def dict_to_csv(data: dict[str, Any]) -> str:
    flat = _flatten(data)
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(flat.keys()))
    writer.writeheader()
    writer.writerow(flat)
    return stream.getvalue()

