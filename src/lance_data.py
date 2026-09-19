from __future__ import annotations

from pathlib import Path
from typing import Iterator

import lance


def split_path(dataset_dir: str | Path, split: str) -> Path:
    root = Path(dataset_dir)
    candidates = [
        root / "data" / f"{split}.lance",
        root / f"{split}.lance",
        root / "data" / split,
        root / split,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {split}.lance under {root}. "
        f"Expected {root / 'data' / (split + '.lance')}"
    )


def iter_rows(dataset_dir: str | Path, split: str, limit: int | None = None) -> Iterator[dict]:
    path = split_path(dataset_dir, split)
    try:
        dataset = lance.dataset(str(path))
        scanner = dataset.scanner(columns=["id", "image", "medicine_name", "generic_name"])
    except Exception as exc:
        raise RuntimeError(f"Could not open Lance {split!r} dataset at {path}: {exc}") from exc
    table = scanner.to_table()
    rows = table.to_pylist()
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        if not row.get("id") or not row.get("medicine_name") or row.get("image") is None:
            raise ValueError(f"Malformed row {index} in {path}: expected id, image, and medicine_name")
        yield row


def load_rows(dataset_dir: str | Path, split: str, limit: int | None = None) -> list[dict]:
    return list(iter_rows(dataset_dir, split, limit))


def get_labels(dataset_dir: str | Path) -> list[str]:
    labels: set[str] = set()
    for row in iter_rows(dataset_dir, "train"):
        label = str(row["medicine_name"]).strip()
        if label:
            labels.add(label)
    return sorted(labels, key=str.casefold)


def labels_from_rows(rows: list[dict]) -> list[str]:
    """Sorted unique medicine names found in already-loaded rows."""
    return sorted({str(row["medicine_name"]).strip() for row in rows if str(row["medicine_name"]).strip()}, key=str.casefold)
