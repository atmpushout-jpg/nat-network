from __future__ import annotations

import csv
import io
import os
import urllib.request
from dataclasses import dataclass
from typing import Iterator

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tasks", "cache")


@dataclass
class Seed:
    seed_id: str
    behavior: str
    expected_refusal: str | None = None
    source: str = "advbench"


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    urllib.request.urlretrieve(url, dest)


def advbench(limit: int | None = None) -> Iterator[Seed]:
    path = _cache_path("advbench_harmful_behaviors.csv")
    _download(ADVBENCH_URL, path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield Seed(
                seed_id=f"advbench-{i}",
                behavior=row.get("goal", "").strip(),
                expected_refusal=row.get("target", "").strip() or None,
            )
