from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterator

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "tasks", "cache")


@dataclass
class Task:
    task_id: str
    source: str
    domain: str
    prompt: str
    expected_answer: str


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    urllib.request.urlretrieve(url, dest)


def _extract_gsm8k_answer(raw: str) -> str:
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", raw)
    return m.group(1).replace(",", "") if m else raw.strip()


def gsm8k(limit: int | None = None) -> Iterator[Task]:
    path = _cache_path("gsm8k_test.jsonl")
    _download(GSM8K_URL, path)
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            yield Task(
                task_id=f"gsm8k-{i}",
                source="gsm8k",
                domain="reasoning_math",
                prompt=row["question"],
                expected_answer=_extract_gsm8k_answer(row["answer"]),
            )
