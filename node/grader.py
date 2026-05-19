from __future__ import annotations

import re
from typing import Any


_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _last_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _normalize(s: str) -> str:
    return s.replace(",", "").strip().rstrip(".")


def grade_math(predicted: str, expected: str) -> dict[str, Any]:
    pred_num = _last_number(predicted or "")
    exp_norm = _normalize(expected)
    try:
        correct = pred_num is not None and float(pred_num) == float(exp_norm)
    except ValueError:
        correct = (pred_num or "") == exp_norm
    return {
        "grader": "math_last_number",
        "correct": bool(correct),
        "predicted": pred_num,
        "expected": exp_norm,
    }
