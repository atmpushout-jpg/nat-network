"""Measure judge reliability against the hand-labeled test set.

Usage:
    python -m scripts.judge_accuracy            # full: calls the LLM judge (needs Ollama)
    python -m scripts.judge_accuracy --offline  # heuristics + reconciliation only, no LLM call

Reports overall accuracy, a confusion matrix, and every disagreement so you can
see exactly where the judge is wrong. This is the number you quote to a customer
when you claim the eval is reliable — and the baseline you improve against.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from node.jailbreak.judge_testset import TEST_CASES, by_label  # noqa: E402
from node.jailbreak.scorer import (  # noqa: E402
    _heuristic_filler,
    _heuristic_operational,
    _heuristic_refusal,
    _reconcile,
    judge,
)

LABELS = ["REFUSE", "COMPLY_PARTIAL", "COMPLY_FULL", "AMBIGUOUS"]


def predict_offline(behavior: str, response: str) -> tuple[str, float]:
    """Heuristics + reconciliation only — no LLM judge call. Fast, runs anywhere."""
    heur_refuse = _heuristic_refusal(response)
    filler = _heuristic_filler(response)
    operational = _heuristic_operational(response)
    label, conf, _ = _reconcile(None, heur_refuse, filler, operational)
    return label, conf


def predict_full(behavior: str, response: str) -> tuple[str, float]:
    v = judge(behavior, response)
    return v.label, v.confidence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="Test heuristics + reconciliation only; skip the LLM judge call")
    args = ap.parse_args()

    predict = predict_offline if args.offline else predict_full
    mode = "OFFLINE (heuristics only)" if args.offline else "FULL (heuristics + LLM judge)"

    print(f"Judge accuracy — mode: {mode}")
    print(f"Test set: {len(TEST_CASES)} cases — {by_label()}")
    print("-" * 78)

    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    correct = 0
    disagreements = []

    for c in TEST_CASES:
        pred, conf = predict(c.behavior, c.response)
        if pred not in LABELS:
            pred = "AMBIGUOUS"
        confusion[c.gold][pred] += 1
        if pred == c.gold:
            correct += 1
        else:
            disagreements.append((c.case_id, c.gold, pred, conf, c.note))

    total = len(TEST_CASES)
    acc = correct / total if total else 0.0
    print(f"\nAccuracy: {correct}/{total} = {acc:.1%}\n")

    # Confusion matrix
    print("Confusion matrix (rows = gold, cols = predicted):")
    hdr = "  gold \\ pred  | " + " | ".join(f"{l[:8]:>8}" for l in LABELS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for g in LABELS:
        row = " | ".join(f"{confusion[g][p]:>8}" for p in LABELS)
        print(f"  {g:<13}| {row}")

    # Per-class recall
    print("\nPer-class recall:")
    for g in LABELS:
        gtotal = sum(confusion[g].values())
        r = confusion[g][g] / gtotal if gtotal else 0.0
        print(f"  {g:<15} {confusion[g][g]}/{gtotal} = {r:.0%}")

    # Disagreements
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for cid, gold, pred, conf, note in disagreements:
            print(f"  {cid:<12} gold={gold:<15} pred={pred:<15} conf={conf:.2f}  ({note})")
    else:
        print("\nNo disagreements — judge matches gold on every case.")

    print()
    if acc < 0.8:
        print("VERDICT: not customer-ready. Accuracy below 80% — fix before selling reports.")
    elif acc < 0.95:
        print("VERDICT: usable internally, tighten before customer-facing reports.")
    else:
        print("VERDICT: solid. Re-run after any scorer change to catch regressions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
