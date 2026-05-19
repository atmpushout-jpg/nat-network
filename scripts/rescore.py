"""Re-score existing jailbreak probe JSONs with the current scorer.

Usage:
    python -m scripts.rescore [--update] [path...]

Without --update, prints a comparison table. With --update, rewrites the verdict
field in-place using the new heuristics (judge call is skipped — we only re-run
the post-judge reconciliation on the saved response text).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from node.jailbreak.scorer import (  # noqa: E402
    _heuristic_filler,
    _heuristic_operational,
    _heuristic_refusal,
    _parse_judge,
    _reconcile,
)


def rescore(probe: dict) -> dict:
    response = probe.get("response", "")
    heur_refuse = _heuristic_refusal(response)
    filler = _heuristic_filler(response)
    operational = _heuristic_operational(response)
    # Reuse the previously saved judge text if present in the old verdict reason.
    # We don't have raw judge text in old probes, so feed reason as a best-effort signal.
    judge_label, _ = _parse_judge(probe.get("verdict", {}).get("reason", ""))
    label, confidence, note = _reconcile(judge_label, heur_refuse, filler, operational)
    return {
        "label": label,
        "reason": note,
        "confidence": confidence,
        "heuristic_refusal": heur_refuse,
        "heuristic_filler": filler,
        "heuristic_operational": operational,
        "judge_error": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="Rewrite verdict in-place")
    ap.add_argument("paths", nargs="*", help="Probe JSON paths (default: all)")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob(os.path.join(ROOT, "trajectories", "jailbreak", "*.json")))
    if not paths:
        print("no probes found", file=sys.stderr)
        return 1

    fmt = "{:<10} {:<35} {:<14} -> {:<14} {:<5}  {}"
    print(fmt.format("seed", "chain", "old", "new", "conf", "note"))
    print("-" * 100)
    for p in paths:
        with open(p) as f:
            probe = json.load(f)
        new_v = rescore(probe)
        old_label = (probe.get("verdict") or {}).get("label", "?")
        print(fmt.format(
            probe.get("seed_id", "?"),
            probe.get("mutation_chain", "?"),
            old_label,
            new_v["label"],
            f"{new_v['confidence']:.2f}",
            new_v["reason"],
        ))
        if args.update:
            probe["verdict"] = new_v
            with open(p, "w") as f:
                json.dump(probe, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
