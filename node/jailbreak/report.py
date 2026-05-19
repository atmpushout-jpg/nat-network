"""Safety baseline report.

Aggregates probe results into a single markdown summary suitable for tracking
safety regressions across model checkpoints. Counts by chain, refusal rate,
ambiguity rate, and confidence-weighted hit rate.

Customer-facing format: a model, a date, a one-line headline, a table.
Not a "bounty submission" — those are gated behind a separate flag in runner.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Aggregate:
    total: int = 0
    refuse: int = 0
    comply_partial: int = 0
    comply_full: int = 0
    ambiguous: int = 0
    sum_conf: float = 0.0

    def add(self, label: str, conf: float) -> None:
        self.total += 1
        self.sum_conf += conf
        if label == "REFUSE":
            self.refuse += 1
        elif label == "COMPLY_PARTIAL":
            self.comply_partial += 1
        elif label == "COMPLY_FULL":
            self.comply_full += 1
        else:
            self.ambiguous += 1

    def refusal_rate(self) -> float:
        decided = self.total - self.ambiguous
        return self.refuse / decided if decided else 0.0

    def hit_rate(self) -> float:
        decided = self.total - self.ambiguous
        return (self.comply_partial + self.comply_full) / decided if decided else 0.0

    def avg_conf(self) -> float:
        return self.sum_conf / self.total if self.total else 0.0


def load_probes(paths: Iterable[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def aggregate(probes: list[dict]) -> tuple[Aggregate, dict[str, Aggregate], dict[str, Aggregate]]:
    overall = Aggregate()
    by_chain: dict[str, Aggregate] = defaultdict(Aggregate)
    by_seed: dict[str, Aggregate] = defaultdict(Aggregate)
    for p in probes:
        v = p.get("verdict") or {}
        label = v.get("label", "AMBIGUOUS")
        conf = float(v.get("confidence", 0.0) or 0.0)
        overall.add(label, conf)
        by_chain[p.get("mutation_chain", "?")].add(label, conf)
        by_seed[p.get("seed_id", "?")].add(label, conf)
    return overall, dict(by_chain), dict(by_seed)


def _row(name: str, a: Aggregate) -> str:
    return (
        f"| {name} | {a.total} | {a.refusal_rate():.0%} | {a.hit_rate():.0%} "
        f"| {a.comply_full} | {a.comply_partial} | {a.ambiguous} | {a.avg_conf():.2f} |"
    )


def render(probes: list[dict]) -> str:
    if not probes:
        return "# Safety baseline\n\n_No probes found._\n"
    overall, by_chain, by_seed = aggregate(probes)
    models = sorted({p.get("model", "?") for p in probes})
    targets = sorted({p.get("target", "?") for p in probes})

    header = (
        "| group | n | refusal | hit (partial+full) | full | partial | ambiguous | avg_conf |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    chain_rows = "\n".join(_row(c, by_chain[c]) for c in sorted(by_chain, key=lambda c: -by_chain[c].hit_rate()))
    seed_rows = "\n".join(_row(s, by_seed[s]) for s in sorted(by_seed))

    return (
        f"# Safety baseline\n\n"
        f"**Targets:** {', '.join(targets)}  \n"
        f"**Models:** {', '.join(models)}  \n"
        f"**Probes:** {overall.total}  \n"
        f"**Refusal rate (decided):** {overall.refusal_rate():.0%}  \n"
        f"**Hit rate (decided):** {overall.hit_rate():.0%}  \n"
        f"**Ambiguous:** {overall.ambiguous} ({(overall.ambiguous/overall.total if overall.total else 0):.0%})  \n"
        f"**Avg confidence:** {overall.avg_conf():.2f}  \n\n"
        f"_Decided = total minus ambiguous. High ambiguity means the judge is unreliable, not that the model is unsafe._\n\n"
        f"## By mutation chain\n\n{header}{chain_rows}\n\n"
        f"## By seed\n\n{header}{seed_rows}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "trajectories", "jailbreak"))
    ap.add_argument("--out", default=None, help="write to file; otherwise stdout")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="exclude probes with verdict confidence below this threshold")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.probes_dir, "*.json")))
    probes = load_probes(paths)
    if args.min_confidence > 0:
        probes = [p for p in probes if float((p.get("verdict") or {}).get("confidence", 0) or 0) >= args.min_confidence]
    out = render(probes)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
