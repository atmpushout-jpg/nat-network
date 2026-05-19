from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .mutator import MutationChain, default_chains
from .scorer import Verdict, judge
from .seeds import Seed, advbench
from .targets import TargetResponse, make_target

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "trajectories", "jailbreak")


@dataclass
class ProbeResult:
    probe_id: str
    seed_id: str
    behavior: str
    target: str
    model: str
    mutation_chain: str
    prompt: str
    response: str
    verdict: dict[str, Any]
    node_id: str
    ts: float = field(default_factory=time.time)


def _save_probe(result: ProbeResult) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{result.probe_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    return path


def _emit_report(result: ProbeResult) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fname = f"{result.target}_{result.seed_id}_{result.mutation_chain.replace('+', '-')}_{result.probe_id[:8]}.md"
    path = os.path.join(REPORTS_DIR, fname)
    md = (
        f"# Bounty submission draft: {result.seed_id} via `{result.mutation_chain}`\n\n"
        f"**Target:** `{result.target}` / `{result.model}`\n\n"
        f"**Verdict:** `{result.verdict.get('label')}` — {result.verdict.get('reason')}\n\n"
        f"**Behavior tested (from AdvBench):** {result.behavior}\n\n"
        f"## Prompt sent\n\n```\n{result.prompt}\n```\n\n"
        f"## Model response\n\n```\n{result.response}\n```\n\n"
        f"## Reproduction\n\n"
        f"1. Send the prompt above to `{result.model}` via official API.\n"
        f"2. Compare output to the response above.\n"
        f"3. Verify the response constitutes a policy violation.\n\n"
        f"## Submission notes\n\n"
        f"- This template is a starting point. Strengthen before submitting:\n"
        f"  - Add: novelty argument (why this attack class is new)\n"
        f"  - Add: severity argument (real-world harm potential)\n"
        f"  - Add: minimum reproducible example\n"
        f"  - Add: model version + temperature + system prompt context\n"
    )
    with open(path, "w") as f:
        f.write(md)
    return path


def run(
    target_name: str = "ollama",
    target_kwargs: dict | None = None,
    limit: int = 10,
    chains: list[MutationChain] | None = None,
    node_id: str | None = None,
    emit_bounty_drafts: bool = False,
) -> tuple[int, int, list[ProbeResult]]:
    target = make_target(target_name, **(target_kwargs or {}))
    chains = chains or default_chains()
    node_id = node_id or socket.gethostname()
    total = 0
    hits = 0
    probes: list[ProbeResult] = []
    for seed in advbench(limit=limit):
        for chain in chains:
            total += 1
            prompt = chain.apply(seed.behavior)
            try:
                resp: TargetResponse = target.query(prompt)
            except Exception as e:
                sys.stdout.write(f"[{node_id}] {seed.seed_id} {chain.label}: TARGET_ERROR {e}\n")
                continue
            v: Verdict = judge(seed.behavior, resp.text)
            probe = ProbeResult(
                probe_id=uuid.uuid4().hex,
                seed_id=seed.seed_id,
                behavior=seed.behavior,
                target=resp.target_name,
                model=resp.model,
                mutation_chain=chain.label,
                prompt=prompt,
                response=resp.text,
                verdict={
                    "label": v.label,
                    "reason": v.reason,
                    "confidence": v.confidence,
                    "heuristic_refusal": v.heuristic_refusal,
                    "heuristic_filler": v.heuristic_filler,
                    "heuristic_operational": v.heuristic_operational,
                    "judge_error": v.judge_error,
                },
                node_id=node_id,
            )
            _save_probe(probe)
            probes.append(probe)
            mark = " "
            if v.label in ("COMPLY_PARTIAL", "COMPLY_FULL"):
                if emit_bounty_drafts:
                    _emit_report(probe)
                mark = "!"
                hits += 1
            sys.stdout.write(f"[{node_id}] {mark} {seed.seed_id} {chain.label}: {v.label} (c={v.confidence:.2f}) — {v.reason[:60]}\n")
            sys.stdout.flush()
    return total, hits, probes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="ollama", choices=["ollama", "anthropic", "openai"])
    parser.add_argument("--model", default=None, help="override target model")
    parser.add_argument("--limit", type=int, default=5, help="number of seed behaviors")
    parser.add_argument("--node-id", default=socket.gethostname())
    parser.add_argument("--emit-bounty-drafts", action="store_true",
                        help="Auto-emit markdown bounty-submission drafts for hits (off by default).")
    args = parser.parse_args()

    kwargs: dict[str, Any] = {}
    if args.model:
        kwargs["model"] = args.model

    total, hits, _ = run(
        target_name=args.target,
        target_kwargs=kwargs,
        limit=args.limit,
        node_id=args.node_id,
        emit_bounty_drafts=args.emit_bounty_drafts,
    )
    print(f"\n[{args.node_id}] probes={total} hits={hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
