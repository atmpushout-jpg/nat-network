"""Submit jobs to the orchestrator from AdvBench seeds × default chains.

Example:
    python -m orchestrator.submit --server http://127.0.0.1:8765 --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from node.jailbreak.mutator import default_chains  # noqa: E402
from node.jailbreak.seeds import advbench  # noqa: E402


def submit(server: str, behavior: str, chain: list[str], seed_id: str) -> str:
    payload = json.dumps({"behavior": behavior, "chain": chain, "seed_id": seed_id}).encode()
    req = urllib.request.Request(f"{server}/jobs/submit", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())["job_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("ORCH_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    chains = default_chains()
    n = 0
    for seed in advbench(limit=args.limit):
        for chain in chains:
            jid = submit(args.server, seed.behavior, list(chain.names), seed.seed_id)
            n += 1
            print(f"submitted {jid[:8]} {seed.seed_id} {chain.label}")
    print(f"total submitted: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
