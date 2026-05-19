from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Iterable

from .grader import grade_math
from .llm import ollama_generate
from .task_sources import Task, gsm8k
from .trajectory import Trajectory

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5-coder:1.5b")
TRAJECTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "trajectories")

COT_TEMPLATE = """You are a careful math problem solver.

Solve this problem step by step. After your reasoning, write a single line:
ANSWER: <number>

Problem:
{question}
"""


def run_one(task: Task, model: str, node_id: str) -> Trajectory:
    traj = Trajectory(
        task_id=task.task_id,
        task_source=task.source,
        task_domain=task.domain,
        task_prompt=task.prompt,
        expected_answer=task.expected_answer,
        model=model,
        node_id=node_id,
    )
    prompt = COT_TEMPLATE.format(question=task.prompt)
    traj.add_step(role="user", content=prompt)
    response = ollama_generate(prompt, model=model)
    traj.add_step(role="assistant", content=response)
    traj.final_answer = response.strip()
    traj.grade = grade_math(response, task.expected_answer or "")
    import time
    traj.finished_at = time.time()
    return traj


def run_batch(tasks: Iterable[Task], model: str, node_id: str) -> list[Trajectory]:
    results: list[Trajectory] = []
    for t in tasks:
        sys.stdout.write(f"[{node_id}] {t.task_id} ... ")
        sys.stdout.flush()
        try:
            traj = run_one(t, model=model, node_id=node_id)
            path = traj.save(TRAJECTORY_DIR)
            grade = traj.grade or {}
            sys.stdout.write(
                f"{'OK' if grade.get('correct') else 'WRONG'} "
                f"(pred={grade.get('predicted')} exp={grade.get('expected')}) -> {os.path.basename(path)}\n"
            )
            results.append(traj)
        except Exception as e:
            sys.stdout.write(f"ERROR: {e}\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="gsm8k")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--node-id", default=socket.gethostname())
    args = parser.parse_args()

    if args.source == "gsm8k":
        tasks = list(gsm8k(limit=args.limit))
    else:
        print(f"unknown source: {args.source}", file=sys.stderr)
        return 2

    results = run_batch(tasks, model=args.model, node_id=args.node_id)
    correct = sum(1 for t in results if (t.grade or {}).get("correct"))
    print(f"\n[{args.node_id}] score: {correct}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
