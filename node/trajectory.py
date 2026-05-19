from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    role: str
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    ts: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    task_id: str
    task_source: str
    task_domain: str
    task_prompt: str
    expected_answer: str | None
    model: str
    node_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    grade: dict[str, Any] | None = None
    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add_step(self, **kwargs: Any) -> None:
        self.steps.append(Step(**kwargs))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def save(self, root: str) -> str:
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"{self.trajectory_id}.json")
        with open(path, "w") as f:
            f.write(self.to_json())
        return path
