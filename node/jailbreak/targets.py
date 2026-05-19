from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class TargetResponse:
    target_name: str
    model: str
    text: str
    raw: dict | None = None


class Target(Protocol):
    name: str

    def query(self, prompt: str) -> TargetResponse:
        ...


class OllamaTarget:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5-coder:1.5b", url: str = "http://127.0.0.1:11434/api/generate", timeout: int = 180):
        self.model = model
        self.url = url
        self.timeout = timeout

    def query(self, prompt: str) -> TargetResponse:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        return TargetResponse(target_name=self.name, model=self.model, text=data.get("response", ""), raw=data)


class AnthropicTarget:
    """Use only against Anthropic models you are authorized to test
    (e.g., via HackerOne bug bounty program)."""
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7", api_key_env: str = "ANTHROPIC_API_KEY", timeout: int = 60):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        self.timeout = timeout

    def query(self, prompt: str) -> TargetResponse:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Only use this target against models you are "
                "authorized to test (e.g., HackerOne bounty program)."
            )
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return TargetResponse(target_name=self.name, model=self.model, text=text, raw=data)


class OpenAITarget:
    """Use only against OpenAI models you are authorized to test
    (e.g., via Bugcrowd bug bounty program)."""
    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key_env: str = "OPENAI_API_KEY", timeout: int = 60):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        self.timeout = timeout

    def query(self, prompt: str) -> TargetResponse:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Only use this target against models you are "
                "authorized to test (e.g., Bugcrowd bounty program)."
            )
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        text = data["choices"][0]["message"]["content"]
        return TargetResponse(target_name=self.name, model=self.model, text=text, raw=data)


TARGETS: dict[str, type[Target]] = {
    "ollama": OllamaTarget,
    "anthropic": AnthropicTarget,
    "openai": OpenAITarget,
}


def make_target(name: str, **kwargs) -> Target:
    cls = TARGETS[name]
    return cls(**kwargs)
