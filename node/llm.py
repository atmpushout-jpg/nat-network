from __future__ import annotations

import json
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def ollama_generate(prompt: str, model: str = "qwen2.5-coder:1.5b", timeout: int = 600) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data.get("response", "")
