"""Worker: pulls jobs from the orchestrator, runs target+judge, posts result back.

Run one per inference node (e.g., one per Mac mini, one per workstation).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# In-memory ring buffer of recently-completed contributions for the chat UI.
# Worker thread appends; HTTP thread reads. deque.append is thread-safe in CPython.
RECENT_CONTRIBS: "deque[dict]" = deque(maxlen=50)
CONTRIB_COUNTS = {"REFUSE": 0, "COMPLY_PARTIAL": 0, "COMPLY_FULL": 0, "AMBIGUOUS": 0, "ERROR": 0}

# Support both layouts:
#   1) repo checkout:   agent-farm/orchestrator/worker.py + agent-farm/node/...
#   2) bootstrap host:  $WORKDIR/worker.py + $WORKDIR/node/...
_HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (_HERE, os.path.join(_HERE, "..")):
    if os.path.isdir(os.path.join(candidate, "node")):
        sys.path.insert(0, candidate)
        break

from node.jailbreak.mutator import MutationChain  # noqa: E402
from node.jailbreak.scorer import judge  # noqa: E402
from node.jailbreak.targets import make_target  # noqa: E402


OLLAMA_BASE = "http://127.0.0.1:11434"
NAT_AI_PORT_DEFAULT = 11500


def _start_nat_ai_server(workdir: str, port: int) -> None:
    """Spawn an HTTP server (in this thread) that serves chat.html and proxies /api/* to local Ollama.

    Runs on 127.0.0.1 only — never exposed beyond the loopback interface.
    Same-origin with the chat UI, so no CORS friction with Ollama.
    """
    chat_path = os.path.join(workdir, "chat.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _serve_chat(self):
            if not os.path.exists(chat_path):
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"chat.html not present in workdir")
                return
            with open(chat_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self, method: str):
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n) if n else None
            url = OLLAMA_BASE + self.path
            req = urllib.request.Request(url, data=data, method=method)
            if self.headers.get("Content-Type"):
                req.add_header("Content-Type", self.headers["Content-Type"])
            try:
                resp = urllib.request.urlopen(req, timeout=900)
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(e.read())
                return
            except Exception as e:
                print(f"[nat-ai proxy] error talking to Ollama: {type(e).__name__}: {e}", file=sys.stderr)
                self.send_response(502)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"proxy error: {type(e).__name__}: {e}".encode())
                return
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() in ("transfer-encoding", "connection", "content-length"):
                    continue
                self.send_header(k, v)
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self):
            if self.path in ("/", "/chat", "/chat.html"):
                return self._serve_chat()
            if self.path == "/contributions":
                body = json.dumps({
                    "recent": list(RECENT_CONTRIBS),
                    "counts": CONTRIB_COUNTS,
                    "total": sum(CONTRIB_COUNTS.values()),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/api/"):
                return self._proxy("GET")
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path.startswith("/api/"):
                return self._proxy("POST")
            self.send_response(404)
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    try:
        srv.serve_forever()
    except Exception:
        pass


def _http(url: str, method: str = "GET", body: dict | None = None, headers: dict | None = None, timeout: int = 30) -> tuple[int, bytes]:
    """Make an HTTP request. Returns (status, body). Network errors return (0, b'<reason>')
    so the worker can backoff instead of crashing when the orchestrator is briefly down."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}".encode()


def run(server: str, target_name: str, target_kwargs: dict, worker_id: str, poll_interval: float, share_content: bool = False, nat_ai_port: int = NAT_AI_PORT_DEFAULT) -> None:
    headers = {"X-Worker-Id": worker_id}
    target = make_target(target_name, **target_kwargs)
    privacy_mode = "FULL" if share_content else "MINIMAL"
    # Spin up the local NAT AI server (chat UI + Ollama proxy) on a daemon thread.
    if nat_ai_port:
        t = threading.Thread(target=_start_nat_ai_server, args=(_HERE, nat_ai_port), daemon=True)
        t.start()
        print(f"[{worker_id}] NAT AI chat ready at http://127.0.0.1:{nat_ai_port}/")
    print(f"[{worker_id}] worker started target={target_name} share={privacy_mode}")
    backoff = poll_interval
    while True:
        status, body = _http(f"{server}/jobs/next", headers=headers)
        if status == 0:
            # Network error — orchestrator down, retry with backoff up to 30s
            print(f"[{worker_id}] orchestrator unreachable: {body[:120].decode(errors='replace')}, retrying in {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        backoff = poll_interval  # reset on success
        if status == 204:
            time.sleep(poll_interval)
            continue
        if status != 200:
            print(f"[{worker_id}] poll error {status} {body[:200]!r}", file=sys.stderr)
            time.sleep(poll_interval)
            continue
        job = json.loads(body)
        chain = MutationChain(tuple(job["chain"]))
        prompt = chain.apply(job["behavior"])
        try:
            resp = target.query(prompt)
            v = judge(job["behavior"], resp.text)
            probe = {
                "probe_id": uuid.uuid4().hex,
                "job_id": job["job_id"],
                "seed_id": job["seed_id"],
                "target": resp.target_name,
                "model": resp.model,
                "mutation_chain": chain.label,
                "verdict": {
                    "label": v.label,
                    "reason": v.reason,
                    "confidence": v.confidence,
                    "heuristic_refusal": v.heuristic_refusal,
                    "heuristic_filler": v.heuristic_filler,
                    "heuristic_operational": v.heuristic_operational,
                    "judge_error": v.judge_error,
                },
                "node_id": worker_id,
                "share_mode": privacy_mode,
            }
            if share_content:
                probe["behavior"] = job["behavior"]
                probe["prompt"] = prompt
                probe["response"] = resp.text
            _http(f"{server}/jobs/{job['job_id']}/result", method="POST", body=probe)
            print(f"[{worker_id}] {job['seed_id']} {chain.label}: {v.label} (c={v.confidence:.2f})")
            RECENT_CONTRIBS.append({
                "ts": time.time(),
                "seed": job["seed_id"],
                "chain": chain.label,
                "label": v.label,
                "confidence": round(v.confidence, 2),
            })
            CONTRIB_COUNTS[v.label] = CONTRIB_COUNTS.get(v.label, 0) + 1
        except Exception as e:
            err = {"error": f"{type(e).__name__}: {e}", "job_id": job["job_id"]}
            _http(f"{server}/jobs/{job['job_id']}/result", method="POST", body=err)
            print(f"[{worker_id}] {job['seed_id']} ERROR {e}", file=sys.stderr)
            RECENT_CONTRIBS.append({
                "ts": time.time(),
                "seed": job.get("seed_id", "?"),
                "chain": chain.label,
                "label": "ERROR",
                "confidence": 0.0,
            })
            CONTRIB_COUNTS["ERROR"] = CONTRIB_COUNTS.get("ERROR", 0) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.environ.get("ORCH_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--target", default="ollama", choices=["ollama", "anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--share-content", action="store_true",
                    help="Send full prompt+response to orchestrator (default: send only verdict labels for privacy)")
    ap.add_argument("--nat-ai-port", type=int, default=NAT_AI_PORT_DEFAULT,
                    help="Port for local NAT AI chat server. 0 disables.")
    args = ap.parse_args()
    target_kwargs = {}
    if args.model:
        target_kwargs["model"] = args.model
    run(args.server, args.target, target_kwargs, args.worker_id, args.poll_interval,
        share_content=args.share_content, nat_ai_port=args.nat_ai_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
