# NAT NETWORK

A distributed LLM safety-evaluation harness with one-click Windows installers and a local AI suite.

NAT NETWORK lets community-operator nodes run AI-safety evaluation probes against local Ollama models and report verdicts back to a central orchestrator. Operators get a free local AI suite (chat with persistent memory, AI pair-programmer, visual workflow builder) in exchange for contributing compute when jobs are queued.

## What's included

- **Orchestrator** — stdlib-Python HTTP work-queue. Serves the join page, all installers, and the job API.
- **Worker** — pulls jobs, runs target+judge against local Ollama, posts verdicts back. Also hosts a same-origin NAT AI chat server on `127.0.0.1:11500` (proxies to Ollama, so no CORS headaches).
- **Jailbreak harness** — AdvBench seeds × mutation chains × an ensemble (heuristic+LLM-judge) scorer with explicit AMBIGUOUS labels and confidence scores. Emits markdown safety-baseline reports.
- **One-click Windows installers**:
  - `nat-network-joiner.bat` — joins the fleet (consent screen, installs Python + Ollama + model, starts worker)
  - `nat-coding-mode.bat` — installs Aider on Python 3.11 in an isolated pipx venv (pair-programmer)
  - `nat-n8n.bat` — installs n8n for visual workflows wired to local Ollama
  - `nat-claude-install.bat` — installs Anthropic Claude Code CLI
  - `nat-reactivate.bat` — reconnect-existing-install one-click
  - `nat-handoff.bat` — migrate the project (and Claude memory) to a new machine

## NAT AI chat features

- Mode presets: plain / node specialist / web3 educator / explain code / debug / quick rewrite / brainstorm / docs prep
- Persistent local memory in `localStorage` (per-device) with auto-summarization at 30 turns
- Model selector auto-populated from your local Ollama
- Live "your node contributions" feed showing the worker's recent verdicts
- File-paste box for single-file context
- Web3 mode is education-only (no buy/sell recommendations); banner makes that explicit

## Architecture quick-reference

```
Phone or VPS (orchestrator host)
  http://<host>:8765/
    /                    join page
    /chat                NAT AI chat HTML (served by worker, not orchestrator, when via 127.0.0.1:11500)
    /joiner.bat          full install
    /reactivate.bat      reconnect
    /handoff.bat         migrate to a new machine
    /coding-mode.bat     install NAT CODE
    /n8n.bat             install NAT FLOW
    /claude.bat          install NAT CLAUDE
    /project.tar         full project bundle for handoff
    /jobs/next           worker pulls jobs
    /jobs/<id>/result    worker posts probe results
    /status              queue counts
    /worker.py           worker code
    /bundle.tar          node/ package for worker bootstrap
    /contributions       (worker, on port 11500) recent verdicts

Worker (per node)
  http://127.0.0.1:11500/
    /                    NAT AI chat
    /api/*               proxy to local Ollama (same-origin)
    /contributions       recent verdicts JSON
```

## Quick start

1. Run the orchestrator: `python -m orchestrator.server --port 8765`
2. Open `http://<your-host>:8765/` from any browser
3. Click **Download node installer** on Windows; double-click the bat; press Y
4. Submit work: `python -m orchestrator.submit --limit 5`
5. Generate report: `python -m node.jailbreak.report`

## Privacy

- Worker default `share_mode=MINIMAL`: only verdict labels go back to the orchestrator. Local prompts/responses stay on the device.
- NAT AI chat memory is `localStorage`-only; never uploaded anywhere.
- Auto-summarization happens locally via the user's own Ollama model.

## What this isn't

- Not a coding agent. Use Aider (NAT CODE) for that.
- Not a financial advisor. Web3 mode is educational only; no buy/sell signals.
- Not a crypto miner / not a bandwidth-sharing SDK / not for ad-fraud automation.
- Not for unauthorized red-teaming. The Anthropic/OpenAI target classes exist for authorized bounty programs only.

## License

MIT. See `LICENSE`.
