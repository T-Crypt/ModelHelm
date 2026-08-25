# ModelHelm

Intelligent orchestration for local and frontier AI.

ModelHelm is an MCP server that lets Claude Code delegate coding tasks to
locally hosted models (via LM Studio) while Claude handles planning,
architecture, and final review. Model selection is informed by
[llmfit](https://github.com/) hardware/model fit scoring.

## Status

Phase 1: MCP server + local agent loop. See
`docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md` for
scope and `docs/ModelHelm-Spec.md` for the full project vision.

## Requirements

- Python >= 3.11
- [LM Studio](https://lmstudio.ai/) running locally with a tool-use-capable
  model loaded (developed against `qwen3-coder-30b-a3b`)
- [llmfit](https://github.com/) installed and resolvable (via PATH, or set
  `llmfit.binary_path` in `modelhelm.yaml`)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Running tests

```powershell
.\.venv\Scripts\pytest.exe tests/unit/ -v
.\.venv\Scripts\pytest.exe tests/integration/ -v   # requires LM Studio running
```

## Running the MCP server

```powershell
.\.venv\Scripts\python.exe -m modelhelm.mcp.server
```

Configure via `modelhelm.yaml` at the repository root (see that file for
the full default configuration, including the safety policy).
