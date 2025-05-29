# PyMCP – Python Model Context Protocol Implementation

PyMCP is a **learning-oriented, MIT-licensed** re-implementation of the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/specification) in pure Python ≥ 3.11.  
It is inspired by (but not a direct port of) the excellent [go-go-mcp](https://github.com/go-go-golems/go-go-mcp) project and aims to demonstrate:

* idiomatic Python design patterns for protocol, plugin and transport layers  
* async (or green-thread) concurrency compared with Go goroutines  
* dynamic tool loading from YAML definitions  
* clean project structure, CI, and documentation suitable for real-world OSS work

---

## ✨ Key Features (MVP)

| Category        | Capability                                                     |
|-----------------|----------------------------------------------------------------|
| **Protocol**    | JSON-RPC 2.0 messages, batch processing, MCP core methods (`initialize`, `prompts/*`, `tools/*`) |
| **Transports**  | `stdio` (CLI ↔ LLM), `HTTP` + Server-Sent Events (SSE) |
| **Server Core** | Session store, request router, graceful shutdown |
| **Tools**       | Pluggable registry, schema discovery, streaming responses, YAML-defined shell adapters |
| **Client**      | Minimal client library + Typer-based CLI (`client tools list`, `client tools call …`) |
| **Config**      | YAML profiles, environment overrides, hot reload |
| **DevX**        | Type hints, Ruff/Black, pre-commit hooks, GitHub Actions matrix |
| **Docs**        | Architecture diagrams & API reference in `/docs` |

See the full milestone roadmap in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

---

## 📦 Installation

> Requires **Python 3.11+** and a working C compiler for optional dependencies.

### 1 – Clone & bootstrap

```bash
git clone https://github.com/<your-user>/pymcp.git
cd pymcp
# Recommended: use Poetry or Hatch
poetry install  # or: pip install -e ".[dev]"
pre-commit install  # lint/format on every commit
```

### 2 – Run unit tests

```bash
poetry run pytest
```

---

## 🚀 Quick Start

### Start a local MCP server (stdio)

```bash
poetry run mcp server start --transport stdio
```

### Start an HTTP/SSE server

```bash
poetry run mcp server start --transport http --port 8000
```

### List available tools (from default profile)

```bash
poetry run mcp client tools list
```

### Call a tool over HTTP

```bash
http POST http://localhost:8000/mcp jsonrpc=2.0 id=1 method=tools/call \
     params:='{"name": "echo", "args": {"message": "Hello, MCP!"}}'
```

You should receive a JSON-RPC response containing the tool output.

---

## 🗂️ Project Structure

```
pymcp/
├── mcp/                 # Library code
│   ├── protocol/        # JSON-RPC & MCP message models
│   ├── transport/       # stdio, http, sse
│   ├── server/          # core server + session handling
│   ├── client/          # client utilities
│   ├── tools/           # base classes, registry, loaders
│   ├── prompts/         # prompt registry
│   ├── config/          # YAML profile loader & models
│   └── utils/           # logging, typing helpers
├── cli/                 # Typer CLI entry point (`mcp`)
├── docs/                # Extended documentation
├── tests/               # pytest test suite
└── examples/            # Sample YAML tools & demo scripts
```

---

## 🏗️ Architecture Overview

1. **Protocol Layer** – validates & serialises JSON-RPC requests.
2. **Transport Layer** – abstracts stdio, HTTP, SSE streams.
3. **Server Core** – routes parsed requests to method handlers, maintains sessions.
4. **Tool System** – registry of `Tool` objects (native Python classes or shell wrappers) discovered via YAML or entry points.
5. **Client Library** – thin wrapper around transport for synchronous or async calls.

A simplified flow:

```
Client (CLI/HTTP) ──▶ Transport ──▶ RequestHandler ──▶ ToolRegistry ──▶ Tool ──▶ Response
```

---

## 🤝 Contributing

We welcome issues, discussion, and PRs!

1. **Fork** the repo and create a feature branch:  
   `git checkout -b feat/my-awesome-thing`
2. Ensure `pre-commit run --all-files` passes (Black, Ruff, MyPy, etc.).
3. Add/modify **tests** in `tests/` with good coverage.
4. Use **Conventional Commits** (`feat: ...`, `fix: ...`) for clear history.
5. Open a pull request – the GitHub Actions CI must be green.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full guidelines (code style, DCO).

---

## 📜 License

PyMCP is released under the **MIT License** – see [`LICENSE`](LICENSE) for details.

Happy hacking! 🎉
