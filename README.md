# PyMCP – Python Model Context Protocol Implementation

PyMCP is a **learning-oriented, MIT-licensed** re-implementation of the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/specification) in pure Python ≥ 3.11.  
It is inspired by (but not a direct port of) the excellent [go-go-mcp](https://github.com/go-go-golems/go-go-mcp) project and aims to demonstrate:

* idiomatic Python design patterns for protocol, plugin and transport layers  
* async (or green-thread) concurrency compared with Go goroutines  
* dynamic tool loading from YAML definitions  
* clean project structure, CI, and documentation suitable for real-world OSS work  

---

## 📦 Repository

* **GitHub:** <https://github.com/mohsincsv/pymcp>  
* **Default branch:** `main`  
* **License:** MIT  
* **Latest release:** _in development_  
* **CI status:** _coming soon_ <!-- badge placeholder -->

---

## 🚀 Project Status & Progress

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Project scaffold, Poetry/Hatch config, linting, pre-commit, MIT license | **✅ Complete** |
| 1 | Core JSON-RPC 2.0 & MCP protocol models with validation and full test suite | **✅ Complete** |
| 2 | Transport layer:<br>• StdioTransport (CLI)<br>• SSETransport (FastAPI) with streaming | **✅ Complete** |
| 3 | Server core: session store, request router, graceful shutdown | 🛠 _Next up_ |
| 4 | Tool system: registry, YAML loader, shell adapter, example tools | ⏳ Planned |
| 5 | Prompts & resources registry + handlers | ⏳ Planned |
| 6 | Client library & Typer CLI | ⏳ Planned |
| 7 | Configuration profiles, hot reload | ⏳ Planned |
| 8 | Docs, CI, first `v0.1.0` release | ⏳ Planned |

---

## ✨ Key Features (MVP)

| Category        | Capability                                                     |
|-----------------|----------------------------------------------------------------|
| **Protocol**    | JSON-RPC 2.0 messages, batch processing, MCP core methods (`initialize`, `prompts/*`, `tools/*`) |
| **Transports**  | `stdio` (CLI ↔ LLM), `HTTP` + Server-Sent Events (SSE) |
| **Server Core** | Session store, request router, graceful shutdown *(next)* |
| **Tools**       | Pluggable registry, schema discovery, streaming responses *(planned)* |
| **Client**      | Minimal client library + Typer-based CLI *(planned)* |
| **Config**      | YAML profiles, environment overrides, hot reload *(planned)* |
| **DevX**        | Type hints, Ruff/Black, pre-commit hooks, GitHub Actions matrix *(CI coming)* |
| **Docs**        | Architecture diagrams & API reference in `/docs` *(later)* |

---

## 📦 Installation

> Requires **Python 3.11+** and a working C compiler for optional dependencies.

### 1 – Clone & bootstrap

```bash
git clone https://github.com/mohsincsv/pymcp.git
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
│   ├── server/          # core server + session handling (WIP)
│   ├── client/          # client utilities (planned)
│   ├── tools/           # base classes, registry, loaders (planned)
│   ├── prompts/         # prompt registry (planned)
│   ├── config/          # YAML profile loader & models (planned)
│   └── utils/           # logging, typing helpers
├── cli/                 # Typer CLI entry point (`mcp`) (planned)
├── docs/                # Extended documentation (to be written)
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
