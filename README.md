# PyMCP – Python Model Context Protocol Implementation

PyMCP is a **MIT-licensed** re-implementation of the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/specification) in pure Python ≥ 3.11.  
It is inspired by (but not a direct port of) the [go-go-mcp](https://github.com/go-go-golems/go-go-mcp) project and demonstrates:

* idiomatic Python design for protocol, plugin and transport layers  
* asynchronous concurrency compared with Go goroutines  
* dynamic tool loading from YAML definitions  
* a clear project structure suitable for production-grade open source work  

---

## Repository

* **GitHub:** <https://github.com/mohsincsv/pymcp>  
* **Default branch:** `main`  
* **License:** MIT  

---

## Current Status

All core functionality is implemented and tested:

| Category        | Capability                                                          | Status |
|-----------------|---------------------------------------------------------------------|--------|
| Protocol        | JSON-RPC 2.0 messages, batch processing, MCP core methods           | Complete |
| Transports      | `stdio` (CLI ↔ LLM), `HTTP` + Server-Sent Events (SSE)              | Complete |
| Server Core     | Session store, request router, graceful shutdown                    | Complete |
| Tool System     | Pluggable registry, YAML loader, shell adapter, streaming results   | Complete |
| Client Library  | Full client with stdio and HTTP transports, CLI integration         | Complete |
| Configuration   | YAML profiles, environment overrides, hot reload                    | Complete |
| Documentation   | Reference material in `/docs` (continuously updated)               | Ongoing |

---

## Installation

> Requires **Python 3.11+** and a working C compiler for optional dependencies.

1. Clone and install dependencies:

```bash
git clone https://github.com/mohsincsv/pymcp.git
cd pymcp
poetry install            # or: pip install -e ".[dev]"
pre-commit install        # optional – lint/format on commit
```

2. Run the test suite:

```bash
poetry run pytest
```

---

## Quick Start

### Start a local MCP server (stdio)

```bash
poetry run mcp server start --transport stdio
```

### Start an HTTP/SSE server

```bash
poetry run mcp server start --transport sse --port 8000
```

### List available tools

```bash
poetry run mcp client tools list
```

### Call a tool over HTTP

```bash
http POST http://localhost:8000/mcp \
     jsonrpc=2.0 id=1 method=tools/call \
     params:='{"name":"echo","args":{"message":"Hello, MCP!"}}'
```

---

## Project Structure

```
pymcp/
├── mcp/                 # Library code
│   ├── protocol/        # JSON-RPC & MCP message models
│   ├── transport/       # stdio, http, sse
│   ├── server/          # server core + session handling
│   ├── client/          # client utilities
│   ├── tools/           # base classes, registry, loaders
│   ├── prompts/         # prompt registry
│   ├── resources/       # resource registry
│   └── utils/           # logging, typing helpers
├── cli/                 # Typer CLI entry point (`mcp`)
├── docs/                # Extended documentation
├── tests/               # pytest test suite
└── examples/            # Sample YAML tools & demo scripts
```

---

## Architecture Overview

1. **Protocol Layer** – validates and serialises JSON-RPC requests.  
2. **Transport Layer** – abstracts stdio, HTTP and SSE streams.  
3. **Server Core** – routes parsed requests to method handlers, maintains sessions.  
4. **Tool System** – registry of `Tool` objects (native Python or shell wrappers) discovered via YAML or entry points.  
5. **Client Library** – thin wrapper around transports for synchronous or asynchronous calls.

A simplified data flow:

```
Client ─▶ Transport ─▶ RequestHandler ─▶ ToolRegistry ─▶ Tool ─▶ Response
```

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
