# PyMCP – Python Model Context Protocol Implementation

PyMCP is a **MIT-licensed** re-implementation of the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/specification) in pure Python.

It is inspired by (but not a direct port of) the [go-go-mcp](https://github.com/go-go-golems/go-go-mcp) project and demonstrates:

* idiomatic Python design for protocol, plugin and transport layers  
* asynchronous concurrency compared with Go goroutines  
* dynamic tool loading from YAML definitions  
* a clear project structure suitable for production-grade open source work  


## Installation

### 1. Development checkout (recommended)

```bash
git clone https://github.com/mohsincsv/pymcp.git
cd pymcp
pip install -e ".[dev]"      # or: poetry install
pre-commit install           # lint / format on every commit
```

### 2. Editable install from local path

```bash
pip install -e /path/to/pymcp
```

(There’s no PyPI package yet—ship coming when the API freezes.)

---

## Overview – what’s inside

| Component            | Path                   | Role |
|----------------------|------------------------|------|
| Protocol layer       | `mcp.protocol`         | Pydantic models for JSON-RPC + MCP messages |
| Transport layer      | `mcp.transport`        | `stdio` and `sse` (FastAPI) strategies |
| Server core          | `mcp.server`           | Request router, session store, graceful shutdown |
| Tool system          | `mcp.tools`            | `Tool` ABC, registry, YAML loader, streaming |
| Prompt & resources   | `mcp.prompts` / `resources` | Same registry pattern, lighter weight |
| Client library       | `mcp.client`           | Sync / async calls, facade over transports |
| CLI                  | `cli/mcp.py`           | Typer entry-point wrapping everything |

You can drop a YAML file in a watched folder and the server picks it up with no restart.

---

## Supported MCP methods

| Method              | Purpose                                  |
|---------------------|------------------------------------------|
| `initialize`        | handshake / capability negotiation       |
| `ping`              | health check                             |
| `prompts/list`      | list prompts                             |
| `prompts/get`       | get prompt content                       |
| `resources/list`    | list static resources                    |
| `resources/read`    | read resource                            |
| `tools/list`        | list registered tools                    |
| `tools/call`        | run a tool (streaming supported)         |

_Not yet implemented_: notifications, `resources/subscribe`.

---

## Running

### Basic usage (stdio server + client)

```bash
# terminal 1 – start a stdio server
python -m mcp.server start --transport stdio

# terminal 2 – connect with a client via the same pipe
python -m mcp.client tools list
```

### Server mode

```bash
# stdio (default)
python -m mcp.server start --transport stdio

# HTTP + SSE on port 3000
python -m mcp.server start --transport sse --port 3000
```

The server watches `--tool-dir` folders and reloads on file changes.

### Server tools (no long-running server)

```bash
python -m mcp.server tools list --tool-dir ./examples
python -m mcp.server tools call echo --args message="Hello"
```

### Client mode

```bash
# list tools from a running HTTP server
python -m mcp.client tools list --transport http --url http://localhost:3000

# call a tool with JSON parameters
python -m mcp.client tools call echo --json '{"message": "Hi"}'
```

---

## Debug mode

Add `--debug` to any command for verbose logs:

```bash
python -m mcp.server start --transport sse --debug
```

---

## Configuration

A config file can hold multiple **profiles**.  
Create a skeleton:

```bash
python -m mcp.config init
```

Example `mcp.yaml`:

```yaml
default_profile: dev
profiles:
  dev:
    transport: stdio
    tool_dirs: ["./examples"]
    debug: true
  http:
    transport: sse
    host: 0.0.0.0
    port: 9000
    tool_dirs: ["./examples"]
```

Helpful commands:

```bash
python -m mcp.config list-profiles
python -m mcp.config show-profile dev
python -m mcp.config set-default-profile http
```

---

## Shell commands (YAML tools)

Create `examples/echo.yaml`:

```yaml
name: echo
description: Echo a message
parameters:
  message:
    type: string
command: |
  echo "{{ message }}"
```

Run it directly:

```bash
python -m mcp.tool load examples/echo.yaml
python -m mcp.tool call echo --args '{"message":"works"}'
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

1. Client serialises a JSON-RPC request.  
2. Transport (stdio or HTTP/SSE) feeds it to the `Server`.  
3. `MCPRequestHandler` routes the call (`tools/call`, `prompts/get`, …).  
4. Registries return the right object; it runs and optionally streams back chunks.  
5. The same path in reverse sends responses to the client.

Everything is async and testable—no global mysticism.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
