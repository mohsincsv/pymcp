# Python Model Context Protocol (MCP) – Project Plan  
*(file: `pymcp/PROJECT_PLAN.md`)*

---

## 1. Vision & Objectives
| Goal | Description |
|------|-------------|
| Learning-oriented | Re-implement core ideas of [go-go-mcp] in idiomatic Python to understand protocol, plugin architecture, concurrency, and tooling. |
| Minimal Viable MCP | Support JSON-RPC 2.0 requests, batch calls, `initialize`, `tools/list`, `tools/call`, and `prompts/list` over stdio & HTTP/SSE. |
| Extensible | Modular package layout allowing new transports, providers, and UI front-ends. |
| OSS-ready | Clean licensing (MIT), README, CI, semantic versioning. |

---

## 2. Planned Directory Layout

```
pymcp/
├── README.md
├── pyproject.toml          # Poetry / Hatch project config
├── mcp/                    # Python package root
│   ├── __init__.py
│   ├── protocol/           # JSON-RPC & MCP messages
│   │   ├── base.py
│   │   ├── methods.py
│   │   └── validation.py
│   ├── transport/          # Abstractions & impls
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── stdio.py
│   │   ├── sse.py
│   │   └── http.py
│   ├── server/             # MCP server core
│   │   ├── __init__.py
│   │   ├── session.py
│   │   ├── handler.py
│   │   └── server.py
│   ├── client/             # MCP client lib
│   │   ├── __init__.py
│   │   └── client.py
│   ├── tools/              # Tool system
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── shell_adapter.py
│   │   └── loaders/
│   │       └── yaml_loader.py
│   ├── prompts/            # Prompt registry
│   │   └── registry.py
│   ├── config/             # YAML profile handling
│   │   └── loader.py
│   └── utils/              # Shared helpers
│       ├── logging.py
│       └── typing.py
├── cli/                    # Command-line entry points
│   └── mcp.py
├── tests/
│   └── ...
└── docs/
    ├── architecture.md
    ├── usage.md
    └── api_reference.md
```

---

## 3. Phase-by-Phase Implementation Roadmap

### Phase 0 – Prerequisites & Repo Bootstrapping
1. **Create repo** under personal GitHub (`pymcp`).
2. Initialize Poetry/Hatch; set Python ≥ 3.11.
3. Add pre-commit (ruff/black, mypy), `.gitignore`, `LICENSE`.

### Phase 1 – Core Protocol Layer  
*Goal: JSON-RPC 2.0 + MCP base messages.*

| Step | Details |
|------|---------|
| 1.1 | `protocol/base.py`: `Request`, `Response`, `Error`, `Notification`, `BatchRequest/Response`. |
| 1.2 | Validation helpers (`jsonschema` or `pydantic`). |
| 1.3 | Enum/consts for MCP method names. |
| 1.4 | Unit tests mirroring Go originals (happy path + error cases). |

### Phase 2 – Transport Abstraction
1. Define `Transport` ABC with `listen(ctx, handler)` and `send(...)`.
2. **Stdio transport** (`asyncio` streams).
3. **HTTP/SSE transport** using `FastAPI` + `sse-starlette`.
4. Shared capability descriptor (`TransportInfo`).

### Phase 3 – Server Core
1. `session.SessionStore` (in-memory, UUID tokens).  
2. `handler.RequestHandler` routing MCP methods → controllers.  
3. `server.Server` orchestrating transport, providers, lifecycle signals.  
4. Logging (structlog) & graceful shutdown.

### Phase 4 – Tool System
| Step | Deliverable |
|------|-------------|
| 4.1 | `tools.base.Tool` interface (`schema()`, `call()` returning streaming iterator). |
| 4.2 | `tools.registry.Registry` w/ dynamic registration. |
| 4.3 | YAML loader producing shell-command tools (use `subprocess` & `jinja2` for templating). |
| 4.4 | Example tools (`echo`, `fetch_url`). |
| 4.5 | `tools/call`, `tools/list` MCP handlers. |

### Phase 5 – Prompts & Resources (Minimal)
1. Registry pattern mirroring tool system.  
2. Implement `prompts/list` & `prompts/get`.

### Phase 6 – Client Library
1. `client.Client` supporting stdio & HTTP transports.  
2. High-level helpers: `list_tools()`, `call_tool()`.  
3. CLI sub-commands for quick testing.

### Phase 7 – Configuration & Profiles
1. `config.loader` reading YAML with profiles (tool paths, env).  
2. In-CLI flags `--profile`, env var overrides.  
3. Validation via `pydantic` models.

### Phase 8 – CLI Application
1. Use **Typer** for ergonomic commands.  
2. Commands: `server start`, `client tools list`, `client tools call`, `schema`, `config ...`.  
3. Add option for auto-reload on file changes (`watchdog`).

### Phase 9 – Optional Enhancements
- Web UI (React/Vite) consuming SSE stream.  
- Async task queue (e.g. `anyio`, `trio`) for long-running tools.  
- Database session store (SQLite).

### Phase 10 – Testing & CI
| Task | Tooling |
|------|---------|
| Unit tests | `pytest`, `pytest-asyncio` |
| Coverage  | `pytest-cov` |
| Lint/format | `ruff`, `black` |
| Type-check | `mypy` |
| CI | GitHub Actions matrix (3.11/3.12) |

### Phase 11 – Docs & Release
1. `docs/architecture.md` with diagrams (PlantUML).  
2. Usage guide & examples.  
3. Versioning strategy (SemVer), `CHANGELOG.md`.  
4. First `v0.1.0` release.

---

## 4. Milestone Timeline (Estimate)

| Milestone | Deliverables | ETA |
|-----------|--------------|-----|
| M0 | Repo bootstrapped, CI green | Day 1 |
| M1 | Protocol + tests | Day 3 |
| M2 | Stdio transport, basic server | Day 6 |
| M3 | Tool system with echo tool | Day 10 |
| M4 | HTTP/SSE transport, client lib | Day 14 |
| M5 | Profiles & CLI UX polish | Day 18 |
| M6 | Docs, v0.1.0 release | Day 21 |

---

## 5. Contribution & Branch Strategy
- `main` = stable, protected.
- Feature branches `feat/<component>` → PR → review (self).
- Conventional Commits: `feat:`, `fix:`, `docs:` etc.
- Use `pre-commit` hooks to enforce lint/format.

---

## 6. Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Async complexity | Start with synchronous stdio; add async later |
| Spec drift vs Go impl | Write conformance tests using captured JSON fixtures |
| License compliance for sample tools | Use MIT-licensed or self-written examples |

---

## 7. Next Action Checklist
- [ ] Confirm personal Git config in project directory  
- [ ] Create GitHub repo `pymcp` under personal account  
- [ ] Push initial scaffold (`Phase 0`)  
- [ ] Move to **Phase 1: Protocol implementation**

---

*Happy hacking!*
