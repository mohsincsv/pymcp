"""
Transport layer for MCP.

This package provides transport implementations for MCP communication,
including abstractions and concrete implementations for different
transport mechanisms (stdio, HTTP/SSE).
"""

# Version
__version__ = "0.1.0"

# Base transport types
from mcp.transport.base import (
    Transport,
    TransportInfo,
    TransportType,
    TransportCapability,
    RequestContext,
    RequestHandler,
    SessionStore,
    StreamingResponse,
    TransportError,
)

# Stdio transport
from mcp.transport.stdio import (
    StdioTransport,
)

# SSE transport
from mcp.transport.sse import (
    SSETransport,
    SSEStream,
)

# Update package exports
__all__ = [
    # Base types
    "Transport",
    "TransportInfo",
    "TransportType",
    "TransportCapability",
    "RequestContext",
    "RequestHandler",
    "SessionStore",
    "StreamingResponse",
    "TransportError",
    
    # Stdio transport
    "StdioTransport",
    
    # SSE transport
    "SSETransport",
    "SSEStream",
]
