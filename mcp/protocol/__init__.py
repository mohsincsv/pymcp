"""
MCP Protocol Package

This package provides the core protocol implementation for the Model Context Protocol (MCP),
including JSON-RPC 2.0 message types, MCP-specific method definitions, and validation utilities.
"""

# Version
__version__ = "0.1.0"

# Export core protocol types from base
from mcp.protocol.base import (
    Request,
    Response,
    Notification,
    Error,
    ErrorCode,
    BatchRequest,
    BatchResponse,
    RequestMetadata,
)

# Export MCP method definitions
from mcp.protocol.methods import (
    # Method enum
    MCPMethod,
    
    # Initialize
    ServerInfo,
    ServerCapabilities,
    ClientInfo,
    ClientCapabilities,
    InitializeParams,
    InitializeResult,
    
    # Ping
    PingParams,
    PingResult,
    
    # Tools
    ToolParameterSchema,
    ToolSchema,
    ToolsListParams,
    ToolsListResult,
    ToolCallParams,
    ToolCallResult,
    ToolCallStreamChunk,
    
    # Prompts
    PromptMetadata,
    PromptsListParams,
    PromptsListResult,
    PromptsGetParams,
    PromptsGetResult,
    
    # Resources
    ResourceMetadata,
    ResourcesListParams,
    ResourcesListResult,
    ResourcesReadParams,
    ResourcesReadResult,
    ResourcesSubscribeParams,
    ResourcesSubscribeResult,
    
    # Notifications
    CancelRequestParams,
    ProgressParams,
)

# Export validation utilities
from mcp.protocol.validation import (
    MCPValidationError,
    parse_json_rpc,
    parse_json_rpc_object,
    create_error_response,
    create_result_response,
    create_batch_response,
    parse_params_model,
    serialize_model,
    handle_validation_error,
)

# Update package exports
__all__ = [
    # Base types
    "Request",
    "Response",
    "Notification",
    "Error",
    "ErrorCode",
    "BatchRequest",
    "BatchResponse",
    "RequestMetadata",
    
    # Method enum
    "MCPMethod",
    
    # Initialize
    "ServerInfo",
    "ServerCapabilities",
    "ClientInfo",
    "ClientCapabilities",
    "InitializeParams",
    "InitializeResult",
    
    # Ping
    "PingParams",
    "PingResult",
    
    # Tools
    "ToolParameterSchema",
    "ToolSchema",
    "ToolsListParams",
    "ToolsListResult",
    "ToolCallParams",
    "ToolCallResult",
    "ToolCallStreamChunk",
    
    # Prompts
    "PromptMetadata",
    "PromptsListParams",
    "PromptsListResult",
    "PromptsGetParams",
    "PromptsGetResult",
    
    # Resources
    "ResourceMetadata",
    "ResourcesListParams",
    "ResourcesListResult",
    "ResourcesReadParams",
    "ResourcesReadResult",
    "ResourcesSubscribeParams",
    "ResourcesSubscribeResult",
    
    # Notifications
    "CancelRequestParams",
    "ProgressParams",
    
    # Validation utilities
    "MCPValidationError",
    "parse_json_rpc",
    "parse_json_rpc_object",
    "create_error_response",
    "create_result_response",
    "create_batch_response",
    "parse_params_model",
    "serialize_model",
    "handle_validation_error",
]
