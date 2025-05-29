"""
MCP-specific method definitions and message types.

This module defines the specific method names, request parameters, and response types
for the Model Context Protocol (MCP) specification.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# Method name constants
class MCPMethod(str, Enum):
    """Standard MCP method names."""
    
    # Core methods
    INITIALIZE = "initialize"
    PING = "ping"
    
    # Tool methods
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # Prompt methods
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    
    # Resource methods
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    
    # Notification methods
    CANCEL_REQUEST = "$/cancelRequest"
    PROGRESS = "$/progress"


# ---- Initialize Method ----

class ServerInfo(BaseModel):
    """Server information returned during initialization."""
    
    name: str = Field(..., description="Name of the server implementation")
    version: str = Field(..., description="Version of the server implementation")


class ServerCapabilities(BaseModel):
    """Capabilities supported by the server."""
    
    supports_tool_streaming: bool = Field(
        False, description="Whether the server supports streaming tool results"
    )
    supports_batch_requests: bool = Field(
        False, description="Whether the server supports batch requests"
    )
    supports_cancellation: bool = Field(
        False, description="Whether the server supports cancellation of requests"
    )
    supports_progress: bool = Field(
        False, description="Whether the server supports progress notifications"
    )
    supports_resources: bool = Field(
        False, description="Whether the server supports resources"
    )
    supports_prompts: bool = Field(
        False, description="Whether the server supports prompts"
    )


class ClientInfo(BaseModel):
    """Client information sent during initialization."""
    
    name: str = Field(..., description="Name of the client implementation")
    version: str = Field(..., description="Version of the client implementation")


class ClientCapabilities(BaseModel):
    """Capabilities supported by the client."""
    
    supports_tool_streaming: bool = Field(
        False, description="Whether the client supports streaming tool results"
    )
    supports_batch_requests: bool = Field(
        False, description="Whether the client supports batch requests"
    )
    supports_cancellation: bool = Field(
        False, description="Whether the client supports cancellation of requests"
    )
    supports_progress: bool = Field(
        False, description="Whether the client supports progress notifications"
    )


class InitializeParams(BaseModel):
    """Parameters for the initialize method."""
    
    client_info: ClientInfo = Field(..., description="Information about the client")
    client_capabilities: Optional[ClientCapabilities] = Field(
        None, description="Capabilities supported by the client"
    )


class InitializeResult(BaseModel):
    """Result of the initialize method."""
    
    server_info: ServerInfo = Field(..., description="Information about the server")
    server_capabilities: ServerCapabilities = Field(
        ..., description="Capabilities supported by the server"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for the client to use in subsequent requests"
    )


# ---- Ping Method ----

class PingParams(BaseModel):
    """Parameters for the ping method."""
    
    pass  # No parameters needed


class PingResult(BaseModel):
    """Result of the ping method."""
    
    timestamp: int = Field(..., description="Server timestamp in milliseconds since epoch")


# ---- Tools Methods ----

class ToolParameterSchema(BaseModel):
    """Schema for a tool parameter."""
    
    name: str = Field(..., description="Name of the parameter")
    type: str = Field(..., description="Type of the parameter (string, number, boolean, etc.)")
    description: Optional[str] = Field(None, description="Description of the parameter")
    required: bool = Field(False, description="Whether the parameter is required")
    default: Optional[Any] = Field(None, description="Default value for the parameter")
    enum: Optional[List[Any]] = Field(None, description="Enumeration of possible values")


class ToolSchema(BaseModel):
    """Schema for a tool."""
    
    name: str = Field(..., description="Name of the tool")
    description: Optional[str] = Field(None, description="Description of the tool")
    parameters: List[ToolParameterSchema] = Field(
        default_factory=list, description="Parameters accepted by the tool"
    )
    returns: Optional[Dict[str, Any]] = Field(
        None, description="Schema for the return value of the tool"
    )
    streaming: bool = Field(
        False, description="Whether the tool supports streaming results"
    )


class ToolsListParams(BaseModel):
    """Parameters for the tools/list method."""
    
    pass  # No parameters needed


class ToolsListResult(BaseModel):
    """Result of the tools/list method."""
    
    tools: List[ToolSchema] = Field(..., description="List of available tools")


class ToolCallParams(BaseModel):
    """Parameters for the tools/call method."""
    
    name: str = Field(..., description="Name of the tool to call")
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )
    stream: bool = Field(
        False, description="Whether to stream the results of the tool call"
    )


class ToolCallResult(BaseModel):
    """Result of the tools/call method."""
    
    result: Any = Field(..., description="Result of the tool call")


class ToolCallStreamChunk(BaseModel):
    """Chunk of a streaming tool call result."""
    
    chunk: Any = Field(..., description="Chunk of the streaming result")
    is_last: bool = Field(
        False, description="Whether this is the last chunk of the stream"
    )


# ---- Prompts Methods ----

class PromptMetadata(BaseModel):
    """Metadata about a prompt."""
    
    name: str = Field(..., description="Name of the prompt")
    description: Optional[str] = Field(None, description="Description of the prompt")
    version: Optional[str] = Field(None, description="Version of the prompt")
    tags: Optional[List[str]] = Field(None, description="Tags associated with the prompt")
    parameters: Optional[List[ToolParameterSchema]] = Field(
        None, description="Parameters accepted by the prompt"
    )


class PromptsListParams(BaseModel):
    """Parameters for the prompts/list method."""
    
    tags: Optional[List[str]] = Field(
        None, description="Filter prompts by tags"
    )


class PromptsListResult(BaseModel):
    """Result of the prompts/list method."""
    
    prompts: List[PromptMetadata] = Field(..., description="List of available prompts")


class PromptsGetParams(BaseModel):
    """Parameters for the prompts/get method."""
    
    name: str = Field(..., description="Name of the prompt to get")
    args: Optional[Dict[str, Any]] = Field(
        None, description="Arguments to fill in the prompt template"
    )


class PromptsGetResult(BaseModel):
    """Result of the prompts/get method."""
    
    content: str = Field(..., description="Content of the prompt")
    metadata: PromptMetadata = Field(..., description="Metadata about the prompt")


# ---- Resources Methods ----

class ResourceMetadata(BaseModel):
    """Metadata about a resource."""
    
    name: str = Field(..., description="Name of the resource")
    description: Optional[str] = Field(None, description="Description of the resource")
    version: Optional[str] = Field(None, description="Version of the resource")
    tags: Optional[List[str]] = Field(None, description="Tags associated with the resource")
    mime_type: Optional[str] = Field(None, description="MIME type of the resource")
    size: Optional[int] = Field(None, description="Size of the resource in bytes")


class ResourcesListParams(BaseModel):
    """Parameters for the resources/list method."""
    
    tags: Optional[List[str]] = Field(
        None, description="Filter resources by tags"
    )


class ResourcesListResult(BaseModel):
    """Result of the resources/list method."""
    
    resources: List[ResourceMetadata] = Field(..., description="List of available resources")


class ResourcesReadParams(BaseModel):
    """Parameters for the resources/read method."""
    
    name: str = Field(..., description="Name of the resource to read")


class ResourcesReadResult(BaseModel):
    """Result of the resources/read method."""
    
    content: str = Field(..., description="Content of the resource")
    metadata: ResourceMetadata = Field(..., description="Metadata about the resource")


class ResourcesSubscribeParams(BaseModel):
    """Parameters for the resources/subscribe method."""
    
    name: str = Field(..., description="Name of the resource to subscribe to")


class ResourcesSubscribeResult(BaseModel):
    """Result of the resources/subscribe method."""
    
    subscription_id: str = Field(..., description="ID of the subscription")


# ---- Notification Methods ----

class CancelRequestParams(BaseModel):
    """Parameters for the $/cancelRequest notification."""
    
    id: Union[str, int] = Field(..., description="ID of the request to cancel")


class ProgressParams(BaseModel):
    """Parameters for the $/progress notification."""
    
    token: str = Field(..., description="Progress token from the original request")
    value: Union[int, float] = Field(
        ..., description="Progress value (0-100 for percentage)"
    )
    message: Optional[str] = Field(None, description="Progress message")
