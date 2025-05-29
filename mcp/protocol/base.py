"""
Base JSON-RPC 2.0 protocol models for MCP.

This module defines the core protocol types used for MCP communication,
implementing the JSON-RPC 2.0 specification with Pydantic models.
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Union, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator, RootModel


class ErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP specific error codes
    TRANSPORT_ERROR = -32000
    UNAUTHORIZED = -32001
    RATE_LIMITED = -32002
    TOOL_NOT_FOUND = -32003
    TOOL_EXECUTION_ERROR = -32004
    PROMPT_NOT_FOUND = -32005
    RESOURCE_NOT_FOUND = -32006
    SESSION_NOT_FOUND = -32007
    BATCH_PARTIAL_ERROR = -32008
    CANCELLATION_ERROR = -32009


class Error(BaseModel):
    """JSON-RPC 2.0 error object."""
    
    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    data: Optional[Any] = Field(None, description="Additional error data")
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.data:
            return f"code: {self.code}, message: {self.message}, data: {self.data}"
        return f"code: {self.code}, message: {self.message}"


class Request(BaseModel):
    """JSON-RPC 2.0 request."""
    
    jsonrpc: Literal["2.0"] = Field("2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(..., description="Request ID")
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")
    
    @model_validator(mode='after')
    def validate_notification(self) -> 'Request':
        """Validate that notifications don't have an ID."""
        # This is just a sanity check - in practice, notifications should use the Notification class
        if self.id is None and not isinstance(self, Notification):
            raise ValueError("Request must have an ID unless it's a notification")
        return self


class Response(BaseModel):
    """JSON-RPC 2.0 response."""
    
    jsonrpc: Literal["2.0"] = Field("2.0", description="JSON-RPC version")
    id: Optional[Union[str, int]] = Field(..., description="Request ID")
    result: Optional[Any] = Field(None, description="Result data")
    error: Optional[Error] = Field(None, description="Error information")
    
    @model_validator(mode='after')
    def validate_result_or_error(self) -> 'Response':
        """Validate that response has either result or error, not both."""
        if self.result is not None and self.error is not None:
            raise ValueError("Response cannot have both result and error")
        if self.result is None and self.error is None:
            raise ValueError("Response must have either result or error")
        return self


class Notification(BaseModel):
    """JSON-RPC 2.0 notification (request without ID)."""
    
    jsonrpc: Literal["2.0"] = Field("2.0", description="JSON-RPC version")
    method: str = Field(..., description="Method name")
    params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")


class BatchRequest(RootModel[List[Request]]):
    """JSON-RPC 2.0 batch request."""
    
    def __iter__(self):
        """Allow iterating through requests."""
        return iter(self.root)
    
    def __len__(self) -> int:
        """Return the number of requests."""
        return len(self.root)
    
    @field_validator('root')
    @classmethod
    def validate_batch(cls, requests: List[Request]) -> List[Request]:
        """Validate batch request."""
        if not requests:
            raise ValueError("Batch request cannot be empty")
        # Ensure all requests have the correct JSON-RPC version
        for i, req in enumerate(requests):
            if req.jsonrpc != "2.0":
                raise ValueError(f"Request {i} has invalid JSON-RPC version: {req.jsonrpc}")
        return requests
    
    def get_request_by_id(self, id_value: Union[str, int]) -> Optional[Request]:
        """Find a request in the batch by its ID."""
        for req in self.root:
            if req.id == id_value:
                return req
        return None


class BatchResponse(RootModel[List[Response]]):
    """JSON-RPC 2.0 batch response."""
    
    def __iter__(self):
        """Allow iterating through responses."""
        return iter(self.root)
    
    def __len__(self) -> int:
        """Return the number of responses."""
        return len(self.root)
    
    @field_validator('root')
    @classmethod
    def validate_batch(cls, responses: List[Response]) -> List[Response]:
        """Validate batch response."""
        # Batch responses can be empty if all requests were notifications
        for i, resp in enumerate(responses):
            if resp.jsonrpc != "2.0":
                raise ValueError(f"Response {i} has invalid JSON-RPC version: {resp.jsonrpc}")
        return responses


class RequestMetadata(BaseModel):
    """Common metadata that can be included in requests."""
    
    progress_token: Optional[str] = Field(
        None, 
        description="Token for tracking progress of long-running operations"
    )
    
    cancellation_token: Optional[str] = Field(
        None,
        description="Token that can be used to cancel an ongoing operation"
    )
