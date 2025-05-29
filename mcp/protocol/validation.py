"""
Validation utilities for MCP protocol messages.

This module provides functions to validate JSON-RPC messages,
parse messages from JSON, serialize messages to JSON, and
handle validation errors gracefully.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from pydantic import BaseModel, ValidationError

from mcp.protocol.base import (
    BatchRequest, 
    BatchResponse, 
    Error, 
    ErrorCode, 
    Notification, 
    Request, 
    Response
)

# Type variable for generic model validation
T = TypeVar('T', bound=BaseModel)


class MCPValidationError(Exception):
    """Exception raised for MCP protocol validation errors."""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INVALID_REQUEST):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


def parse_json_rpc(data: str) -> Union[Request, Notification, BatchRequest, Dict[str, Any]]:
    """
    Parse a JSON-RPC message from a JSON string.
    
    Args:
        data: JSON string containing a JSON-RPC message
        
    Returns:
        Parsed Request, Notification, BatchRequest, or raw dict if parsing fails
        
    Raises:
        MCPValidationError: If the JSON is invalid
    """
    try:
        parsed_data = json.loads(data)
        return parse_json_rpc_object(parsed_data)
    except json.JSONDecodeError as e:
        raise MCPValidationError(
            f"Invalid JSON: {str(e)}", 
            ErrorCode.PARSE_ERROR
        )


def parse_json_rpc_object(
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Union[Request, Notification, BatchRequest, Dict[str, Any]]:
    """
    Parse a JSON-RPC message from a Python object.
    
    Args:
        data: Python object (dict or list) containing a JSON-RPC message
        
    Returns:
        Parsed Request, Notification, BatchRequest, or raw dict if validation fails
        
    Raises:
        MCPValidationError: If the object doesn't conform to JSON-RPC spec
    """
    # Handle batch requests
    if isinstance(data, list):
        try:
            return BatchRequest(__root__=data)
        except ValidationError as e:
            raise MCPValidationError(
                f"Invalid batch request: {str(e)}", 
                ErrorCode.INVALID_REQUEST
            )
    
    # Handle single requests/notifications
    if not isinstance(data, dict):
        raise MCPValidationError(
            "JSON-RPC message must be an object or an array", 
            ErrorCode.INVALID_REQUEST
        )
    
    # Check for required fields
    if "jsonrpc" not in data:
        raise MCPValidationError(
            "Missing 'jsonrpc' field", 
            ErrorCode.INVALID_REQUEST
        )
    
    if data.get("jsonrpc") != "2.0":
        raise MCPValidationError(
            f"Invalid JSON-RPC version: {data.get('jsonrpc')}", 
            ErrorCode.INVALID_REQUEST
        )
    
    if "method" not in data:
        raise MCPValidationError(
            "Missing 'method' field", 
            ErrorCode.INVALID_REQUEST
        )
    
    # Determine if it's a request or notification
    if "id" in data:
        try:
            return Request(**data)
        except ValidationError as e:
            raise MCPValidationError(
                f"Invalid request: {str(e)}", 
                ErrorCode.INVALID_REQUEST
            )
    else:
        try:
            return Notification(**data)
        except ValidationError as e:
            raise MCPValidationError(
                f"Invalid notification: {str(e)}", 
                ErrorCode.INVALID_REQUEST
            )


def create_error_response(
    request_id: Optional[Union[str, int]],
    error_code: ErrorCode,
    message: str,
    data: Optional[Any] = None
) -> Response:
    """
    Create an error response.
    
    Args:
        request_id: ID of the request that caused the error
        error_code: Error code
        message: Error message
        data: Additional error data
        
    Returns:
        Response object with error information
    """
    error = Error(code=error_code, message=message, data=data)
    return Response(jsonrpc="2.0", id=request_id, error=error)


def create_result_response(
    request_id: Union[str, int],
    result: Any
) -> Response:
    """
    Create a successful response.
    
    Args:
        request_id: ID of the request
        result: Result data
        
    Returns:
        Response object with result
    """
    return Response(jsonrpc="2.0", id=request_id, result=result)


def create_batch_response(responses: List[Response]) -> BatchResponse:
    """
    Create a batch response.
    
    Args:
        responses: List of Response objects
        
    Returns:
        BatchResponse object
    """
    return BatchResponse(__root__=responses)


def parse_params_model(
    params: Optional[Dict[str, Any]],
    model_class: Type[T]
) -> Tuple[Optional[T], Optional[Error]]:
    """
    Parse and validate parameters against a Pydantic model.
    
    Args:
        params: Parameters to validate
        model_class: Pydantic model class to validate against
        
    Returns:
        Tuple of (parsed model, error) - if validation succeeds, error is None
    """
    if params is None:
        try:
            # Try to create model with empty dict (will work if all fields have defaults)
            return model_class(), None
        except ValidationError:
            return None, Error(
                code=ErrorCode.INVALID_PARAMS,
                message=f"Missing required parameters for {model_class.__name__}",
            )
    
    try:
        return model_class(**params), None
    except ValidationError as e:
        return None, Error(
            code=ErrorCode.INVALID_PARAMS,
            message=f"Invalid parameters for {model_class.__name__}",
            data=str(e),
        )


def serialize_model(model: BaseModel) -> str:
    """
    Serialize a Pydantic model to a JSON string.
    
    Args:
        model: Pydantic model to serialize
        
    Returns:
        JSON string representation of the model
    """
    return model.model_dump_json()


def handle_validation_error(
    e: Union[ValidationError, MCPValidationError],
    request_id: Optional[Union[str, int]] = None
) -> Response:
    """
    Handle a validation error and create an appropriate error response.
    
    Args:
        e: Validation error
        request_id: ID of the request that caused the error
        
    Returns:
        Response object with error information
    """
    if isinstance(e, MCPValidationError):
        return create_error_response(
            request_id,
            e.error_code,
            e.message
        )
    else:  # ValidationError
        return create_error_response(
            request_id,
            ErrorCode.INVALID_PARAMS,
            "Validation error",
            str(e)
        )
