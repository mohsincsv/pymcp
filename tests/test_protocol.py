"""
Unit tests for the MCP protocol implementation.

Tests JSON-RPC message parsing, validation, serialization, error handling,
and Pydantic model validation for the MCP protocol.
"""

import json
import pytest
from typing import Dict, Any, List, Union

from mcp.protocol import (
    # Base types
    Request,
    Response,
    Notification,
    Error,
    ErrorCode,
    BatchRequest,
    BatchResponse,
    RequestMetadata,
    
    # Method enum and models
    MCPMethod,
    ServerInfo,
    ServerCapabilities,
    ClientInfo,
    ClientCapabilities,
    InitializeParams,
    InitializeResult,
    ToolSchema,
    ToolParameterSchema,
    
    # Validation utilities
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


class TestBaseProtocol:
    """Tests for the base JSON-RPC 2.0 protocol types."""
    
    def test_request_creation(self):
        """Test creating a valid Request object."""
        request = Request(
            jsonrpc="2.0",
            id="test-1",
            method="test/method",
            params={"foo": "bar"}
        )
        assert request.jsonrpc == "2.0"
        assert request.id == "test-1"
        assert request.method == "test/method"
        assert request.params == {"foo": "bar"}
    
    def test_request_validation(self):
        """Test validation of Request objects."""
        # Missing ID should raise error
        with pytest.raises(ValueError):
            Request(jsonrpc="2.0", method="test/method")
        
        # Invalid jsonrpc version should raise error
        with pytest.raises(ValueError):
            Request(jsonrpc="1.0", id="test-1", method="test/method")
    
    def test_response_creation(self):
        """Test creating Response objects."""
        # Success response
        success = Response(
            jsonrpc="2.0",
            id="test-1",
            result={"status": "ok"}
        )
        assert success.jsonrpc == "2.0"
        assert success.id == "test-1"
        assert success.result == {"status": "ok"}
        assert success.error is None
        
        # Error response
        error = Response(
            jsonrpc="2.0",
            id="test-1",
            error=Error(code=ErrorCode.INVALID_REQUEST, message="Invalid request")
        )
        assert error.jsonrpc == "2.0"
        assert error.id == "test-1"
        assert error.result is None
        assert error.error.code == ErrorCode.INVALID_REQUEST
        assert error.error.message == "Invalid request"
    
    def test_response_validation(self):
        """Test validation of Response objects."""
        # Response must have either result or error
        with pytest.raises(ValueError):
            Response(jsonrpc="2.0", id="test-1")
        
        # Response cannot have both result and error
        with pytest.raises(ValueError):
            Response(
                jsonrpc="2.0", 
                id="test-1", 
                result={"status": "ok"}, 
                error=Error(code=ErrorCode.INVALID_REQUEST, message="Invalid request")
            )
    
    def test_notification_creation(self):
        """Test creating Notification objects."""
        notification = Notification(
            jsonrpc="2.0",
            method="test/notification",
            params={"event": "update"}
        )
        assert notification.jsonrpc == "2.0"
        assert notification.method == "test/notification"
        assert notification.params == {"event": "update"}
    
    def test_batch_request(self):
        """Test batch request creation and validation."""
        requests = [
            Request(jsonrpc="2.0", id="1", method="method1"),
            Request(jsonrpc="2.0", id="2", method="method2", params={"foo": "bar"})
        ]
        batch = BatchRequest(__root__=requests)
        
        assert len(batch) == 2
        assert batch.get_request_by_id("1").method == "method1"
        assert batch.get_request_by_id("2").params == {"foo": "bar"}
        assert batch.get_request_by_id("3") is None
        
        # Empty batch should fail validation
        with pytest.raises(ValueError):
            BatchRequest(__root__=[])
        
        # Invalid jsonrpc version in any request should fail
        with pytest.raises(ValueError):
            BatchRequest(__root__=[
                {"jsonrpc": "1.0", "id": "1", "method": "method1"},
                {"jsonrpc": "2.0", "id": "2", "method": "method2"}
            ])
    
    def test_batch_response(self):
        """Test batch response creation and validation."""
        responses = [
            Response(jsonrpc="2.0", id="1", result={"status": "ok"}),
            Response(
                jsonrpc="2.0", 
                id="2", 
                error=Error(code=ErrorCode.INVALID_PARAMS, message="Invalid params")
            )
        ]
        batch = BatchResponse(__root__=responses)
        
        assert len(batch) == 2
        # Verify we can iterate through the batch
        response_list = list(batch)
        assert response_list[0].id == "1"
        assert response_list[1].id == "2"
        
        # Invalid jsonrpc version in any response should fail
        with pytest.raises(ValueError):
            BatchResponse(__root__=[
                {"jsonrpc": "1.0", "id": "1", "result": {"status": "ok"}},
                {"jsonrpc": "2.0", "id": "2", "result": {"status": "ok"}}
            ])


class TestMCPMethods:
    """Tests for MCP-specific method models."""
    
    def test_initialize_params(self):
        """Test InitializeParams model."""
        params = InitializeParams(
            client_info=ClientInfo(name="test-client", version="1.0.0"),
            client_capabilities=ClientCapabilities(
                supports_tool_streaming=True,
                supports_batch_requests=True
            )
        )
        assert params.client_info.name == "test-client"
        assert params.client_info.version == "1.0.0"
        assert params.client_capabilities.supports_tool_streaming is True
        assert params.client_capabilities.supports_batch_requests is True
        assert params.client_capabilities.supports_cancellation is False
    
    def test_initialize_result(self):
        """Test InitializeResult model."""
        result = InitializeResult(
            server_info=ServerInfo(name="test-server", version="1.0.0"),
            server_capabilities=ServerCapabilities(
                supports_tool_streaming=True,
                supports_batch_requests=True,
                supports_prompts=True
            ),
            session_id="session-123"
        )
        assert result.server_info.name == "test-server"
        assert result.server_info.version == "1.0.0"
        assert result.server_capabilities.supports_tool_streaming is True
        assert result.server_capabilities.supports_prompts is True
        assert result.session_id == "session-123"
    
    def test_tool_schema(self):
        """Test ToolSchema model."""
        schema = ToolSchema(
            name="test-tool",
            description="A test tool",
            parameters=[
                ToolParameterSchema(
                    name="param1",
                    type="string",
                    description="A string parameter",
                    required=True
                ),
                ToolParameterSchema(
                    name="param2",
                    type="number",
                    description="A number parameter",
                    required=False,
                    default=42
                )
            ],
            streaming=True
        )
        
        assert schema.name == "test-tool"
        assert schema.description == "A test tool"
        assert len(schema.parameters) == 2
        assert schema.parameters[0].name == "param1"
        assert schema.parameters[0].type == "string"
        assert schema.parameters[0].required is True
        assert schema.parameters[1].default == 42
        assert schema.streaming is True


class TestValidation:
    """Tests for protocol validation utilities."""
    
    def test_parse_json_rpc_request(self):
        """Test parsing a JSON-RPC request from a string."""
        json_str = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-1",
            "method": "test/method",
            "params": {"foo": "bar"}
        })
        
        request = parse_json_rpc(json_str)
        assert isinstance(request, Request)
        assert request.id == "test-1"
        assert request.method == "test/method"
        assert request.params == {"foo": "bar"}
    
    def test_parse_json_rpc_notification(self):
        """Test parsing a JSON-RPC notification from a string."""
        json_str = json.dumps({
            "jsonrpc": "2.0",
            "method": "test/notification",
            "params": {"event": "update"}
        })
        
        notification = parse_json_rpc(json_str)
        assert isinstance(notification, Notification)
        assert notification.method == "test/notification"
        assert notification.params == {"event": "update"}
    
    def test_parse_json_rpc_batch(self):
        """Test parsing a JSON-RPC batch request from a string."""
        json_str = json.dumps([
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "method1"
            },
            {
                "jsonrpc": "2.0",
                "id": "2",
                "method": "method2",
                "params": {"foo": "bar"}
            }
        ])
        
        batch = parse_json_rpc(json_str)
        assert isinstance(batch, BatchRequest)
        assert len(batch) == 2
        requests = list(batch)
        assert requests[0].id == "1"
        assert requests[1].params == {"foo": "bar"}
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        with pytest.raises(MCPValidationError) as exc_info:
            parse_json_rpc("{invalid json")
        
        assert exc_info.value.error_code == ErrorCode.PARSE_ERROR
    
    def test_parse_invalid_jsonrpc(self):
        """Test parsing invalid JSON-RPC messages."""
        # Missing jsonrpc field
        with pytest.raises(MCPValidationError) as exc_info:
            parse_json_rpc_object({"id": "1", "method": "test"})
        assert exc_info.value.error_code == ErrorCode.INVALID_REQUEST
        
        # Invalid jsonrpc version
        with pytest.raises(MCPValidationError) as exc_info:
            parse_json_rpc_object({"jsonrpc": "1.0", "id": "1", "method": "test"})
        assert exc_info.value.error_code == ErrorCode.INVALID_REQUEST
        
        # Missing method field
        with pytest.raises(MCPValidationError) as exc_info:
            parse_json_rpc_object({"jsonrpc": "2.0", "id": "1"})
        assert exc_info.value.error_code == ErrorCode.INVALID_REQUEST
    
    def test_create_error_response(self):
        """Test creating error responses."""
        response = create_error_response(
            "test-1",
            ErrorCode.METHOD_NOT_FOUND,
            "Method not found",
            {"requested_method": "unknown/method"}
        )
        
        assert response.jsonrpc == "2.0"
        assert response.id == "test-1"
        assert response.result is None
        assert response.error.code == ErrorCode.METHOD_NOT_FOUND
        assert response.error.message == "Method not found"
        assert response.error.data == {"requested_method": "unknown/method"}
    
    def test_create_result_response(self):
        """Test creating result responses."""
        response = create_result_response("test-1", {"status": "ok", "value": 42})
        
        assert response.jsonrpc == "2.0"
        assert response.id == "test-1"
        assert response.result == {"status": "ok", "value": 42}
        assert response.error is None
    
    def test_create_batch_response(self):
        """Test creating batch responses."""
        responses = [
            create_result_response("1", {"status": "ok"}),
            create_error_response("2", ErrorCode.INVALID_PARAMS, "Invalid params")
        ]
        
        batch = create_batch_response(responses)
        assert isinstance(batch, BatchResponse)
        assert len(batch) == 2
        response_list = list(batch)
        assert response_list[0].id == "1"
        assert response_list[0].result == {"status": "ok"}
        assert response_list[1].id == "2"
        assert response_list[1].error.code == ErrorCode.INVALID_PARAMS
    
    def test_parse_params_model(self):
        """Test parsing and validating parameters against a model."""
        # Valid parameters
        params = {"client_info": {"name": "test-client", "version": "1.0.0"}}
        model, error = parse_params_model(params, InitializeParams)
        
        assert error is None
        assert isinstance(model, InitializeParams)
        assert model.client_info.name == "test-client"
        
        # Invalid parameters
        params = {"client_info": {"name": "test-client"}}  # Missing version
        model, error = parse_params_model(params, InitializeParams)
        
        assert model is None
        assert error is not None
        assert error.code == ErrorCode.INVALID_PARAMS
    
    def test_serialize_model(self):
        """Test serializing models to JSON."""
        model = ServerInfo(name="test-server", version="1.0.0")
        json_str = serialize_model(model)
        
        # Parse the JSON string back to a dict for comparison
        data = json.loads(json_str)
        assert data == {"name": "test-server", "version": "1.0.0"}
    
    def test_handle_validation_error(self):
        """Test handling validation errors."""
        # MCPValidationError
        error = MCPValidationError("Test error", ErrorCode.INVALID_REQUEST)
        response = handle_validation_error(error, "test-1")
        
        assert response.id == "test-1"
        assert response.error.code == ErrorCode.INVALID_REQUEST
        assert response.error.message == "Test error"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_batch(self):
        """Test handling empty batch requests."""
        with pytest.raises(MCPValidationError) as exc_info:
            parse_json_rpc("[]")
        
        assert exc_info.value.error_code == ErrorCode.INVALID_REQUEST
    
    def test_non_object_request(self):
        """Test handling non-object requests."""
        with pytest.raises(MCPValidationError) as exc_info:
            parse_json_rpc_object("not an object")
        
        assert exc_info.value.error_code == ErrorCode.INVALID_REQUEST
        assert "must be an object or an array" in str(exc_info.value)
    
    def test_parse_params_with_defaults(self):
        """Test parsing parameters with default values."""
        # Empty params should use defaults
        model, error = parse_params_model(None, ClientCapabilities)
        
        assert error is None
        assert isinstance(model, ClientCapabilities)
        assert model.supports_tool_streaming is False
        assert model.supports_batch_requests is False
        
        # Model without defaults should fail with empty params
        model, error = parse_params_model(None, ClientInfo)
        
        assert model is None
        assert error is not None
        assert error.code == ErrorCode.INVALID_PARAMS
