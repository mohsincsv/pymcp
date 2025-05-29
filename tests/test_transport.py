"""
Unit tests for the MCP transport implementations.

Tests the transport layer, including the base abstractions, stdio transport,
and SSE transport implementations.
"""

import asyncio
import json
import io
import pytest
from typing import Dict, Any, List, Optional, Set, Union, AsyncIterator, Tuple
from unittest.mock import MagicMock, patch, AsyncMock

import pytest_asyncio
from fastapi.testclient import TestClient

from mcp.protocol import (
    Request,
    Response,
    Notification,
    BatchRequest,
    BatchResponse,
    Error,
    ErrorCode,
    create_error_response,
    create_result_response,
)
from mcp.transport import (
    Transport,
    TransportInfo,
    TransportType,
    TransportCapability,
    RequestContext,
    RequestHandler,
    SessionStore,
    StreamingResponse,
    TransportError,
    StdioTransport,
    SSETransport,
    SSEStream,
)


class MockRequestHandler:
    """Mock implementation of RequestHandler for testing."""
    
    def __init__(self):
        self.handled_requests = []
        self.handled_notifications = []
        self.handled_batches = []
    
    async def handle_request(self, request: Request, context: RequestContext) -> Response:
        """Handle a request and return a mock response."""
        self.handled_requests.append((request, context))
        
        # Return a success response for testing
        return create_result_response(
            request.id,
            {"status": "ok", "method": request.method}
        )
    
    async def handle_notification(self, notification: Notification, context: RequestContext) -> None:
        """Handle a notification."""
        self.handled_notifications.append((notification, context))
    
    async def handle_batch_request(self, batch: BatchRequest, context: RequestContext) -> BatchResponse:
        """Handle a batch request and return a mock batch response."""
        self.handled_batches.append((batch, context))
        
        # Create a response for each request in the batch
        responses = []
        for req in batch:
            responses.append(
                create_result_response(
                    req.id,
                    {"status": "ok", "method": req.method}
                )
            )
        
        return BatchResponse(__root__=responses)


class MockSessionStore:
    """Mock implementation of SessionStore for testing."""
    
    def __init__(self):
        self.sessions = {}
    
    async def create_session(self) -> str:
        """Create a new session with a test ID."""
        session_id = f"test-session-{len(self.sessions) + 1}"
        self.sessions[session_id] = {}
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self.sessions.get(session_id)
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        if session_id in self.sessions:
            self.sessions[session_id].update(data)
            return True
        return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


@pytest_asyncio.fixture
async def mock_handler():
    """Fixture for a mock request handler."""
    return MockRequestHandler()


@pytest_asyncio.fixture
async def mock_session_store():
    """Fixture for a mock session store."""
    return MockSessionStore()


@pytest_asyncio.fixture
async def stdio_transport():
    """Fixture for a stdio transport with string IO streams."""
    input_stream = io.StringIO()
    output_stream = io.StringIO()
    transport = StdioTransport(
        input_stream=input_stream,
        output_stream=output_stream,
        pretty_print=False,
    )
    yield transport, input_stream, output_stream


@pytest_asyncio.fixture
async def sse_transport():
    """Fixture for an SSE transport."""
    transport = SSETransport(
        host="127.0.0.1",
        port=8000,
        cors_origins=["*"],
        path_prefix="",
        enable_docs=True,
    )
    # Create the app but don't start the server
    transport._app = transport._create_app()
    transport._handler = MockRequestHandler()
    yield transport


@pytest.mark.asyncio
class TestBaseTransport:
    """Tests for the base Transport class and related types."""
    
    def test_transport_info(self):
        """Test TransportInfo creation and capability checking."""
        # Create transport info with capabilities
        info = TransportInfo(
            type=TransportType.STDIO,
            capabilities={
                TransportCapability.BATCH_REQUESTS,
                TransportCapability.STREAMING,
            }
        )
        
        # Check type and capabilities
        assert info.type == TransportType.STDIO
        assert TransportCapability.BATCH_REQUESTS in info.capabilities
        assert TransportCapability.STREAMING in info.capabilities
        assert TransportCapability.CANCELLATION not in info.capabilities
        
        # Test supports method
        assert info.supports(TransportCapability.BATCH_REQUESTS) is True
        assert info.supports(TransportCapability.STREAMING) is True
        assert info.supports(TransportCapability.CANCELLATION) is False
    
    def test_request_context(self):
        """Test RequestContext creation and properties."""
        # Create a request context
        context = RequestContext(
            transport_id="test-transport",
            session_id="test-session",
            request_id="test-request",
            method="test/method",
            user_id="test-user",
            client_info={"user_agent": "test-agent"},
            timestamp=123456789,
        )
        
        # Check properties
        assert context.transport_id == "test-transport"
        assert context.session_id == "test-session"
        assert context.request_id == "test-request"
        assert context.method == "test/method"
        assert context.user_id == "test-user"
        assert context.client_info == {"user_agent": "test-agent"}
        assert context.timestamp == 123456789
    
    def test_transport_error(self):
        """Test TransportError creation and conversion to Error."""
        # Create a transport error
        error = TransportError(
            "Test error",
            ErrorCode.INVALID_REQUEST,
            {"detail": "Additional error details"}
        )
        
        # Check properties
        assert error.message == "Test error"
        assert error.code == ErrorCode.INVALID_REQUEST
        assert error.data == {"detail": "Additional error details"}
        
        # Convert to Error
        mcp_error = error.to_error()
        assert isinstance(mcp_error, Error)
        assert mcp_error.code == ErrorCode.INVALID_REQUEST
        assert mcp_error.message == "Test error"
        assert mcp_error.data == {"detail": "Additional error details"}


@pytest.mark.asyncio
class TestStdioTransport:
    """Tests for the StdioTransport implementation."""
    
    async def test_stdio_transport_info(self, stdio_transport):
        """Test StdioTransport info and capabilities."""
        transport, _, _ = stdio_transport
        
        # Check transport info
        info = transport.info
        assert info.type == TransportType.STDIO
        assert TransportCapability.BATCH_REQUESTS in info.capabilities
        # Stdio doesn't support true streaming
        assert TransportCapability.STREAMING not in info.capabilities
    
    async def test_stdio_transport_session_store(self, stdio_transport, mock_session_store):
        """Test setting and using session store with StdioTransport."""
        transport, _, _ = stdio_transport
        
        # Set session store
        transport.set_session_store(mock_session_store)
        
        # Create a session
        session_id = await transport.create_session()
        assert session_id.startswith("test-session-")
        
        # Get session data
        session_data = await transport.get_session(session_id)
        assert session_data == {}
        
        # Update session data
        success = await transport.update_session(session_id, {"key": "value"})
        assert success is True
        
        # Get updated session data
        session_data = await transport.get_session(session_id)
        assert session_data == {"key": "value"}
    
    async def test_stdio_transport_send(self, stdio_transport):
        """Test sending messages through StdioTransport."""
        transport, _, output_stream = stdio_transport
        
        # Create a response to send
        response = create_result_response("test-1", {"status": "ok"})
        
        # Send the response
        await transport.send(response)
        
        # Check output
        output = output_stream.getvalue()
        assert output.strip()
        
        # Parse the output as JSON
        output_json = json.loads(output)
        assert output_json["jsonrpc"] == "2.0"
        assert output_json["id"] == "test-1"
        assert output_json["result"]["status"] == "ok"
    
    async def test_stdio_transport_process_line(self, stdio_transport, mock_handler):
        """Test processing a line of input with StdioTransport."""
        transport, _, _ = stdio_transport
        transport._handler = mock_handler
        
        # Process a valid request
        request_line = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-1",
            "method": "test/method",
            "params": {"foo": "bar"}
        })
        
        await transport._process_line(request_line)
        
        # Check that the handler was called
        assert len(mock_handler.handled_requests) == 1
        request, context = mock_handler.handled_requests[0]
        assert request.id == "test-1"
        assert request.method == "test/method"
        assert request.params == {"foo": "bar"}
    
    async def test_stdio_transport_invalid_json(self, stdio_transport, mock_handler):
        """Test handling invalid JSON input with StdioTransport."""
        transport, _, output_stream = stdio_transport
        transport._handler = mock_handler
        
        # Process invalid JSON
        await transport._process_line("{invalid json")
        
        # Check output for error response
        output = output_stream.getvalue()
        output_json = json.loads(output)
        assert output_json["jsonrpc"] == "2.0"
        assert output_json["error"]["code"] == ErrorCode.PARSE_ERROR
        assert "Invalid JSON" in output_json["error"]["message"]
    
    async def test_stdio_transport_batch_request(self, stdio_transport, mock_handler):
        """Test processing a batch request with StdioTransport."""
        transport, _, output_stream = stdio_transport
        transport._handler = mock_handler
        
        # Process a batch request
        batch_line = json.dumps([
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "method1",
            },
            {
                "jsonrpc": "2.0",
                "id": "2",
                "method": "method2",
                "params": {"foo": "bar"}
            }
        ])
        
        await transport._process_line(batch_line)
        
        # Check that the handler was called with a batch
        assert len(mock_handler.handled_batches) == 1
        batch, context = mock_handler.handled_batches[0]
        assert len(batch) == 2
        assert batch.get_request_by_id("1").method == "method1"
        assert batch.get_request_by_id("2").params == {"foo": "bar"}
        
        # Check output for batch response
        output = output_stream.getvalue()
        output_json = json.loads(output)
        assert isinstance(output_json, list)
        assert len(output_json) == 2
        assert output_json[0]["id"] == "1"
        assert output_json[1]["id"] == "2"
    
    async def test_stdio_transport_notification(self, stdio_transport, mock_handler):
        """Test processing a notification with StdioTransport."""
        transport, _, output_stream = stdio_transport
        transport._handler = mock_handler
        
        # Process a notification
        notification_line = json.dumps({
            "jsonrpc": "2.0",
            "method": "test/notification",
            "params": {"event": "update"}
        })
        
        await transport._process_line(notification_line)
        
        # Check that the handler was called with a notification
        assert len(mock_handler.handled_notifications) == 1
        notification, context = mock_handler.handled_notifications[0]
        assert notification.method == "test/notification"
        assert notification.params == {"event": "update"}
        
        # No response should be sent for notifications
        assert output_stream.getvalue() == ""


@pytest.mark.asyncio
class TestSSETransport:
    """Tests for the SSETransport implementation."""
    
    async def test_sse_transport_info(self, sse_transport):
        """Test SSETransport info and capabilities."""
        transport = sse_transport
        
        # Check transport info
        info = transport.info
        assert info.type == TransportType.SSE
        assert TransportCapability.BATCH_REQUESTS in info.capabilities
        assert TransportCapability.STREAMING in info.capabilities
        assert TransportCapability.BIDIRECTIONAL in info.capabilities
    
    async def test_sse_transport_app_creation(self, sse_transport):
        """Test SSETransport FastAPI app creation."""
        transport = sse_transport
        
        # Get the app
        app = transport.app
        
        # Check that the app has the expected routes
        routes = [route.path for route in app.routes]
        assert "/mcp" in routes
        assert "/session" in routes
        assert "/health" in routes
    
    async def test_sse_transport_health_endpoint(self, sse_transport):
        """Test SSETransport health endpoint."""
        transport = sse_transport
        
        # Create a test client
        client = TestClient(transport.app)
        
        # Call the health endpoint
        response = client.get("/health")
        
        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["transport"]["type"] == "sse"
        assert "active_streams" in data
        assert "sessions" in data
    
    async def test_sse_transport_session_endpoint(self, sse_transport):
        """Test SSETransport session endpoint."""
        transport = sse_transport
        
        # Set a mock session store
        session_store = MockSessionStore()
        transport.set_session_store(session_store)
        
        # Create a test client
        client = TestClient(transport.app)
        
        # Call the session endpoint
        response = client.post("/session")
        
        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        
        # Check that a cookie was set
        assert "mcp_session" in response.cookies
        assert response.cookies["mcp_session"] == data["session_id"]
    
    async def test_sse_transport_mcp_endpoint(self, sse_transport):
        """Test SSETransport MCP endpoint for regular requests."""
        transport = sse_transport
        
        # Create a test client
        client = TestClient(transport.app)
        
        # Call the MCP endpoint with a request
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "test/method",
                "params": {"foo": "bar"}
            }
        )
        
        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-1"
        assert data["result"]["status"] == "ok"
        assert data["result"]["method"] == "test/method"
    
    async def test_sse_transport_mcp_endpoint_batch(self, sse_transport):
        """Test SSETransport MCP endpoint for batch requests."""
        transport = sse_transport
        
        # Create a test client
        client = TestClient(transport.app)
        
        # Call the MCP endpoint with a batch request
        response = client.post(
            "/mcp",
            json=[
                {
                    "jsonrpc": "2.0",
                    "id": "1",
                    "method": "method1",
                },
                {
                    "jsonrpc": "2.0",
                    "id": "2",
                    "method": "method2",
                    "params": {"foo": "bar"}
                }
            ]
        )
        
        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "1"
        assert data[1]["id"] == "2"
    
    async def test_sse_transport_mcp_endpoint_notification(self, sse_transport):
        """Test SSETransport MCP endpoint for notifications."""
        transport = sse_transport
        
        # Create a test client
        client = TestClient(transport.app)
        
        # Call the MCP endpoint with a notification
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "test/notification",
                "params": {"event": "update"}
            }
        )
        
        # Check the response (no content for notifications)
        assert response.status_code == 204
    
    async def test_sse_transport_mcp_endpoint_error(self, sse_transport):
        """Test SSETransport MCP endpoint error handling."""
        transport = sse_transport
        
        # Create a test client
        client = TestClient(transport.app)
        
        # Call the MCP endpoint with an invalid request
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "1.0",  # Invalid version
                "id": "test-1",
                "method": "test/method"
            }
        )
        
        # Check the response
        assert response.status_code == 400
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["error"]["code"] == ErrorCode.INVALID_REQUEST
    
    async def test_sse_stream(self):
        """Test SSEStream implementation."""
        # Create a stream
        stream = SSEStream("test-stream")
        
        # Write some chunks
        await stream.write("chunk1")
        await stream.write("chunk2")
        
        # Read the chunks
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            if len(chunks) == 2:
                break
        
        # Check the chunks
        assert chunks == ["chunk1", "chunk2"]
        
        # Close the stream
        await stream.close()
        
        # Stream should be closed
        assert stream.closed is True
        
        # Reading from a closed stream should raise StopAsyncIteration
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
