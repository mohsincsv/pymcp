"""
MCP client implementation.

This module provides a comprehensive client for interacting with MCP servers,
supporting both stdio and HTTP/SSE transports. It handles all MCP protocol
methods and provides a convenient API for client applications.
"""

import asyncio
import json
import logging
import sys
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable, TypeVar, Generic

import httpx

from mcp.protocol import (
    # Base types
    Request,
    Response,
    Notification,
    Error,
    ErrorCode,
    BatchRequest,
    BatchResponse,
    
    # Method enum and models
    MCPMethod,
    ServerInfo,
    ServerCapabilities,
    ClientInfo,
    ClientCapabilities,
    InitializeParams,
    InitializeResult,
    PingParams,
    PingResult,
    ToolsListParams,
    ToolsListResult,
    ToolCallParams,
    ToolCallResult,
    ToolCallStreamChunk,
    PromptsListParams,
    PromptsListResult,
    PromptsGetParams,
    PromptsGetResult,
    ResourcesListParams,
    ResourcesListResult,
    ResourcesReadParams,
    ResourcesReadResult,
    
    # Validation utilities
    parse_json_rpc,
    create_error_response,
    MCPValidationError,
)


# Type variable for generic result types
T = TypeVar('T')


class ClientError(Exception):
    """Exception raised for client-related errors."""
    
    def __init__(self, message: str, error_code: Optional[int] = None, data: Optional[Any] = None):
        """
        Initialize a client error.
        
        Args:
            message: Error message
            error_code: Optional error code
            data: Optional error data
        """
        self.message = message
        self.error_code = error_code
        self.data = data
        super().__init__(message)


class ClientResponse(Generic[T]):
    """
    Response from an MCP client request.
    
    This class wraps the raw JSON-RPC response and provides access to
    the result or error information.
    """
    
    def __init__(self, response: Response, result_type: Optional[type] = None):
        """
        Initialize a client response.
        
        Args:
            response: Raw JSON-RPC response
            result_type: Optional type to cast the result to
        """
        self.response = response
        self.result_type = result_type
    
    @property
    def id(self) -> Union[str, int]:
        """Get the request ID."""
        return self.response.id
    
    @property
    def has_result(self) -> bool:
        """Check if the response has a result."""
        return self.response.result is not None
    
    @property
    def has_error(self) -> bool:
        """Check if the response has an error."""
        return self.response.error is not None
    
    @property
    def result(self) -> T:
        """
        Get the response result.
        
        Returns:
            Response result
            
        Raises:
            ClientError: If the response has an error
        """
        if self.has_error:
            error = self.response.error
            raise ClientError(
                message=error.message,
                error_code=error.code,
                data=error.data,
            )
        
        if self.result_type and isinstance(self.response.result, dict):
            # Try to cast the result to the specified type
            try:
                return self.result_type(**self.response.result)
            except Exception as e:
                # If casting fails, return the raw result
                return self.response.result
        
        return self.response.result
    
    @property
    def error(self) -> Optional[Error]:
        """Get the response error, if any."""
        return self.response.error
    
    def __str__(self) -> str:
        """String representation of the response."""
        if self.has_result:
            return f"ClientResponse(id={self.id}, result={self.result})"
        else:
            return f"ClientResponse(id={self.id}, error={self.error})"


class StreamingResponse(AsyncIterator[Any]):
    """
    Streaming response from an MCP client request.
    
    This class provides an async iterator interface for streaming responses,
    such as those from tool calls with streaming enabled.
    """
    
    def __init__(self, iterator: AsyncIterator[Any]):
        """
        Initialize a streaming response.
        
        Args:
            iterator: Async iterator providing response chunks
        """
        self._iterator = iterator
        self._closed = False
    
    async def __anext__(self) -> Any:
        """Get the next chunk of the streaming response."""
        if self._closed:
            raise StopAsyncIteration
        
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise
    
    async def close(self) -> None:
        """Close the streaming response."""
        self._closed = True


class StdioTransport:
    """
    Stdio transport for MCP client.
    
    This class provides a transport implementation that communicates with
    an MCP server over standard input/output.
    """
    
    def __init__(
        self,
        command: Optional[List[str]] = None,
        input_stream=None,
        output_stream=None,
    ):
        """
        Initialize the stdio transport.
        
        Args:
            command: Optional command to start the server process
            input_stream: Input stream to read from (defaults to sys.stdin)
            output_stream: Output stream to write to (defaults to sys.stdout)
        """
        self.command = command
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self.process = None
        self._request_queue = asyncio.Queue()
        self._response_handlers = {}
        self._notification_handlers = {}
        self._running = False
        self._reader_task = None
        self._writer_task = None
        self._logger = logging.getLogger("pymcp.client.stdio")
    
    async def connect(self) -> None:
        """
        Connect to the server.
        
        If a command is provided, starts the server process.
        
        Raises:
            ClientError: If connection fails
        """
        if self._running:
            return
        
        try:
            if self.command:
                # Start the server process
                self.process = await asyncio.create_subprocess_exec(
                    *self.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                # Use the process streams
                self.input_stream = self.process.stdout
                self.output_stream = self.process.stdin
            
            # Start the reader and writer tasks
            self._running = True
            self._reader_task = asyncio.create_task(self._read_loop())
            self._writer_task = asyncio.create_task(self._write_loop())
        
        except Exception as e:
            self._running = False
            raise ClientError(f"Failed to connect: {str(e)}")
    
    async def disconnect(self) -> None:
        """
        Disconnect from the server.
        
        If a server process was started, terminates it.
        """
        if not self._running:
            return
        
        self._running = False
        
        # Cancel the reader and writer tasks
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
            self._writer_task = None
        
        # Terminate the server process if it was started
        if self.process:
            self.process.terminate()
            try:
                await self.process.wait()
            except Exception:
                pass
            self.process = None
    
    async def send_request(self, request: Request) -> Response:
        """
        Send a request to the server.
        
        Args:
            request: Request to send
            
        Returns:
            Server response
            
        Raises:
            ClientError: If the request fails
        """
        if not self._running:
            raise ClientError("Not connected to server")
        
        # Create a future to receive the response
        future = asyncio.Future()
        self._response_handlers[request.id] = future
        
        # Queue the request for sending
        await self._request_queue.put(request)
        
        try:
            # Wait for the response
            return await future
        except asyncio.CancelledError:
            # Remove the response handler
            if request.id in self._response_handlers:
                del self._response_handlers[request.id]
            raise
    
    async def send_notification(self, notification: Notification) -> None:
        """
        Send a notification to the server.
        
        Args:
            notification: Notification to send
            
        Raises:
            ClientError: If the notification fails
        """
        if not self._running:
            raise ClientError("Not connected to server")
        
        # Queue the notification for sending
        await self._request_queue.put(notification)
    
    async def _read_loop(self) -> None:
        """
        Read loop for receiving messages from the server.
        
        This method runs in a separate task and processes messages
        from the server, dispatching them to the appropriate handlers.
        """
        try:
            while self._running:
                # Read a line from the input stream
                if isinstance(self.input_stream, asyncio.StreamReader):
                    line = await self.input_stream.readline()
                    if not line:
                        # EOF reached
                        self._running = False
                        break
                    line = line.decode("utf-8").strip()
                else:
                    # For non-asyncio streams, use readline
                    line = self.input_stream.readline().strip()
                    if not line:
                        # EOF reached
                        self._running = False
                        break
                
                # Process the line
                await self._process_line(line)
        
        except asyncio.CancelledError:
            # Task was cancelled
            self._running = False
        
        except Exception as e:
            # Unexpected error
            self._logger.error(f"Error in read loop: {str(e)}")
            self._running = False
    
    async def _write_loop(self) -> None:
        """
        Write loop for sending messages to the server.
        
        This method runs in a separate task and sends messages
        from the request queue to the server.
        """
        try:
            while self._running:
                # Get a message from the queue
                message = await self._request_queue.get()
                
                # Convert the message to JSON
                json_str = message.model_dump_json(exclude_none=True)
                
                # Write the JSON to the output stream
                if isinstance(self.output_stream, asyncio.StreamWriter):
                    self.output_stream.write((json_str + "\n").encode("utf-8"))
                    await self.output_stream.drain()
                else:
                    # For non-asyncio streams, use write
                    self.output_stream.write(json_str + "\n")
                    self.output_stream.flush()
                
                # Mark the task as done
                self._request_queue.task_done()
        
        except asyncio.CancelledError:
            # Task was cancelled
            self._running = False
        
        except Exception as e:
            # Unexpected error
            self._logger.error(f"Error in write loop: {str(e)}")
            self._running = False
    
    async def _process_line(self, line: str) -> None:
        """
        Process a line from the server.
        
        Args:
            line: Line to process
        """
        if not line:
            return
        
        try:
            # Parse the JSON-RPC message
            message = parse_json_rpc(line)
            
            if isinstance(message, Response):
                # Handle response
                if message.id in self._response_handlers:
                    # Resolve the future with the response
                    future = self._response_handlers.pop(message.id)
                    future.set_result(message)
                else:
                    # Unexpected response
                    self._logger.warning(f"Received response with unknown ID: {message.id}")
            
            elif isinstance(message, Notification):
                # Handle notification
                if message.method in self._notification_handlers:
                    # Call the notification handler
                    handler = self._notification_handlers[message.method]
                    asyncio.create_task(handler(message))
                else:
                    # Unexpected notification
                    self._logger.debug(f"Received notification with unknown method: {message.method}")
            
            elif isinstance(message, BatchResponse):
                # Handle batch response
                for response in message:
                    if response.id in self._response_handlers:
                        # Resolve the future with the response
                        future = self._response_handlers.pop(response.id)
                        future.set_result(response)
                    else:
                        # Unexpected response
                        self._logger.warning(f"Received response with unknown ID: {response.id}")
            
            else:
                # Unexpected message type
                self._logger.warning(f"Received unexpected message type: {type(message)}")
        
        except MCPValidationError as e:
            # Invalid JSON-RPC message
            self._logger.error(f"Invalid JSON-RPC message: {str(e)}")
        
        except json.JSONDecodeError as e:
            # Invalid JSON
            self._logger.error(f"Invalid JSON: {str(e)}")
        
        except Exception as e:
            # Unexpected error
            self._logger.error(f"Error processing message: {str(e)}")
    
    def register_notification_handler(
        self,
        method: str,
        handler: Callable[[Notification], None],
    ) -> None:
        """
        Register a handler for notifications.
        
        Args:
            method: Notification method
            handler: Handler function
        """
        self._notification_handlers[method] = handler


class HTTPTransport:
    """
    HTTP transport for MCP client.
    
    This class provides a transport implementation that communicates with
    an MCP server over HTTP, with support for SSE streaming.
    """
    
    def __init__(
        self,
        url: str,
        session_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the HTTP transport.
        
        Args:
            url: URL of the MCP server
            session_id: Optional session ID
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip("/")
        self.session_id = session_id
        self.headers = headers or {}
        self.timeout = timeout
        self.client = None
        self._notification_handlers = {}
        self._logger = logging.getLogger("pymcp.client.http")
    
    async def connect(self) -> None:
        """
        Connect to the server.
        
        Creates an HTTP client session.
        
        Raises:
            ClientError: If connection fails
        """
        try:
            # Create an HTTP client
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.headers,
            )
            
            # Check if the server is reachable
            response = await self.client.get(f"{self.url}/health")
            if response.status_code != 200:
                raise ClientError(f"Server returned status code {response.status_code}")
        
        except Exception as e:
            if self.client:
                await self.client.aclose()
                self.client = None
            raise ClientError(f"Failed to connect: {str(e)}")
    
    async def disconnect(self) -> None:
        """
        Disconnect from the server.
        
        Closes the HTTP client session.
        """
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def send_request(self, request: Request) -> Response:
        """
        Send a request to the server.
        
        Args:
            request: Request to send
            
        Returns:
            Server response
            
        Raises:
            ClientError: If the request fails
        """
        if not self.client:
            raise ClientError("Not connected to server")
        
        # Prepare the request data
        data = request.model_dump_json(exclude_none=True)
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.session_id:
            headers["X-MCP-Session"] = self.session_id
        
        try:
            # Send the request
            response = await self.client.post(
                f"{self.url}/mcp",
                content=data,
                headers=headers,
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                raise ClientError(f"Server returned status code {response.status_code}")
            
            # Parse the response
            try:
                response_data = response.json()
                return Response(**response_data)
            except Exception as e:
                raise ClientError(f"Invalid response: {str(e)}")
        
        except httpx.RequestError as e:
            # HTTP request error
            raise ClientError(f"HTTP request failed: {str(e)}")
        
        except Exception as e:
            # Unexpected error
            raise ClientError(f"Error sending request: {str(e)}")
    
    async def send_notification(self, notification: Notification) -> None:
        """
        Send a notification to the server.
        
        Args:
            notification: Notification to send
            
        Raises:
            ClientError: If the notification fails
        """
        if not self.client:
            raise ClientError("Not connected to server")
        
        # Prepare the notification data
        data = notification.model_dump_json(exclude_none=True)
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.session_id:
            headers["X-MCP-Session"] = self.session_id
        
        try:
            # Send the notification
            response = await self.client.post(
                f"{self.url}/mcp",
                content=data,
                headers=headers,
            )
            
            # Check if the notification was accepted
            if response.status_code not in (200, 202, 204):
                raise ClientError(f"Server returned status code {response.status_code}")
        
        except httpx.RequestError as e:
            # HTTP request error
            raise ClientError(f"HTTP request failed: {str(e)}")
        
        except Exception as e:
            # Unexpected error
            raise ClientError(f"Error sending notification: {str(e)}")
    
    async def stream_request(self, request: Request) -> AsyncIterator[Any]:
        """
        Send a streaming request to the server.
        
        Args:
            request: Request to send
            
        Returns:
            Async iterator of response chunks
            
        Raises:
            ClientError: If the request fails
        """
        if not self.client:
            raise ClientError("Not connected to server")
        
        # Prepare the request data
        data = request.model_dump_json(exclude_none=True)
        
        # Set up headers for SSE
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        if self.session_id:
            headers["X-MCP-Session"] = self.session_id
        
        try:
            # Send the request with streaming response
            async with self.client.stream(
                "POST",
                f"{self.url}/mcp",
                content=data,
                headers=headers,
            ) as response:
                # Check if the request was accepted
                if response.status_code != 200:
                    raise ClientError(f"Server returned status code {response.status_code}")
                
                # Process the SSE stream
                async for line in response.aiter_lines():
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Parse SSE event
                    if line.startswith("data:"):
                        # Extract the data
                        data = line[5:].strip()
                        
                        try:
                            # Parse the JSON data
                            event_data = json.loads(data)
                            
                            # Check if it's the last chunk
                            if event_data.get("is_last", False):
                                # Check for errors
                                if "error" in event_data:
                                    raise ClientError(
                                        message=event_data["error"],
                                        error_code=ErrorCode.TOOL_EXECUTION_ERROR,
                                    )
                                break
                            
                            # Yield the chunk
                            yield event_data
                        
                        except json.JSONDecodeError as e:
                            # Invalid JSON
                            self._logger.error(f"Invalid JSON in SSE event: {str(e)}")
                            continue
        
        except httpx.RequestError as e:
            # HTTP request error
            raise ClientError(f"HTTP request failed: {str(e)}")
        
        except Exception as e:
            # Unexpected error
            if not isinstance(e, ClientError):
                raise ClientError(f"Error streaming request: {str(e)}")
            raise
    
    def register_notification_handler(
        self,
        method: str,
        handler: Callable[[Notification], None],
    ) -> None:
        """
        Register a handler for notifications.
        
        Note: HTTP transport does not support server-initiated notifications.
        This method is provided for compatibility with the StdioTransport interface.
        
        Args:
            method: Notification method
            handler: Handler function
        """
        self._logger.warning("HTTP transport does not support server-initiated notifications")
        self._notification_handlers[method] = handler


class Client:
    """
    MCP client for interacting with MCP servers.
    
    This class provides a high-level API for interacting with MCP servers,
    supporting both stdio and HTTP/SSE transports. It handles all MCP protocol
    methods and provides a convenient API for client applications.
    """
    
    def __init__(
        self,
        transport: Union[StdioTransport, HTTPTransport],
        client_info: Optional[ClientInfo] = None,
        client_capabilities: Optional[ClientCapabilities] = None,
    ):
        """
        Initialize the MCP client.
        
        Args:
            transport: Transport to use for communication
            client_info: Optional client information
            client_capabilities: Optional client capabilities
        """
        self.transport = transport
        self.client_info = client_info or ClientInfo(
            name="pymcp-client",
            version="0.1.0",
        )
        self.client_capabilities = client_capabilities or ClientCapabilities(
            supports_tool_streaming=True,
            supports_batch_requests=True,
            supports_cancellation=True,
            supports_progress=True,
        )
        
        self.server_info = None
        self.server_capabilities = None
        self.session_id = None
        
        self._logger = logging.getLogger("pymcp.client")
        self._next_id = 0
    
    async def connect(self) -> None:
        """
        Connect to the server and initialize the session.
        
        Raises:
            ClientError: If connection or initialization fails
        """
        # Connect to the server
        await self.transport.connect()
        
        # Initialize the session
        await self.initialize()
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        await self.transport.disconnect()
    
    async def initialize(self) -> ClientResponse[InitializeResult]:
        """
        Initialize the session with the server.
        
        Returns:
            Server information and capabilities
            
        Raises:
            ClientError: If initialization fails
        """
        # Create the initialize request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.INITIALIZE,
            params=InitializeParams(
                client_info=self.client_info,
                client_capabilities=self.client_capabilities,
            ).model_dump(),
        )
        
        # Send the request
        response = await self.transport.send_request(request)
        
        # Check for errors
        if response.error:
            raise ClientError(
                message=f"Initialization failed: {response.error.message}",
                error_code=response.error.code,
                data=response.error.data,
            )
        
        # Parse the result
        try:
            result = InitializeResult(**response.result)
            
            # Store the server information
            self.server_info = result.server_info
            self.server_capabilities = result.server_capabilities
            
            # Store the session ID if provided
            if result.session_id:
                self.session_id = result.session_id
                
                # Set the session ID for HTTP transport
                if isinstance(self.transport, HTTPTransport):
                    self.transport.session_id = result.session_id
            
            return ClientResponse(response, InitializeResult)
        
        except Exception as e:
            raise ClientError(f"Invalid initialization response: {str(e)}")
    
    async def ping(self) -> ClientResponse[PingResult]:
        """
        Ping the server to check connection health.
        
        Returns:
            Server timestamp
            
        Raises:
            ClientError: If the ping fails
        """
        # Create the ping request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.PING,
            params=PingParams().model_dump(),
        )
        
        # Send the request
        response = await self.transport.send_request(request)
        
        return ClientResponse(response, PingResult)
    
    async def list_tools(self) -> ClientResponse[ToolsListResult]:
        """
        List available tools on the server.
        
        Returns:
            List of available tools
            
        Raises:
            ClientError: If the request fails
        """
        # Create the tools/list request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.TOOLS_LIST,
            params=ToolsListParams().model_dump(),
        )
        
        # Send the request
        response = await self.transport.send_request(request)
        
        return ClientResponse(response, ToolsListResult)
    
    async def call_tool(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Union[ClientResponse[ToolCallResult], StreamingResponse]:
        """
        Call a tool on the server.
        
        Args:
            name: Name of the tool to call
            args: Arguments to pass to the tool
            stream: Whether to stream the results
            
        Returns:
            Tool result or streaming response
            
        Raises:
            ClientError: If the request fails
        """
        # Check if streaming is supported
        if stream and (
            not self.server_capabilities or
            not self.server_capabilities.supports_tool_streaming
        ):
            raise ClientError("Server does not support tool streaming")
        
        # Create the tools/call request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.TOOLS_CALL,
            params=ToolCallParams(
                name=name,
                args=args or {},
                stream=stream,
            ).model_dump(),
        )
        
        if stream:
            # Send a streaming request
            if isinstance(self.transport, HTTPTransport):
                # Use HTTP streaming
                iterator = self.transport.stream_request(request)
                return StreamingResponse(iterator)
            else:
                # Stdio transport doesn't support true streaming
                # Fall back to non-streaming request
                self._logger.warning("Stdio transport does not support true streaming, falling back to non-streaming request")
                stream = False
        
        if not stream:
            # Send a regular request
            response = await self.transport.send_request(request)
            return ClientResponse(response, ToolCallResult)
    
    async def list_prompts(
        self,
        tags: Optional[List[str]] = None,
    ) -> ClientResponse[PromptsListResult]:
        """
        List available prompts on the server.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of available prompts
            
        Raises:
            ClientError: If the request fails or prompts are not supported
        """
        # Check if prompts are supported
        if (
            not self.server_capabilities or
            not self.server_capabilities.supports_prompts
        ):
            raise ClientError("Server does not support prompts")
        
        # Create the prompts/list request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.PROMPTS_LIST,
            params=PromptsListParams(
                tags=tags,
            ).model_dump(),
        )
        
        # Send the request
        response = await self.transport.send_request(request)
        
        return ClientResponse(response, PromptsListResult)
    
    async def get_prompt(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> ClientResponse[PromptsGetResult]:
        """
        Get a prompt from the server.
        
        Args:
            name: Name of the prompt to get
            args: Optional arguments to fill in the prompt template
            
        Returns:
            Prompt content and metadata
            
        Raises:
            ClientError: If the request fails or prompts are not supported
        """
        # Check if prompts are supported
        if (
            not self.server_capabilities or
            not self.server_capabilities.supports_prompts
        ):
            raise ClientError("Server does not support prompts")
        
        # Create the prompts/get request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.PROMPTS_GET,
            params=PromptsGetParams(
                name=name,
                args=args,
            ).model_dump(),
        )
        
        # Send the request
        response = await self.transport.send_request(request)
        
        return ClientResponse(response, PromptsGetResult)
    
    async def list_resources(
        self,
        tags: Optional[List[str]] = None,
    ) -> ClientResponse[ResourcesListResult]:
        """
        List available resources on the server.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of available resources
            
        Raises:
            ClientError: If the request fails or resources are not supported
        """
        # Check if resources are supported
        if (
            not self.server_capabilities or
            not self.server_capabilities.supports_resources
        ):
            raise ClientError("Server does not support resources")
        
        # Create the resources/list request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.RESOURCES_LIST,
            params=ResourcesListParams(
                tags=tags,
            ).model_dump(),
        )
        
        # Send the request
        response = await self.transport.send_request(request)
        
        return ClientResponse(response, ResourcesListResult)
    
    async def read_resource(
        self,
        name: str,
    ) -> ClientResponse[ResourcesReadResult]:
        """
        Read a resource from the server.
        
        Args:
            name: Name of the resource to read
            
        Returns:
            Resource content and metadata
            
        Raises:
            ClientError: If the request fails or resources are not supported
        """
        # Check if resources are supported
        if (
            not self.server_capabilities or
            not self.server_capabilities.supports_resources
        ):
            raise ClientError("Server does not support resources")
        
        # Create the resources/read request
        request = Request(
            jsonrpc="2.0",
            id=self._get_next_id(),
            method=MCPMethod.RESOURCES_READ,
            params=ResourcesReadParams(
                name=name,
            ).model_dump(),
        )
        
        # Send the request
        response = await self.transport.send_request(request)
        
        return ClientResponse(response, ResourcesReadResult)
    
    async def cancel_request(self, request_id: Union[str, int]) -> None:
        """
        Cancel a request.
        
        Args:
            request_id: ID of the request to cancel
            
        Raises:
            ClientError: If the cancellation fails or is not supported
        """
        # Check if cancellation is supported
        if (
            not self.server_capabilities or
            not self.server_capabilities.supports_cancellation
        ):
            raise ClientError("Server does not support cancellation")
        
        # Create the cancellation notification
        notification = Notification(
            jsonrpc="2.0",
            method=MCPMethod.CANCEL_REQUEST,
            params={
                "id": request_id,
            },
        )
        
        # Send the notification
        await self.transport.send_notification(notification)
    
    def _get_next_id(self) -> str:
        """
        Get the next request ID.
        
        Returns:
            Unique request ID
        """
        self._next_id += 1
        return str(self._next_id)
    
    async def __aenter__(self) -> "Client":
        """
        Enter the async context.
        
        Returns:
            The client instance
        """
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the async context.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        await self.disconnect()


# Convenience functions for creating clients
async def create_stdio_client(
    command: Optional[List[str]] = None,
    client_info: Optional[ClientInfo] = None,
    client_capabilities: Optional[ClientCapabilities] = None,
) -> Client:
    """
    Create a client that communicates with an MCP server over stdio.
    
    Args:
        command: Optional command to start the server process
        client_info: Optional client information
        client_capabilities: Optional client capabilities
        
    Returns:
        Connected MCP client
        
    Raises:
        ClientError: If connection or initialization fails
    """
    # Create the transport
    transport = StdioTransport(command=command)
    
    # Create the client
    client = Client(
        transport=transport,
        client_info=client_info,
        client_capabilities=client_capabilities,
    )
    
    # Connect to the server
    await client.connect()
    
    return client


async def create_http_client(
    url: str,
    client_info: Optional[ClientInfo] = None,
    client_capabilities: Optional[ClientCapabilities] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Client:
    """
    Create a client that communicates with an MCP server over HTTP.
    
    Args:
        url: URL of the MCP server
        client_info: Optional client information
        client_capabilities: Optional client capabilities
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        
    Returns:
        Connected MCP client
        
    Raises:
        ClientError: If connection or initialization fails
    """
    # Create the transport
    transport = HTTPTransport(
        url=url,
        headers=headers,
        timeout=timeout,
    )
    
    # Create the client
    client = Client(
        transport=transport,
        client_info=client_info,
        client_capabilities=client_capabilities,
    )
    
    # Connect to the server
    await client.connect()
    
    return client
