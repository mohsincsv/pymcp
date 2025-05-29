"""
Stdio transport implementation for MCP.

This module provides a transport implementation that communicates over
standard input/output streams, suitable for CLI applications and integration
with language models that use stdio for communication.
"""

import asyncio
import json
import sys
import uuid
from typing import Any, Dict, Optional, Set, Union

from mcp.protocol import (
    BatchRequest,
    BatchResponse,
    Error,
    ErrorCode,
    Notification,
    Request,
    Response,
    parse_json_rpc,
    create_error_response,
    MCPValidationError,
)
from mcp.transport.base import (
    RequestContext,
    RequestHandler,
    Transport,
    TransportCapability,
    TransportError,
    TransportInfo,
    TransportType,
)


class StdioTransport(Transport):
    """
    Transport implementation that communicates over stdin/stdout.
    
    This transport reads JSON-RPC messages from stdin (one per line)
    and writes responses to stdout. It's suitable for CLI applications
    and integration with language models that use stdio.
    """
    
    def __init__(
        self,
        input_stream=None,
        output_stream=None,
        pretty_print: bool = False,
    ) -> None:
        """
        Initialize the stdio transport.
        
        Args:
            input_stream: Input stream to read from (defaults to sys.stdin)
            output_stream: Output stream to write to (defaults to sys.stdout)
            pretty_print: Whether to pretty-print JSON output
        """
        super().__init__()
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout
        self._pretty_print = pretty_print
        self._transport_id = f"stdio-{uuid.uuid4()}"
        self._running = False
        self._handler: Optional[RequestHandler] = None
    
    @property
    def info(self) -> TransportInfo:
        """Get information about the transport."""
        capabilities: Set[TransportCapability] = {
            TransportCapability.BATCH_REQUESTS,
            # Stdio doesn't support true streaming, but we can simulate it with chunked responses
            # TransportCapability.STREAMING,
        }
        return TransportInfo(
            type=TransportType.STDIO,
            capabilities=capabilities,
        )
    
    async def listen(
        self,
        handler: RequestHandler,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start listening for incoming requests on stdin.
        
        Args:
            handler: Handler to process incoming requests
            context: Optional context information
            
        Raises:
            TransportError: If the transport fails to start listening
        """
        if self._running:
            raise TransportError("Stdio transport is already running")
        
        self._running = True
        self._handler = handler
        
        try:
            # Create a task to read from stdin
            read_task = asyncio.create_task(self._read_loop())
            
            # Wait for the read task to complete
            await read_task
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            self._running = False
            raise
        except Exception as e:
            self._running = False
            raise TransportError(f"Error in stdio transport: {str(e)}")
    
    async def _read_loop(self) -> None:
        """
        Read lines from stdin and process them as JSON-RPC messages.
        
        This method runs in a loop until the transport is closed.
        """
        if not self._handler:
            raise TransportError("No request handler set")
        
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        loop = asyncio.get_event_loop()
        
        # Get a file descriptor for stdin
        if hasattr(self._input, "fileno"):
            fd = self._input.fileno()
            await loop.connect_read_pipe(lambda: protocol, open(fd, "rb", buffering=0))
        else:
            # For testing with custom input streams
            transport, _ = await loop.connect_read_pipe(lambda: protocol, self._input)
        
        while self._running:
            try:
                # Read a line from stdin
                line = await reader.readline()
                if not line:
                    # EOF reached
                    self._running = False
                    break
                
                # Process the line
                await self._process_line(line.decode("utf-8").strip())
            except asyncio.CancelledError:
                self._running = False
                break
            except Exception as e:
                # Log the error and continue
                error_response = create_error_response(
                    None,  # We don't know the request ID
                    ErrorCode.INTERNAL_ERROR,
                    f"Error processing request: {str(e)}",
                )
                await self.send(error_response)
    
    async def _process_line(self, line: str) -> None:
        """
        Process a line from stdin as a JSON-RPC message.
        
        Args:
            line: JSON-RPC message as a string
        """
        if not self._handler:
            raise TransportError("No request handler set")
        
        if not line.strip():
            # Skip empty lines
            return
        
        try:
            # Parse the JSON-RPC message
            message = parse_json_rpc(line)
            
            # Create a request context
            context = RequestContext(
                transport_id=self._transport_id,
                session_id=None,  # Stdio doesn't maintain sessions
                request_id=getattr(message, "id", None),
                method=getattr(message, "method", None),
            )
            
            # Handle the message based on its type
            if isinstance(message, Request):
                response = await self._handler.handle_request(message, context)
                await self.send(response)
            elif isinstance(message, Notification):
                await self._handler.handle_notification(message, context)
                # No response for notifications
            elif isinstance(message, BatchRequest):
                response = await self._handler.handle_batch_request(message, context)
                await self.send(response)
            else:
                # This shouldn't happen if parse_json_rpc is working correctly
                error_response = create_error_response(
                    None,
                    ErrorCode.INVALID_REQUEST,
                    "Invalid JSON-RPC message",
                )
                await self.send(error_response)
        except MCPValidationError as e:
            # Handle validation errors
            error_response = create_error_response(
                None,  # We don't know the request ID
                e.error_code,
                e.message,
            )
            await self.send(error_response)
        except json.JSONDecodeError:
            # Handle JSON parsing errors
            error_response = create_error_response(
                None,
                ErrorCode.PARSE_ERROR,
                "Invalid JSON",
            )
            await self.send(error_response)
        except Exception as e:
            # Handle other errors
            error_response = create_error_response(
                None,
                ErrorCode.INTERNAL_ERROR,
                f"Error processing request: {str(e)}",
            )
            await self.send(error_response)
    
    async def send(
        self,
        message: Union[Response, Notification, BatchResponse],
        session_id: Optional[str] = None,
    ) -> None:
        """
        Send a message over stdout.
        
        Args:
            message: Message to send
            session_id: Ignored for stdio transport
            
        Raises:
            TransportError: If the message cannot be sent
        """
        try:
            # Convert the message to JSON
            if self._pretty_print:
                json_str = json.dumps(
                    message.model_dump(exclude_none=True),
                    indent=2,
                )
            else:
                json_str = json.dumps(message.model_dump(exclude_none=True))
            
            # Write the JSON to stdout
            print(json_str, file=self._output, flush=True)
        except Exception as e:
            raise TransportError(f"Error sending message: {str(e)}")
    
    async def close(self) -> None:
        """Close the transport and release resources."""
        self._running = False
        # No resources to release for stdio
