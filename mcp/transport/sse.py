"""
HTTP/SSE transport implementation for MCP.

This module provides a transport implementation that communicates over
HTTP with Server-Sent Events (SSE) for streaming responses, suitable for
web applications and browser-based clients.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Set, Union, Callable, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request, Response, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse

from mcp.protocol import (
    BatchRequest,
    BatchResponse,
    Error,
    ErrorCode,
    Notification,
    Request as MCPRequest,
    Response as MCPResponse,
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
    StreamingResponse,
)


class SSEStream(StreamingResponse):
    """
    Implementation of StreamingResponse for SSE.
    
    This class provides an async iterator interface over a queue
    of SSE events, allowing for streaming responses to clients.
    """
    
    def __init__(self, request_id: Union[str, int]):
        """
        Initialize the SSE stream.
        
        Args:
            request_id: ID of the request associated with this stream
        """
        self.request_id = request_id
        self.queue = asyncio.Queue()
        self.closed = False
    
    async def __anext__(self) -> str:
        """Get the next chunk of the streaming response."""
        if self.closed:
            raise StopAsyncIteration
        
        try:
            # Wait for the next chunk with a timeout
            chunk = await asyncio.wait_for(self.queue.get(), timeout=60.0)
            if chunk is None:
                # None is used as a sentinel to indicate the end of the stream
                self.closed = True
                raise StopAsyncIteration
            return chunk
        except asyncio.TimeoutError:
            # If we timeout, close the stream
            self.closed = True
            raise StopAsyncIteration
    
    async def write(self, chunk: str) -> None:
        """
        Write a chunk to the stream.
        
        Args:
            chunk: Chunk to write
        """
        if not self.closed:
            await self.queue.put(chunk)
    
    async def close(self) -> None:
        """Close the stream."""
        if not self.closed:
            self.closed = True
            await self.queue.put(None)


class SSETransport(Transport):
    """
    Transport implementation that communicates over HTTP with SSE for streaming.
    
    This transport provides a FastAPI server with endpoints for JSON-RPC
    communication and SSE for streaming responses.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        cors_origins: List[str] = None,
        path_prefix: str = "",
        enable_docs: bool = True,
    ) -> None:
        """
        Initialize the SSE transport.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            cors_origins: List of allowed CORS origins
            path_prefix: Prefix for all API paths
            enable_docs: Whether to enable FastAPI docs
        """
        super().__init__()
        self._host = host
        self._port = port
        self._cors_origins = cors_origins or ["*"]
        self._path_prefix = path_prefix.rstrip("/")
        self._enable_docs = enable_docs
        self._transport_id = f"sse-{uuid.uuid4()}"
        self._running = False
        self._handler: Optional[RequestHandler] = None
        self._app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._active_streams: Dict[Union[str, int], SSEStream] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    @property
    def info(self) -> TransportInfo:
        """Get information about the transport."""
        capabilities: Set[TransportCapability] = {
            TransportCapability.BATCH_REQUESTS,
            TransportCapability.STREAMING,
            TransportCapability.BIDIRECTIONAL,
        }
        return TransportInfo(
            type=TransportType.SSE,
            capabilities=capabilities,
        )
    
    @property
    def app(self) -> FastAPI:
        """
        Get the FastAPI application.
        
        Returns:
            FastAPI application
            
        Raises:
            TransportError: If the app hasn't been initialized
        """
        if self._app is None:
            raise TransportError("SSE transport app not initialized")
        return self._app
    
    def _create_app(self) -> FastAPI:
        """
        Create the FastAPI application.
        
        Returns:
            FastAPI application
        """
        # Create the FastAPI app
        app = FastAPI(
            title="MCP Server",
            description="Model Context Protocol server with SSE transport",
            docs_url="/docs" if self._enable_docs else None,
            redoc_url="/redoc" if self._enable_docs else None,
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define request model for FastAPI
        class MCPRequestModel(BaseModel):
            jsonrpc: str = Field(..., description="JSON-RPC version")
            id: Optional[Union[str, int]] = Field(None, description="Request ID")
            method: str = Field(..., description="Method name")
            params: Optional[Dict[str, Any]] = Field(None, description="Method parameters")
        
        # Define batch request model for FastAPI
        class MCPBatchRequestModel(BaseModel):
            __root__: List[MCPRequestModel] = Field(..., description="List of requests")
        
        # Add routes
        
        # MCP endpoint
        @app.post(f"{self._path_prefix}/mcp")
        async def handle_mcp_request(
            request_data: Union[MCPRequestModel, MCPBatchRequestModel, Dict[str, Any], List[Dict[str, Any]]],
            request: Request,
            background_tasks: BackgroundTasks,
        ):
            """
            Handle an MCP request.
            
            Args:
                request_data: Request data
                request: FastAPI request object
                background_tasks: FastAPI background tasks
                
            Returns:
                JSON-RPC response
            """
            # Get session ID from cookie or header
            session_id = request.cookies.get("mcp_session") or request.headers.get("X-MCP-Session")
            
            # Create a request context
            context = RequestContext(
                transport_id=self._transport_id,
                session_id=session_id,
                request_id=None,  # Will be set later
                method=None,  # Will be set later
                client_info={
                    "user_agent": request.headers.get("User-Agent"),
                    "remote_addr": request.client.host if request.client else None,
                },
                timestamp=int(asyncio.get_event_loop().time() * 1000),
            )
            
            try:
                # Handle batch request
                if isinstance(request_data, list) or (
                    hasattr(request_data, "__root__") and isinstance(request_data.__root__, list)
                ):
                    # Convert to BatchRequest
                    if hasattr(request_data, "__root__"):
                        batch_data = [item.dict() for item in request_data.__root__]
                    else:
                        batch_data = request_data
                    
                    batch = BatchRequest(__root__=batch_data)
                    response = await self._handler.handle_batch_request(batch, context)
                    return JSONResponse(content=response.model_dump(exclude_none=True))
                
                # Handle single request
                if isinstance(request_data, dict):
                    request_dict = request_data
                else:
                    request_dict = request_data.dict()
                
                # Check if it's a notification (no ID)
                if "id" not in request_dict or request_dict["id"] is None:
                    notification = Notification(**request_dict)
                    context.method = notification.method
                    await self._handler.handle_notification(notification, context)
                    # No response for notifications
                    return Response(status_code=204)
                
                # It's a regular request
                mcp_request = MCPRequest(**request_dict)
                context.request_id = mcp_request.id
                context.method = mcp_request.method
                
                # Check if this is a streaming request
                streaming = False
                if mcp_request.method == "tools/call" and mcp_request.params:
                    streaming = mcp_request.params.get("stream", False)
                
                if streaming:
                    # Create a stream for this request
                    stream = SSEStream(mcp_request.id)
                    self._active_streams[mcp_request.id] = stream
                    
                    # Handle the request in the background
                    background_tasks.add_task(
                        self._handle_streaming_request, mcp_request, context, stream
                    )
                    
                    # Return an SSE response
                    return EventSourceResponse(
                        self._sse_generator(stream),
                        media_type="text/event-stream",
                    )
                
                # Handle regular request
                response = await self._handler.handle_request(mcp_request, context)
                
                # If we have a session ID in the response, set it as a cookie
                result_dict = response.model_dump(exclude_none=True)
                if (
                    response.result 
                    and isinstance(response.result, dict) 
                    and "session_id" in response.result
                ):
                    json_response = JSONResponse(content=result_dict)
                    json_response.set_cookie(
                        key="mcp_session",
                        value=response.result["session_id"],
                        httponly=True,
                        samesite="strict",
                    )
                    return json_response
                
                return JSONResponse(content=result_dict)
            
            except MCPValidationError as e:
                # Handle validation errors
                error_response = create_error_response(
                    getattr(request_data, "id", None),
                    e.error_code,
                    e.message,
                )
                return JSONResponse(
                    content=error_response.model_dump(exclude_none=True),
                    status_code=400,
                )
            except Exception as e:
                # Handle other errors
                error_response = create_error_response(
                    getattr(request_data, "id", None),
                    ErrorCode.INTERNAL_ERROR,
                    f"Error processing request: {str(e)}",
                )
                return JSONResponse(
                    content=error_response.model_dump(exclude_none=True),
                    status_code=500,
                )
        
        # Session endpoint
        @app.post(f"{self._path_prefix}/session")
        async def create_session():
            """
            Create a new session.
            
            Returns:
                Session ID
            """
            if self._session_store:
                session_id = await self._session_store.create_session()
            else:
                session_id = str(uuid.uuid4())
                self._sessions[session_id] = {}
            
            response = JSONResponse(content={"session_id": session_id})
            response.set_cookie(
                key="mcp_session",
                value=session_id,
                httponly=True,
                samesite="strict",
            )
            return response
        
        # Health check endpoint
        @app.get(f"{self._path_prefix}/health")
        async def health_check():
            """
            Health check endpoint.
            
            Returns:
                Status information
            """
            return {
                "status": "ok",
                "transport": self.info.model_dump(),
                "active_streams": len(self._active_streams),
                "sessions": len(self._sessions),
            }
        
        return app
    
    async def _sse_generator(self, stream: SSEStream) -> AsyncIterator[Dict[str, str]]:
        """
        Generate SSE events from a stream.
        
        Args:
            stream: Stream to generate events from
            
        Yields:
            SSE event data
        """
        try:
            async for chunk in stream:
                yield {
                    "event": "message",
                    "data": chunk,
                }
        except Exception as e:
            # Send an error event
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": f"Stream error: {str(e)}",
                    "code": ErrorCode.INTERNAL_ERROR,
                }),
            }
        finally:
            # Clean up the stream
            if stream.request_id in self._active_streams:
                del self._active_streams[stream.request_id]
    
    async def _handle_streaming_request(
        self,
        request: MCPRequest,
        context: RequestContext,
        stream: SSEStream,
    ) -> None:
        """
        Handle a streaming request.
        
        Args:
            request: Request to handle
            context: Request context
            stream: Stream to write response chunks to
        """
        try:
            # Handle the request
            response = await self._handler.handle_request(request, context)
            
            # Write the response as a JSON string
            await stream.write(json.dumps(response.model_dump(exclude_none=True)))
        except Exception as e:
            # Handle errors
            error_response = create_error_response(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Error processing streaming request: {str(e)}",
            )
            await stream.write(json.dumps(error_response.model_dump(exclude_none=True)))
        finally:
            # Close the stream
            await stream.close()
    
    async def listen(
        self,
        handler: RequestHandler,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start listening for incoming requests.
        
        Args:
            handler: Handler to process incoming requests
            context: Optional context information
            
        Raises:
            TransportError: If the transport fails to start listening
        """
        if self._running:
            raise TransportError("SSE transport is already running")
        
        self._running = True
        self._handler = handler
        
        # Create the FastAPI app
        self._app = self._create_app()
        
        # Configure Uvicorn server
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            loop="asyncio",
        )
        
        # Create and start the server
        self._server = uvicorn.Server(config)
        
        try:
            await self._server.serve()
        except Exception as e:
            self._running = False
            raise TransportError(f"Error starting SSE transport: {str(e)}")
        finally:
            self._running = False
    
    async def send(
        self,
        message: Union[MCPResponse, Notification, BatchResponse],
        session_id: Optional[str] = None,
    ) -> None:
        """
        Send a message over the transport.
        
        For SSE transport, this only works for streaming responses where
        the client is still connected. For regular responses, the response
        is sent directly in the HTTP response.
        
        Args:
            message: Message to send
            session_id: Optional session ID to target a specific client
            
        Raises:
            TransportError: If the message cannot be sent
        """
        # For SSE, we can only send messages to active streams
        if isinstance(message, MCPResponse) and message.id in self._active_streams:
            stream = self._active_streams[message.id]
            try:
                await stream.write(json.dumps(message.model_dump(exclude_none=True)))
            except Exception as e:
                raise TransportError(f"Error sending message: {str(e)}")
        else:
            # For regular responses, we can't send them outside of the HTTP response cycle
            # This is a no-op
            pass
    
    async def close(self) -> None:
        """
        Close the transport and release resources.
        
        Raises:
            TransportError: If the transport fails to close
        """
        self._running = False
        
        # Close all active streams
        for stream in self._active_streams.values():
            await stream.close()
        self._active_streams.clear()
        
        # Stop the server
        if self._server:
            self._server.should_exit = True
