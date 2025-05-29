"""
Transport layer abstractions for MCP.

This module defines the base Transport interface and related types
that all transport implementations must conform to. It provides
a common abstraction over different communication mechanisms
(stdio, HTTP, SSE) used to exchange MCP messages.
"""

import abc
import asyncio
import enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Union, AsyncIterator

from pydantic import BaseModel, Field

from mcp.protocol import (
    Request, 
    Response, 
    Notification, 
    BatchRequest, 
    BatchResponse,
    Error,
    ErrorCode
)


class TransportType(str, enum.Enum):
    """Enumeration of supported transport types."""
    
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    WEBSOCKET = "websocket"


class TransportCapability(str, enum.Enum):
    """Capabilities that a transport may support."""
    
    STREAMING = "streaming"
    BATCH_REQUESTS = "batch_requests"
    CANCELLATION = "cancellation"
    PROGRESS = "progress"
    BIDIRECTIONAL = "bidirectional"


class TransportInfo(BaseModel):
    """Information about a transport implementation."""
    
    type: TransportType = Field(..., description="Type of transport")
    capabilities: Set[TransportCapability] = Field(
        default_factory=set, 
        description="Capabilities supported by this transport"
    )
    
    def supports(self, capability: TransportCapability) -> bool:
        """Check if the transport supports a specific capability."""
        return capability in self.capabilities


class RequestContext(BaseModel):
    """Context information for a request being processed."""
    
    transport_id: str = Field(..., description="Unique identifier for the transport instance")
    session_id: Optional[str] = Field(None, description="Session ID if available")
    request_id: Optional[Union[str, int]] = Field(None, description="ID of the request being processed")
    method: Optional[str] = Field(None, description="Method being called")
    
    # Additional metadata that might be useful for handlers
    user_id: Optional[str] = Field(None, description="User ID if authentication is used")
    client_info: Optional[Dict[str, Any]] = Field(None, description="Client information")
    timestamp: Optional[int] = Field(None, description="Timestamp when the request was received")


class RequestHandler(Protocol):
    """Protocol for request handlers that process MCP requests."""
    
    async def handle_request(
        self, 
        request: Request, 
        context: RequestContext
    ) -> Response:
        """
        Handle a single MCP request.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response to the request
        """
        ...
    
    async def handle_notification(
        self, 
        notification: Notification, 
        context: RequestContext
    ) -> None:
        """
        Handle an MCP notification.
        
        Args:
            notification: The notification to handle
            context: Context information for the notification
        """
        ...
    
    async def handle_batch_request(
        self, 
        batch: BatchRequest, 
        context: RequestContext
    ) -> BatchResponse:
        """
        Handle a batch of MCP requests.
        
        Args:
            batch: The batch request to handle
            context: Context information for the batch
            
        Returns:
            Batch response with results for each request
        """
        ...


class SessionStore(Protocol):
    """Protocol for session stores that manage transport sessions."""
    
    async def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            Session ID
        """
        ...
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: ID of the session to get
            
        Returns:
            Session data or None if not found
        """
        ...
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: ID of the session to update
            data: New session data
            
        Returns:
            True if successful, False if session not found
        """
        ...
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if successful, False if session not found
        """
        ...


class Transport(abc.ABC):
    """
    Abstract base class for MCP transports.
    
    A transport is responsible for receiving MCP requests and sending responses
    over a specific communication channel (stdio, HTTP, SSE, etc.).
    """
    
    def __init__(self) -> None:
        """Initialize the transport."""
        self._session_store: Optional[SessionStore] = None
    
    @property
    @abc.abstractmethod
    def info(self) -> TransportInfo:
        """
        Get information about the transport.
        
        Returns:
            TransportInfo object describing the transport
        """
        ...
    
    @abc.abstractmethod
    async def listen(
        self, 
        handler: RequestHandler, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start listening for incoming requests.
        
        Args:
            handler: Handler to process incoming requests
            context: Optional context information for the transport
            
        Raises:
            TransportError: If the transport fails to start listening
        """
        ...
    
    @abc.abstractmethod
    async def send(
        self, 
        message: Union[Response, Notification, BatchResponse], 
        session_id: Optional[str] = None
    ) -> None:
        """
        Send a message over the transport.
        
        Args:
            message: Message to send
            session_id: Optional session ID to target a specific client
            
        Raises:
            TransportError: If the message cannot be sent
        """
        ...
    
    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the transport and release resources.
        
        Raises:
            TransportError: If the transport fails to close
        """
        ...
    
    def set_session_store(self, session_store: SessionStore) -> None:
        """
        Set the session store for the transport.
        
        Args:
            session_store: Session store to use
        """
        self._session_store = session_store
    
    async def create_session(self) -> str:
        """
        Create a new session using the session store.
        
        Returns:
            Session ID
            
        Raises:
            ValueError: If no session store is set
        """
        if self._session_store is None:
            raise ValueError("No session store set")
        return await self._session_store.create_session()
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data using the session store.
        
        Args:
            session_id: ID of the session to get
            
        Returns:
            Session data or None if not found
            
        Raises:
            ValueError: If no session store is set
        """
        if self._session_store is None:
            raise ValueError("No session store set")
        return await self._session_store.get_session(session_id)
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update session data using the session store.
        
        Args:
            session_id: ID of the session to update
            data: New session data
            
        Returns:
            True if successful, False if session not found
            
        Raises:
            ValueError: If no session store is set
        """
        if self._session_store is None:
            raise ValueError("No session store set")
        return await self._session_store.update_session(session_id, data)


class StreamingResponse(AsyncIterator[str]):
    """
    Base class for streaming responses.
    
    This provides an async iterator interface for transports that support
    streaming responses, such as SSE or WebSockets.
    """
    
    @abc.abstractmethod
    async def __anext__(self) -> str:
        """Get the next chunk of the streaming response."""
        ...
    
    async def close(self) -> None:
        """Close the streaming response."""
        pass


class TransportError(Exception):
    """Exception raised for transport-related errors."""
    
    def __init__(
        self, 
        message: str, 
        code: ErrorCode = ErrorCode.TRANSPORT_ERROR, 
        data: Optional[Any] = None
    ) -> None:
        """
        Initialize a transport error.
        
        Args:
            message: Error message
            code: Error code
            data: Additional error data
        """
        self.message = message
        self.code = code
        self.data = data
        super().__init__(message)
    
    def to_error(self) -> Error:
        """
        Convert to an MCP Error object.
        
        Returns:
            Error object
        """
        return Error(code=self.code, message=self.message, data=self.data)
