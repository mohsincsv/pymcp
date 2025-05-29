"""
MCP server implementation.

This module provides the main Server class that orchestrates the MCP components,
including transport, session management, request handling, and provider setup.
It manages the server lifecycle and provides a clean API for starting and stopping
the server.
"""

import asyncio
import signal
import sys
from typing import Any, Dict, List, Optional, Set, Type, Union

import structlog
from pydantic import BaseModel, Field

from mcp.protocol import (
    ServerInfo,
    ServerCapabilities,
    TransportCapability,
)
from mcp.server.handler import (
    MCPRequestHandler,
    ToolProvider,
    PromptProvider,
    ResourceProvider,
)
from mcp.server.session import InMemorySessionStore
from mcp.transport.base import Transport, SessionStore


class ServerOptions(BaseModel):
    """Configuration options for the MCP server."""
    
    name: str = Field("pymcp-server", description="Name of the server")
    version: str = Field("0.1.0", description="Version of the server")
    session_expiration_seconds: Optional[int] = Field(
        3600, description="Session expiration time in seconds (None for no expiration)"
    )
    log_level: str = Field("info", description="Log level")
    debug: bool = Field(False, description="Enable debug mode")


class Server:
    """
    Main MCP server class.
    
    This class orchestrates the MCP components, including transport, session management,
    request handling, and provider setup. It manages the server lifecycle and provides
    a clean API for starting and stopping the server.
    """
    
    def __init__(
        self,
        transport: Transport,
        options: Optional[ServerOptions] = None,
        session_store: Optional[SessionStore] = None,
        tool_provider: Optional[ToolProvider] = None,
        prompt_provider: Optional[PromptProvider] = None,
        resource_provider: Optional[ResourceProvider] = None,
    ) -> None:
        """
        Initialize the server.
        
        Args:
            transport: Transport implementation to use
            options: Server configuration options
            session_store: Session store implementation (defaults to InMemorySessionStore)
            tool_provider: Provider for tool-related functionality
            prompt_provider: Provider for prompt-related functionality
            resource_provider: Provider for resource-related functionality
        """
        self.transport = transport
        self.options = options or ServerOptions()
        self.session_store = session_store or InMemorySessionStore(
            expiration_seconds=self.options.session_expiration_seconds
        )
        self.tool_provider = tool_provider
        self.prompt_provider = prompt_provider
        self.resource_provider = resource_provider
        
        # Set up logging
        self.logger = structlog.get_logger("pymcp.server")
        
        # Set up server info and capabilities
        self.server_info = ServerInfo(
            name=self.options.name,
            version=self.options.version,
        )
        
        # Determine server capabilities based on transport capabilities
        transport_capabilities = self.transport.info.capabilities
        self.server_capabilities = ServerCapabilities(
            supports_tool_streaming=TransportCapability.STREAMING in transport_capabilities,
            supports_batch_requests=TransportCapability.BATCH_REQUESTS in transport_capabilities,
            supports_cancellation=TransportCapability.CANCELLATION in transport_capabilities,
            supports_progress=TransportCapability.PROGRESS in transport_capabilities,
            supports_resources=self.resource_provider is not None,
            supports_prompts=self.prompt_provider is not None,
        )
        
        # Create request handler
        self.handler = MCPRequestHandler(
            server_info=self.server_info,
            server_capabilities=self.server_capabilities,
            tool_provider=self.tool_provider,
            prompt_provider=self.prompt_provider,
            resource_provider=self.resource_provider,
        )
        
        # Set session store for transport
        self.transport.set_session_store(self.session_store)
        
        # Server state
        self._running = False
        self._server_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """
        Start the server.
        
        This method starts the transport and begins listening for requests.
        It also sets up signal handlers for graceful shutdown.
        
        Raises:
            RuntimeError: If the server is already running
        """
        if self._running:
            raise RuntimeError("Server is already running")
        
        self.logger.info(
            "Starting MCP server",
            server_name=self.server_info.name,
            server_version=self.server_info.version,
            transport_type=self.transport.info.type,
            transport_capabilities=list(self.transport.info.capabilities),
        )
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Start the server
        self._running = True
        self._server_task = asyncio.create_task(self._run_server())
        
        try:
            # Wait for the server to start
            await asyncio.sleep(0.1)
            self.logger.info("Server started successfully")
        except Exception as e:
            # If there's an error, clean up
            self._running = False
            if self._server_task:
                self._server_task.cancel()
                self._server_task = None
            self.logger.error("Error starting server", error=str(e))
            raise
    
    async def _run_server(self) -> None:
        """
        Run the server loop.
        
        This method is called by start() and runs in a separate task.
        """
        try:
            # Start the transport
            await self.transport.listen(self.handler)
        except asyncio.CancelledError:
            # Server was cancelled, clean up
            self.logger.info("Server task cancelled")
            self._running = False
        except Exception as e:
            # Unexpected error
            self.logger.error("Error in server loop", error=str(e))
            self._running = False
            # Signal shutdown
            self._shutdown_event.set()
    
    async def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the server gracefully.
        
        This method stops the transport and cleans up resources.
        
        Args:
            timeout: Timeout in seconds for graceful shutdown
            
        Raises:
            RuntimeError: If the server is not running
        """
        if not self._running:
            raise RuntimeError("Server is not running")
        
        self.logger.info("Stopping server")
        
        # Set the shutdown event
        self._shutdown_event.set()
        
        # Cancel the server task
        if self._server_task:
            self._server_task.cancel()
            try:
                await asyncio.wait_for(self._server_task, timeout=timeout)
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Timeout waiting for server task to cancel, forcing shutdown"
                )
            except Exception as e:
                self.logger.error("Error cancelling server task", error=str(e))
        
        # Close the transport
        try:
            await self.transport.close()
        except Exception as e:
            self.logger.error("Error closing transport", error=str(e))
        
        self._running = False
        self._server_task = None
        self.logger.info("Server stopped")
    
    def _setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for graceful shutdown.
        
        This method sets up handlers for SIGINT and SIGTERM to trigger
        a graceful shutdown of the server.
        """
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )
    
    async def _handle_signal(self, sig: signal.Signals) -> None:
        """
        Handle a signal.
        
        This method is called when a signal is received and triggers
        a graceful shutdown of the server.
        
        Args:
            sig: Signal that was received
        """
        self.logger.info(f"Received signal {sig.name}, shutting down")
        
        # Stop the server
        try:
            await self.stop()
        except Exception as e:
            self.logger.error("Error stopping server", error=str(e))
        
        # Exit the process
        sys.exit(0)
    
    @property
    def is_running(self) -> bool:
        """
        Check if the server is running.
        
        Returns:
            True if the server is running, False otherwise
        """
        return self._running
    
    async def wait_for_shutdown(self) -> None:
        """
        Wait for the server to shut down.
        
        This method waits for the shutdown event to be set, which happens
        when the server is stopped or encounters an error.
        """
        await self._shutdown_event.wait()
