"""
Request handler for MCP server.

This module provides the main request handler for the MCP server,
which routes requests to the appropriate method handlers based on
the method name. It handles all MCP protocol methods including
initialize, ping, tools/list, tools/call, prompts/list, etc.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type, Union

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
    CancelRequestParams,
    ProgressParams,
    
    # Validation utilities
    parse_params_model,
    create_error_response,
    create_result_response,
)
from mcp.transport.base import RequestContext, RequestHandler


class ToolProvider(Protocol):
    """Protocol for tool providers."""
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools.
        
        Returns:
            List of tool schemas
        """
        ...
    
    async def call_tool(
        self, 
        name: str, 
        args: Dict[str, Any], 
        stream: bool = False
    ) -> Union[Dict[str, Any], asyncio.Queue]:
        """
        Call a tool with the given arguments.
        
        Args:
            name: Name of the tool to call
            args: Arguments to pass to the tool
            stream: Whether to stream the results
            
        Returns:
            Tool result or a queue for streaming results
            
        Raises:
            ValueError: If the tool is not found
        """
        ...


class PromptProvider(Protocol):
    """Protocol for prompt providers."""
    
    async def list_prompts(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all available prompts.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of prompt metadata
        """
        ...
    
    async def get_prompt(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a prompt with the given name.
        
        Args:
            name: Name of the prompt to get
            args: Optional arguments to fill in the prompt template
            
        Returns:
            Prompt content and metadata
            
        Raises:
            ValueError: If the prompt is not found
        """
        ...


class ResourceProvider(Protocol):
    """Protocol for resource providers."""
    
    async def list_resources(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all available resources.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of resource metadata
        """
        ...
    
    async def read_resource(self, name: str) -> Dict[str, Any]:
        """
        Read a resource with the given name.
        
        Args:
            name: Name of the resource to read
            
        Returns:
            Resource content and metadata
            
        Raises:
            ValueError: If the resource is not found
        """
        ...


class MCPRequestHandler(RequestHandler):
    """
    Main request handler for MCP server.
    
    This class implements the RequestHandler protocol from the transport layer
    and routes requests to the appropriate method handlers based on the method name.
    """
    
    def __init__(
        self,
        server_info: ServerInfo,
        server_capabilities: ServerCapabilities,
        tool_provider: Optional[ToolProvider] = None,
        prompt_provider: Optional[PromptProvider] = None,
        resource_provider: Optional[ResourceProvider] = None,
    ) -> None:
        """
        Initialize the request handler.
        
        Args:
            server_info: Information about the server
            server_capabilities: Capabilities supported by the server
            tool_provider: Provider for tool-related functionality
            prompt_provider: Provider for prompt-related functionality
            resource_provider: Provider for resource-related functionality
        """
        self.server_info = server_info
        self.server_capabilities = server_capabilities
        self.tool_provider = tool_provider
        self.prompt_provider = prompt_provider
        self.resource_provider = resource_provider
        
        # Map of method names to handler methods
        self._method_handlers: Dict[str, Callable] = {
            MCPMethod.INITIALIZE: self._handle_initialize,
            MCPMethod.PING: self._handle_ping,
            MCPMethod.TOOLS_LIST: self._handle_tools_list,
            MCPMethod.TOOLS_CALL: self._handle_tools_call,
            MCPMethod.PROMPTS_LIST: self._handle_prompts_list,
            MCPMethod.PROMPTS_GET: self._handle_prompts_get,
            MCPMethod.RESOURCES_LIST: self._handle_resources_list,
            MCPMethod.RESOURCES_READ: self._handle_resources_read,
        }
        
        # Map of notification methods to handler methods
        self._notification_handlers: Dict[str, Callable] = {
            MCPMethod.CANCEL_REQUEST: self._handle_cancel_request,
        }
        
        # Active requests that can be cancelled
        self._active_requests: Dict[Union[str, int], asyncio.Task] = {}
    
    async def handle_request(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle a single MCP request.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response to the request
        """
        # Check if the method is supported
        if request.method not in self._method_handlers:
            return create_error_response(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                f"Method not found: {request.method}",
            )
        
        try:
            # Create a task for the request handler
            handler = self._method_handlers[request.method]
            task = asyncio.create_task(handler(request, context))
            
            # Store the task if it can be cancelled
            if request.id is not None:
                self._active_requests[request.id] = task
            
            # Wait for the task to complete
            response = await task
            
            # Remove the task from active requests
            if request.id is not None and request.id in self._active_requests:
                del self._active_requests[request.id]
            
            return response
        except asyncio.CancelledError:
            # Request was cancelled
            return create_error_response(
                request.id,
                ErrorCode.CANCELLATION_ERROR,
                "Request was cancelled",
            )
        except Exception as e:
            # Handle unexpected errors
            return create_error_response(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Internal error: {str(e)}",
            )
    
    async def handle_notification(
        self,
        notification: Notification,
        context: RequestContext,
    ) -> None:
        """
        Handle an MCP notification.
        
        Args:
            notification: The notification to handle
            context: Context information for the notification
        """
        # Check if the method is supported
        if notification.method not in self._notification_handlers:
            # Silently ignore unsupported notification methods
            return
        
        try:
            # Call the notification handler
            handler = self._notification_handlers[notification.method]
            await handler(notification, context)
        except Exception:
            # Silently ignore errors in notification handlers
            pass
    
    async def handle_batch_request(
        self,
        batch: BatchRequest,
        context: RequestContext,
    ) -> BatchResponse:
        """
        Handle a batch of MCP requests.
        
        Args:
            batch: The batch request to handle
            context: Context information for the batch
            
        Returns:
            Batch response with results for each request
        """
        # Create tasks for each request in the batch
        tasks = []
        for request in batch:
            # Create a new context for each request with the correct request ID
            request_context = RequestContext(
                transport_id=context.transport_id,
                session_id=context.session_id,
                request_id=request.id,
                method=request.method,
                user_id=context.user_id,
                client_info=context.client_info,
                timestamp=context.timestamp,
            )
            
            # Create a task for the request
            task = asyncio.create_task(self.handle_request(request, request_context))
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out responses for notifications (which don't have an ID)
        # and handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Convert exceptions to error responses
                request = batch[i]
                valid_responses.append(create_error_response(
                    request.id,
                    ErrorCode.INTERNAL_ERROR,
                    f"Internal error: {str(response)}",
                ))
            elif isinstance(response, Response):
                valid_responses.append(response)
        
        return BatchResponse(__root__=valid_responses)
    
    async def _handle_initialize(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the initialize method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with server information and capabilities
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, InitializeParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Create the result
        result = InitializeResult(
            server_info=self.server_info,
            server_capabilities=self.server_capabilities,
            session_id=context.session_id,
        )
        
        return create_result_response(request.id, result.model_dump())
    
    async def _handle_ping(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the ping method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with server timestamp
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, PingParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Create the result
        result = PingResult(
            timestamp=int(time.time() * 1000),
        )
        
        return create_result_response(request.id, result.model_dump())
    
    async def _handle_tools_list(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the tools/list method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with list of available tools
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, ToolsListParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Check if tool provider is available
        if not self.tool_provider:
            return create_error_response(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                "Tool provider not available",
            )
        
        try:
            # Get the list of tools
            tools = await self.tool_provider.list_tools()
            
            # Create the result
            result = ToolsListResult(tools=tools)
            
            return create_result_response(request.id, result.model_dump())
        except Exception as e:
            return create_error_response(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Error listing tools: {str(e)}",
            )
    
    async def _handle_tools_call(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the tools/call method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with tool result
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, ToolCallParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Check if tool provider is available
        if not self.tool_provider:
            return create_error_response(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                "Tool provider not available",
            )
        
        try:
            # Call the tool
            result = await self.tool_provider.call_tool(
                params.name,
                params.args,
                params.stream,
            )
            
            # Handle streaming results
            if params.stream and isinstance(result, asyncio.Queue):
                # This is handled by the transport layer
                # We just return a success response
                return create_result_response(
                    request.id,
                    {"status": "streaming"}
                )
            
            # Create the result for non-streaming calls
            tool_result = ToolCallResult(result=result)
            
            return create_result_response(request.id, tool_result.model_dump())
        except ValueError as e:
            # Tool not found
            return create_error_response(
                request.id,
                ErrorCode.TOOL_NOT_FOUND,
                str(e),
            )
        except Exception as e:
            # Other errors
            return create_error_response(
                request.id,
                ErrorCode.TOOL_EXECUTION_ERROR,
                f"Error calling tool: {str(e)}",
            )
    
    async def _handle_prompts_list(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the prompts/list method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with list of available prompts
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, PromptsListParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Check if prompt provider is available
        if not self.prompt_provider:
            return create_error_response(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                "Prompt provider not available",
            )
        
        try:
            # Get the list of prompts
            prompts = await self.prompt_provider.list_prompts(params.tags)
            
            # Create the result
            result = PromptsListResult(prompts=prompts)
            
            return create_result_response(request.id, result.model_dump())
        except Exception as e:
            return create_error_response(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Error listing prompts: {str(e)}",
            )
    
    async def _handle_prompts_get(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the prompts/get method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with prompt content and metadata
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, PromptsGetParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Check if prompt provider is available
        if not self.prompt_provider:
            return create_error_response(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                "Prompt provider not available",
            )
        
        try:
            # Get the prompt
            prompt = await self.prompt_provider.get_prompt(params.name, params.args)
            
            # Create the result
            result = PromptsGetResult(**prompt)
            
            return create_result_response(request.id, result.model_dump())
        except ValueError as e:
            # Prompt not found
            return create_error_response(
                request.id,
                ErrorCode.PROMPT_NOT_FOUND,
                str(e),
            )
        except Exception as e:
            # Other errors
            return create_error_response(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Error getting prompt: {str(e)}",
            )
    
    async def _handle_resources_list(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the resources/list method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with list of available resources
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, ResourcesListParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Check if resource provider is available
        if not self.resource_provider:
            return create_error_response(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                "Resource provider not available",
            )
        
        try:
            # Get the list of resources
            resources = await self.resource_provider.list_resources(params.tags)
            
            # Create the result
            result = ResourcesListResult(resources=resources)
            
            return create_result_response(request.id, result.model_dump())
        except Exception as e:
            return create_error_response(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Error listing resources: {str(e)}",
            )
    
    async def _handle_resources_read(
        self,
        request: Request,
        context: RequestContext,
    ) -> Response:
        """
        Handle the resources/read method.
        
        Args:
            request: The request to handle
            context: Context information for the request
            
        Returns:
            Response with resource content and metadata
        """
        # Parse and validate the parameters
        params, error = parse_params_model(request.params, ResourcesReadParams)
        if error:
            return create_error_response(request.id, error.code, error.message, error.data)
        
        # Check if resource provider is available
        if not self.resource_provider:
            return create_error_response(
                request.id,
                ErrorCode.METHOD_NOT_FOUND,
                "Resource provider not available",
            )
        
        try:
            # Read the resource
            resource = await self.resource_provider.read_resource(params.name)
            
            # Create the result
            result = ResourcesReadResult(**resource)
            
            return create_result_response(request.id, result.model_dump())
        except ValueError as e:
            # Resource not found
            return create_error_response(
                request.id,
                ErrorCode.RESOURCE_NOT_FOUND,
                str(e),
            )
        except Exception as e:
            # Other errors
            return create_error_response(
                request.id,
                ErrorCode.INTERNAL_ERROR,
                f"Error reading resource: {str(e)}",
            )
    
    async def _handle_cancel_request(
        self,
        notification: Notification,
        context: RequestContext,
    ) -> None:
        """
        Handle the $/cancelRequest notification.
        
        Args:
            notification: The notification to handle
            context: Context information for the notification
        """
        # Parse and validate the parameters
        params, error = parse_params_model(notification.params, CancelRequestParams)
        if error:
            # Silently ignore invalid notifications
            return
        
        # Check if the request is active
        if params.id in self._active_requests:
            # Cancel the task
            self._active_requests[params.id].cancel()
            # Remove it from active requests
            del self._active_requests[params.id]
