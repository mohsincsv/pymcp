"""
Tool registry implementation for MCP.

This module provides a comprehensive tool registry that implements the ToolProvider
protocol and provides tool discovery, execution, and management capabilities.
It serves as the central component for registering, discovering, and executing
tools in the MCP server.
"""

import asyncio
import importlib
import inspect
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, AsyncIterator

from mcp.tools.base import (
    Tool,
    ToolRegistry,
    ToolSchema,
    ToolError,
    ToolValidationError,
    ToolExecutionError,
    FunctionTool,
)


class MCPToolRegistry:
    """
    Comprehensive tool registry for MCP.
    
    This class implements the ToolProvider protocol and provides a central
    registry for tools in the MCP server. It supports tool discovery, execution,
    and management, and integrates with the core ToolRegistry.
    """
    
    def __init__(self) -> None:
        """Initialize the MCP tool registry."""
        self._registry = ToolRegistry()
        self._streaming_queues: Dict[str, asyncio.Queue] = {}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools.
        
        Returns:
            List of tool schemas as dictionaries
        """
        schemas = self._registry.list_tools()
        return [schema.to_dict() for schema in schemas]
    
    async def list_tools_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        List tools with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of tool schemas as dictionaries
        """
        schemas = self._registry.list_tools_by_tag(tag)
        return [schema.to_dict() for schema in schemas]
    
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
            ToolValidationError: If the arguments are invalid
            ToolExecutionError: If the tool execution fails
        """
        try:
            # Get the tool
            tool = self._registry.get_tool(name)
            
            # Check if streaming is requested but not supported
            if stream and not tool.get_schema().streaming:
                raise ValueError(f"Tool '{name}' does not support streaming")
            
            if stream:
                # Create a queue for streaming results
                queue = asyncio.Queue()
                
                # Start a task to execute the tool and stream results
                asyncio.create_task(self._stream_tool_results(tool, args, queue))
                
                # Store the queue for potential cancellation
                self._streaming_queues[name] = queue
                
                return queue
            else:
                # Execute the tool normally
                result = await tool.execute(args)
                
                # Convert the result to a dictionary if it's not already
                if isinstance(result, dict):
                    return result
                else:
                    return {"result": result}
        
        except ValueError as e:
            # Tool not found
            raise ValueError(f"Tool '{name}' not found: {str(e)}")
        
        except ToolValidationError as e:
            # Invalid arguments
            raise ToolValidationError(
                message=f"Invalid arguments for tool '{name}': {e.message}",
                parameter=e.parameter,
                details=e.details
            )
        
        except ToolExecutionError as e:
            # Tool execution failed
            raise ToolExecutionError(
                message=f"Error executing tool '{name}': {e.message}",
                details=e.details
            )
        
        except Exception as e:
            # Other errors
            raise ToolExecutionError(
                message=f"Unexpected error executing tool '{name}': {str(e)}",
                details={"exception": str(e)}
            )
    
    async def _stream_tool_results(
        self,
        tool: Tool,
        args: Dict[str, Any],
        queue: asyncio.Queue
    ) -> None:
        """
        Stream tool results to a queue.
        
        Args:
            tool: Tool to execute
            args: Tool arguments
            queue: Queue to stream results to
        """
        try:
            # Execute the tool with streaming
            async for chunk in tool.execute_streaming(args):
                # Convert the chunk to a dictionary if it's not already
                if isinstance(chunk, dict):
                    chunk_dict = chunk
                else:
                    chunk_dict = {"chunk": chunk}
                
                # Add a flag for the last chunk
                await queue.put(chunk_dict)
            
            # Signal the end of the stream with a special marker
            await queue.put({"is_last": True})
        
        except Exception as e:
            # Send error information
            await queue.put({
                "error": str(e),
                "is_last": True
            })
        
        finally:
            # Clean up
            tool_name = tool.get_schema().name
            if tool_name in self._streaming_queues:
                del self._streaming_queues[tool_name]
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        self._registry.register_tool(tool)
    
    def register_function(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        streaming: bool = False,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> FunctionTool:
        """
        Register a function as a tool.
        
        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            streaming: Whether the tool supports streaming results
            tags: Tags for the tool
            version: Version of the tool
            
        Returns:
            The created function tool
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        return self._registry.register_function(
            func=func,
            name=name,
            description=description,
            streaming=streaming,
            tags=tags,
            version=version,
        )
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Name of the tool to unregister
            
        Returns:
            True if the tool was unregistered, False if it wasn't found
        """
        if name in self._registry:
            # Get the tool schema
            tool = self._registry.get_tool(name)
            
            # Cancel any streaming tasks
            if name in self._streaming_queues:
                # Signal the end of the stream
                self._streaming_queues[name].put_nowait({"is_last": True, "cancelled": True})
                del self._streaming_queues[name]
            
            # Remove the tool from the registry
            self._registry._tools.pop(name)
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all registered tools."""
        # Cancel any streaming tasks
        for queue in self._streaming_queues.values():
            queue.put_nowait({"is_last": True, "cancelled": True})
        
        self._streaming_queues.clear()
        self._registry.clear()
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to get
            
        Returns:
            The tool, or None if not found
        """
        try:
            return self._registry.get_tool(name)
        except ValueError:
            return None
    
    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Name of the tool to check
            
        Returns:
            True if the tool is registered, False otherwise
        """
        return name in self._registry
    
    def load_tools_from_module(self, module_name: str) -> int:
        """
        Load tools from a Python module.
        
        This method imports the specified module and registers any Tool
        instances or functions decorated with @tool found in the module.
        
        Args:
            module_name: Name of the module to load tools from
            
        Returns:
            Number of tools loaded
            
        Raises:
            ImportError: If the module cannot be imported
        """
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Count the number of tools loaded
            count = 0
            
            # Register Tool instances
            for name in dir(module):
                obj = getattr(module, name)
                
                # Check if it's a Tool instance
                if isinstance(obj, Tool):
                    self.register_tool(obj)
                    count += 1
                
                # Check if it's a function decorated with @tool
                elif callable(obj) and hasattr(obj, "_is_tool"):
                    # The function is already wrapped as a FunctionTool
                    self.register_tool(obj._tool)
                    count += 1
            
            return count
        
        except ImportError as e:
            raise ImportError(f"Error importing module '{module_name}': {str(e)}")
    
    def load_tools_from_directory(self, directory: str) -> int:
        """
        Load tools from Python files in a directory.
        
        This method imports all Python files in the specified directory
        and registers any Tool instances or functions decorated with @tool
        found in the files.
        
        Args:
            directory: Path to the directory to load tools from
            
        Returns:
            Number of tools loaded
            
        Raises:
            ValueError: If the directory does not exist
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory '{directory}' does not exist")
        
        # Add the directory to the Python path
        sys.path.insert(0, os.path.abspath(directory))
        
        # Count the number of tools loaded
        count = 0
        
        try:
            # Iterate over Python files in the directory
            for filename in os.listdir(directory):
                if filename.endswith(".py") and not filename.startswith("_"):
                    # Get the module name
                    module_name = filename[:-3]
                    
                    try:
                        # Load tools from the module
                        count += self.load_tools_from_module(module_name)
                    except ImportError:
                        # Skip modules that cannot be imported
                        pass
            
            return count
        
        finally:
            # Remove the directory from the Python path
            sys.path.pop(0)
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._registry)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._registry


# Tool decorator
def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    streaming: bool = False,
    tags: Optional[List[str]] = None,
    version: Optional[str] = None,
) -> Callable:
    """
    Decorator to register a function as a tool.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        streaming: Whether the tool supports streaming results
        tags: Tags for the tool
        version: Version of the tool
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Create a FunctionTool from the function
        tool_obj = FunctionTool(
            func=func,
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or "",
            streaming=streaming,
            tags=tags or [],
            version=version,
        )
        
        # Store the tool on the function for later registration
        func._is_tool = True
        func._tool = tool_obj
        
        return func
    
    return decorator


# Global tool registry instance
global_registry = MCPToolRegistry()


# Convenience functions for the global registry
def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    global_registry.register_tool(tool)


def register_function(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    streaming: bool = False,
    tags: Optional[List[str]] = None,
    version: Optional[str] = None,
) -> FunctionTool:
    """Register a function as a tool in the global registry."""
    return global_registry.register_function(
        func=func,
        name=name,
        description=description,
        streaming=streaming,
        tags=tags,
        version=version,
    )


def get_registry() -> MCPToolRegistry:
    """Get the global tool registry."""
    return global_registry
