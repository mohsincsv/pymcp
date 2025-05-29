"""
Base tool system for MCP.

This module provides the core abstractions for the MCP tool system,
including the Tool abstract base class, parameter definitions, schema
representation, and utilities for tool registration and execution.
"""

import abc
import asyncio
import inspect
import json
from enum import Enum
from typing import (
    Any, 
    AsyncIterator, 
    Callable, 
    Dict, 
    Generic, 
    List, 
    Optional, 
    Set, 
    Type, 
    TypeVar, 
    Union, 
    get_type_hints,
    get_origin,
    get_args,
)

from pydantic import BaseModel, Field, create_model, ValidationError

# Type variable for generic tool result
T = TypeVar("T")


class ParameterType(str, Enum):
    """Enumeration of supported parameter types."""
    
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    
    @classmethod
    def from_python_type(cls, type_hint: Type) -> "ParameterType":
        """
        Convert a Python type hint to a parameter type.
        
        Args:
            type_hint: Python type hint
            
        Returns:
            Parameter type
            
        Raises:
            ValueError: If the type hint is not supported
        """
        if type_hint == str:
            return cls.STRING
        elif type_hint == int:
            return cls.INTEGER
        elif type_hint == float:
            return cls.NUMBER
        elif type_hint == bool:
            return cls.BOOLEAN
        elif get_origin(type_hint) == list:
            return cls.ARRAY
        elif get_origin(type_hint) == dict:
            return cls.OBJECT
        elif type_hint == type(None):
            return cls.NULL
        else:
            # Check for Optional[X]
            if get_origin(type_hint) == Union:
                args = get_args(type_hint)
                if len(args) == 2 and type(None) in args:
                    # It's an Optional[X]
                    non_none_type = args[0] if args[1] == type(None) else args[1]
                    return cls.from_python_type(non_none_type)
            
            # Default to string for unsupported types
            return cls.STRING


class ToolParameter(BaseModel):
    """
    Definition of a tool parameter.
    
    This class represents a parameter that a tool accepts, including
    its name, type, description, and validation constraints.
    """
    
    name: str = Field(..., description="Name of the parameter")
    type: ParameterType = Field(..., description="Type of the parameter")
    description: str = Field("", description="Description of the parameter")
    required: bool = Field(False, description="Whether the parameter is required")
    default: Optional[Any] = Field(None, description="Default value for the parameter")
    enum: Optional[List[Any]] = Field(None, description="Enumeration of possible values")
    
    # Additional constraints for specific types
    min_length: Optional[int] = Field(None, description="Minimum length for strings")
    max_length: Optional[int] = Field(None, description="Maximum length for strings")
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")
    minimum: Optional[float] = Field(None, description="Minimum value for numbers")
    maximum: Optional[float] = Field(None, description="Maximum value for numbers")
    
    # Array-specific constraints
    items: Optional[Dict[str, Any]] = Field(None, description="Schema for array items")
    min_items: Optional[int] = Field(None, description="Minimum number of items in array")
    max_items: Optional[int] = Field(None, description="Maximum number of items in array")
    
    # Object-specific constraints
    properties: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, description="Schema for object properties"
    )
    required_properties: Optional[List[str]] = Field(
        None, description="Required properties for objects"
    )
    
    @classmethod
    def from_type_hint(
        cls, 
        name: str, 
        type_hint: Type, 
        description: str = "", 
        required: bool = False,
        default: Optional[Any] = None,
        **kwargs
    ) -> "ToolParameter":
        """
        Create a parameter from a Python type hint.
        
        Args:
            name: Name of the parameter
            type_hint: Python type hint
            description: Description of the parameter
            required: Whether the parameter is required
            default: Default value for the parameter
            **kwargs: Additional constraints
            
        Returns:
            Tool parameter
        """
        param_type = ParameterType.from_python_type(type_hint)
        
        # Handle array items
        items = None
        if param_type == ParameterType.ARRAY and get_origin(type_hint) == list:
            item_type = get_args(type_hint)[0] if get_args(type_hint) else Any
            items = {
                "type": ParameterType.from_python_type(item_type).value
            }
        
        # Handle object properties
        properties = None
        required_properties = None
        if param_type == ParameterType.OBJECT and get_origin(type_hint) == dict:
            # For now, just use a generic object schema
            properties = {}
        
        return cls(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=default,
            items=items,
            properties=properties,
            required_properties=required_properties,
            **kwargs
        )


class ToolSchema(BaseModel):
    """
    Schema definition for a tool.
    
    This class represents the metadata and parameter schema for a tool,
    which can be used for discovery, validation, and documentation.
    """
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field("", description="Description of the tool")
    parameters: List[ToolParameter] = Field(
        default_factory=list, description="Parameters accepted by the tool"
    )
    returns: Optional[Dict[str, Any]] = Field(
        None, description="Schema for the return value of the tool"
    )
    streaming: bool = Field(
        False, description="Whether the tool supports streaming results"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for the tool")
    version: Optional[str] = Field(None, description="Version of the tool")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the schema to a dictionary.
        
        Returns:
            Dictionary representation of the schema
        """
        return self.model_dump(exclude_none=True)


class ToolError(Exception):
    """Exception raised for tool-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a tool error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ToolValidationError(ToolError):
    """Exception raised for tool parameter validation errors."""
    
    def __init__(
        self, 
        message: str, 
        parameter: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a tool validation error.
        
        Args:
            message: Error message
            parameter: Name of the parameter that failed validation
            details: Additional error details
        """
        self.parameter = parameter
        super().__init__(message, details)


class ToolExecutionError(ToolError):
    """Exception raised for tool execution errors."""
    
    pass


class Tool(Generic[T], abc.ABC):
    """
    Abstract base class for MCP tools.
    
    This class defines the interface that all tools must implement,
    including methods for getting the tool schema and executing the tool.
    """
    
    @abc.abstractmethod
    def get_schema(self) -> ToolSchema:
        """
        Get the tool schema.
        
        Returns:
            Tool schema
        """
        ...
    
    @abc.abstractmethod
    async def execute(self, args: Dict[str, Any]) -> T:
        """
        Execute the tool with the given arguments.
        
        Args:
            args: Tool arguments
            
        Returns:
            Tool result
            
        Raises:
            ToolValidationError: If the arguments are invalid
            ToolExecutionError: If the tool execution fails
        """
        ...
    
    async def execute_streaming(self, args: Dict[str, Any]) -> AsyncIterator[T]:
        """
        Execute the tool with streaming results.
        
        Args:
            args: Tool arguments
            
        Yields:
            Tool result chunks
            
        Raises:
            ToolValidationError: If the arguments are invalid
            ToolExecutionError: If the tool execution fails
            NotImplementedError: If the tool doesn't support streaming
        """
        # Default implementation just yields the result as a single chunk
        if not self.get_schema().streaming:
            raise NotImplementedError("Tool does not support streaming")
        
        result = await self.execute(args)
        yield result
    
    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the tool arguments against the schema.
        
        Args:
            args: Tool arguments
            
        Returns:
            Validated arguments (with defaults applied)
            
        Raises:
            ToolValidationError: If the arguments are invalid
        """
        schema = self.get_schema()
        
        # Create a Pydantic model from the parameters
        fields = {}
        for param in schema.parameters:
            field_info = Field(
                default=param.default if not param.required else ...,
                description=param.description,
            )
            
            # Determine the field type
            if param.type == ParameterType.STRING:
                field_type = str
            elif param.type == ParameterType.INTEGER:
                field_type = int
            elif param.type == ParameterType.NUMBER:
                field_type = float
            elif param.type == ParameterType.BOOLEAN:
                field_type = bool
            elif param.type == ParameterType.ARRAY:
                field_type = List[Any]
            elif param.type == ParameterType.OBJECT:
                field_type = Dict[str, Any]
            else:
                field_type = Any
            
            # Make it Optional if not required
            if not param.required:
                field_type = Optional[field_type]
            
            fields[param.name] = (field_type, field_info)
        
        # Create the model
        ParamsModel = create_model("ToolParams", **fields)
        
        try:
            # Validate the arguments
            validated = ParamsModel(**args)
            return validated.model_dump()
        except ValidationError as e:
            # Convert Pydantic validation error to ToolValidationError
            errors = e.errors()
            if errors:
                error = errors[0]
                param_name = error["loc"][0] if error["loc"] else None
                message = error["msg"]
                raise ToolValidationError(
                    message=message,
                    parameter=param_name,
                    details={"errors": errors}
                )
            else:
                raise ToolValidationError(
                    message="Invalid arguments",
                    details={"errors": errors}
                )


class FunctionTool(Tool[T]):
    """
    Tool implementation that wraps a Python function.
    
    This class provides a convenient way to create tools from existing
    Python functions, automatically generating the schema from the function
    signature and docstring.
    """
    
    def __init__(
        self,
        func: Callable[..., T],
        name: Optional[str] = None,
        description: Optional[str] = None,
        streaming: bool = False,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
    ) -> None:
        """
        Initialize a function tool.
        
        Args:
            func: Function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            streaming: Whether the tool supports streaming results
            tags: Tags for the tool
            version: Version of the tool
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""
        self.streaming = streaming
        self.tags = tags or []
        self.version = version
        
        # Cache the schema
        self._schema: Optional[ToolSchema] = None
    
    def get_schema(self) -> ToolSchema:
        """
        Get the tool schema.
        
        Returns:
            Tool schema
        """
        if self._schema is None:
            # Generate the schema from the function signature
            signature = inspect.signature(self.func)
            type_hints = get_type_hints(self.func)
            
            parameters = []
            for name, param in signature.parameters.items():
                # Skip self parameter for methods
                if name == "self":
                    continue
                
                # Get the type hint for the parameter
                type_hint = type_hints.get(name, Any)
                
                # Determine if the parameter is required
                required = param.default == inspect.Parameter.empty
                
                # Get the default value
                default = None if required else param.default
                
                # Create the parameter
                parameter = ToolParameter.from_type_hint(
                    name=name,
                    type_hint=type_hint,
                    required=required,
                    default=default,
                )
                parameters.append(parameter)
            
            # Create the schema
            self._schema = ToolSchema(
                name=self.name,
                description=self.description,
                parameters=parameters,
                streaming=self.streaming,
                tags=self.tags,
                version=self.version,
            )
        
        return self._schema
    
    async def execute(self, args: Dict[str, Any]) -> T:
        """
        Execute the tool with the given arguments.
        
        Args:
            args: Tool arguments
            
        Returns:
            Tool result
            
        Raises:
            ToolValidationError: If the arguments are invalid
            ToolExecutionError: If the tool execution fails
        """
        # Validate the arguments
        validated_args = self.validate_args(args)
        
        try:
            # Call the function
            result = self.func(**validated_args)
            
            # Handle coroutines
            if inspect.iscoroutine(result):
                result = await result
            
            return result
        except Exception as e:
            # Convert exceptions to ToolExecutionError
            raise ToolExecutionError(
                message=f"Tool execution failed: {str(e)}",
                details={"exception": str(e)}
            ) from e


class ToolRegistry:
    """
    Registry for MCP tools.
    
    This class provides a central registry for tools, allowing them to be
    discovered and accessed by name. It also provides methods for registering
    tools and retrieving tool schemas.
    """
    
    def __init__(self) -> None:
        """Initialize the tool registry."""
        self._tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        schema = tool.get_schema()
        if schema.name in self._tools:
            raise ValueError(f"Tool with name '{schema.name}' is already registered")
        
        self._tools[schema.name] = tool
    
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
        tool = FunctionTool(
            func=func,
            name=name,
            description=description,
            streaming=streaming,
            tags=tags,
            version=version,
        )
        
        self.register_tool(tool)
        return tool
    
    def get_tool(self, name: str) -> Tool:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to get
            
        Returns:
            The tool
            
        Raises:
            ValueError: If the tool is not found
        """
        if name not in self._tools:
            raise ValueError(f"Tool with name '{name}' not found")
        
        return self._tools[name]
    
    def list_tools(self) -> List[ToolSchema]:
        """
        List all registered tools.
        
        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self._tools.values()]
    
    def list_tools_by_tag(self, tag: str) -> List[ToolSchema]:
        """
        List tools with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of tool schemas
        """
        return [
            tool.get_schema() 
            for tool in self._tools.values() 
            if tag in tool.get_schema().tags
        ]
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
