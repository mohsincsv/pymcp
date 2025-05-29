"""
YAML tool loader for MCP.

This module provides functionality for loading tools from YAML files,
particularly shell commands and scripts. It supports Jinja2 templating
for command generation and parameter validation against schemas.
"""

import asyncio
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, AsyncIterator

import jinja2
import yaml
from pydantic import BaseModel, Field, ValidationError

from mcp.tools.base import (
    Tool,
    ToolSchema,
    ToolParameter,
    ParameterType,
    ToolError,
    ToolValidationError,
    ToolExecutionError,
)


class ShellCommandParameter(BaseModel):
    """Parameter definition for a shell command."""
    
    name: str = Field(..., description="Name of the parameter")
    type: str = Field("string", description="Type of the parameter")
    description: str = Field("", description="Description of the parameter")
    required: bool = Field(False, description="Whether the parameter is required")
    default: Optional[Any] = Field(None, description="Default value for the parameter")
    enum: Optional[List[Any]] = Field(None, description="Enumeration of possible values")
    
    # Additional constraints
    min_length: Optional[int] = Field(None, description="Minimum length for strings")
    max_length: Optional[int] = Field(None, description="Maximum length for strings")
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")
    minimum: Optional[float] = Field(None, description="Minimum value for numbers")
    maximum: Optional[float] = Field(None, description="Maximum value for numbers")


class ShellCommandDefinition(BaseModel):
    """Definition of a shell command from YAML."""
    
    name: str = Field(..., description="Name of the command")
    description: str = Field("", description="Description of the command")
    parameters: List[ShellCommandParameter] = Field(
        default_factory=list, description="Parameters for the command"
    )
    command: Optional[str] = Field(None, description="Command template to execute")
    commands: Optional[List[str]] = Field(None, description="List of command templates to execute")
    script: Optional[str] = Field(None, description="Script template to execute")
    working_dir: Optional[str] = Field(None, description="Working directory for the command")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables for the command")
    shell: bool = Field(True, description="Whether to use a shell to execute the command")
    streaming: bool = Field(False, description="Whether to stream the command output")
    tags: List[str] = Field(default_factory=list, description="Tags for the command")
    version: Optional[str] = Field(None, description="Version of the command")
    
    @property
    def has_command(self) -> bool:
        """Check if the definition has a command."""
        return self.command is not None
    
    @property
    def has_commands(self) -> bool:
        """Check if the definition has multiple commands."""
        return self.commands is not None and len(self.commands) > 0
    
    @property
    def has_script(self) -> bool:
        """Check if the definition has a script."""
        return self.script is not None
    
    def validate(self) -> None:
        """
        Validate the command definition.
        
        Raises:
            ValueError: If the definition is invalid
        """
        # Check that at least one of command, commands, or script is defined
        if not self.has_command and not self.has_commands and not self.has_script:
            raise ValueError(
                f"Command '{self.name}' must define at least one of 'command', 'commands', or 'script'"
            )
        
        # Check that only one of command, commands, or script is defined
        count = sum([self.has_command, self.has_commands, self.has_script])
        if count > 1:
            raise ValueError(
                f"Command '{self.name}' must define only one of 'command', 'commands', or 'script'"
            )


class ShellCommandTool(Tool[Dict[str, Any]]):
    """
    Tool implementation that executes shell commands.
    
    This class provides a tool that executes shell commands or scripts,
    with support for templating and parameter validation.
    """
    
    def __init__(self, definition: ShellCommandDefinition) -> None:
        """
        Initialize a shell command tool.
        
        Args:
            definition: Shell command definition
        """
        self.definition = definition
        
        # Validate the definition
        self.definition.validate()
        
        # Set up Jinja2 environment
        self.jinja_env = jinja2.Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Cache the schema
        self._schema: Optional[ToolSchema] = None
    
    def get_schema(self) -> ToolSchema:
        """
        Get the tool schema.
        
        Returns:
            Tool schema
        """
        if self._schema is None:
            # Convert parameters to ToolParameter objects
            parameters = []
            for param in self.definition.parameters:
                # Map YAML parameter type to ParameterType
                param_type = ParameterType.STRING
                if param.type == "string":
                    param_type = ParameterType.STRING
                elif param.type == "number":
                    param_type = ParameterType.NUMBER
                elif param.type == "integer":
                    param_type = ParameterType.INTEGER
                elif param.type == "boolean":
                    param_type = ParameterType.BOOLEAN
                elif param.type == "array":
                    param_type = ParameterType.ARRAY
                elif param.type == "object":
                    param_type = ParameterType.OBJECT
                
                # Create the parameter
                tool_param = ToolParameter(
                    name=param.name,
                    type=param_type,
                    description=param.description,
                    required=param.required,
                    default=param.default,
                    enum=param.enum,
                    min_length=param.min_length,
                    max_length=param.max_length,
                    pattern=param.pattern,
                    minimum=param.minimum,
                    maximum=param.maximum,
                )
                parameters.append(tool_param)
            
            # Create the schema
            self._schema = ToolSchema(
                name=self.definition.name,
                description=self.definition.description,
                parameters=parameters,
                streaming=self.definition.streaming,
                tags=self.definition.tags,
                version=self.definition.version,
            )
        
        return self._schema
    
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the shell command with the given arguments.
        
        Args:
            args: Command arguments
            
        Returns:
            Command output
            
        Raises:
            ToolValidationError: If the arguments are invalid
            ToolExecutionError: If the command execution fails
        """
        # Validate the arguments
        validated_args = self.validate_args(args)
        
        try:
            # Render the command template
            if self.definition.has_command:
                # Single command
                command = self._render_template(self.definition.command, validated_args)
                output = await self._execute_command(command)
                return {"output": output}
            
            elif self.definition.has_commands:
                # Multiple commands
                outputs = []
                for cmd_template in self.definition.commands:
                    command = self._render_template(cmd_template, validated_args)
                    output = await self._execute_command(command)
                    outputs.append(output)
                return {"outputs": outputs}
            
            elif self.definition.has_script:
                # Script
                script = self._render_template(self.definition.script, validated_args)
                output = await self._execute_script(script)
                return {"output": output}
            
            else:
                # This shouldn't happen due to validation
                raise ToolExecutionError(
                    message=f"Command '{self.definition.name}' has no command, commands, or script defined"
                )
        
        except jinja2.exceptions.TemplateError as e:
            # Template rendering error
            raise ToolExecutionError(
                message=f"Error rendering command template: {str(e)}",
                details={"exception": str(e)}
            )
        
        except subprocess.SubprocessError as e:
            # Command execution error
            raise ToolExecutionError(
                message=f"Error executing command: {str(e)}",
                details={"exception": str(e)}
            )
        
        except Exception as e:
            # Other errors
            raise ToolExecutionError(
                message=f"Error executing command: {str(e)}",
                details={"exception": str(e)}
            )
    
    async def execute_streaming(self, args: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute the shell command with streaming output.
        
        Args:
            args: Command arguments
            
        Yields:
            Command output chunks
            
        Raises:
            ToolValidationError: If the arguments are invalid
            ToolExecutionError: If the command execution fails
            NotImplementedError: If the tool doesn't support streaming
        """
        if not self.definition.streaming:
            raise NotImplementedError("Command does not support streaming")
        
        # Validate the arguments
        validated_args = self.validate_args(args)
        
        try:
            # Render the command template
            if self.definition.has_command:
                # Single command
                command = self._render_template(self.definition.command, validated_args)
                async for chunk in self._execute_command_streaming(command):
                    yield {"chunk": chunk}
            
            elif self.definition.has_commands:
                # Multiple commands
                for cmd_template in self.definition.commands:
                    command = self._render_template(cmd_template, validated_args)
                    async for chunk in self._execute_command_streaming(command):
                        yield {"chunk": chunk, "command": command}
            
            elif self.definition.has_script:
                # Script
                script = self._render_template(self.definition.script, validated_args)
                async for chunk in self._execute_script_streaming(script):
                    yield {"chunk": chunk}
            
            else:
                # This shouldn't happen due to validation
                raise ToolExecutionError(
                    message=f"Command '{self.definition.name}' has no command, commands, or script defined"
                )
        
        except jinja2.exceptions.TemplateError as e:
            # Template rendering error
            yield {
                "error": f"Error rendering command template: {str(e)}",
                "is_last": True
            }
        
        except subprocess.SubprocessError as e:
            # Command execution error
            yield {
                "error": f"Error executing command: {str(e)}",
                "is_last": True
            }
        
        except Exception as e:
            # Other errors
            yield {
                "error": f"Error executing command: {str(e)}",
                "is_last": True
            }
    
    def _render_template(self, template_str: str, args: Dict[str, Any]) -> str:
        """
        Render a Jinja2 template with the given arguments.
        
        Args:
            template_str: Template string
            args: Template arguments
            
        Returns:
            Rendered template
            
        Raises:
            jinja2.exceptions.TemplateError: If template rendering fails
        """
        template = self.jinja_env.from_string(template_str)
        return template.render(**args)
    
    async def _execute_command(self, command: str) -> str:
        """
        Execute a shell command.
        
        Args:
            command: Command to execute
            
        Returns:
            Command output
            
        Raises:
            subprocess.SubprocessError: If command execution fails
        """
        # Prepare environment variables
        env = os.environ.copy()
        if self.definition.env:
            env.update(self.definition.env)
        
        # Determine working directory
        cwd = self.definition.working_dir
        
        # Execute the command
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
            shell=self.definition.shell,
        )
        
        # Wait for the command to complete
        stdout, stderr = await process.communicate()
        
        # Check if the command succeeded
        if process.returncode != 0:
            raise subprocess.SubprocessError(
                f"Command failed with exit code {process.returncode}: {stderr.decode()}"
            )
        
        # Return the command output
        return stdout.decode()
    
    async def _execute_command_streaming(self, command: str) -> AsyncIterator[str]:
        """
        Execute a shell command with streaming output.
        
        Args:
            command: Command to execute
            
        Yields:
            Command output chunks
            
        Raises:
            subprocess.SubprocessError: If command execution fails
        """
        # Prepare environment variables
        env = os.environ.copy()
        if self.definition.env:
            env.update(self.definition.env)
        
        # Determine working directory
        cwd = self.definition.working_dir
        
        # Execute the command
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
            shell=self.definition.shell,
        )
        
        # Stream stdout
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            yield line.decode()
        
        # Wait for the command to complete
        await process.wait()
        
        # Check if the command succeeded
        if process.returncode != 0:
            stderr = await process.stderr.read()
            raise subprocess.SubprocessError(
                f"Command failed with exit code {process.returncode}: {stderr.decode()}"
            )
    
    async def _execute_script(self, script: str) -> str:
        """
        Execute a shell script.
        
        Args:
            script: Script to execute
            
        Returns:
            Script output
            
        Raises:
            subprocess.SubprocessError: If script execution fails
        """
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            # Execute the script
            return await self._execute_command(script_path)
        
        finally:
            # Clean up the temporary file
            os.unlink(script_path)
    
    async def _execute_script_streaming(self, script: str) -> AsyncIterator[str]:
        """
        Execute a shell script with streaming output.
        
        Args:
            script: Script to execute
            
        Yields:
            Script output chunks
            
        Raises:
            subprocess.SubprocessError: If script execution fails
        """
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            # Execute the script with streaming
            async for chunk in self._execute_command_streaming(script_path):
                yield chunk
        
        finally:
            # Clean up the temporary file
            os.unlink(script_path)


class YAMLToolLoader:
    """
    Loader for YAML-defined tools.
    
    This class provides functionality for loading tools from YAML files,
    particularly shell commands and scripts.
    """
    
    def __init__(self) -> None:
        """Initialize the YAML tool loader."""
        pass
    
    def load_tool_from_yaml(self, yaml_path: Union[str, Path]) -> ShellCommandTool:
        """
        Load a tool from a YAML file.
        
        Args:
            yaml_path: Path to the YAML file
            
        Returns:
            Shell command tool
            
        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the YAML file is invalid
        """
        # Convert to Path object
        path = Path(yaml_path)
        
        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"YAML file '{yaml_path}' does not exist")
        
        # Load the YAML file
        with open(path, "r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file '{yaml_path}': {str(e)}")
        
        # Create a command definition
        try:
            definition = ShellCommandDefinition(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid command definition in '{yaml_path}': {str(e)}")
        
        # Create a shell command tool
        return ShellCommandTool(definition)
    
    def load_tools_from_directory(self, directory: Union[str, Path]) -> List[ShellCommandTool]:
        """
        Load tools from YAML files in a directory.
        
        Args:
            directory: Path to the directory
            
        Returns:
            List of shell command tools
            
        Raises:
            FileNotFoundError: If the directory does not exist
        """
        # Convert to Path object
        path = Path(directory)
        
        # Check if the directory exists
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory}' does not exist")
        
        # Check if it's a directory
        if not path.is_dir():
            raise FileNotFoundError(f"'{directory}' is not a directory")
        
        # Load tools from YAML files
        tools = []
        for yaml_path in path.glob("*.yaml"):
            try:
                tool = self.load_tool_from_yaml(yaml_path)
                tools.append(tool)
            except (ValueError, FileNotFoundError):
                # Skip invalid files
                pass
        
        return tools


# Global YAML tool loader instance
yaml_tool_loader = YAMLToolLoader()


# Convenience functions
def load_tool_from_yaml(yaml_path: Union[str, Path]) -> ShellCommandTool:
    """Load a tool from a YAML file."""
    return yaml_tool_loader.load_tool_from_yaml(yaml_path)


def load_tools_from_directory(directory: Union[str, Path]) -> List[ShellCommandTool]:
    """Load tools from YAML files in a directory."""
    return yaml_tool_loader.load_tools_from_directory(directory)
