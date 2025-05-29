"""
MCP CLI application.

This module provides a comprehensive command-line interface for the MCP client
and server, built using Typer. It includes commands for starting servers,
managing tools, interacting with prompts and resources, and more.
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

from mcp import __version__
from mcp.protocol import (
    ServerInfo,
    ServerCapabilities,
    ClientInfo,
    ClientCapabilities,
    TransportCapability,
)
from mcp.server.server import Server, ServerOptions
from mcp.server.session import InMemorySessionStore
from mcp.client.client import (
    Client,
    StdioTransport,
    HTTPTransport,
    ClientError,
    create_stdio_client,
    create_http_client,
)
from mcp.transport.stdio import StdioTransport as ServerStdioTransport
from mcp.transport.sse import SSETransport
from mcp.tools.registry import (
    MCPToolRegistry,
    load_tool_from_yaml,
    load_tools_from_directory,
    get_registry as get_tool_registry,
)
from mcp.prompts.registry import (
    PromptRegistry,
    get_registry as get_prompt_registry,
)
from mcp.resources.registry import (
    ResourceRegistry,
    get_registry as get_resource_registry,
)

# Create the Typer app
app = typer.Typer(
    name="mcp",
    help="Model Context Protocol (MCP) client and server",
    add_completion=False,
)

# Create the console for rich output
console = Console()

# Create command groups
server_app = typer.Typer(help="Server commands")
client_app = typer.Typer(help="Client commands")
tool_app = typer.Typer(help="Tool commands")
prompt_app = typer.Typer(help="Prompt commands")
resource_app = typer.Typer(help="Resource commands")
config_app = typer.Typer(help="Configuration commands")

# Add command groups to the main app
app.add_typer(server_app, name="server")
app.add_typer(client_app, name="client")
app.add_typer(tool_app, name="tool")
app.add_typer(prompt_app, name="prompt")
app.add_typer(resource_app, name="resource")
app.add_typer(config_app, name="config")

# Create subcommand groups
server_tools_app = typer.Typer(help="Server tool commands")
client_tools_app = typer.Typer(help="Client tool commands")
client_prompts_app = typer.Typer(help="Client prompt commands")
client_resources_app = typer.Typer(help="Client resource commands")

# Add subcommand groups
server_app.add_typer(server_tools_app, name="tools")
client_app.add_typer(client_tools_app, name="tools")
client_app.add_typer(client_prompts_app, name="prompts")
client_app.add_typer(client_resources_app, name="resources")


# Configure logging
def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging with the specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


# Helper function to run async functions
def run_async(func, *args, **kwargs):
    """
    Run an async function in the event loop.
    
    Args:
        func: Async function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the async function
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(func(*args, **kwargs))


# Helper function to load configuration
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration paths
    default_paths = [
        Path.cwd() / "mcp.yaml",
        Path.cwd() / "mcp.yml",
        Path.cwd() / ".mcp.yaml",
        Path.cwd() / ".mcp.yml",
        Path.home() / ".mcp.yaml",
        Path.home() / ".mcp.yml",
        Path.home() / ".config" / "mcp" / "config.yaml",
    ]
    
    # Use the specified path or try default paths
    paths_to_try = [config_path] if config_path else default_paths
    
    for path in paths_to_try:
        if path and path.exists():
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                console.print(f"[bold red]Error loading configuration from {path}: {str(e)}[/]")
                return {}
    
    # No configuration found
    if config_path:
        console.print(f"[bold yellow]Configuration file not found: {config_path}[/]")
    
    return {}


# Helper function to get the default profile from configuration
def get_default_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the default profile from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Default profile dictionary
    """
    profiles = config.get("profiles", {})
    default_profile_name = config.get("default_profile")
    
    if default_profile_name and default_profile_name in profiles:
        return profiles[default_profile_name]
    elif profiles:
        # Use the first profile if no default is specified
        return next(iter(profiles.values()))
    else:
        return {}


# Helper function to create a server
async def create_server(
    transport_type: str,
    host: str = "127.0.0.1",
    port: int = 3000,
    name: str = "pymcp-server",
    version: str = __version__,
    debug: bool = False,
    tool_dirs: List[str] = None,
    prompt_dirs: List[str] = None,
    resource_dirs: List[str] = None,
) -> Server:
    """
    Create an MCP server with the specified options.
    
    Args:
        transport_type: Transport type (stdio, sse)
        host: Host to bind to (for sse transport)
        port: Port to bind to (for sse transport)
        name: Server name
        version: Server version
        debug: Enable debug mode
        tool_dirs: Directories to load tools from
        prompt_dirs: Directories to load prompts from
        resource_dirs: Directories to load resources from
        
    Returns:
        MCP server instance
        
    Raises:
        ValueError: If the transport type is invalid
    """
    # Set up logging
    setup_logging("DEBUG" if debug else "INFO")
    
    # Create the tool registry
    tool_registry = MCPToolRegistry()
    
    # Load tools if directories are specified
    if tool_dirs:
        for tool_dir in tool_dirs:
            try:
                tools = load_tools_from_directory(tool_dir)
                for tool in tools:
                    tool_registry.register_tool(tool)
                console.print(f"[green]Loaded {len(tools)} tools from {tool_dir}[/]")
            except Exception as e:
                console.print(f"[bold red]Error loading tools from {tool_dir}: {str(e)}[/]")
    
    # Create the prompt registry
    prompt_registry = get_prompt_registry()
    
    # Load prompts if directories are specified
    if prompt_dirs:
        for prompt_dir in prompt_dirs:
            try:
                prompts = prompt_registry.load_prompts_from_directory(prompt_dir)
                console.print(f"[green]Loaded {len(prompts)} prompts from {prompt_dir}[/]")
            except Exception as e:
                console.print(f"[bold red]Error loading prompts from {prompt_dir}: {str(e)}[/]")
    
    # Create the resource registry
    resource_registry = get_resource_registry()
    
    # Load resources if directories are specified
    if resource_dirs:
        for resource_dir in resource_dirs:
            try:
                resources = resource_registry.load_resources_from_directory(resource_dir)
                console.print(f"[green]Loaded {len(resources)} resources from {resource_dir}[/]")
            except Exception as e:
                console.print(f"[bold red]Error loading resources from {resource_dir}: {str(e)}[/]")
    
    # Create the transport
    if transport_type == "stdio":
        transport = ServerStdioTransport()
    elif transport_type == "sse":
        transport = SSETransport(host=host, port=port)
    else:
        raise ValueError(f"Invalid transport type: {transport_type}")
    
    # Create the server options
    options = ServerOptions(
        name=name,
        version=version,
        debug=debug,
    )
    
    # Create the session store
    session_store = InMemorySessionStore()
    
    # Create the server
    server = Server(
        transport=transport,
        options=options,
        session_store=session_store,
        tool_provider=tool_registry,
        prompt_provider=prompt_registry,
        resource_provider=resource_registry,
    )
    
    return server


# Version command
@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
):
    """
    Model Context Protocol (MCP) client and server.
    """
    # Set up logging
    setup_logging("DEBUG" if debug else "INFO")
    
    # Show version and exit if requested
    if version:
        console.print(f"[bold]PyMCP[/] version [bold blue]{__version__}[/]")
        raise typer.Exit()


# Server commands
@server_app.command("start")
def server_start(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, sse)",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to (for sse transport)",
    ),
    port: int = typer.Option(
        3000,
        "--port",
        "-p",
        help="Port to bind to (for sse transport)",
    ),
    name: str = typer.Option(
        "pymcp-server",
        "--name",
        "-n",
        help="Server name",
    ),
    version: str = typer.Option(
        __version__,
        "--server-version",
        help="Server version",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Configuration profile to use",
    ),
    tool_dir: List[str] = typer.Option(
        [],
        "--tool-dir",
        help="Directory to load tools from (can be specified multiple times)",
    ),
    prompt_dir: List[str] = typer.Option(
        [],
        "--prompt-dir",
        help="Directory to load prompts from (can be specified multiple times)",
    ),
    resource_dir: List[str] = typer.Option(
        [],
        "--resource-dir",
        help="Directory to load resources from (can be specified multiple times)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug output",
    ),
):
    """
    Start an MCP server.
    """
    # Load configuration
    cfg = load_config(config)
    
    # Get the profile
    profile_cfg = {}
    if profile and profile in cfg.get("profiles", {}):
        profile_cfg = cfg["profiles"][profile]
    elif not profile and "default_profile" in cfg:
        default_profile = cfg["default_profile"]
        if default_profile in cfg.get("profiles", {}):
            profile_cfg = cfg["profiles"][default_profile]
    
    # Merge command-line options with profile configuration
    transport = transport or profile_cfg.get("transport", "stdio")
    host = host or profile_cfg.get("host", "127.0.0.1")
    port = port or profile_cfg.get("port", 3000)
    name = name or profile_cfg.get("name", "pymcp-server")
    version = version or profile_cfg.get("version", __version__)
    debug = debug or profile_cfg.get("debug", False)
    
    # Merge tool directories
    tool_dirs = list(tool_dir)
    if "tool_dirs" in profile_cfg:
        tool_dirs.extend(profile_cfg["tool_dirs"])
    
    # Merge prompt directories
    prompt_dirs = list(prompt_dir)
    if "prompt_dirs" in profile_cfg:
        prompt_dirs.extend(profile_cfg["prompt_dirs"])
    
    # Merge resource directories
    resource_dirs = list(resource_dir)
    if "resource_dirs" in profile_cfg:
        resource_dirs.extend(profile_cfg["resource_dirs"])
    
    try:
        # Create and start the server
        console.print(f"[bold]Starting MCP server[/] with [bold blue]{transport}[/] transport")
        
        if transport == "sse":
            console.print(f"Listening on [bold]http://{host}:{port}[/]")
        
        # Create the server
        server = run_async(
            create_server,
            transport_type=transport,
            host=host,
            port=port,
            name=name,
            version=version,
            debug=debug,
            tool_dirs=tool_dirs,
            prompt_dirs=prompt_dirs,
            resource_dirs=resource_dirs,
        )
        
        # Start the server
        run_async(server.start)
        
        # Wait for the server to shut down
        run_async(server.wait_for_shutdown)
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Server stopped by user[/]")
    
    except Exception as e:
        console.print(f"[bold red]Error starting server: {str(e)}[/]")
        raise typer.Exit(1)


@server_tools_app.command("list")
def server_tools_list(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Configuration profile to use",
    ),
    tool_dir: List[str] = typer.Option(
        [],
        "--tool-dir",
        help="Directory to load tools from (can be specified multiple times)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
):
    """
    List available tools.
    """
    # Load configuration
    cfg = load_config(config)
    
    # Get the profile
    profile_cfg = {}
    if profile and profile in cfg.get("profiles", {}):
        profile_cfg = cfg["profiles"][profile]
    elif not profile and "default_profile" in cfg:
        default_profile = cfg["default_profile"]
        if default_profile in cfg.get("profiles", {}):
            profile_cfg = cfg["profiles"][default_profile]
    
    # Merge tool directories
    tool_dirs = list(tool_dir)
    if "tool_dirs" in profile_cfg:
        tool_dirs.extend(profile_cfg["tool_dirs"])
    
    # Create the tool registry
    tool_registry = MCPToolRegistry()
    
    # Load tools if directories are specified
    if tool_dirs:
        for tool_dir in tool_dirs:
            try:
                tools = load_tools_from_directory(tool_dir)
                for tool in tools:
                    tool_registry.register_tool(tool)
                console.print(f"[green]Loaded {len(tools)} tools from {tool_dir}[/]")
            except Exception as e:
                console.print(f"[bold red]Error loading tools from {tool_dir}: {str(e)}[/]")
    
    # Get the list of tools
    tools = run_async(tool_registry.list_tools)
    
    # Output the tools
    if format == "json":
        console.print_json(json.dumps(tools))
    else:
        # Create a table
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Parameters", style="yellow")
        table.add_column("Tags", style="blue")
        
        for tool in tools:
            # Format parameters
            params = ", ".join([p["name"] for p in tool.get("parameters", [])])
            
            # Format tags
            tags = ", ".join(tool.get("tags", []))
            
            # Add the tool to the table
            table.add_row(
                tool["name"],
                tool.get("description", ""),
                params,
                tags,
            )
        
        console.print(table)


@server_tools_app.command("call")
def server_tools_call(
    name: str = typer.Argument(..., help="Name of the tool to call"),
    args_str: str = typer.Option(
        "{}",
        "--args",
        "-a",
        help="Tool arguments as a JSON string",
    ),
    args_file: Optional[Path] = typer.Option(
        None,
        "--args-file",
        "-f",
        help="File containing tool arguments as JSON",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Configuration profile to use",
    ),
    tool_dir: List[str] = typer.Option(
        [],
        "--tool-dir",
        help="Directory to load tools from (can be specified multiple times)",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        "-s",
        help="Stream the tool output",
    ),
):
    """
    Call a tool directly.
    """
    # Load configuration
    cfg = load_config(config)
    
    # Get the profile
    profile_cfg = {}
    if profile and profile in cfg.get("profiles", {}):
        profile_cfg = cfg["profiles"][profile]
    elif not profile and "default_profile" in cfg:
        default_profile = cfg["default_profile"]
        if default_profile in cfg.get("profiles", {}):
            profile_cfg = cfg["profiles"][default_profile]
    
    # Merge tool directories
    tool_dirs = list(tool_dir)
    if "tool_dirs" in profile_cfg:
        tool_dirs.extend(profile_cfg["tool_dirs"])
    
    # Create the tool registry
    tool_registry = MCPToolRegistry()
    
    # Load tools if directories are specified
    if tool_dirs:
        for tool_dir in tool_dirs:
            try:
                tools = load_tools_from_directory(tool_dir)
                for tool in tools:
                    tool_registry.register_tool(tool)
                console.print(f"[green]Loaded {len(tools)} tools from {tool_dir}[/]")
            except Exception as e:
                console.print(f"[bold red]Error loading tools from {tool_dir}: {str(e)}[/]")
    
    # Parse arguments
    args = {}
    if args_file:
        try:
            with open(args_file, "r") as f:
                args = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading arguments from {args_file}: {str(e)}[/]")
            raise typer.Exit(1)
    else:
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            console.print(f"[bold red]Invalid JSON arguments: {args_str}[/]")
            raise typer.Exit(1)
    
    try:
        # Call the tool
        if stream:
            # Stream the tool output
            console.print(f"[bold]Calling tool[/] [bold blue]{name}[/] with streaming...")
            
            result_queue = run_async(tool_registry.call_tool, name, args, True)
            
            # Process the streaming results
            while True:
                try:
                    chunk = run_async(result_queue.get)
                    
                    # Check if it's the last chunk
                    if chunk.get("is_last", False):
                        # Check for errors
                        if "error" in chunk:
                            console.print(f"[bold red]Error: {chunk['error']}[/]")
                        break
                    
                    # Print the chunk
                    if "chunk" in chunk:
                        console.print(chunk["chunk"], end="", highlight=False)
                    else:
                        console.print_json(json.dumps(chunk))
                
                except asyncio.CancelledError:
                    break
                
                except Exception as e:
                    console.print(f"[bold red]Error processing chunk: {str(e)}[/]")
                    break
        
        else:
            # Call the tool normally
            console.print(f"[bold]Calling tool[/] [bold blue]{name}[/]...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Executing tool...", total=None)
                result = run_async(tool_registry.call_tool, name, args)
            
            # Print the result
            if isinstance(result, dict):
                console.print_json(json.dumps(result))
            else:
                console.print(result)
    
    except ValueError as e:
        console.print(f"[bold red]Tool not found: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error calling tool: {str(e)}[/]")
        raise typer.Exit(1)


# Client commands
@client_app.command("connect")
def client_connect(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, http)",
    ),
    url: str = typer.Option(
        "http://localhost:3000",
        "--url",
        "-u",
        help="Server URL (for http transport)",
    ),
    command: str = typer.Option(
        None,
        "--command",
        "-c",
        help="Server command (for stdio transport)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
):
    """
    Connect to an MCP server and show server information.
    """
    try:
        # Create the client
        if transport == "stdio":
            # Parse the command
            cmd_args = command.split(",") if command else None
            
            # Create the client
            client = run_async(
                create_stdio_client,
                command=cmd_args,
            )
        
        elif transport == "http":
            # Create the client
            client = run_async(
                create_http_client,
                url=url,
            )
        
        else:
            console.print(f"[bold red]Invalid transport type: {transport}[/]")
            raise typer.Exit(1)
        
        # Get server information
        server_info = client.server_info
        server_capabilities = client.server_capabilities
        
        # Output the server information
        if format == "json":
            info = {
                "server_info": server_info.model_dump(),
                "server_capabilities": server_capabilities.model_dump(),
                "session_id": client.session_id,
            }
            console.print_json(json.dumps(info))
        
        else:
            # Create a table for server information
            info_table = Table(title="Server Information")
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value", style="green")
            
            info_table.add_row("Name", server_info.name)
            info_table.add_row("Version", server_info.version)
            info_table.add_row("Session ID", client.session_id or "None")
            
            # Create a table for server capabilities
            cap_table = Table(title="Server Capabilities")
            cap_table.add_column("Capability", style="cyan")
            cap_table.add_column("Supported", style="green")
            
            cap_table.add_row(
                "Tool Streaming",
                "✓" if server_capabilities.supports_tool_streaming else "✗",
            )
            cap_table.add_row(
                "Batch Requests",
                "✓" if server_capabilities.supports_batch_requests else "✗",
            )
            cap_table.add_row(
                "Cancellation",
                "✓" if server_capabilities.supports_cancellation else "✗",
            )
            cap_table.add_row(
                "Progress",
                "✓" if server_capabilities.supports_progress else "✗",
            )
            cap_table.add_row(
                "Resources",
                "✓" if server_capabilities.supports_resources else "✗",
            )
            cap_table.add_row(
                "Prompts",
                "✓" if server_capabilities.supports_prompts else "✗",
            )
            
            console.print(info_table)
            console.print(cap_table)
        
        # Disconnect from the server
        run_async(client.disconnect)
    
    except ClientError as e:
        console.print(f"[bold red]Client error: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error connecting to server: {str(e)}[/]")
        raise typer.Exit(1)


@client_tools_app.command("list")
def client_tools_list(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, http)",
    ),
    url: str = typer.Option(
        "http://localhost:3000",
        "--url",
        "-u",
        help="Server URL (for http transport)",
    ),
    command: str = typer.Option(
        None,
        "--command",
        "-c",
        help="Server command (for stdio transport)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
):
    """
    List tools available on the server.
    """
    try:
        # Create the client
        if transport == "stdio":
            # Parse the command
            cmd_args = command.split(",") if command else None
            
            # Create the client
            client = run_async(
                create_stdio_client,
                command=cmd_args,
            )
        
        elif transport == "http":
            # Create the client
            client = run_async(
                create_http_client,
                url=url,
            )
        
        else:
            console.print(f"[bold red]Invalid transport type: {transport}[/]")
            raise typer.Exit(1)
        
        # List tools
        response = run_async(client.list_tools)
        tools = response.result.tools
        
        # Output the tools
        if format == "json":
            console.print_json(json.dumps(tools))
        
        else:
            # Create a table
            table = Table(title="Available Tools")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Parameters", style="yellow")
            table.add_column("Streaming", style="blue")
            
            for tool in tools:
                # Format parameters
                params = ", ".join([p["name"] for p in tool.get("parameters", [])])
                
                # Add the tool to the table
                table.add_row(
                    tool["name"],
                    tool.get("description", ""),
                    params,
                    "✓" if tool.get("streaming", False) else "✗",
                )
            
            console.print(table)
        
        # Disconnect from the server
        run_async(client.disconnect)
    
    except ClientError as e:
        console.print(f"[bold red]Client error: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error listing tools: {str(e)}[/]")
        raise typer.Exit(1)


@client_tools_app.command("call")
def client_tools_call(
    name: str = typer.Argument(..., help="Name of the tool to call"),
    args_str: str = typer.Option(
        "{}",
        "--args",
        "-a",
        help="Tool arguments as a JSON string",
    ),
    args_file: Optional[Path] = typer.Option(
        None,
        "--args-file",
        "-f",
        help="File containing tool arguments as JSON",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, http)",
    ),
    url: str = typer.Option(
        "http://localhost:3000",
        "--url",
        "-u",
        help="Server URL (for http transport)",
    ),
    command: str = typer.Option(
        None,
        "--command",
        "-c",
        help="Server command (for stdio transport)",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        "-s",
        help="Stream the tool output",
    ),
):
    """
    Call a tool on the server.
    """
    # Parse arguments
    args = {}
    if args_file:
        try:
            with open(args_file, "r") as f:
                args = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading arguments from {args_file}: {str(e)}[/]")
            raise typer.Exit(1)
    else:
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            console.print(f"[bold red]Invalid JSON arguments: {args_str}[/]")
            raise typer.Exit(1)
    
    try:
        # Create the client
        if transport == "stdio":
            # Parse the command
            cmd_args = command.split(",") if command else None
            
            # Create the client
            client = run_async(
                create_stdio_client,
                command=cmd_args,
            )
        
        elif transport == "http":
            # Create the client
            client = run_async(
                create_http_client,
                url=url,
            )
        
        else:
            console.print(f"[bold red]Invalid transport type: {transport}[/]")
            raise typer.Exit(1)
        
        # Call the tool
        if stream:
            # Stream the tool output
            console.print(f"[bold]Calling tool[/] [bold blue]{name}[/] with streaming...")
            
            streaming_response = run_async(client.call_tool, name, args, True)
            
            # Process the streaming results
            async def process_stream():
                try:
                    async for chunk in streaming_response:
                        # Print the chunk
                        if "chunk" in chunk:
                            console.print(chunk["chunk"], end="", highlight=False)
                        else:
                            console.print_json(json.dumps(chunk))
                
                except ClientError as e:
                    console.print(f"[bold red]Client error: {str(e)}[/]")
                
                except Exception as e:
                    console.print(f"[bold red]Error processing chunk: {str(e)}[/]")
            
            # Run the stream processor
            run_async(process_stream)
        
        else:
            # Call the tool normally
            console.print(f"[bold]Calling tool[/] [bold blue]{name}[/]...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Executing tool...", total=None)
                response = run_async(client.call_tool, name, args)
            
            # Print the result
            result = response.result
            if isinstance(result, dict):
                console.print_json(json.dumps(result))
            else:
                console.print(result)
        
        # Disconnect from the server
        run_async(client.disconnect)
    
    except ClientError as e:
        console.print(f"[bold red]Client error: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error calling tool: {str(e)}[/]")
        raise typer.Exit(1)


@client_prompts_app.command("list")
def client_prompts_list(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, http)",
    ),
    url: str = typer.Option(
        "http://localhost:3000",
        "--url",
        "-u",
        help="Server URL (for http transport)",
    ),
    command: str = typer.Option(
        None,
        "--command",
        "-c",
        help="Server command (for stdio transport)",
    ),
    tags: List[str] = typer.Option(
        [],
        "--tag",
        help="Filter prompts by tag (can be specified multiple times)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
):
    """
    List prompts available on the server.
    """
    try:
        # Create the client
        if transport == "stdio":
            # Parse the command
            cmd_args = command.split(",") if command else None
            
            # Create the client
            client = run_async(
                create_stdio_client,
                command=cmd_args,
            )
        
        elif transport == "http":
            # Create the client
            client = run_async(
                create_http_client,
                url=url,
            )
        
        else:
            console.print(f"[bold red]Invalid transport type: {transport}[/]")
            raise typer.Exit(1)
        
        # List prompts
        response = run_async(client.list_prompts, tags if tags else None)
        prompts = response.result.prompts
        
        # Output the prompts
        if format == "json":
            console.print_json(json.dumps(prompts))
        
        else:
            # Create a table
            table = Table(title="Available Prompts")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Parameters", style="yellow")
            table.add_column("Tags", style="blue")
            
            for prompt in prompts:
                # Format parameters
                params = ", ".join([p["name"] for p in prompt.get("parameters", [])])
                
                # Format tags
                tags = ", ".join(prompt.get("tags", []))
                
                # Add the prompt to the table
                table.add_row(
                    prompt["name"],
                    prompt.get("description", ""),
                    params,
                    tags,
                )
            
            console.print(table)
        
        # Disconnect from the server
        run_async(client.disconnect)
    
    except ClientError as e:
        console.print(f"[bold red]Client error: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error listing prompts: {str(e)}[/]")
        raise typer.Exit(1)


@client_prompts_app.command("get")
def client_prompts_get(
    name: str = typer.Argument(..., help="Name of the prompt to get"),
    args_str: str = typer.Option(
        "{}",
        "--args",
        "-a",
        help="Prompt arguments as a JSON string",
    ),
    args_file: Optional[Path] = typer.Option(
        None,
        "--args-file",
        "-f",
        help="File containing prompt arguments as JSON",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, http)",
    ),
    url: str = typer.Option(
        "http://localhost:3000",
        "--url",
        "-u",
        help="Server URL (for http transport)",
    ),
    command: str = typer.Option(
        None,
        "--command",
        "-c",
        help="Server command (for stdio transport)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="File to write the prompt content to",
    ),
):
    """
    Get a prompt from the server.
    """
    # Parse arguments
    args = {}
    if args_file:
        try:
            with open(args_file, "r") as f:
                args = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading arguments from {args_file}: {str(e)}[/]")
            raise typer.Exit(1)
    else:
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            console.print(f"[bold red]Invalid JSON arguments: {args_str}[/]")
            raise typer.Exit(1)
    
    try:
        # Create the client
        if transport == "stdio":
            # Parse the command
            cmd_args = command.split(",") if command else None
            
            # Create the client
            client = run_async(
                create_stdio_client,
                command=cmd_args,
            )
        
        elif transport == "http":
            # Create the client
            client = run_async(
                create_http_client,
                url=url,
            )
        
        else:
            console.print(f"[bold red]Invalid transport type: {transport}[/]")
            raise typer.Exit(1)
        
        # Get the prompt
        console.print(f"[bold]Getting prompt[/] [bold blue]{name}[/]...")
        
        response = run_async(client.get_prompt, name, args)
        result = response.result
        
        # Get the content and metadata
        content = result.content
        metadata = result.metadata
        
        # Output the content
        if output_file:
            # Write to file
            with open(output_file, "w") as f:
                f.write(content)
            console.print(f"[green]Prompt content written to {output_file}[/]")
        else:
            # Print to console
            syntax = Syntax(content, "markdown", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=f"Prompt: {name}", border_style="cyan"))
        
        # Print metadata
        metadata_table = Table(title="Prompt Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="green")
        
        metadata_table.add_row("Name", metadata.name)
        metadata_table.add_row("Description", metadata.description or "")
        metadata_table.add_row("Version", metadata.version or "")
        metadata_table.add_row("Tags", ", ".join(metadata.tags))
        
        console.print(metadata_table)
        
        # Disconnect from the server
        run_async(client.disconnect)
    
    except ClientError as e:
        console.print(f"[bold red]Client error: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error getting prompt: {str(e)}[/]")
        raise typer.Exit(1)


@client_resources_app.command("list")
def client_resources_list(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, http)",
    ),
    url: str = typer.Option(
        "http://localhost:3000",
        "--url",
        "-u",
        help="Server URL (for http transport)",
    ),
    command: str = typer.Option(
        None,
        "--command",
        "-c",
        help="Server command (for stdio transport)",
    ),
    tags: List[str] = typer.Option(
        [],
        "--tag",
        help="Filter resources by tag (can be specified multiple times)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
):
    """
    List resources available on the server.
    """
    try:
        # Create the client
        if transport == "stdio":
            # Parse the command
            cmd_args = command.split(",") if command else None
            
            # Create the client
            client = run_async(
                create_stdio_client,
                command=cmd_args,
            )
        
        elif transport == "http":
            # Create the client
            client = run_async(
                create_http_client,
                url=url,
            )
        
        else:
            console.print(f"[bold red]Invalid transport type: {transport}[/]")
            raise typer.Exit(1)
        
        # List resources
        response = run_async(client.list_resources, tags if tags else None)
        resources = response.result.resources
        
        # Output the resources
        if format == "json":
            console.print_json(json.dumps(resources))
        
        else:
            # Create a table
            table = Table(title="Available Resources")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("MIME Type", style="yellow")
            table.add_column("Size", style="blue")
            table.add_column("Tags", style="magenta")
            
            for resource in resources:
                # Format size
                size = resource.get("size", 0)
                if size:
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                else:
                    size_str = "Unknown"
                
                # Format tags
                tags = ", ".join(resource.get("tags", []))
                
                # Add the resource to the table
                table.add_row(
                    resource["name"],
                    resource.get("description", ""),
                    resource.get("mime_type", "Unknown"),
                    size_str,
                    tags,
                )
            
            console.print(table)
        
        # Disconnect from the server
        run_async(client.disconnect)
    
    except ClientError as e:
        console.print(f"[bold red]Client error: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error listing resources: {str(e)}[/]")
        raise typer.Exit(1)


@client_resources_app.command("read")
def client_resources_read(
    name: str = typer.Argument(..., help="Name of the resource to read"),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, http)",
    ),
    url: str = typer.Option(
        "http://localhost:3000",
        "--url",
        "-u",
        help="Server URL (for http transport)",
    ),
    command: str = typer.Option(
        None,
        "--command",
        "-c",
        help="Server command (for stdio transport)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="File to write the resource content to",
    ),
):
    """
    Read a resource from the server.
    """
    try:
        # Create the client
        if transport == "stdio":
            # Parse the command
            cmd_args = command.split(",") if command else None
            
            # Create the client
            client = run_async(
                create_stdio_client,
                command=cmd_args,
            )
        
        elif transport == "http":
            # Create the client
            client = run_async(
                create_http_client,
                url=url,
            )
        
        else:
            console.print(f"[bold red]Invalid transport type: {transport}[/]")
            raise typer.Exit(1)
        
        # Read the resource
        console.print(f"[bold]Reading resource[/] [bold blue]{name}[/]...")
        
        response = run_async(client.read_resource, name)
        result = response.result
        
        # Get the content and metadata
        content = result.content
        metadata = result.metadata
        
        # Check if the content is base64 encoded
        is_binary = metadata.encoding == "base64"
        
        # Output the content
        if output_file:
            # Write to file
            if is_binary:
                # Decode base64 and write as binary
                import base64
                with open(output_file, "wb") as f:
                    f.write(base64.b64decode(content))
            else:
                # Write as text
                with open(output_file, "w") as f:
                    f.write(content)
            
            console.print(f"[green]Resource content written to {output_file}[/]")
        
        else:
            # Print to console
            if is_binary:
                console.print("[yellow]Binary content (base64 encoded)[/]")
                console.print(content[:100] + "..." if len(content) > 100 else content)
            else:
                # Try to determine the syntax highlighting language
                lang = "text"
                if metadata.mime_type:
                    if metadata.mime_type.startswith("text/plain"):
                        lang = "text"
                    elif metadata.mime_type.startswith("text/html"):
                        lang = "html"
                    elif metadata.mime_type.startswith("application/json"):
                        lang = "json"
                    elif metadata.mime_type.startswith("application/xml"):
                        lang = "xml"
                    elif metadata.mime_type.startswith("text/markdown"):
                        lang = "markdown"
                
                syntax = Syntax(content, lang, theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"Resource: {name}", border_style="cyan"))
        
        # Print metadata
        metadata_table = Table(title="Resource Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="green")
        
        metadata_table.add_row("Name", metadata.name)
        metadata_table.add_row("Description", metadata.description or "")
        metadata_table.add_row("MIME Type", metadata.mime_type or "Unknown")
        metadata_table.add_row("Size", str(metadata.size or "Unknown"))
        metadata_table.add_row("Encoding", metadata.encoding or "None")
        metadata_table.add_row("Version", metadata.version or "")
        metadata_table.add_row("Tags", ", ".join(metadata.tags))
        
        console.print(metadata_table)
        
        # Disconnect from the server
        run_async(client.disconnect)
    
    except ClientError as e:
        console.print(f"[bold red]Client error: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error reading resource: {str(e)}[/]")
        raise typer.Exit(1)


# Tool commands
@tool_app.command("load")
def tool_load(
    path: Path = typer.Argument(..., help="Path to YAML tool definition or directory"),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively load tools from directories",
    ),
):
    """
    Load a tool from a YAML file or directory.
    """
    try:
        # Check if the path exists
        if not path.exists():
            console.print(f"[bold red]Path does not exist: {path}[/]")
            raise typer.Exit(1)
        
        # Get the tool registry
        tool_registry = get_tool_registry()
        
        if path.is_dir():
            # Load tools from directory
            console.print(f"[bold]Loading tools from directory[/] [bold blue]{path}[/]...")
            
            # Count the number of tools loaded
            count = 0
            
            if recursive:
                # Walk the directory tree
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith((".yaml", ".yml")):
                            file_path = Path(root) / file
                            try:
                                tool = load_tool_from_yaml(file_path)
                                tool_registry.register_tool(tool)
                                console.print(f"[green]Loaded tool {tool.get_schema().name} from {file_path}[/]")
                                count += 1
                            except Exception as e:
                                console.print(f"[yellow]Error loading tool from {file_path}: {str(e)}[/]")
            else:
                # Load tools from the directory
                for file in path.glob("*.yaml"):
                    try:
                        tool = load_tool_from_yaml(file)
                        tool_registry.register_tool(tool)
                        console.print(f"[green]Loaded tool {tool.get_schema().name} from {file}[/]")
                        count += 1
                    except Exception as e:
                        console.print(f"[yellow]Error loading tool from {file}: {str(e)}[/]")
                
                for file in path.glob("*.yml"):
                    try:
                        tool = load_tool_from_yaml(file)
                        tool_registry.register_tool(tool)
                        console.print(f"[green]Loaded tool {tool.get_schema().name} from {file}[/]")
                        count += 1
                    except Exception as e:
                        console.print(f"[yellow]Error loading tool from {file}: {str(e)}[/]")
            
            console.print(f"[bold green]Loaded {count} tools from {path}[/]")
        
        else:
            # Load a single tool
            console.print(f"[bold]Loading tool from file[/] [bold blue]{path}[/]...")
            
            tool = load_tool_from_yaml(path)
            tool_registry.register_tool(tool)
            
            console.print(f"[bold green]Loaded tool {tool.get_schema().name} from {path}[/]")
    
    except Exception as e:
        console.print(f"[bold red]Error loading tool: {str(e)}[/]")
        raise typer.Exit(1)


@tool_app.command("list")
def tool_list(
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table, json)",
    ),
):
    """
    List registered tools.
    """
    try:
        # Get the tool registry
        tool_registry = get_tool_registry()
        
        # Get the list of tools
        tools = run_async(tool_registry.list_tools)
        
        # Output the tools
        if format == "json":
            console.print_json(json.dumps(tools))
        else:
            # Create a table
            table = Table(title="Registered Tools")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Parameters", style="yellow")
            table.add_column("Tags", style="blue")
            
            for tool in tools:
                # Format parameters
                params = ", ".join([p["name"] for p in tool.get("parameters", [])])
                
                # Format tags
                tags = ", ".join(tool.get("tags", []))
                
                # Add the tool to the table
                table.add_row(
                    tool["name"],
                    tool.get("description", ""),
                    params,
                    tags,
                )
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[bold red]Error listing tools: {str(e)}[/]")
        raise typer.Exit(1)


@tool_app.command("call")
def tool_call(
    name: str = typer.Argument(..., help="Name of the tool to call"),
    args_str: str = typer.Option(
        "{}",
        "--args",
        "-a",
        help="Tool arguments as a JSON string",
    ),
    args_file: Optional[Path] = typer.Option(
        None,
        "--args-file",
        "-f",
        help="File containing tool arguments as JSON",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        "-s",
        help="Stream the tool output",
    ),
):
    """
    Call a registered tool.
    """
    # Parse arguments
    args = {}
    if args_file:
        try:
            with open(args_file, "r") as f:
                args = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading arguments from {args_file}: {str(e)}[/]")
            raise typer.Exit(1)
    else:
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            console.print(f"[bold red]Invalid JSON arguments: {args_str}[/]")
            raise typer.Exit(1)
    
    try:
        # Get the tool registry
        tool_registry = get_tool_registry()
        
        # Call the tool
        if stream:
            # Stream the tool output
            console.print(f"[bold]Calling tool[/] [bold blue]{name}[/] with streaming...")
            
            result_queue = run_async(tool_registry.call_tool, name, args, True)
            
            # Process the streaming results
            while True:
                try:
                    chunk = run_async(result_queue.get)
                    
                    # Check if it's the last chunk
                    if chunk.get("is_last", False):
                        # Check for errors
                        if "error" in chunk:
                            console.print(f"[bold red]Error: {chunk['error']}[/]")
                        break
                    
                    # Print the chunk
                    if "chunk" in chunk:
                        console.print(chunk["chunk"], end="", highlight=False)
                    else:
                        console.print_json(json.dumps(chunk))
                
                except asyncio.CancelledError:
                    break
                
                except Exception as e:
                    console.print(f"[bold red]Error processing chunk: {str(e)}[/]")
                    break
        
        else:
            # Call the tool normally
            console.print(f"[bold]Calling tool[/] [bold blue]{name}[/]...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Executing tool...", total=None)
                result = run_async(tool_registry.call_tool, name, args)
            
            # Print the result
            if isinstance(result, dict):
                console.print_json(json.dumps(result))
            else:
                console.print(result)
    
    except ValueError as e:
        console.print(f"[bold red]Tool not found: {str(e)}[/]")
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[bold red]Error calling tool: {str(e)}[/]")
        raise typer.Exit(1)


# Configuration commands
@config_app.command("init")
def config_init(
    output: Path = typer.Option(
        Path.cwd() / "mcp.yaml",
        "--output",
        "-o",
        help="Path to write the configuration file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration file",
    ),
):
    """
    Initialize a new configuration file.
    """
    try:
        # Check if the file already exists
        if output.exists() and not force:
            console.print(f"[bold yellow]Configuration file already exists: {output}[/]")
            console.print("Use --force to overwrite")
            raise typer.Exit(1)
        
        # Create the configuration
        config = {
            "default_profile": "default",
            "profiles": {
                "default": {
                    "name": "pymcp-server",
                    "version": __version__,
                    "transport": "stdio",
                    "host": "127.0.0.1",
                    "port": 3000,
                    "debug": False,
                    "tool_dirs": ["./tools"],
                    "prompt_dirs": ["./prompts"],
                    "resource_dirs": ["./resources"],
                },
                "http": {
                    "name": "pymcp-http-server",
                    "version": __version__,
                    "transport": "sse",
                    "host": "127.0.0.1",
                    "port": 3000,
                    "debug": False,
                    "tool_dirs": ["./tools"],
                    "prompt_dirs": ["./prompts"],
                    "resource_dirs": ["./resources"],
                },
            },
        }
        
        # Write the configuration file
        with open(output, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        console.print(f"[bold green]Configuration file created: {output}[/]")
    
    except Exception as e:
        console.print(f"[bold red]Error creating configuration file: {str(e)}[/]")
        raise typer.Exit(1)


@config_app.command("list-profiles")
def config_list_profiles(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """
    List available configuration profiles.
    """
    try:
        # Load configuration
        cfg = load_config(config)
        
        # Get profiles
        profiles = cfg.get("profiles", {})
        default_profile = cfg.get("default_profile")
        
        if not profiles:
            console.print("[yellow]No profiles found in configuration[/]")
            raise typer.Exit(0)
        
        # Create a table
        table = Table(title="Configuration Profiles")
        table.add_column("Name", style="cyan")
        table.add_column("Transport", style="green")
        table.add_column("Host:Port", style="yellow")
        table.add_column("Tool Dirs", style="blue")
        table.add_column("Default", style="magenta")
        
        for name, profile in profiles.items():
            # Format transport
            transport = profile.get("transport", "stdio")
            
            # Format host:port
            host_port = ""
            if transport == "sse":
                host = profile.get("host", "127.0.0.1")
                port = profile.get("port", 3000)
                host_port = f"{host}:{port}"
            
            # Format tool directories
            tool_dirs = ", ".join(profile.get("tool_dirs", []))
            
            # Format default indicator
            is_default = "✓" if name == default_profile else ""
            
            # Add the profile to the table
            table.add_row(
                name,
                transport,
                host_port,
                tool_dirs,
                is_default,
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[bold red]Error listing profiles: {str(e)}[/]")
        raise typer.Exit(1)


@config_app.command("show-profile")
def config_show_profile(
    name: str = typer.Argument(..., help="Name of the profile to show"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format (yaml, json)",
    ),
):
    """
    Show details of a configuration profile.
    """
    try:
        # Load configuration
        cfg = load_config(config)
        
        # Get profiles
        profiles = cfg.get("profiles", {})
        
        if name not in profiles:
            console.print(f"[bold red]Profile not found: {name}[/]")
            raise typer.Exit(1)
        
        # Get the profile
        profile = profiles[name]
        
        # Output the profile
        if format == "json":
            console.print_json(json.dumps(profile))
        else:
            # Format as YAML
            yaml_str = yaml.dump(profile, default_flow_style=False)
            syntax = Syntax(yaml_str, "yaml", theme="monokai")
            console.print(Panel(syntax, title=f"Profile: {name}", border_style="cyan"))
    
    except Exception as e:
        console.print(f"[bold red]Error showing profile: {str(e)}[/]")
        raise typer.Exit(1)


@config_app.command("add-profile")
def config_add_profile(
    name: str = typer.Argument(..., help="Name of the profile to add"),
    description: str = typer.Argument(..., help="Description of the profile"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport type (stdio, sse)",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to (for sse transport)",
    ),
    port: int = typer.Option(
        3000,
        "--port",
        "-p",
        help="Port to bind to (for sse transport)",
    ),
    tool_dir: List[str] = typer.Option(
        [],
        "--tool-dir",
        help="Directory to load tools from (can be specified multiple times)",
    ),
    prompt_dir: List[str] = typer.Option(
        [],
        "--prompt-dir",
        help="Directory to load prompts from (can be specified multiple times)",
    ),
    resource_dir: List[str] = typer.Option(
        [],
        "--resource-dir",
        help="Directory to load resources from (can be specified multiple times)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug output",
    ),
    set_default: bool = typer.Option(
        False,
        "--set-default",
        help="Set this profile as the default",
    ),
):
    """
    Add a new configuration profile.
    """
    try:
        # Load configuration
        cfg_path = config
        if not cfg_path:
            # Try to find an existing configuration file
            for path in [
                Path.cwd() / "mcp.yaml",
                Path.cwd() / "mcp.yml",
                Path.cwd() / ".mcp.yaml",
                Path.cwd() / ".mcp.yml",
                Path.home() / ".mcp.yaml",
                Path.home() / ".mcp.yml",
                Path.home() / ".config" / "mcp" / "config.yaml",
            ]:
                if path.exists():
                    cfg_path = path
                    break
            
            if not cfg_path:
                # Create a new configuration file
                cfg_path = Path.cwd() / "mcp.yaml"
        
        cfg = load_config(cfg_path)
        
        # Get profiles
        profiles = cfg.get("profiles", {})
        
        # Check if the profile already exists
        if name in profiles:
            console.print(f"[bold red]Profile already exists: {name}[/]")
            raise typer.Exit(1)
        
        # Create the profile
        profile = {
            "name": f"pymcp-{name}-server",
            "version": __version__,
            "description": description,
            "transport": transport,
            "host": host,
            "port": port,
            "debug": debug,
            "tool_dirs": list(tool_dir),
            "prompt_dirs": list(prompt_dir),
            "resource_dirs": list(resource_dir),
        }
        
        # Add the profile to the configuration
        if "profiles" not in cfg:
            cfg["profiles"] = {}
        
        cfg["profiles"][name] = profile
        
        # Set as default if requested
        if set_default:
            cfg["default_profile"] = name
        
        # Write the configuration file
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        console.print(f"[bold green]Profile added: {name}[/]")
        console.print(f"Configuration file updated: {cfg_path}")
    
    except Exception as e:
        console.print(f"[bold red]Error adding profile: {str(e)}[/]")
        raise typer.Exit(1)


@config_app.command("set-default-profile")
def config_set_default_profile(
    name: str = typer.Argument(..., help="Name of the profile to set as default"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """
    Set the default configuration profile.
    """
    try:
        # Load configuration
        cfg_path = config
        if not cfg_path:
            # Try to find an existing configuration file
            for path in [
                Path.cwd() / "mcp.yaml",
                Path.cwd() / "mcp.yml",
                Path.cwd() / ".mcp.yaml",
                Path.cwd() / ".mcp.yml",
                Path.home() / ".mcp.yaml",
                Path.home() / ".mcp.yml",
                Path.home() / ".config" / "mcp" / "config.yaml",
            ]:
                if path.exists():
                    cfg_path = path
                    break
            
            if not cfg_path:
                console.print("[bold red]No configuration file found[/]")
                raise typer.Exit(1)
        
        cfg = load_config(cfg_path)
        
        # Get profiles
        profiles = cfg.get("profiles", {})
        
        # Check if the profile exists
        if name not in profiles:
            console.print(f"[bold red]Profile not found: {name}[/]")
            raise typer.Exit(1)
        
        # Set the default profile
        cfg["default_profile"] = name
        
        # Write the configuration file
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        console.print(f"[bold green]Default profile set to: {name}[/]")
        console.print(f"Configuration file updated: {cfg_path}")
    
    except Exception as e:
        console.print(f"[bold red]Error setting default profile: {str(e)}[/]")
        raise typer.Exit(1)


# Run the app
if __name__ == "__main__":
    app()
