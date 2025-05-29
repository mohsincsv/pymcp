"""
Prompt registry implementation for MCP.

This module provides a comprehensive prompt registry that implements the PromptProvider
protocol and provides prompt discovery, rendering, and management capabilities.
It serves as the central component for registering, discovering, and retrieving
prompts in the MCP server.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import jinja2
import yaml
from pydantic import BaseModel, Field, ValidationError


class PromptMetadata(BaseModel):
    """Metadata for a prompt."""
    
    name: str = Field(..., description="Name of the prompt")
    description: str = Field("", description="Description of the prompt")
    version: Optional[str] = Field(None, description="Version of the prompt")
    tags: List[str] = Field(default_factory=list, description="Tags for the prompt")
    parameters: List[Dict[str, Any]] = Field(
        default_factory=list, description="Parameters for the prompt"
    )


class Prompt(BaseModel):
    """
    Representation of a prompt.
    
    A prompt consists of content (the template) and metadata describing
    the prompt and its parameters.
    """
    
    content: str = Field(..., description="Content of the prompt")
    metadata: PromptMetadata = Field(..., description="Metadata for the prompt")
    
    def render(self, args: Optional[Dict[str, Any]] = None) -> str:
        """
        Render the prompt with the given arguments.
        
        Args:
            args: Arguments to fill in the prompt template
            
        Returns:
            Rendered prompt
            
        Raises:
            jinja2.exceptions.TemplateError: If template rendering fails
        """
        if not args:
            return self.content
        
        # Set up Jinja2 environment
        env = jinja2.Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Render the template
        template = env.from_string(self.content)
        return template.render(**args)


class PromptRegistry:
    """
    Comprehensive prompt registry for MCP.
    
    This class implements the PromptProvider protocol and provides a central
    registry for prompts in the MCP server. It supports prompt discovery,
    rendering, and management.
    """
    
    def __init__(self) -> None:
        """Initialize the prompt registry."""
        self._prompts: Dict[str, Prompt] = {}
    
    async def list_prompts(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all available prompts.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of prompt metadata as dictionaries
        """
        if not tags:
            # Return all prompts
            return [prompt.metadata.model_dump() for prompt in self._prompts.values()]
        
        # Filter by tags
        filtered_prompts = []
        for prompt in self._prompts.values():
            # Check if the prompt has all the requested tags
            if all(tag in prompt.metadata.tags for tag in tags):
                filtered_prompts.append(prompt.metadata.model_dump())
        
        return filtered_prompts
    
    async def get_prompt(
        self, 
        name: str, 
        args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a prompt with the given name.
        
        Args:
            name: Name of the prompt to get
            args: Optional arguments to fill in the prompt template
            
        Returns:
            Prompt content and metadata
            
        Raises:
            ValueError: If the prompt is not found
            jinja2.exceptions.TemplateError: If template rendering fails
        """
        if name not in self._prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        prompt = self._prompts[name]
        
        # Render the prompt if args are provided
        content = prompt.render(args)
        
        # Return the prompt content and metadata
        return {
            "content": content,
            "metadata": prompt.metadata.model_dump(),
        }
    
    def register_prompt(self, prompt: Prompt) -> None:
        """
        Register a prompt.
        
        Args:
            prompt: Prompt to register
            
        Raises:
            ValueError: If a prompt with the same name is already registered
        """
        if prompt.metadata.name in self._prompts:
            raise ValueError(f"Prompt with name '{prompt.metadata.name}' is already registered")
        
        self._prompts[prompt.metadata.name] = prompt
    
    def register_prompt_from_content(
        self,
        name: str,
        content: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
    ) -> Prompt:
        """
        Register a prompt from content.
        
        Args:
            name: Name of the prompt
            content: Content of the prompt
            description: Description of the prompt
            tags: Tags for the prompt
            version: Version of the prompt
            parameters: Parameters for the prompt
            
        Returns:
            The registered prompt
            
        Raises:
            ValueError: If a prompt with the same name is already registered
        """
        # Create the prompt metadata
        metadata = PromptMetadata(
            name=name,
            description=description,
            tags=tags or [],
            version=version,
            parameters=parameters or [],
        )
        
        # Create the prompt
        prompt = Prompt(content=content, metadata=metadata)
        
        # Register the prompt
        self.register_prompt(prompt)
        
        return prompt
    
    def unregister_prompt(self, name: str) -> bool:
        """
        Unregister a prompt.
        
        Args:
            name: Name of the prompt to unregister
            
        Returns:
            True if the prompt was unregistered, False if it wasn't found
        """
        if name in self._prompts:
            del self._prompts[name]
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all registered prompts."""
        self._prompts.clear()
    
    def get_prompt_sync(self, name: str) -> Optional[Prompt]:
        """
        Get a prompt by name (synchronous version).
        
        Args:
            name: Name of the prompt to get
            
        Returns:
            The prompt, or None if not found
        """
        return self._prompts.get(name)
    
    def has_prompt(self, name: str) -> bool:
        """
        Check if a prompt is registered.
        
        Args:
            name: Name of the prompt to check
            
        Returns:
            True if the prompt is registered, False otherwise
        """
        return name in self._prompts
    
    def load_prompt_from_file(self, file_path: Union[str, Path]) -> Prompt:
        """
        Load a prompt from a file.
        
        The file can be a YAML file with metadata and content, or a plain text file
        with just the content. For plain text files, the name is derived from the
        filename.
        
        Args:
            file_path: Path to the file
            
        Returns:
            The loaded prompt
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is invalid
        """
        # Convert to Path object
        path = Path(file_path)
        
        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"File '{file_path}' does not exist")
        
        # Determine file type
        if path.suffix.lower() in (".yaml", ".yml"):
            # Load YAML file
            with open(path, "r") as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML file '{file_path}': {str(e)}")
            
            # Check if it has the expected structure
            if not isinstance(data, dict):
                raise ValueError(f"Invalid prompt file '{file_path}': expected a dictionary")
            
            if "content" not in data:
                raise ValueError(f"Invalid prompt file '{file_path}': missing 'content' field")
            
            if "metadata" not in data:
                raise ValueError(f"Invalid prompt file '{file_path}': missing 'metadata' field")
            
            # Create the prompt
            try:
                metadata = PromptMetadata(**data["metadata"])
                prompt = Prompt(content=data["content"], metadata=metadata)
            except ValidationError as e:
                raise ValueError(f"Invalid prompt metadata in '{file_path}': {str(e)}")
        
        else:
            # Assume it's a plain text file
            with open(path, "r") as f:
                content = f.read()
            
            # Use the filename as the prompt name
            name = path.stem
            
            # Create the prompt
            metadata = PromptMetadata(name=name, description=f"Prompt loaded from {path.name}")
            prompt = Prompt(content=content, metadata=metadata)
        
        # Register the prompt
        self.register_prompt(prompt)
        
        return prompt
    
    def load_prompts_from_directory(self, directory: Union[str, Path]) -> List[Prompt]:
        """
        Load prompts from files in a directory.
        
        Args:
            directory: Path to the directory
            
        Returns:
            List of loaded prompts
            
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
        
        # Load prompts from files
        prompts = []
        for file_path in path.glob("*.*"):
            # Skip non-text files
            if file_path.suffix.lower() not in (".txt", ".md", ".yaml", ".yml", ".json"):
                continue
            
            try:
                prompt = self.load_prompt_from_file(file_path)
                prompts.append(prompt)
            except (ValueError, FileNotFoundError):
                # Skip invalid files
                pass
        
        return prompts
    
    def __len__(self) -> int:
        """Get the number of registered prompts."""
        return len(self._prompts)
    
    def __contains__(self, name: str) -> bool:
        """Check if a prompt is registered."""
        return name in self._prompts


# Global prompt registry instance
global_registry = PromptRegistry()


# Convenience functions for the global registry
def register_prompt(prompt: Prompt) -> None:
    """Register a prompt in the global registry."""
    global_registry.register_prompt(prompt)


def register_prompt_from_content(
    name: str,
    content: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    version: Optional[str] = None,
    parameters: Optional[List[Dict[str, Any]]] = None,
) -> Prompt:
    """Register a prompt from content in the global registry."""
    return global_registry.register_prompt_from_content(
        name=name,
        content=content,
        description=description,
        tags=tags,
        version=version,
        parameters=parameters,
    )


def get_registry() -> PromptRegistry:
    """Get the global prompt registry."""
    return global_registry
