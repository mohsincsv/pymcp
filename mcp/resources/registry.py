"""
Resource registry implementation for MCP.

This module provides a comprehensive resource registry that implements the ResourceProvider
protocol and provides resource discovery, retrieval, and management capabilities.
It serves as the central component for registering, discovering, and retrieving
resources in the MCP server.
"""

import os
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, BinaryIO

import yaml
from pydantic import BaseModel, Field, ValidationError


class ResourceMetadata(BaseModel):
    """Metadata for a resource."""
    
    name: str = Field(..., description="Name of the resource")
    description: str = Field("", description="Description of the resource")
    version: Optional[str] = Field(None, description="Version of the resource")
    tags: List[str] = Field(default_factory=list, description="Tags for the resource")
    mime_type: Optional[str] = Field(None, description="MIME type of the resource")
    size: Optional[int] = Field(None, description="Size of the resource in bytes")
    encoding: Optional[str] = Field(None, description="Encoding of the resource content")


class Resource(BaseModel):
    """
    Representation of a resource.
    
    A resource consists of content (the data) and metadata describing
    the resource and its properties.
    """
    
    content: str = Field(..., description="Content of the resource (may be base64 encoded)")
    metadata: ResourceMetadata = Field(..., description="Metadata for the resource")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path], name: Optional[str] = None) -> "Resource":
        """
        Create a resource from a file.
        
        Args:
            file_path: Path to the file
            name: Optional name for the resource (defaults to filename)
            
        Returns:
            Resource object
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        # Convert to Path object
        path = Path(file_path)
        
        # Check if the file exists
        if not path.exists():
            raise FileNotFoundError(f"File '{file_path}' does not exist")
        
        # Determine file size
        size = path.stat().st_size
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        
        # Use the filename as the resource name if not provided
        if name is None:
            name = path.stem
        
        # Determine if the file is text or binary
        is_text = False
        if mime_type:
            is_text = mime_type.startswith("text/")
        
        # Read the file content
        if is_text:
            with open(path, "r") as f:
                content = f.read()
                encoding = None
        else:
            with open(path, "rb") as f:
                content = base64.b64encode(f.read()).decode("ascii")
                encoding = "base64"
        
        # Create the resource metadata
        metadata = ResourceMetadata(
            name=name,
            description=f"Resource loaded from {path.name}",
            mime_type=mime_type,
            size=size,
            encoding=encoding,
        )
        
        # Create the resource
        return cls(content=content, metadata=metadata)


class ResourceRegistry:
    """
    Comprehensive resource registry for MCP.
    
    This class implements the ResourceProvider protocol and provides a central
    registry for resources in the MCP server. It supports resource discovery,
    retrieval, and management.
    """
    
    def __init__(self) -> None:
        """Initialize the resource registry."""
        self._resources: Dict[str, Resource] = {}
    
    async def list_resources(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all available resources.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of resource metadata as dictionaries
        """
        if not tags:
            # Return all resources
            return [resource.metadata.model_dump() for resource in self._resources.values()]
        
        # Filter by tags
        filtered_resources = []
        for resource in self._resources.values():
            # Check if the resource has all the requested tags
            if all(tag in resource.metadata.tags for tag in tags):
                filtered_resources.append(resource.metadata.model_dump())
        
        return filtered_resources
    
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
        if name not in self._resources:
            raise ValueError(f"Resource '{name}' not found")
        
        resource = self._resources[name]
        
        # Return the resource content and metadata
        return {
            "content": resource.content,
            "metadata": resource.metadata.model_dump(),
        }
    
    def register_resource(self, resource: Resource) -> None:
        """
        Register a resource.
        
        Args:
            resource: Resource to register
            
        Raises:
            ValueError: If a resource with the same name is already registered
        """
        if resource.metadata.name in self._resources:
            raise ValueError(f"Resource with name '{resource.metadata.name}' is already registered")
        
        self._resources[resource.metadata.name] = resource
    
    def register_resource_from_content(
        self,
        name: str,
        content: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        mime_type: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> Resource:
        """
        Register a resource from content.
        
        Args:
            name: Name of the resource
            content: Content of the resource
            description: Description of the resource
            tags: Tags for the resource
            mime_type: MIME type of the resource
            encoding: Encoding of the resource content
            
        Returns:
            The registered resource
            
        Raises:
            ValueError: If a resource with the same name is already registered
        """
        # Determine size
        size = len(content.encode("utf-8") if isinstance(content, str) else content)
        
        # Create the resource metadata
        metadata = ResourceMetadata(
            name=name,
            description=description,
            tags=tags or [],
            mime_type=mime_type,
            size=size,
            encoding=encoding,
        )
        
        # Create the resource
        resource = Resource(content=content, metadata=metadata)
        
        # Register the resource
        self.register_resource(resource)
        
        return resource
    
    def register_resource_from_file(
        self,
        file_path: Union[str, Path],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Resource:
        """
        Register a resource from a file.
        
        Args:
            file_path: Path to the file
            name: Optional name for the resource (defaults to filename)
            description: Optional description for the resource
            tags: Optional tags for the resource
            
        Returns:
            The registered resource
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If a resource with the same name is already registered
        """
        # Create the resource from the file
        resource = Resource.from_file(file_path, name)
        
        # Update description and tags if provided
        if description:
            resource.metadata.description = description
        if tags:
            resource.metadata.tags = tags
        
        # Register the resource
        self.register_resource(resource)
        
        return resource
    
    def unregister_resource(self, name: str) -> bool:
        """
        Unregister a resource.
        
        Args:
            name: Name of the resource to unregister
            
        Returns:
            True if the resource was unregistered, False if it wasn't found
        """
        if name in self._resources:
            del self._resources[name]
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all registered resources."""
        self._resources.clear()
    
    def get_resource_sync(self, name: str) -> Optional[Resource]:
        """
        Get a resource by name (synchronous version).
        
        Args:
            name: Name of the resource to get
            
        Returns:
            The resource, or None if not found
        """
        return self._resources.get(name)
    
    def has_resource(self, name: str) -> bool:
        """
        Check if a resource is registered.
        
        Args:
            name: Name of the resource to check
            
        Returns:
            True if the resource is registered, False otherwise
        """
        return name in self._resources
    
    def load_resources_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = False,
    ) -> List[Resource]:
        """
        Load resources from files in a directory.
        
        Args:
            directory: Path to the directory
            pattern: Glob pattern for matching files
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of loaded resources
            
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
        
        # Load resources from files
        resources = []
        
        # Determine the glob method based on recursion
        glob_method = path.rglob if recursive else path.glob
        
        for file_path in glob_method(pattern):
            # Skip directories
            if file_path.is_dir():
                continue
            
            try:
                resource = self.register_resource_from_file(file_path)
                resources.append(resource)
            except (ValueError, FileNotFoundError):
                # Skip invalid files
                pass
        
        return resources
    
    def __len__(self) -> int:
        """Get the number of registered resources."""
        return len(self._resources)
    
    def __contains__(self, name: str) -> bool:
        """Check if a resource is registered."""
        return name in self._resources


# Global resource registry instance
global_registry = ResourceRegistry()


# Convenience functions for the global registry
def register_resource(resource: Resource) -> None:
    """Register a resource in the global registry."""
    global_registry.register_resource(resource)


def register_resource_from_content(
    name: str,
    content: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    mime_type: Optional[str] = None,
    encoding: Optional[str] = None,
) -> Resource:
    """Register a resource from content in the global registry."""
    return global_registry.register_resource_from_content(
        name=name,
        content=content,
        description=description,
        tags=tags,
        mime_type=mime_type,
        encoding=encoding,
    )


def register_resource_from_file(
    file_path: Union[str, Path],
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Resource:
    """Register a resource from a file in the global registry."""
    return global_registry.register_resource_from_file(
        file_path=file_path,
        name=name,
        description=description,
        tags=tags,
    )


def get_registry() -> ResourceRegistry:
    """Get the global resource registry."""
    return global_registry
