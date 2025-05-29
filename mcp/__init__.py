"""
PyMCP - Python Model Context Protocol Implementation

A Python implementation of the Model Context Protocol (MCP) specification,
providing standardized communication between AI applications and language models.
"""

__version__ = "0.1.0"
__author__ = "Mohsin Iqbal"
__license__ = "MIT"

# Package-level exports for convenient imports
from typing import List, Dict, Any, Optional, Union, Callable, AsyncIterator

# Version information
def get_version() -> str:
    """Return the current version of the package."""
    return __version__

# These will be populated as we implement the components
__all__: List[str] = [
    "get_version",
    # Will add Server, Client, Tool, etc. as they're implemented
]
