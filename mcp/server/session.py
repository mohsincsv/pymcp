"""
Session management for MCP server.

This module provides session management functionality for the MCP server,
including a session store interface and an in-memory implementation.
Sessions are used to maintain state across requests, especially for
transports like HTTP/SSE where clients may reconnect.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, Set

from mcp.transport.base import SessionStore


class InMemorySessionStore(SessionStore):
    """
    In-memory implementation of SessionStore.
    
    This implementation stores sessions in memory using a dictionary.
    It's suitable for development and testing, but not for production
    use with multiple server instances.
    """
    
    def __init__(self, expiration_seconds: Optional[int] = None) -> None:
        """
        Initialize the in-memory session store.
        
        Args:
            expiration_seconds: Optional time in seconds after which sessions expire.
                                If None, sessions never expire.
        """
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._expiration_seconds = expiration_seconds
        self._last_accessed: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        # Start expiration task if needed
        if expiration_seconds is not None and expiration_seconds > 0:
            self._start_expiration_task()
    
    async def create_session(self) -> str:
        """
        Create a new session with a unique ID.
        
        Returns:
            Session ID as a string
        """
        session_id = str(uuid.uuid4())
        
        async with self._lock:
            self._sessions[session_id] = {}
            self._last_accessed[session_id] = time.time()
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by ID.
        
        Args:
            session_id: ID of the session to get
            
        Returns:
            Session data or None if not found
        """
        async with self._lock:
            if session_id not in self._sessions:
                return None
            
            # Update last accessed time
            self._last_accessed[session_id] = time.time()
            
            # Return a copy of the session data to prevent concurrent modification
            return self._sessions[session_id].copy()
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: ID of the session to update
            data: New session data (will be merged with existing data)
            
        Returns:
            True if successful, False if session not found
        """
        async with self._lock:
            if session_id not in self._sessions:
                return False
            
            # Update last accessed time
            self._last_accessed[session_id] = time.time()
            
            # Merge the new data with existing data
            self._sessions[session_id].update(data)
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if successful, False if session not found
        """
        async with self._lock:
            if session_id not in self._sessions:
                return False
            
            del self._sessions[session_id]
            if session_id in self._last_accessed:
                del self._last_accessed[session_id]
            
            return True
    
    async def get_all_session_ids(self) -> Set[str]:
        """
        Get all active session IDs.
        
        Returns:
            Set of session IDs
        """
        async with self._lock:
            return set(self._sessions.keys())
    
    async def clear_all_sessions(self) -> None:
        """
        Clear all sessions from the store.
        """
        async with self._lock:
            self._sessions.clear()
            self._last_accessed.clear()
    
    def _start_expiration_task(self) -> None:
        """
        Start a background task to expire old sessions.
        """
        async def expire_sessions() -> None:
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    await self._expire_old_sessions()
                except asyncio.CancelledError:
                    break
                except Exception:
                    # Log the error but don't crash the task
                    pass
        
        # Start the task
        asyncio.create_task(expire_sessions())
    
    async def _expire_old_sessions(self) -> None:
        """
        Expire sessions that haven't been accessed recently.
        """
        if self._expiration_seconds is None:
            return
        
        now = time.time()
        expired_sessions = []
        
        async with self._lock:
            for session_id, last_accessed in self._last_accessed.items():
                if now - last_accessed > self._expiration_seconds:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
                del self._last_accessed[session_id]
