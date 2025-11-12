"""
SQLite database layer for Movie Recommender Chatbot.

Handles user management, chat session tracking, and database lifecycle.
Uses connection pooling for efficient async operations.
"""

import aiosqlite
from uuid import uuid4
from enum import Enum
from typing import Optional
from contextlib import asynccontextmanager
from config.settings import API_DB
from logging_config.logger import get_logger

logger = get_logger(__name__)

# Database configuration
DB_PATH = API_DB


class ChatStatus(str, Enum):
    """Enum for chat status values to avoid magic strings."""
    ACTIVE = "active"
    ARCHIVED = "archived"


# Module-level connection pool
_db_connection: Optional[aiosqlite.Connection] = None


async def init_db() -> None:
    """
    Initialize database schema and connection pool.
    
    Creates users and chats tables with proper foreign key constraints
    and indices for optimized queries. Enables foreign key enforcement.
    
    Raises:
        aiosqlite.Error: If database initialization fails
    """
    global _db_connection
    
    try:
        _db_connection = await aiosqlite.connect(DB_PATH)
        _db_connection.row_factory = aiosqlite.Row
        
        # Enable foreign key constraints (critical for SQLite)
        await _db_connection.execute("PRAGMA foreign_keys = ON")
        
        # Create users table
        await _db_connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chats table with foreign key constraint
        await _db_connection.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT CHECK(status IN ('active', 'archived')) DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        
        # Create composite index for (user_id, chat_id) lookups
        await _db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chats_user_chat 
            ON chats(user_id, chat_id)
        """)
        
        # Create index for user_id lookups
        await _db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)
        """)
        
        # Create index for status filtering
        await _db_connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_chats_status ON chats(status)
        """)
        
        await _db_connection.commit()
        logger.info("Database initialized successfully with schema and indices")
        
    except aiosqlite.Error as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db() -> None:
    """
    Close database connection pool.
    
    Should be called during application shutdown to ensure
    proper resource cleanup.
    """
    global _db_connection
    if _db_connection:
        try:
            await _db_connection.close()
            logger.info("Database connection closed")
        except aiosqlite.Error as e:
            logger.error(f"Error closing database connection: {e}")


@asynccontextmanager
async def get_db():
    """
    Context manager for database access with automatic connection handling.
    
    Yields:
        aiosqlite.Connection: Active database connection
    """
    if _db_connection is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    yield _db_connection


async def create_user() -> str:
    """
    Create a new user with a generated UUID.
    
    Returns:
        str: The newly created user_id
        
    Raises:
        aiosqlite.IntegrityError: If user_id collision occurs (extremely rare)
        aiosqlite.Error: If database operation fails
    """
    user_id = str(uuid4())
    
    try:
        async with get_db() as db:
            await db.execute(
                "INSERT INTO users (user_id) VALUES (?)",
                (user_id,)
            )
            await db.commit()
        
        logger.info(f"Created new user: {user_id}")
        return user_id
        
    except aiosqlite.IntegrityError as e:
        logger.error(f"User creation failed - integrity error: {e}")
        raise
    except aiosqlite.Error as e:
        logger.error(f"User creation failed: {e}")
        raise


async def get_user(user_id: str) -> Optional[dict]:
    """
    Retrieve user by user_id.
    
    Args:
        user_id: The user identifier to look up
        
    Returns:
        dict with user_id and created_at if found, None otherwise
        
    Raises:
        aiosqlite.Error: If database query fails
    """
    try:
        async with get_db() as db:
            async with db.execute(
                "SELECT user_id, created_at FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
        if row:
            user_data = dict(row)
            logger.debug(f"Retrieved user: {user_id}")
            return user_data
        
        logger.debug(f"User not found: {user_id}")
        return None
        
    except aiosqlite.Error as e:
        logger.error(f"Failed to retrieve user {user_id}: {e}")
        raise


async def create_chat(user_id: str) -> str:
    """
    Create a new chat session for a user.
    
    Args:
        user_id: The user who owns this chat session
        
    Returns:
        str: The newly created chat_id
        
    Raises:
        aiosqlite.IntegrityError: If foreign key constraint fails or chat_id collision
        aiosqlite.Error: If database operation fails
    """
    chat_id = str(uuid4())
    
    try:
        async with get_db() as db:
            await db.execute(
                "INSERT INTO chats (chat_id, user_id, status) VALUES (?, ?, ?)",
                (chat_id, user_id, ChatStatus.ACTIVE.value)
            )
            await db.commit()
        
        logger.info(f"Created new chat {chat_id} for user {user_id}")
        return chat_id
        
    except aiosqlite.IntegrityError as e:
        logger.error(f"Chat creation failed for user {user_id} - integrity error: {e}")
        raise
    except aiosqlite.Error as e:
        logger.error(f"Chat creation failed for user {user_id}: {e}")
        raise


async def get_chat(chat_id: str, user_id: str) -> Optional[dict]:
    """
    Retrieve chat by chat_id and user_id pair.
    
    Validates that the chat exists and belongs to the specified user.
    
    Args:
        chat_id: The chat identifier
        user_id: The user who owns the chat (for authorization)
        
    Returns:
        dict with chat details if found and authorized, None otherwise
        
    Raises:
        aiosqlite.Error: If database query fails
    """
    try:
        async with get_db() as db:
            async with db.execute(
                """SELECT chat_id, user_id, status, created_at, updated_at 
                   FROM chats 
                   WHERE chat_id = ? AND user_id = ?""",
                (chat_id, user_id)
            ) as cursor:
                row = await cursor.fetchone()
        
        if row:
            chat_data = dict(row)
            logger.debug(f"Retrieved chat {chat_id} for user {user_id}")
            return chat_data
        
        logger.debug(f"Chat not found or unauthorized: {chat_id} for user {user_id}")
        return None
        
    except aiosqlite.Error as e:
        logger.error(f"Failed to retrieve chat {chat_id}: {e}")
        raise


async def update_chat_status(chat_id: str, user_id: str, status: ChatStatus) -> bool:
    """
    Update chat status (archive or unarchive).
    
    Args:
        chat_id: The chat to update
        user_id: The user who owns the chat (for authorization)
        status: New status to set
        
    Returns:
        bool: True if chat status was updated, False if not found or unauthorized
        
    Raises:
        aiosqlite.Error: If database operation fails
    """
    try:
        async with get_db() as db:
            cursor = await db.execute(
                """UPDATE chats 
                   SET status = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE chat_id = ? AND user_id = ?""",
                (status.value, chat_id, user_id)
            )
            await db.commit()
            
            rows_affected = cursor.rowcount
            
        if rows_affected > 0:
            logger.info(f"Updated chat {chat_id} status to {status.value} for user {user_id}")
            return True
        
        logger.warning(f"Failed to update chat {chat_id} - not found or unauthorized")
        return False
        
    except aiosqlite.Error as e:
        logger.error(f"Failed to update chat {chat_id} status: {e}")
        raise
