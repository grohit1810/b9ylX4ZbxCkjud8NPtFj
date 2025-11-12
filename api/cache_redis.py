"""
Redis caching layer for Movie Recommender Chatbot.

Implements cache-aside pattern for user existence and (user_id, chat_id) pair validation.
Uses semantic key naming and configurable TTLs.
"""

import redis.asyncio as redis
from enum import Enum
from typing import Optional
from logging_config.logger import get_logger

logger = get_logger(__name__)


class CacheTTL(int, Enum):
    """Cache TTL values in seconds for different entity types."""
    USER = 3600  # 1 hour - users rarely deleted
    CHAT_PAIR = 900  # 15 minutes - chat ownership validation


class CacheKeyPrefix(str, Enum):
    """Redis key prefixes for namespacing and clarity."""
    USER = "user"
    CHAT_PAIR = "chat_pair"  # Format: chat_pair:{user_id}:{chat_id} -> status


# Module-level connection pool
_redis_pool: Optional[redis.ConnectionPool] = None


async def init_redis() -> redis.Redis:
    """
    Initialize Redis connection pool.
    
    Creates a connection pool that can be shared across multiple
    Redis client instances for efficient connection reuse.
    
    Returns:
        redis.Redis: Redis client instance using the connection pool
        
    Raises:
        redis.ConnectionError: If Redis connection fails
    """
    global _redis_pool
    
    try:
        _redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            max_connections=10,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        
        logger.info("Redis connection pool initialized successfully")
        return redis.Redis(connection_pool=_redis_pool)
        
    except redis.ConnectionError as e:
        logger.error(f"Failed to initialize Redis pool: {e}")
        raise


async def close_redis(redis_client: redis.Redis) -> None:
    """
    Close Redis client and connection pool.
    
    Args:
        redis_client: The Redis client to close
    """
    try:
        if redis_client:
            await redis_client.aclose()
        
        if _redis_pool:
            await _redis_pool.aclose()
        
        logger.info("Redis connections closed")
        
    except redis.RedisError as e:
        logger.error(f"Error closing Redis connections: {e}")


def _make_user_key(user_id: str) -> str:
    """Generate Redis key for user existence check."""
    return f"{CacheKeyPrefix.USER.value}:{user_id}"


def _make_chat_pair_key(user_id: str, chat_id: str) -> str:
    """Generate Redis key for (user_id, chat_id) pair with status."""
    return f"{CacheKeyPrefix.CHAT_PAIR.value}:{user_id}:{chat_id}"


# ---------- User Cache ----------

async def cache_set_user(redis_client: redis.Redis, user_id: str) -> None:
    """
    Cache that a user exists in the system.
    
    Args:
        redis_client: Redis client instance
        user_id: The user identifier to cache
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        key = _make_user_key(user_id)
        await redis_client.setex(
            key,
            CacheTTL.USER.value,
            "1"  # Simple existence flag
        )
        logger.debug(f"Cached user: {user_id}")
        
    except redis.RedisError as e:
        logger.error(f"Failed to cache user {user_id}: {e}")
        # Don't raise - cache failures shouldn't break the application


async def cache_get_user(redis_client: redis.Redis, user_id: str) -> bool:
    """
    Check if user exists in cache.
    
    Args:
        redis_client: Redis client instance
        user_id: The user identifier to check
        
    Returns:
        bool: True if user exists in cache, False otherwise
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        key = _make_user_key(user_id)
        result = await redis_client.get(key)
        
        exists = result is not None
        logger.debug(f"User cache {'hit' if exists else 'miss'}: {user_id}")
        return exists
        
    except redis.RedisError as e:
        logger.error(f"Failed to get user cache for {user_id}: {e}")
        # On cache failure, assume miss to fall back to DB
        return False


# ---------- Chat Pair Cache (user_id, chat_id) -> status ----------

async def cache_set_chat_pair(
    redis_client: redis.Redis,
    user_id: str,
    chat_id: str,
    status: str
) -> None:
    """
    Cache a (user_id, chat_id) pair with its status.
    
    This validates chat ownership and stores the chat status.
    
    Args:
        redis_client: Redis client instance
        user_id: The user identifier
        chat_id: The chat identifier
        status: Chat status value (e.g., 'active', 'archived')
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        key = _make_chat_pair_key(user_id, chat_id)
        await redis_client.setex(
            key,
            CacheTTL.CHAT_PAIR.value,
            status
        )
        logger.debug(f"Cached chat pair ({user_id}, {chat_id}): {status}")
        
    except redis.RedisError as e:
        logger.error(f"Failed to cache chat pair ({user_id}, {chat_id}): {e}")


async def cache_get_chat_pair(
    redis_client: redis.Redis,
    user_id: str,
    chat_id: str
) -> Optional[str]:
    """
    Get cached status for (user_id, chat_id) pair.
    
    Args:
        redis_client: Redis client instance
        user_id: The user identifier
        chat_id: The chat identifier
        
    Returns:
        str: Chat status if cached, None if cache miss
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        key = _make_chat_pair_key(user_id, chat_id)
        status = await redis_client.get(key)
        
        logger.debug(f"Chat pair cache {'hit' if status else 'miss'}: ({user_id}, {chat_id})")
        return status
        
    except redis.RedisError as e:
        logger.error(f"Failed to get chat pair cache for ({user_id}, {chat_id}): {e}")
        return None


async def cache_invalidate_chat_pair(
    redis_client: redis.Redis,
    user_id: str,
    chat_id: str
) -> None:
    """
    Invalidate cached (user_id, chat_id) pair (e.g., when status changes).
    
    Args:
        redis_client: Redis client instance
        user_id: The user identifier
        chat_id: The chat identifier
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        key = _make_chat_pair_key(user_id, chat_id)
        await redis_client.delete(key)
        logger.debug(f"Invalidated chat pair cache: ({user_id}, {chat_id})")
        
    except redis.RedisError as e:
        logger.error(f"Failed to invalidate chat pair cache for ({user_id}, {chat_id}): {e}")
