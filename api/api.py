"""
FastAPI REST API for Movie Recommender Chatbot.

Provides endpoints for conversational movie recommendations using
LLM-powered agent with structured data retrieval from SQLite.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import redis.asyncio as redis

from api.db_sqlite import (
    init_db,
    close_db,
    create_user,
    get_user,
    create_chat,
    get_chat,
    update_chat_status,
    ChatStatus,
)
from api.cache_redis import (
    init_redis,
    close_redis,
    cache_set_user,
    cache_get_user,
    cache_get_chat_pair,
    cache_set_chat_pair,
)
from api.metrics import setup_metrics, update_service_health
from movie_assistant.agent import MovieAgent
from config.settings import LLM_CONFIG, MCP_SERVER_CONFIG
from logging_config.logger import get_logger

logger = get_logger(__name__)

# Global state managed via lifespan
redis_client: Optional[redis.Redis] = None
movie_agent: Optional[MovieAgent] = None


# Pydantic models for request/response validation

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    user_id: Optional[str] = Field(
        None,
        description="User identifier. If not provided, a new user will be created.",
    )
    chat_id: Optional[str] = Field(
        None,
        description="Chat session identifier. If not provided, a new chat will be created.",
    )
    message: str = Field(
        ...,
        description="User's movie-related query"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Chat session identifier")
    message: str = Field(..., description="User's original message")
    reply: str = Field(..., description="Agent's response")
    is_new_user: bool = Field(..., description="Whether a new user was created")
    is_new_chat: bool = Field(..., description="Whether a new chat was created")


class ChatStatusRequest(BaseModel):
    """Request model for archive/unarchive endpoints."""
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Chat session identifier")


class ChatStatusResponse(BaseModel):
    """Response model for archive/unarchive endpoints."""
    message: str = Field(..., description="Status message")
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Chat identifier")
    status: str = Field(..., description="New chat status")


class HistoryRequest(BaseModel):
    """Request model for chat history endpoint."""
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Chat session identifier")


class HistoryResponse(BaseModel):
    """Response model for chat history endpoint."""
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Chat identifier")
    conversation_history: List[Dict] = Field(..., description="Chat conversation history")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service health status")
    database: str = Field(..., description="Database connection status")
    cache: str = Field(..., description="Redis cache status")
    agent: str = Field(..., description="Movie agent status")


# Application lifespan management

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.
    
    Initializes database, Redis cache, and movie agent on startup.
    Ensures proper cleanup of all resources on shutdown.
    """
    global redis_client, movie_agent
    
    # Startup
    try:
        logger.info("Starting Movie Recommender Chatbot API")
        
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize Redis
        redis_client = await init_redis()
        logger.info("Redis cache initialized")
        
        # Initialize movie agent
        mcp_url = (
            f"http://{MCP_SERVER_CONFIG['mcp_server_host']}:"
            f"{MCP_SERVER_CONFIG['mcp_server_port']}"
            f"{MCP_SERVER_CONFIG['mcp_http_path']}"
        )
        movie_agent = MovieAgent(
            llm_memory_db=LLM_CONFIG["conversation_checkpoint_db"],
            mcp_url=mcp_url,
            llm_host=LLM_CONFIG['host'],
            llm_model=LLM_CONFIG['model'],
        )
        logger.info("Movie agent initialized")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        logger.info("Shutting down Movie Recommender Chatbot API")
        
        if redis_client:
            await close_redis(redis_client)
            logger.info("Redis connections closed")
        
        await close_db()
        logger.info("Database connection closed")
        
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# FastAPI app initialization

app = FastAPI(
    title="Movie Recommender Chatbot",
    description=(
        "REST API for conversational movie recommendations using LLM-powered agent "
        "with structured data retrieval from movie database."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Initialize Prometheus metrics
setup_metrics(app)
logger.info("Prometheus metrics instrumentation configured")


# Helper functions for business logic

async def validate_or_create_user(user_id: Optional[str]) -> tuple[str, bool]:
    """
    Validate existing user or create a new one.
    
    Args:
        user_id: Optional user identifier
        
    Returns:
        tuple[str, bool]: (user_id, is_new_user)
        
    Raises:
        HTTPException: If user validation fails
    """
    is_new_user = False
    if not redis_client:
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize redis cache"
            )
    
    if not user_id:
        # Create new user
        try:
            user_id = await create_user()
            await cache_set_user(redis_client, user_id)
            is_new_user = True
            logger.info(f"Created new user: {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    else:
        # Validate existing user
        try:
            # Check cache first
            user_exists = await cache_get_user(redis_client, user_id)
            
            if not user_exists:
                # Cache miss - check database
                user = await get_user(user_id)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"User not found: {user_id}"
                    )
                # Update cache
                await cache_set_user(redis_client, user_id)
            
            logger.debug(f"Validated existing user: {user_id}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to validate user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to validate user"
            )
    
    return user_id, is_new_user


async def validate_chat_pair(user_id: str, chat_id: str) -> str:
    """
    Validate that (user_id, chat_id) pair exists and chat is active.
    
    Args:
        user_id: User identifier
        chat_id: Chat identifier
        
    Returns:
        str: Chat status
        
    Raises:
        HTTPException: If validation fails
    """
    if not redis_client:
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize redis cache"
            )
    try:
        # Check cache first
        cached_status = await cache_get_chat_pair(redis_client, user_id, chat_id)
        
        if cached_status:
            logger.debug(f"Validated chat pair from cache: ({user_id}, {chat_id})")
            return cached_status
        
        # Cache miss - check database
        chat = await get_chat(chat_id, user_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat not found or unauthorized: {chat_id}"
            )
        
        # Update cache
        await cache_set_chat_pair(redis_client, user_id, chat_id, chat['status'])
        logger.debug(f"Validated chat pair from database: ({user_id}, {chat_id})")
        
        return chat['status']
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate chat pair ({user_id}, {chat_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate chat session"
        )


# API Endpoints

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Movie Recommender Chatbot",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "chat": "/chat",
            "history": "/chat/history",
            "archive": "/chat/archive",
            "unarchive": "/chat/unarchive",
        }
    }



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Enhanced health check endpoint with metrics tracking.
    
    Verifies that database, cache, and agent are operational.
    Updates Prometheus health gauges for monitoring.
    """
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "cache": "unknown",
        "agent": "unknown",
    }
    
    # Check database
    try:
        await get_user("health_check_dummy")
        health_status["database"] = "connected"
        update_service_health("database", True)
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = "error"
        health_status["status"] = "degraded"
        update_service_health("database", False)
    
    # Check Redis
    try:
        if redis_client:
            await redis_client.ping() # type: ignore
            health_status["cache"] = "connected"
            update_service_health("cache", True)
        else:
            health_status["cache"] = "not_initialized"
            health_status["status"] = "degraded"
            update_service_health("cache", False)
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["cache"] = "error"
        health_status["status"] = "degraded"
        update_service_health("cache", False)
    
    # Check agent
    if movie_agent:
        health_status["agent"] = "initialized"
        update_service_health("agent", True)
    else:
        health_status["agent"] = "not_initialized"
        health_status["status"] = "degraded"
        update_service_health("agent", False)
    
    return health_status





@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for movie recommendations.
    
    Handles user validation, chat session management, and routes
    queries to the movie agent for LLM-powered responses.
    
    Request validation logic:
    - No user_id, no chat_id: Create both
    - Only user_id: Create chat
    - Only chat_id: Invalid request (user_id required with chat_id)
    - Both user_id and chat_id: Validate pair and check status
    
    Args:
        request: ChatRequest containing user_id, chat_id, and message
        
    Returns:
        ChatResponse with user info, chat info, and agent reply
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    logger.info(f"Received chat request: user={request.user_id}, chat={request.chat_id}")
    
    is_new_user = False
    is_new_chat = False

    if not redis_client:
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize redis cache"
            )
    
    # Case 1: No user_id and no chat_id -> create both
    if not request.user_id and not request.chat_id:
        user_id, is_new_user = await validate_or_create_user(None)
        
        # Create new chat for new user
        try:
            chat_id = await create_chat(user_id)
            await cache_set_chat_pair(redis_client, user_id, chat_id, ChatStatus.ACTIVE.value)
            is_new_chat = True
            logger.info(f"Created new chat {chat_id} for new user {user_id}")
        except Exception as e:
            logger.error(f"Failed to create chat for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )
    
    # Case 2: Only user_id -> create chat
    elif request.user_id and not request.chat_id:
        user_id, is_new_user = await validate_or_create_user(request.user_id)
        
        # Create new chat for existing user
        try:
            chat_id = await create_chat(user_id)
            await cache_set_chat_pair(redis_client, user_id, chat_id, ChatStatus.ACTIVE.value)
            is_new_chat = True
            logger.info(f"Created new chat {chat_id} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to create chat for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )
    
    # Case 3: Only chat_id -> invalid request
    elif not request.user_id and request.chat_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required when chat_id is provided"
        )
    
    # Case 4: Both user_id and chat_id -> validate pair
    else:
        user_id = request.user_id
        chat_id = request.chat_id
        
        # Validate user exists
        user_id, is_new_user = await validate_or_create_user(user_id)
        
        # Validate (user_id, chat_id) pair and check status
        chat_status = await validate_chat_pair(user_id, chat_id)
        
        if chat_status != ChatStatus.ACTIVE.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chat is not active: {chat_id} (status: {chat_status})"
            )
        
        logger.debug(f"Validated chat pair: ({user_id}, {chat_id})")
    
    # Process message with movie agent
    if not movie_agent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Movie agent not initialized"
        )
    
    try:
        logger.info(f"Processing message for user {user_id}, chat id {chat_id}: {request.message}")
        reply = await movie_agent.answer(request.message, chat_id)
        logger.info(f"Generated reply for chat {chat_id}")
        return ChatResponse(
            user_id=user_id,
            chat_id=chat_id,
            message=request.message,
            reply=reply,
            is_new_user=is_new_user,
            is_new_chat=is_new_chat,
        )
        
    except Exception as e:
        logger.error(f"Movie agent failed to process message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )
    
    


@app.post("/chat/history", response_model=HistoryResponse)
async def get_chat_history(request: HistoryRequest):
    """
    Get conversation history for a chat session.
    
    Validates that the (user_id, chat_id) pair exists and chat is active,
    then retrieves the conversation history from the agent.
    
    Args:
        request: HistoryRequest containing user_id and chat_id
        
    Returns:
        HistoryResponse with conversation history
        
    Raises:
        HTTPException: If validation fails or history retrieval fails
    """
    logger.info(f"Retrieving history for chat {request.chat_id}, user {request.user_id}")
    
    # Validate (user_id, chat_id) pair
    chat_status = await validate_chat_pair(request.user_id, request.chat_id)
    
    # Check if chat is active
    if chat_status != ChatStatus.ACTIVE.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retrieve history: chat is {chat_status}"
        )
    
    # Retrieve history from agent
    if not movie_agent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Movie agent not initialized"
        )
    
    try:
        history = await movie_agent.ahistory(request.chat_id)
        logger.info(f"Retrieved history for chat {request.chat_id}")
        
        return HistoryResponse(
            user_id=request.user_id,
            chat_id=request.chat_id,
            conversation_history=history
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve history for chat {request.chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )


@app.post("/chat/archive", response_model=ChatStatusResponse)
async def archive_chat(request: ChatStatusRequest):
    """
    Archive a chat session.
    
    Validates that the (user_id, chat_id) pair exists, then sets
    chat status to 'archived' and invalidates cache.
    
    Args:
        request: ChatStatusRequest containing user_id and chat_id
        
    Returns:
        ChatStatusResponse confirming the archived chat
        
    Raises:
        HTTPException: If validation or archival fails
    """
    logger.info(f"Archiving chat {request.chat_id} for user {request.user_id}")

    if not redis_client:
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize redis cache"
            )
    
    # Validate (user_id, chat_id) pair exists
    await validate_chat_pair(request.user_id, request.chat_id)
    
    # Update status to archived
    try:
        success = await update_chat_status(
            request.chat_id,
            request.user_id,
            ChatStatus.ARCHIVED
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat not found or unauthorized: {request.chat_id}"
            )
        
        # Invalidate cache
        await cache_set_chat_pair(redis_client, request.user_id, request.chat_id, ChatStatus.ARCHIVED.value)
        
        logger.info(f"Successfully archived chat {request.chat_id}")
        
        return ChatStatusResponse(
            message="Chat archived successfully",
            user_id=request.user_id,
            chat_id=request.chat_id,
            status=ChatStatus.ARCHIVED.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to archive chat {request.chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to archive chat"
        )


@app.post("/chat/unarchive", response_model=ChatStatusResponse)
async def unarchive_chat(request: ChatStatusRequest):
    """
    Unarchive a chat session.
    
    Validates that the (user_id, chat_id) pair exists, then sets
    chat status to 'active' and invalidates cache.
    
    Args:
        request: ChatStatusRequest containing user_id and chat_id
        
    Returns:
        ChatStatusResponse confirming the unarchived chat
        
    Raises:
        HTTPException: If validation or unarchival fails
    """
    logger.info(f"Unarchiving chat {request.chat_id} for user {request.user_id}")

    if not redis_client:
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize redis cache"
            )
    
    # Validate (user_id, chat_id) pair exists
    await validate_chat_pair(request.user_id, request.chat_id)
    
    # Update status to active
    try:
        success = await update_chat_status(
            request.chat_id,
            request.user_id,
            ChatStatus.ACTIVE
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat not found or unauthorized: {request.chat_id}"
            )
        
        # Set user and chat pair in cache
        await cache_set_chat_pair(redis_client, request.user_id, request.chat_id, ChatStatus.ACTIVE.value)
        
        logger.info(f"Successfully unarchived chat {request.chat_id}")
        
        return ChatStatusResponse(
            message="Chat unarchived successfully",
            user_id=request.user_id,
            chat_id=request.chat_id,
            status=ChatStatus.ACTIVE.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unarchive chat {request.chat_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unarchive chat"
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
