"""
Comprehensive tests for MovieAgent and its components.

Tests agent initialization, tool loading, middleware, prompt logic, and integration.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from movie_assistant.agent import MovieAgent
from movie_assistant.middleware import sanitize_sql_args, tool_errors_to_message, log_tools
from movie_assistant.prompts import SYSTEM_PROMPT


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_memory_db(tmp_path):
    """Create temporary memory database."""
    db_path = tmp_path / "test_memory.db"
    return str(db_path)


@pytest.fixture
def mock_mcp_tools():
    """Mock MCP tools."""
    query_sql_tool = Mock()
    query_sql_tool.name = "query_sql_db"
    query_sql_tool.description = "Query movies by structured filters"
    
    query_vector_tool = Mock()
    query_vector_tool.name = "query_vector_db"
    query_vector_tool.description = "Semantic search for movies"
    
    return [query_sql_tool, query_vector_tool]


@pytest.fixture
async def agent_instance(temp_memory_db):
    """Create MovieAgent instance with mocked MCP."""
    with patch('movie_assistant.agent.MultiServerMCPClient') as mock_client:
        # Mock the MCP client
        mock_instance = AsyncMock()
        mock_instance.get_tools = AsyncMock(return_value=[
            Mock(name="query_sql_db", description="SQL query tool"),
            Mock(name="query_vector_db", description="Vector search tool")
        ])
        mock_client.return_value = mock_instance
        
        agent = MovieAgent(
            llm_memory_db=temp_memory_db,
            llm_host="http://localhost:11434",
            llm_model="llama3.2",
            mcp_url="http://localhost:8765/mcp",
            verbose=False
        )
        
        await agent._load_mcp_tools()
        yield agent


# ============================================================================
# AGENT INITIALIZATION TESTS
# ============================================================================

class TestMovieAgentInitialization:
    """Test agent initialization and configuration."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        agent = MovieAgent()
        assert agent.llm_memory_db == "checkpoints.db"
        assert agent.mcp_url == "http://localhost:8765/mcp"
        assert agent.verbose is True
        assert agent._tools is None
    
    def test_init_custom_values(self, temp_memory_db):
        """Test initialization with custom values."""
        agent = MovieAgent(
            llm_memory_db=temp_memory_db,
            llm_host="http://custom:11434",
            llm_model="custom_model",
            temperature=0.7,
            mcp_url="http://custom:9000/mcp",
            verbose=False
        )
        assert agent.llm_memory_db == temp_memory_db
        assert agent.mcp_url == "http://custom:9000/mcp"
        assert agent.verbose is False
    
    def test_middleware_setup_verbose_false(self):
        """Test middleware excludes log_tools when verbose=False."""
        agent = MovieAgent(verbose=False)
        assert sanitize_sql_args in agent._middleware
        assert tool_errors_to_message in agent._middleware
        assert log_tools not in agent._middleware
    
    def test_middleware_setup_verbose_true(self):
        """Test middleware includes log_tools when verbose=True."""
        agent = MovieAgent(verbose=True)
        assert sanitize_sql_args in agent._middleware
        assert tool_errors_to_message in agent._middleware
        assert log_tools in agent._middleware


# ============================================================================
# TOOL LOADING TESTS
# ============================================================================

class TestToolLoading:
    """Test MCP tool loading."""
    
    @pytest.mark.asyncio
    async def test_load_mcp_tools_success(self, temp_memory_db, mock_mcp_tools):
        """Test successful MCP tool loading."""
        with patch('movie_assistant.agent.MultiServerMCPClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_tools = AsyncMock(return_value=mock_mcp_tools)
            mock_client.return_value = mock_instance
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            await agent._load_mcp_tools()
            
            assert agent._tools is not None
            assert len(agent._tools) == 2
            assert agent.tool_names() == ["query_sql_db", "query_vector_db"]
    
    @pytest.mark.asyncio
    async def test_load_mcp_tools_empty(self, temp_memory_db):
        """Test MCP tool loading with no tools."""
        with patch('movie_assistant.agent.MultiServerMCPClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get_tools = AsyncMock(return_value=[])
            mock_client.return_value = mock_instance
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            await agent._load_mcp_tools()
            
            assert agent._tools is not None
            assert len(agent._tools) == 0
            assert agent.tool_names() == []
    
    @pytest.mark.asyncio
    async def test_load_mcp_tools_connection_error(self, temp_memory_db):
        """Test MCP tool loading handles connection errors."""
        with patch('movie_assistant.agent.MultiServerMCPClient') as mock_client:
            mock_client.side_effect = ConnectionError("MCP server unavailable")
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            
            with pytest.raises(ConnectionError):
                await agent._load_mcp_tools()
    
    def test_tool_names_before_loading(self):
        """Test tool_names() before tools are loaded."""
        agent = MovieAgent(verbose=False)
        assert agent.tool_names() == []
    
    def test_tool_descriptions_before_loading(self):
        """Test tool_descriptions() before tools are loaded."""
        agent = MovieAgent(verbose=False)
        assert agent.tool_descriptions() == {}


# ============================================================================
# MIDDLEWARE TESTS
# ============================================================================

class TestMiddleware:
    """Test middleware functions."""
    
    def test_sanitize_sql_args_exists(self):
        """Test sanitize_sql_args middleware exists and is correct type."""
        assert sanitize_sql_args is not None
        # It's a LangChain middleware object
        assert 'sanitize_sql_args' in str(type(sanitize_sql_args))
    
    def test_tool_errors_to_message_exists(self):
        """Test tool_errors_to_message middleware exists and is correct type."""
        assert tool_errors_to_message is not None
        assert 'tool_errors_to_message' in str(type(tool_errors_to_message))
    
    def test_log_tools_exists(self):
        """Test log_tools middleware exists and is correct type."""
        assert log_tools is not None
        assert 'log_tools' in str(type(log_tools))
    
    def test_middleware_registration_in_agent(self):
        """Test middleware is properly registered in agent."""
        agent = MovieAgent(verbose=False)
        assert sanitize_sql_args in agent._middleware
        assert tool_errors_to_message in agent._middleware
        
        agent_verbose = MovieAgent(verbose=True)
        assert log_tools in agent_verbose._middleware
    
    def test_allowed_order_by_values(self):
        """Test ALLOWED_ORDER_BY constant is defined."""
        from movie_assistant.middleware import ALLOWED_ORDER_BY
        assert "year" in ALLOWED_ORDER_BY
        assert "title" in ALLOWED_ORDER_BY
        assert "rating" in ALLOWED_ORDER_BY
        assert "popularity" in ALLOWED_ORDER_BY
        assert len(ALLOWED_ORDER_BY) == 4


# ============================================================================
# ANSWER METHOD TESTS
# ============================================================================

class TestAnswerMethod:
    """Test the main answer() method."""
    
    @pytest.mark.asyncio
    async def test_answer_loads_tools_if_not_loaded(self, temp_memory_db):
        """Test answer() loads tools if not already loaded."""
        with patch('movie_assistant.agent.MultiServerMCPClient') as mock_client, \
             patch('movie_assistant.agent.create_agent') as mock_create_agent, \
             patch('movie_assistant.agent.AsyncSqliteSaver'):
            
            # Setup mocks
            mock_instance = AsyncMock()
            mock_instance.get_tools = AsyncMock(return_value=[])
            mock_client.return_value = mock_instance
            
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [AIMessage(content="Test response")]
            })
            mock_create_agent.return_value = mock_agent
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            assert agent._tools is None
            
            await agent.answer("test query", "thread_123")
            
            # Tools should be loaded
            assert agent._tools is not None
    
    @pytest.mark.asyncio
    async def test_answer_returns_ai_response(self, temp_memory_db):
        """Test answer() extracts and returns AI response."""
        with patch('movie_assistant.agent.MultiServerMCPClient'), \
             patch('movie_assistant.agent.create_agent') as mock_create_agent, \
             patch('movie_assistant.agent.AsyncSqliteSaver'):
            
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [
                    HumanMessage(content="User query"),
                    AIMessage(content="Agent response")
                ]
            })
            mock_create_agent.return_value = mock_agent
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            agent._tools = []  # Pretend tools are loaded
            
            response = await agent.answer("test query", "thread_123")
            
            assert response == "Agent response"
    
    @pytest.mark.asyncio
    async def test_answer_handles_empty_messages(self, temp_memory_db):
        """Test answer() handles empty message list."""
        with patch('movie_assistant.agent.MultiServerMCPClient'), \
             patch('movie_assistant.agent.create_agent') as mock_create_agent, \
             patch('movie_assistant.agent.AsyncSqliteSaver'):
            
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
            mock_create_agent.return_value = mock_agent
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            agent._tools = []
            
            response = await agent.answer("test query", "thread_123")
            
            assert response == ""


# ============================================================================
# HISTORY METHOD TESTS
# ============================================================================

class TestHistoryMethod:
    """Test the ahistory() method."""
    
    @pytest.mark.asyncio
    async def test_ahistory_returns_conversation(self, temp_memory_db):
        """Test ahistory() returns formatted conversation."""
        with patch('movie_assistant.agent.AsyncSqliteSaver') as mock_saver_class:
            mock_saver = AsyncMock()
            mock_saver.aget = AsyncMock(return_value={
                "channel_values": {
                    "messages": [
                        HumanMessage(content="Hello"),
                        AIMessage(content="Hi there!"),
                        HumanMessage(content="Recommend movies"),
                        AIMessage(content="Here are some movies...")
                    ]
                }
            })
            mock_saver_class.from_conn_string = MagicMock(return_value=mock_saver)
            mock_saver.__aenter__ = AsyncMock(return_value=mock_saver)
            mock_saver.__aexit__ = AsyncMock()
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            history = await agent.ahistory("thread_123")
            
            assert len(history) == 4
            assert history[0] == {"role": "user", "content": "Hello"}
            assert history[1] == {"role": "assistant", "content": "Hi there!"}
            assert history[2] == {"role": "user", "content": "Recommend movies"}
            assert history[3] == {"role": "assistant", "content": "Here are some movies..."}
    
    @pytest.mark.asyncio
    async def test_ahistory_empty_conversation(self, temp_memory_db):
        """Test ahistory() with no conversation."""
        with patch('movie_assistant.agent.AsyncSqliteSaver') as mock_saver_class:
            mock_saver = AsyncMock()
            mock_saver.aget = AsyncMock(return_value=None)
            mock_saver_class.from_conn_string = MagicMock(return_value=mock_saver)
            mock_saver.__aenter__ = AsyncMock(return_value=mock_saver)
            mock_saver.__aexit__ = AsyncMock()
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            history = await agent.ahistory("thread_123")
            
            assert history == []
    
    @pytest.mark.asyncio
    async def test_ahistory_filters_empty_messages(self, temp_memory_db):
        """Test ahistory() filters out messages with no content."""
        with patch('movie_assistant.agent.AsyncSqliteSaver') as mock_saver_class:
            mock_saver = AsyncMock()
            mock_saver.aget = AsyncMock(return_value={
                "channel_values": {
                    "messages": [
                        HumanMessage(content="Hello"),
                        AIMessage(content=""),  # Empty content
                        AIMessage(content="Hi there!")
                    ]
                }
            })
            mock_saver_class.from_conn_string = MagicMock(return_value=mock_saver)
            mock_saver.__aenter__ = AsyncMock(return_value=mock_saver)
            mock_saver.__aexit__ = AsyncMock()
            
            agent = MovieAgent(llm_memory_db=temp_memory_db, verbose=False)
            history = await agent.ahistory("thread_123")
            
            # Should only have 2 messages (empty one filtered out)
            assert len(history) == 2
            assert history[0]["content"] == "Hello"
            assert history[1]["content"] == "Hi there!"


# ============================================================================
# PROMPT TESTS
# ============================================================================

class TestSystemPrompt:
    """Test system prompt configuration."""
    
    def test_system_prompt_exists(self):
        """Test SYSTEM_PROMPT is defined."""
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 0
    
    def test_system_prompt_mentions_tools(self):
        """Test system prompt mentions both tools."""
        assert "query_sql_db" in SYSTEM_PROMPT
        assert "query_vector_db" in SYSTEM_PROMPT
    
    def test_system_prompt_has_examples(self):
        """Test system prompt includes few-shot examples."""
        assert "Examples" in SYSTEM_PROMPT or "examples" in SYSTEM_PROMPT
        assert "romance" in SYSTEM_PROMPT.lower()
        assert "thriller" in SYSTEM_PROMPT.lower()
    
    def test_system_prompt_has_ground_rules(self):
        """Test system prompt has grounding instructions."""
        assert "tool" in SYSTEM_PROMPT.lower()
        assert "ONLY" in SYSTEM_PROMPT or "only" in SYSTEM_PROMPT
