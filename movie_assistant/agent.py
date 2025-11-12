# movie_assistant/agent.py
from __future__ import annotations

from typing import Optional, List, Dict, Any
from uuid import uuid4

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from movie_assistant.prompts import SYSTEM_PROMPT
from movie_assistant.middleware import sanitize_sql_args, tool_errors_to_message, log_tools
from config.settings import LLM_CONFIG
from logging_config.logger import get_logger

logger = get_logger(__name__)


class MovieAgent:
    """
    Conversational agent for answering movie-related queries and recommendations.
    """

    def __init__(self, *, 
                 llm_memory_db: str = "checkpoints.db",
                 llm_host: str = "http://localhost:11434", 
                 llm_model: str = "llama3.2",
                 temperature: float = 0.3, 
                 mcp_url: str = "http://localhost:8765/mcp",
                 verbose: bool = True) -> None:
        
        self.llm_memory_db = llm_memory_db 
        self.verbose = verbose
        self.mcp_url = mcp_url

        self._model = ChatOllama(
            base_url=llm_host,
            model=llm_model,
            temperature=temperature,
        )

        self._tools = None
        self._middleware = [sanitize_sql_args, tool_errors_to_message]
        if self.verbose:
            self._middleware.append(log_tools)

        logger.info(f"MovieAgent initialized with llm_memory_db={self.llm_memory_db}, mcp_url={self.mcp_url}, llm_model={llm_model}")

    async def _load_mcp_tools(self) -> None:
        client = MultiServerMCPClient({
            "movies": {"transport": "streamable_http", "url": self.mcp_url}
        })
        self._tools = await client.get_tools()
        logger.info(f"Loaded {len(self._tools)} tools from MCP server")
        if len(self._tools) > 0:
            logger.info(f"Tools description: {self.tool_descriptions()}")

    async def answer(self, text: str, thread_id: str) -> Optional[str | list[str | dict[Any, Any]]]:
        """Handle a single turn: build a fresh agent, persist via AsyncSqliteSaver, return assistant text."""
        if self._tools is None:
            await self._load_mcp_tools()

        logger.info(f"Generating AI response for query - text: {text}, thread_id: {thread_id}")

        async with AsyncSqliteSaver.from_conn_string(self.llm_memory_db) as saver:  

            agent = create_agent(
                model=self._model,
                tools=self._tools,
                system_prompt=SYSTEM_PROMPT,
                checkpointer=saver,
                middleware=self._middleware,
            )

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=text)]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": LLM_CONFIG.get("max_recursion_limit", 25),
                }
            )

        # Extract final assistant message (works with current agent output shape)
        last_ai_response = ""
        msgs = result.get("messages") if isinstance(result, dict) else None
        if isinstance(msgs, list):
            for msg in reversed(msgs):
                if isinstance(msg, AIMessage):
                    last_ai_response = msg.content or ""
                    break
        return last_ai_response

    async def ahistory(self, thread_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Read latest messages for a thread directly from the SQLite saver (no agent needed)."""

        async with AsyncSqliteSaver.from_conn_string(self.llm_memory_db) as saver:
            snap = await saver.aget({"configurable": {"thread_id": thread_id}})

        if not snap or not isinstance(snap, dict):
            return []
            
        channel_values = snap.get("channel_values")
        if not channel_values or not isinstance(channel_values, dict):
            return []
        
        messages = channel_values.get("messages")
        if not messages or not isinstance(messages, list):
            return []

        out: List[Dict[Any, Any]] = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            msg_content = getattr(msg, "content", None)
            if not msg_content:
                continue
            if msg_type == "human":
                out.append({"role": "user", "content": msg_content})
            elif msg_type == "ai":
                out.append({"role": "assistant", "content": msg_content})
        return out
    
    def tool_names(self) -> List[str]:
        return [t.name for t in (self._tools or [])]

    def tool_descriptions(self) -> Dict[str, str]:
        return {t.name: t.description for t in (self._tools or [])}
