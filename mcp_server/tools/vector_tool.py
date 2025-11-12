"""
Vector Query Tool - MCP Tool Definition and Execution
"""
from typing import Dict, Any
from logging_config.logger import get_logger
from mcp_server.handlers.vector_handler import VectorQueryHandler

logger = get_logger(__name__)


class VectorQueryTool:
    """Vector Query MCP Tool."""

    def __init__(self):
        """Initialize vector tool."""
        self.handler = VectorQueryHandler()
        logger.info("Vector Query Tool initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vector query tool.

        Args:
            arguments: Tool arguments
                - query_text: str (required) - Scene description or query
                - top_k: int (optional, default: 5) - Number of results

        Returns:
            Search results
        """
        logger.debug(f"Vector Tool execute: {arguments}")

        try:
            query_text = arguments.get("query_text")
            if not query_text:
                return {
                    "success": False,
                    "error": "query_text is required",
                    "count": 0,
                    "movies": []
                }

            result = self.handler.execute_search(
                query_text=query_text,
                top_k=arguments["top_k"]
            )

            logger.debug(f"Vector Tool result: {result['count']} movies found")
            return result

        except Exception as e:
            logger.error(f"Vector Tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "count": 0,
                "movies": []
            }
