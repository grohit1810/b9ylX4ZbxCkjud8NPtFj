"""
SQL Query Tool - MCP Tool Definition and Execution
"""
from typing import Dict, Any
from logging_config.logger import get_logger
from mcp_server.handlers.sql_handler import SQLQueryHandler

logger = get_logger(__name__)


class SQLQueryTool:
    """SQL Query MCP Tool."""

    def __init__(self):
        """Initialize SQL tool."""
        self.handler = SQLQueryHandler()
        logger.info("SQL Query Tool initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SQL query tool.

        Args:
            arguments: Tool arguments
                - genre: str (optional)
                - year: int (optional)
                - year_min: int (optional)
                - year_max: int (optional)
                - cast: str (optional)
                - director: str (optional)
                - title: str (optional)
                - limit: int (default: 10)
                - offset: int (default: 0)
                - order_by: str (default: "year")
                - order_dir: str (default: "DESC")
                - response_format: str (default: "summary")

        Returns:
            Query results
        """
        logger.debug(f"SQL Tool execute: {arguments}")

        try:
            result = self.handler.execute_query(
                genre=arguments.get("genre"),
                year=arguments.get("year"),
                year_min=arguments.get("year_min"),
                year_max=arguments.get("year_max"),
                cast=arguments.get("cast"),
                director=arguments.get("director"),
                title=arguments.get("title"),
                limit=arguments["limit"],
                offset=arguments["offset"],
                order_by=arguments["order_by"],
                order_dir=arguments["order_dir"],
                response_format=arguments["response_format"]
            )

            logger.debug(f"SQL Tool result: {result['count']} movies found")
            return result

        except Exception as e:
            logger.error(f"SQL Tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "count": 0,
                "movies": []
            }
