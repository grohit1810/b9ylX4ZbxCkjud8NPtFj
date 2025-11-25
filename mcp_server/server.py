"""
Movie Recommender MCP Server (FastMCP + Streamable HTTP)
- Exposes an HTTP endpoint (default: http://0.0.0.0:8765/mcp)
"""

import json
import uvicorn
from typing import Any, Optional, Literal
from pydantic import Field
from typing_extensions import Annotated
from logging_config.logger import get_logger
from config.settings import MCP_SERVER_CONFIG, MCP_SQL_TOOL_CONFIG, MCP_VECTOR_TOOL_CONFIG

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, CallToolResult
from mcp_server.tools.sql_tool import SQLQueryTool
from mcp_server.tools.vector_tool import VectorQueryTool

from starlette.responses import JSONResponse
from starlette.routing import Route

logger = get_logger(__name__)

mcp = FastMCP(MCP_SERVER_CONFIG['mcp_server_name'])
mcp.settings.streamable_http_path = MCP_SERVER_CONFIG['mcp_http_path']
mcp.settings.host = MCP_SERVER_CONFIG['mcp_server_host']
mcp.settings.port = MCP_SERVER_CONFIG['mcp_server_port']

# Initialize tool classes
_sql_tool = SQLQueryTool()
_vector_tool = VectorQueryTool()

OrderBy = Literal[
    "id", "title", "year", "rating", "popularity",
    "vote_count", "revenue", "budget", "runtime", "release_date"
]

def response_to_dict(raw: Any) -> dict:
    """
    Normalize tool output to a dict.
    Accepts dict, JSON str/bytes; everything else becomes an error dict.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"success": False, "error": "Malformed JSON from tool", "raw": raw}
    if isinstance(raw, dict):
        return raw
    # Handle False/None or anything unexpected
    return {"success": False, "error": "Unexpected return type", "raw_type": type(raw).__name__}

def generate_error_response(message: str, payload: Optional[dict] = None) -> CallToolResult:
    sc = {"success": False, "error": message}
    if payload is not None:
        sc["data"] = payload
    return CallToolResult(
        content=[TextContent(type="text", text=f"Error: {message}"),
                 TextContent(type="text", text=json.dumps(sc))],
        structuredContent=sc,
        isError=True,
    )

def generate_success_response(summary: str, payload: dict) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=summary), 
                 TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )

@mcp.tool(
    name="query_sql_db",
    description="Query movie database with structured filters. Supports filtering by genre, year range, cast, director, etc.",
)
async def query_sql_db(
    genre: Annotated[Optional[str], Field(title="genre(s)", description="Movie genre(s) (e.g., 'Action', 'Drama')")] = None,
    year: Annotated[Optional[int], Field(title="year", description="Specific year")] = None,
    year_min: Annotated[Optional[int], Field(title="year_min", description="Minimum release year")] = None,
    year_max: Annotated[Optional[int], Field(title="year_max", description="Maximum release year")] = None,
    cast: Annotated[Optional[str], Field(title="cast(s)", description="Actor/actress name(s)")] = None,
    director: Annotated[Optional[str], Field(title="director", description="Director name")] = None,
    title: Annotated[Optional[str], Field(title="title", description="Movie title (exact or partial)")] = None,
    limit: Annotated[int, Field(title="limit", description="Number of results (default: 10)")] = MCP_SQL_TOOL_CONFIG.get("default_sql_limit", 10),
    offset: Annotated[int, Field(title="offset", description="Pagination offset (default: 0)")] = MCP_SQL_TOOL_CONFIG.get("default_offset", 0),
    order_by: Annotated[Optional[OrderBy], Field(title="order_by", description="Sort field (default: rating)") ] = MCP_SQL_TOOL_CONFIG.get("default_order_by", "rating"),
    order_dir: Annotated[Literal["ASC", "DESC"], Field(title="order_dir", description="Sort direction (default: DESC)")] = MCP_SQL_TOOL_CONFIG.get("default_order_dir", "DESC"),
    response_format: Annotated[Optional[Literal["summary", "detailed"]], Field(title="response_format", description="Response format type")] = MCP_SQL_TOOL_CONFIG.get("default_response_format", "summary"),
) -> CallToolResult:
    """
    Delegates to SQLQueryTool; returns JSON-serializable data.
    """
    logger.info(
        "SQL query invoked: genre=%s, year=%s-%s, cast=%s, director=%s, title=%s, limit=%d, order_by=%s",
        genre, year_min or year, year_max or year, cast, director, title, limit, order_by
    )
    params = {
        "genre": genre,
        "year": year,
        "year_min": year_min,
        "year_max": year_max,
        "cast": cast,
        "director": director,
        "title": title,
        "limit": limit,
        "offset": offset,
        "order_by": order_by,
        "order_dir": order_dir,
        "response_format": response_format,
    }
    try:
        raw = await _sql_tool.execute(params)
    except Exception as e:
        logger.exception("SQL query failed with params: %s", params)
        return generate_error_response("SQL query failed", {"exception": str(e), "params": params})

    result = response_to_dict(raw)

    if not result.get("success", True):
        return generate_error_response(result.get("error", "Unknown error"), result)

    count = result.get("count") or len(result.get("movies", []))
    logger.info("SQL query succeeded: returned %d movies (offset=%d)", count, offset)
    return generate_success_response(f"Found {count} movies", result)

@mcp.tool(
    name="query_vector_db",
    description="Semantic search on movie descriptions. Find movies matching scene descriptions or themes.",
)
async def query_vector_db(
    query_text: Annotated[str, Field(title="query_text", description="Scene description or theme (e.g., 'someone comes back to life')")],
    limit: Annotated[int, Field(title="limit", description="Number of results (default: 5)")] = MCP_VECTOR_TOOL_CONFIG.get("default_top_k", 5),
) -> CallToolResult:
    logger.info("Vector search invoked: query='%s', limit=%d", query_text, limit)
    try:
        raw = await _vector_tool.execute({"query_text": query_text, "top_k": limit})
    except Exception as e:
        logger.exception("Vector search failed for query: '%s'", query_text)
        return generate_error_response("vector search failed", {"exception": str(e)})

    result = response_to_dict(raw)

    # Handle tool-declared failure (e.g., {"success": false, "error": "..."} or bad shape)
    if not result.get("success", True):
        return generate_error_response(result.get("error", "Unknown error"), result)

    count = result.get("count") or len(result.get("movies", []))
    logger.info("Vector search succeeded: returned %d results", count)
    return generate_success_response(f"Found {count} results for: {query_text}", result)

# Build the ASGI app (path configured above)
app = mcp.streamable_http_app()

async def health_check(request):
    """Health check endpoint for MCP server."""
    health_status = {
        "status": "healthy",
        "service": "Movie Recommender MCP Server",
        "tools": {
            "sql_tool": "initialized" if _sql_tool else "not_initialized",
            "vector_tool": "initialized" if _vector_tool else "not_initialized",
        }
    }
    
    # Test SQL tool connection
    try:
        test_result = await _sql_tool.execute({
            "genre": "Action",
            "offset": 0,
            "limit": 1,
            "order_by": "rating",
            "order_dir": "DESC",
            "response_format": "summary"
        })
        health_status["tools"]["sql_tool"] = "connected"
    except Exception as e:
        logger.error(f"SQL tool health check failed: {e}")
        health_status["tools"]["sql_tool"] = "error"
        health_status["status"] = "degraded"
    
    # Test Vector tool connection
    try:
        test_result = await _vector_tool.execute({
            "query_text": "test health check query",
            "top_k": 1
        })
        health_status["tools"]["vector_tool"] = "connected"
    except Exception as e:
        logger.error(f"Vector tool health check failed: {e}")
        health_status["tools"]["vector_tool"] = "error"
        health_status["status"] = "degraded"
    
    return JSONResponse(content=health_status)

app.routes.append(Route("/health", health_check, methods=["GET"]))

def main():
    logger.info(
        "Starting Movie Recommender MCP Server (FastMCP) on %s:%d path=%s",
        MCP_SERVER_CONFIG['mcp_server_host'],
        MCP_SERVER_CONFIG['mcp_server_port'],
        MCP_SERVER_CONFIG['mcp_http_path'],
    )
    logger.info("SQL tool initialized with config: %s", MCP_SQL_TOOL_CONFIG)
    logger.info("Vector tool initialized with config: %s", MCP_VECTOR_TOOL_CONFIG)
    uvicorn.run(app, host=MCP_SERVER_CONFIG['mcp_server_host'], port=MCP_SERVER_CONFIG['mcp_server_port'])

if __name__ == "__main__":
    main()
