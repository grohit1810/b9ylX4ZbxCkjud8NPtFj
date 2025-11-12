"""
Vector search handler logic.
"""
import json
from typing import Dict, Any, List
from logging_config.logger import get_logger
from config.settings import VECTOR_DB_CONFIG, MCP_VECTOR_TOOL_CONFIG
from mcp_server.database.vector_client import VectorDBClient
from mcp_server.utils.validators import validate_vector_params

logger = get_logger(__name__)


class VectorQueryHandler:
    """Handle vector database semantic searches."""

    def __init__(self):
        """Initialize vector handler."""
        self.client = VectorDBClient(
            vector_index_path=VECTOR_DB_CONFIG["index_path"],
            meta_path=VECTOR_DB_CONFIG["meta_path"],
            embedding_model=VECTOR_DB_CONFIG["embedding_model"],
            cache_size=MCP_VECTOR_TOOL_CONFIG["default_cache_size"]
        )
        logger.info("Vector Query Handler initialized")

    def execute_search(
        self,
        query_text: str,
        top_k: int
    ) -> Dict[str, Any]:
        """
        Execute semantic search on movie descriptions.

        Args:
            query_text: Scene description or query
            top_k: Number of results

        Returns:
            Search results
        """
        # Validate parameters
        validate_vector_params(query_text, top_k)

        logger.debug(f"Executing vector search: {query_text}")
        logger.debug(f"Top K: {top_k}")

        # Execute search
        results = self.client.search(query_text, top_k=top_k)

        if not results:
            logger.debug("No semantic matches found")
            return {
                "success": True,
                "count": 0,
                "movies": [],
                "query": query_text,
                "message": "No movies found matching semantic criteria"
            }

        # Format results
        formatted_results = []
        for movie_dict, similarity_score in results:
            formatted_movie = self._format_search_result(movie_dict, similarity_score)
            formatted_results.append(formatted_movie)

        return {
            "success": True,
            "count": len(formatted_results),
            "movies": formatted_results,
            "query": query_text,
            "search_type": "semantic"
        }

    def _parse_json_field(self, field: str) -> List[str]:
        """Parse JSON field."""
        if not field:
            return []
        try:
            return json.loads(field)
        except (json.JSONDecodeError, TypeError):
            return []

    def _format_search_result(
        self,
        movie_dict: Dict[str, Any],
        similarity_score: float
    ) -> Dict[str, Any]:
        """Format search result."""
        cast = self._parse_json_field(movie_dict.get("cast", "[]"))
        keywords = self._parse_json_field(movie_dict.get("keywords", "[]"))

        return {
            "title": movie_dict.get("title"),
            "director": movie_dict.get("director", ""),
            "cast": cast if cast else [],
            "release_date": movie_dict.get("release_date", ""),
            "keywords": keywords if keywords else [],
            "rating": movie_dict.get("rating", 0.0),
            "overview": movie_dict.get("overview", ""),
            "similarity_score": round(similarity_score, 3)
        }
