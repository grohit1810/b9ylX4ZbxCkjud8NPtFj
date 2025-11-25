"""
SQL query handler and logic.
"""
from typing import Dict, Any, Optional
from mcp_server.database.sqlite_client import SQLiteClient
from config.settings import MCP_SQL_TOOL_CONFIG, SQLITE_DB
from mcp_server.utils.validators import validate_sql_params
from logging_config.logger import get_logger

logger = get_logger(__name__)


class SQLQueryHandler:
    """Handle SQL database queries."""

    def __init__(self):
        """Initialize SQL handler."""
        self.client = SQLiteClient(SQLITE_DB,
                                   cache_size=MCP_SQL_TOOL_CONFIG["default_cache_size"])
        logger.info("SQL Query Handler initialized")

    async def execute_query(
        self,
        genre: Optional[str],
        year: Optional[int],
        year_min: Optional[int],
        year_max: Optional[int],
        cast: Optional[str],
        director: Optional[str],
        title: Optional[str],
        limit: int,
        offset: int,
        order_by: str,
        order_dir: str,
        response_format: str
    ) -> Dict[str, Any]:
        """
        Execute SQL query with filters.

        Args:
            genre: Movie genre
            year: Specific year
            year_min: Minimum year
            year_max: Maximum year
            cast: Actor/actress name
            director: Director name
            title: Movie title
            limit: Result limit
            offset: Result offset
            order_by: Field to order by
            order_dir: Order direction ("ASC" or "DESC")
            response_format: "summary" or "detailed"

        Returns:
            Formatted query results
        """
        # Validate parameters
        validate_sql_params(genre, year, year_min, year_max, cast, director, title, offset, order_by, order_dir)

        logger.debug(f"Executing SQL query: genre={genre}, year={year}, "
                    f"cast={cast}, director={director}, limit={limit}")

        # Execute query
        movies = await self.client.query_movies(
            genre=genre,
            year=year,
            year_min=year_min,
            year_max=year_max,
            cast=cast,
            director=director,
            title=title,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_dir=order_dir
        )

        if not movies:
            logger.debug("No movies found")
            return {
                "success": True,
                "count": 0,
                "movies": [],
                "message": "No movies found matching criteria"
            }

        # Format response
        formatted_movies = []
        for movie in movies:
            formatted_movie = self._format_movie(movie, response_format)
            formatted_movies.append(formatted_movie)

        return {
            "success": True,
            "count": len(formatted_movies),
            "movies": formatted_movies,
            "query_filters": {
                "genre": genre,
                "year": year,
                "year_range": (year_min, year_max) if (year_min or year_max) else None,
                "cast": cast,
                "director": director,
                "title": title
            }
        }

    def _format_movie(self, movie: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """Format movie record based on response type."""
        cast = self.client.parse_json_field(movie.get("cast", "[]"))
        cast_top3 = cast[:3] if cast else []
        keywords = self.client.parse_json_field(movie.get("keywords", "[]"))

        if format_type == "detailed":
            # Detailed information
            return {
                "title": movie.get("title"),
                "release_date": movie.get("release_date"),
                "director": movie.get("director"),
                "cast": cast,
                "genres": self.client.parse_json_field(movie.get("genres", "[]")),
                "overview": movie.get("overview"),
                "keywords": keywords if keywords else [],
                "rating": movie.get("rating"),
                "runtime": movie.get("runtime"),
                "popularity": movie.get("popularity")
            }

        else:
            # Summary information
            return {
                "title": movie.get("title"),
                "release_date": movie.get("release_date"),
                "director": movie.get("director"),
                "cast": cast_top3,
                "genres": self.client.parse_json_field(movie.get("genres", "[]")),
                "overview": movie.get("overview"),
                "keywords": keywords[:5] if keywords else [],
                "rating": movie.get("rating")
            }
