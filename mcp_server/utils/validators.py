"""
Input validators for MCP tools.
"""
from typing import Optional
from logging_config.logger import get_logger

logger = get_logger(__name__)


def validate_sql_params(
    genre: Optional[str] = None,
    year: Optional[int] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    cast: Optional[str] = None,
    director: Optional[str] = None,
    title: Optional[str] = None,
    offset: int = 0,
    order_by: str = "year",
    order_dir: str = "DESC"
) -> bool:
    """Validate SQL query parameters."""

    # At least one filter should be provided
    if not any([genre, year, year_min, year_max, cast, director, title]):
        logger.warning("No filters provided for SQL query")

    # Validate year range
    if year_min and year_max and year_min > year_max:
        raise ValueError(f"year_min ({year_min}) cannot be greater than year_max ({year_max})")

    # Validate year types
    if year and not isinstance(year, int):
        raise ValueError("year must be an integer")

    if year_min and not isinstance(year_min, int):
        raise ValueError("year_min must be an integer")

    if year_max and not isinstance(year_max, int):
        raise ValueError("year_max must be an integer")

    # Validate string fields are not empty
    if genre and not isinstance(genre, str):
        raise ValueError("genre must be a string")

    if cast and not isinstance(cast, str):
        raise ValueError("cast must be a string")

    if director and not isinstance(director, str):
        raise ValueError("director must be a string")

    if title and not isinstance(title, str):
        raise ValueError("title must be a string")
    
    if offset < 0:
        raise ValueError("offset must be >= 0")
    
    orderable = {"id", "title", "year", "rating", "popularity", 
                 "vote_count", "revenue", "budget", "runtime", "release_date"}
    if order_by not in orderable:
        raise ValueError(f"order_by must be one of: {orderable}")
    
    if order_dir not in ["ASC", "DESC"]:
        raise ValueError("order_dir must be ASC or DESC")

    return True


def validate_vector_params(
    query_text: str,
    top_k: int = 5
) -> bool:
    """Validate vector search parameters."""

    if not query_text:
        raise ValueError("query_text is required")

    if not isinstance(query_text, str):
        raise ValueError("query_text must be a string")

    if not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")

    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be between 1 and 50")

    return True
