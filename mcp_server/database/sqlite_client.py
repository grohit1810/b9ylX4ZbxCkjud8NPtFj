"""
SQLite database client for structured queries with LRU caching.
"""
import os
import json
import sqlite3
import hashlib
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Sequence
from logging_config.logger import get_logger

logger = get_logger(__name__)


class SQLiteClient:
    """SQLite client for movie database operations with LRU caching."""
    
    def __init__(self, db_path: Path, cache_size: int = 128):
        """
        Initialize SQLite client.
        
        Args:
            db_path: Path to SQLite database
            cache_size: Maximum number of cached queries (default: 128)
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # LRU cache using OrderedDict
        self._cache = OrderedDict()
        self._cache_size = cache_size
        
        # Track DB modification time for cache invalidation
        self._db_mtime = self._get_db_mtime()
        
        logger.info(f"SQLiteClient initialized: {db_path} (cache_size={cache_size})")
    
    def _get_db_mtime(self) -> float:
        """Get database file modification time."""
        try:
            return os.path.getmtime(self.db_path)
        except OSError:
            return 0.0
    
    def _check_db_updated(self):
        """Check if database was updated and clear cache if so."""
        current_mtime = self._get_db_mtime()
        if current_mtime != self._db_mtime:
            logger.info(f"Database updated detected: Clearing cache")
            self._cache.clear()
            self._db_mtime = current_mtime
    
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _normalize_cache_value(self, value: Any) -> Any:
        """
        Normalize value for consistent cache keys.
        
        - Strings: lowercase, stripped
        - Lists/sequences: sorted, lowercased, deduplicated
        - None: remains None
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            # Lowercase and strip whitespace
            return value.lower().strip()
        
        if isinstance(value, (list, tuple)):
            # Convert to sorted tuple of lowercase strings
            normalized = [str(v).lower().strip() for v in value if str(v).strip()]
            return tuple(sorted(set(normalized)))  # Deduplicate and sort
        
        return value
    
    def _make_cache_key(self, **kwargs) -> str:
        """
        Generate normalized cache key from parameters.
        
        Ensures:
        - Case-insensitive: "Action" == "action"
        - Order-independent: ["Action", "Drama"] == ["Drama", "Action"]
        - Whitespace-normalized: "Action " == "Action"
        """
        normalized = {}
        for k, v in kwargs.items():
            normalized[k] = self._normalize_cache_value(v)
        
        # Sort by key for consistent ordering
        items = tuple(sorted(normalized.items()))
        key_str = str(items)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get from cache and mark as recently used."""
        if key in self._cache:
            # Move to end (most recent)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def _put_in_cache(self, key: str, value: List[Dict[str, Any]]):
        """Put in cache with LRU eviction."""
        # If key exists, remove it first (will be re-added at end)
        if key in self._cache:
            del self._cache[key]
        
        # Add to end (most recent)
        self._cache[key] = value
        
        # Evict oldest if cache is full
        if len(self._cache) > self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache full: Evicted oldest entry {oldest_key[:8]}...")
    
    def clear_cache(self):
        """Clear query cache."""
        self._cache.clear()
        logger.info("SQLite cache cleared")
    
    def query_movies(
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
    ) -> List[Dict[str, Any]]:
        """
        Query movies with flexible filters and intelligent ranking.
        
        Features:
        - Automatic LRU caching (128 queries max by default)
        - Case-insensitive cache keys
        - Order-independent cache keys (["Action", "Drama"] == ["Drama", "Action"])
        - Auto cache invalidation when DB is updated
        """
        
        # ========== Check DB Updates ==========
        self._check_db_updated()
        
        # ========== Check Cache ==========
        cache_key = self._make_cache_key(
            genre=genre, year=year, year_min=year_min, year_max=year_max,
            cast=cast, director=director, title=title,
            limit=limit, offset=offset, order_by=order_by, order_dir=order_dir
        )
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache HIT: {cache_key[:8]}... (size={len(self._cache)})")
            return cached_result
        
        # ========== Helper Functions ==========
        
        def escape_like(s: str) -> str:
            """Escape special LIKE characters."""
            return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        
        def normalize_list(val: Optional[Sequence[str] | str]) -> List[str]:
            """Convert string/list to normalized list of strings."""
            if val is None:
                return []
            if isinstance(val, str):
                return [x.strip() for x in val.split(",") if x.strip()]
            return [str(x).strip() for x in val if str(x).strip()]
        
        # ========== Normalize Inputs ==========
        
        genres = normalize_list(genre)
        cast_members = normalize_list(cast)
        
        ORDERABLE_FIELDS = {
            "id", "title", "year", "rating", "popularity",
            "vote_count", "revenue", "budget", "runtime", "release_date"
        }
        order_by = order_by if order_by in ORDERABLE_FIELDS else "year"
        order_dir = "DESC" if str(order_dir).upper() == "DESC" else "ASC"
        
        # ========== Build Base Filters ==========
        
        base_clauses: List[str] = []
        base_params: List[Any] = []
        
        if year is not None:
            base_clauses.append("m.year = ?")
            base_params.append(year)
        else:
            if year_min is not None:
                base_clauses.append("m.year >= ?")
                base_params.append(year_min)
            if year_max is not None:
                base_clauses.append("m.year <= ?")
                base_params.append(year_max)
        
        if director:
            base_clauses.append("m.director LIKE ? ESCAPE '\\' COLLATE NOCASE")
            base_params.append(f"%{escape_like(director)}%")
        
        if title:
            base_clauses.append("m.title LIKE ? ESCAPE '\\' COLLATE NOCASE")
            base_params.append(f"%{escape_like(title)}%")
        
        base_where = f"WHERE {' AND '.join(base_clauses)}" if base_clauses else ""
        
        # ========== Build JSON Match Logic ==========
        
        def build_json_predicates(values: List[str], alias: str):
            if not values:
                return "0", []
            preds = [f"{alias}.value LIKE ? ESCAPE '\\' COLLATE NOCASE" for _ in values]
            params = [f"%{escape_like(v)}%" for v in values]
            return " OR ".join(preds), params
        
        genre_sql, genre_params = build_json_predicates(genres, "ge")
        cast_sql, cast_params = build_json_predicates(cast_members, "ce")
        
        genre_count = len(genres)
        cast_count = len(cast_members)
        
        # ========== Build Query ==========
        
        subquery = f"""
            SELECT
                m.*,
                COALESCE((
                    SELECT COUNT(*)
                    FROM json_each(m.genres) AS ge
                    WHERE ({genre_sql})
                ), 0) AS genre_matches,
                COALESCE((
                    SELECT COUNT(*)
                    FROM json_each(m."cast") AS ce
                    WHERE ({cast_sql})
                ), 0) AS cast_matches
            FROM movies AS m
            {base_where}
        """
        
        outer_where_parts = []
        if genre_count > 0:
            outer_where_parts.append("genre_matches >= 1")
        if cast_count > 0:
            outer_where_parts.append("cast_matches >= 1")
        
        outer_where = f"WHERE {' AND '.join(outer_where_parts)}" if outer_where_parts else ""
        
        ranking = (
            f"CASE WHEN (? = 0 OR genre_matches >= ?) AND "
            f"(? = 0 OR cast_matches >= ?) THEN 1 ELSE 0 END"
        )
        
        full_query = f"""
            SELECT *
            FROM ({subquery}) t
            {outer_where}
            ORDER BY {ranking} DESC, {order_by} {order_dir}
            LIMIT ? OFFSET ?
        """
        
        query_params = (
            base_params + genre_params + cast_params +
            [genre_count, genre_count, cast_count, cast_count] +
            [max(1, int(limit)), max(0, int(offset))]
        )
        
        # ========== Execute Query ==========
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(full_query, query_params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                movie = dict(row)
                movie.pop("genre_matches", None)
                movie.pop("cast_matches", None)
                results.append(movie)
            
            logger.debug(f"Cache MISS: Executed query, {len(results)} results (cache_size={len(self._cache)})")
            
            # Cache result with LRU eviction
            self._put_in_cache(cache_key, results)
            
            return results
            
        finally:
            conn.close()
    
    def get_movie_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get specific movie by exact title."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM movies WHERE title = ?", (title,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def parse_json_field(self, field: str) -> List[str]:
        """Parse JSON field to list of strings."""
        if not field:
            return []
        try:
            return json.loads(field)
        except (json.JSONDecodeError, TypeError):
            return []