"""
MCP Server Configuration.
"""
from pathlib import Path
from config.settings import SQLITE_DB, VECTOR_DB_PATH, VECTOR_META_PATH

# Database paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # movie_recommender root

MCP_CONFIG = {
    # Data paths
    "sqlite_db": SQLITE_DB,
    "vector_index": VECTOR_DB_PATH,
    "vector_meta": VECTOR_META_PATH,

    # Caching
    "sqlite_cache_size": 128,
    "vector_cache_size": 128,
}
