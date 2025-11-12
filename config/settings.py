"""
Application configuration settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
APPLICATION_DATA_DIR = DATA_DIR / "application"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
APPLICATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
MOVIES_CSV = RAW_DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = RAW_DATA_DIR / "tmdb_5000_credits.csv"

SQLITE_DB = PROCESSED_DATA_DIR / "movies.db"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # INFO, DEBUG, WARNING, ERROR
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# API USER and CHAT DATABASE
API_DB = APPLICATION_DATA_DIR / "users.db"

# Vector DB configuration
VECTOR_DB_CONFIG = {
    "index_path": PROCESSED_DATA_DIR / "movie_vectors_hnsw.faiss",
    "meta_path": PROCESSED_DATA_DIR / "movie_vectors_meta.pkl",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "default_m" : 32,
    "default_ef_construction": 200,
    "default_ef_search": 128,
}

MCP_SERVER_CONFIG = {
    "mcp_server_name": "movie-recommender-mcp",
    "mcp_server_host": "127.0.0.1",
    "mcp_server_port": 8765,
    "mcp_http_path": "/mcp",
}

MCP_SQL_TOOL_CONFIG = {
    "default_sql_limit": 10,
    "default_offset": 0,
    "default_order_by": "rating",
    "default_order_dir": "DESC",
    "default_response_format": "summary",
    "default_cache_size": 128,
}

MCP_VECTOR_TOOL_CONFIG = {
    "default_top_k": 5,
    "default_cache_size": 128,
}

# LLM configuration
LLM_CONFIG = {
    "host": "http://localhost:11434",
    "model": "llama3.2",
    "temperature": 0.3,
    "max_recursion_limit": 25,
    "conversation_checkpoint_db": APPLICATION_DATA_DIR / "llm_memory.db",
}