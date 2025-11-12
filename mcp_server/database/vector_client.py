"""
Vector database client for semantic search with LRU caching.
"""
import os
import hashlib
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple
from vector_db.movie_vector_db import FaissHNSWMovieVectorDB
from config.settings import VECTOR_DB_CONFIG
from logging_config.logger import get_logger

logger = get_logger(__name__)


class VectorDBClient:
    """Vector database client for semantic search with LRU caching."""
    
    def __init__(
        self, 
        vector_index_path: Path, 
        meta_path: Path, 
        embedding_model: str,
        cache_size: int = 128
    ):
        """
        Initialize vector DB client.
        
        Args:
            vector_index_path: Path to FAISS index
            meta_path: Path to metadata pickle
            embedding_model: Embedding model name
            cache_size: Maximum number of cached queries (default: 128)
        """
        self.vector_index_path = Path(vector_index_path)
        self.meta_path = Path(meta_path)
        
        if not self.vector_index_path.exists():
            raise FileNotFoundError(f"Vector index not found: {vector_index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        # LRU cache using OrderedDict
        self._cache = OrderedDict()
        self._cache_size = cache_size
        
        # Track index modification times for cache invalidation
        self._index_mtime = self._get_file_mtime(self.vector_index_path)
        self._meta_mtime = self._get_file_mtime(self.meta_path)
        
        logger.info(f"VectorDBClient initialized (cache_size={cache_size})")
        logger.debug(f"Index: {vector_index_path}")
        logger.debug(f"Meta: {meta_path}")
        
        # Initialize vector DB
        self.vector_db = FaissHNSWMovieVectorDB(
            embedding_model=embedding_model,
            vector_index_path=vector_index_path,
            meta_path=meta_path
        )
        
        # Load index
        success = self.vector_db.load()
        if not success:
            raise RuntimeError("Failed to load vector database")
        if self.vector_db.index is None:
            raise RuntimeError("Vector database index is None")
        
        logger.info(f"Vector DB loaded with {self.vector_db.index.ntotal} vectors")
    
    def _get_file_mtime(self, path: Path) -> float:
        """Get file modification time."""
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0
    
    def _check_index_updated(self):
        """Check if index files were updated and clear cache if so."""
        current_index_mtime = self._get_file_mtime(self.vector_index_path)
        current_meta_mtime = self._get_file_mtime(self.meta_path)
        
        if current_index_mtime != self._index_mtime or current_meta_mtime != self._meta_mtime:
            logger.info("Vector index updated detected: Clearing cache")
            self._cache.clear()
            self._index_mtime = current_index_mtime
            self._meta_mtime = current_meta_mtime
    
    def _make_cache_key(self, query_text: str, top_k: int) -> str:
        """
        Generate normalized cache key.
        
        Ensures case-insensitive and whitespace-normalized keys.
        """
        # Normalize: lowercase, strip whitespace, collapse multiple spaces
        normalized_query = " ".join(query_text.lower().split())
        key_str = f"{normalized_query}:{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[List[Tuple[Dict[str, Any], float]]]:
        """Get from cache and mark as recently used."""
        if key in self._cache:
            # Move to end (most recent)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def _put_in_cache(self, key: str, value: List[Tuple[Dict[str, Any], float]]):
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
            logger.debug(f"Cache full: Evicted oldest entry {oldest_key}...")
    
    def clear_cache(self):
        """Clear search cache."""
        self._cache.clear()
        logger.info("Vector search cache cleared")
    
    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Semantic search on movie descriptions.
        
        Features:
        - Automatic LRU caching (128 queries max by default)
        - Case-insensitive cache keys
        - Whitespace-normalized cache keys
        - Auto cache invalidation when index is updated
        
        Args:
            query_text: Scene description or query
            top_k: Number of results
            
        Returns:
            List of movie dictionaries with similarity scores
        """
        # Check for index updates
        self._check_index_updated()
        
        # Check cache
        cache_key = self._make_cache_key(query_text, top_k)
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache HIT: {cache_key} (size={len(self._cache)})")
            return cached_result
        
        # Execute search
        logger.debug(f"Cache MISS: Vector search for '{query_text}' (cache_size={len(self._cache)})")
        results = self.vector_db.search(query_text, k=top_k, efSearch=VECTOR_DB_CONFIG["default_ef_search"])
        
        logger.debug(f"Found {len(results)} results")
        
        # Cache result with LRU eviction
        self._put_in_cache(cache_key, results)
        
        return results
