"""
FAISS HNSW vector database for semantic search.
"""
import json
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from logging_config.logger import get_logger

logger = get_logger(__name__)


class FaissHNSWMovieVectorDB:
    """FAISS HNSW-based vector database for movies."""

    def __init__(self, embedding_model: str, vector_index_path: Path, meta_path: Path):
        """Initialize vector database."""
        self.embedding_model_name = embedding_model
        self.vector_index_path = Path(vector_index_path)
        self.meta_path = Path(meta_path)

        logger.debug(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.index = None
        self.metadata = []

        logger.info(f"Embedding model: {embedding_model} (dim={self.embedding_dim})")

    def create_enriched_document(self, title: str, overview: str, keywords: List[str]) -> str:
        """Create enriched document for embedding."""
        keywords_str = ', '.join(keywords) if keywords else 'N/A'
        enriched_doc = f"""Title: {title}\n\nPlot: {overview}\n\nKey themes and elements: {keywords_str}"""
        return enriched_doc

    def build_index(self, movies_data: List[Dict[str, Any]], M: int = 32, efConstruction: int = 200):
        """Build FAISS HNSW index."""
        logger.info(f"Building HNSW index with {len(movies_data)} movies...")
        logger.debug(f"Parameters: M={M}, efConstruction={efConstruction}")

        logger.debug("Creating enriched documents...")
        documents = []
        for movie in movies_data:
            keywords = movie.get('keywords', [])
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except (json.JSONDecodeError, TypeError):
                    keywords = []

            enriched_doc = self.create_enriched_document(
                title=movie.get('title', 'Unknown'),
                overview=movie.get('overview', ''),
                keywords=keywords
            )
            documents.append(enriched_doc)

        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
        logger.info(f"Generated {len(embeddings)} embeddings")

        logger.debug("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)

        logger.debug("Creating HNSW index...")
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
        self.index.hnsw.efConstruction = efConstruction
        self.index.add(embeddings.astype(np.float32))

        self.metadata = movies_data
        logger.info(f"HNSW index built with {self.index.ntotal} vectors")

    def save(self):
        """Save index and metadata."""
        if self.index is None:
            logger.error("Index not built yet")
            return False

        logger.info("Saving vector database...")
        faiss.write_index(self.index, str(self.vector_index_path))
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved index: {self.vector_index_path}")
        logger.info(f"Saved metadata: {self.meta_path}")
        return True

    def load(self):
        """Load index and metadata."""
        if not self.vector_index_path.exists() or not self.meta_path.exists():
            logger.error("Vector DB files not found")
            return False

        logger.info("Loading vector database...")
        try:
            self.index = faiss.read_index(str(self.vector_index_path))
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded index ({self.index.ntotal} entries)")
            return True
        except Exception as e:
            logger.error(f"Failed to load: {str(e)}")
            return False

    def search(self, query: str, k: int = 10, efSearch: int = 128) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar movies."""
        if self.index is None:
            logger.error("Index not loaded")
            return []

        self.index.hnsw.efSearch = efSearch
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.metadata):
                similarity = 1 - (distance / 2)
                results.append((self.metadata[idx], float(similarity)))

        logger.debug(f"Found {len(results)} results")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get index stats."""
        if self.index is None:
            return {"status": "not_built"}
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": "HNSW",
            "metadata_entries": len(self.metadata)
        }
