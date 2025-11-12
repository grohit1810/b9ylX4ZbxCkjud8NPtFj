"""
Vector database ingestion pipeline.
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from logging_config.logger import get_logger
from vector_db.movie_vector_db import FaissHNSWMovieVectorDB
from config.settings import SQLITE_DB, VECTOR_DB_CONFIG 

logger = get_logger(__name__)


class VectorDBIngestionPipeline:
    """Vector database ingestion pipeline."""

    def read_movies_from_sql(self, db_path: Path) -> List[Dict[str, Any]]:
        """Read movies from SQLite."""
        logger.info(f"Reading from: {db_path}")

        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM movies")
        rows = cursor.fetchall()
        movies = [dict(row) for row in rows]
        conn.close()

        logger.info(f"Read {len(movies)} movies")
        return movies

    def run(self):
        """Execute pipeline."""
        logger.info("=" * 70)
        logger.info("STARTING VECTOR DB INGESTION")
        logger.info("=" * 70)

        logger.info("Step 1: Reading movies...")
        try:
            movies = self.read_movies_from_sql(SQLITE_DB)
        except Exception as e:
            logger.error(f"Failed: {str(e)}")
            logger.info("Run SQL ingestion first: python scripts/run_ingestion.py")
            raise

        logger.info("Step 2: Initializing FAISS HNSW...")
        vector_db = FaissHNSWMovieVectorDB(
            embedding_model=VECTOR_DB_CONFIG["embedding_model"],
            vector_index_path=VECTOR_DB_CONFIG["index_path"],
            meta_path=VECTOR_DB_CONFIG["meta_path"]
        )

        logger.info("Step 3: Building index...")
        logger.info(f"Model: {VECTOR_DB_CONFIG['embedding_model']}")
        logger.info(f"Movies: {len(movies)}")
        vector_db.build_index(movies, M=VECTOR_DB_CONFIG["default_m"], efConstruction=VECTOR_DB_CONFIG["default_ef_construction"])

        logger.info("Step 4: Saving...")
        vector_db.save()

        logger.info("=" * 70)
        logger.info("VECTOR DB CREATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Index: {VECTOR_DB_CONFIG['index_path']}")
        logger.info(f"Metadata: {VECTOR_DB_CONFIG['meta_path']}")
        stats = vector_db.get_stats()
        logger.info(f"Stats: {stats} vectors, dim={stats.get('embedding_dim')}")


def main():
    pipeline = VectorDBIngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
