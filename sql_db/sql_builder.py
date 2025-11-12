"""
SQL database builder for movies.
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from logging_config.logger import get_logger
from sql_db.data_processor import DataProcessor

logger = get_logger(__name__)


class SQLiteMovieDB:
    """Build and populate SQLite database with movie data."""

    def __init__(self, db_path: Path):
        """
        Initialize SQLite database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.processor = DataProcessor()
        logger.info(f"Initialized SQLite DB handler: {db_path}")

    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path))
        logger.debug("Database connection established")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            logger.error(f"Error during database operation: {exc_val}")
            if self.conn:
                self.conn.rollback()
        else:
            if self.conn:
                self.conn.commit()
        self.close()

    def create_schema(self):
        """Create movies table schema."""
        logger.info("Creating database schema...")

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            year INTEGER,
            director TEXT,
            overview TEXT,
            rating FLOAT,
            genres TEXT,       -- JSON array: ["Action", "Thriller"]
            cast TEXT,         -- JSON array: ["Tom Cruise", ...]
            crew TEXT,         -- JSON array: crew names
            keywords TEXT,     -- JSON array: ["space", "culture clash", ...]
            production TEXT,   -- JSON array: ["Disney", "Warner Bros"]
            budget INTEGER,
            revenue INTEGER,
            runtime INTEGER,
            popularity FLOAT,
            vote_count INTEGER,
            release_date TEXT,
            original_language TEXT
        )
        """
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute(create_table_sql)
            self.conn.commit()

            logger.info("Schema created successfully")

    def insert_movie(self, movie_data: Dict[str, Any]):
        """
        Insert a single movie record.

        Args:
            movie_data: Dictionary with movie information
        """
        insert_sql = """
        INSERT OR REPLACE INTO movies (
            id, title, year, director, overview, rating,
            genres, cast, crew, keywords, production,
            budget, revenue, runtime, popularity, vote_count,
            release_date, original_language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            movie_data.get('id'),
            movie_data.get('title'),
            movie_data.get('year'),
            movie_data.get('director'),
            movie_data.get('overview'),
            movie_data.get('rating'),
            movie_data.get('genres'),
            movie_data.get('cast'),
            movie_data.get('crew'),
            movie_data.get('keywords'),
            movie_data.get('production'),
            movie_data.get('budget'),
            movie_data.get('revenue'),
            movie_data.get('runtime'),
            movie_data.get('popularity'),
            movie_data.get('vote_count'),
            movie_data.get('release_date'),
            movie_data.get('original_language')
        )
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute(insert_sql, values)

    def bulk_insert_movies(self, movies_list: List[Dict[str, Any]]):
        """
        Bulk insert multiple movies.

        Args:
            movies_list: List of movie dictionaries
        """
        logger.info(f"Bulk inserting {len(movies_list)} movies...")

        for i, movie in enumerate(movies_list, 1):
            self.insert_movie(movie)

            if i % 100 == 0:
                logger.debug(f"Inserted {i}/{len(movies_list)} movies")
                if self.conn:
                    self.conn.commit()

        if self.conn:
            self.conn.commit()
        logger.info(f"Successfully inserted {len(movies_list)} movies")

    def get_movie_count(self) -> int:
        """Get total number of movies in database."""
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM movies")
            count = cursor.fetchone()[0]
            return count
        return 0

    def verify_data(self):
        """Verify inserted data."""
        count = self.get_movie_count()
        logger.info(f"Total movies in database: {count}")

        # Sample query
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, title, year, director FROM movies LIMIT 5")
            sample = cursor.fetchall()

            logger.debug("Sample records:")
            for record in sample:
                logger.debug(f"  {record}")
