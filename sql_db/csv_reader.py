"""
CSV file reader for TMDB dataset.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple
from logging_config.logger import get_logger

logger = get_logger(__name__)


class TMDBReader:
    """Read and validate TMDB CSV files."""

    def __init__(self, movies_csv: Path, credits_csv: Path):
        """
        Initialize TMDB reader.

        Args:
            movies_csv: Path to tmdb_5000_movies.csv
            credits_csv: Path to tmdb_5000_credits.csv
        """
        self.movies_csv = movies_csv
        self.credits_csv = credits_csv

        # Validate files exist
        if not movies_csv.exists():
            raise FileNotFoundError(f"Movies CSV not found: {movies_csv}")
        if not credits_csv.exists():
            raise FileNotFoundError(f"Credits CSV not found: {credits_csv}")

        logger.info(f"Initialized TMDB reader")
        logger.debug(f"Movies CSV: {movies_csv}")
        logger.debug(f"Credits CSV: {credits_csv}")

    def read_movies(self) -> pd.DataFrame:
        """
        Read tmdb_5000_movies.csv.

        Returns:
            DataFrame with movies data

        Columns:
            - budget, genres, homepage, id, keywords, original_language,
              original_title, overview, popularity, production_companies,
              production_countries, release_date, revenue, runtime,
              spoken_languages, status, tagline, title, vote_average, vote_count
        """
        logger.info(f"Reading movies CSV from: {self.movies_csv}")

        try:
            df = pd.read_csv(self.movies_csv)
            logger.info(f"Loaded {len(df)} movies")
            logger.debug(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to read movies CSV: {str(e)}")
            raise

    def read_credits(self) -> pd.DataFrame:
        """
        Read tmdb_5000_credits.csv.

        Returns:
            DataFrame with credits data

        Columns:
            - movie_id, title, cast, crew
        """
        logger.info(f"Reading credits CSV from: {self.credits_csv}")

        try:
            df = pd.read_csv(self.credits_csv)
            logger.info(f"Loaded {len(df)} movie credits")
            logger.debug(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Failed to read credits CSV: {str(e)}")
            raise

    def read_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read both CSV files.

        Returns:
            Tuple of (movies_df, credits_df)
        """
        movies_df = self.read_movies()
        credits_df = self.read_credits()

        logger.info("Successfully loaded both datasets")
        return movies_df, credits_df

    def validate_datasets(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> bool:
        """
        Validate that datasets are compatible for merging.

        Args:
            movies_df: Movies DataFrame
            credits_df: Credits DataFrame

        Returns:
            True if valid, raises exception otherwise
        """
        # Check if 'id' column exists in movies
        if 'id' not in movies_df.columns:
            raise ValueError("movies_df missing 'id' column")

        # Check if 'movie_id' column exists in credits
        if 'movie_id' not in credits_df.columns:
            raise ValueError("credits_df missing 'movie_id' column")

        # Check for overlapping IDs
        movie_ids = set(movies_df['id'].unique())
        credit_ids = set(credits_df['movie_id'].unique())
        overlap = movie_ids & credit_ids

        logger.info(f"Movie IDs: {len(movie_ids)}")
        logger.info(f"Credit IDs: {len(credit_ids)}")
        logger.info(f"Overlapping IDs: {len(overlap)}")

        if len(overlap) == 0:
            raise ValueError("No overlapping IDs between movies and credits")

        if len(overlap) < min(len(movie_ids), len(credit_ids)) * 0.9:
            logger.warning(f"Low overlap: Only {len(overlap)} common IDs")

        logger.info("Dataset validation passed")
        return True
