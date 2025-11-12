"""
Main data ingestion pipeline.

Orchestrates reading CSV files and populating SQL database.
"""
import json
import pandas as pd
from typing import Dict, Any

from logging_config.logger import get_logger
from sql_db.csv_reader import TMDBReader
from sql_db.sql_builder import SQLiteMovieDB
from sql_db.data_processor import DataProcessor
from config.settings import MOVIES_CSV, CREDITS_CSV, SQLITE_DB

logger = get_logger(__name__)


class IngestionPipeline:
    """Main data ingestion pipeline."""

    def __init__(self):
        """Initialize ingestion pipeline."""
        self.reader = TMDBReader(MOVIES_CSV, CREDITS_CSV)
        self.processor = DataProcessor()
        logger.info("Ingestion pipeline initialized")

    def process_movie_record(
        self, 
        movie_row: pd.Series, 
        credits_row: pd.Series
    ) -> Dict[str, Any]:
        """
        Process a single movie record by combining movies and credits data.

        Args:
            movie_row: Row from movies DataFrame
            credits_row: Row from credits DataFrame (matched by ID)

        Returns:
            Processed movie dictionary
        """
        # Extract basic information
        movie_id = int(movie_row['id'])
        title = movie_row['title']
        overview = movie_row.get('overview', '')

        # Extract year from release_date
        year = self.processor.extract_year_from_date(
            movie_row.get('release_date', '')
        )

        # Extract and process genres
        genres_list = self.processor.extract_genres(
            movie_row.get('genres', '[]')
        )
        genres_json = json.dumps(genres_list)

        # Extract and process keywords
        keywords_list = self.processor.extract_keywords(
            movie_row.get('keywords', '[]')
        )
        keywords_json = json.dumps(keywords_list)

        # Extract and process production companies
        production_list = self.processor.extract_production_companies(
            movie_row.get('production_companies', '[]')
        )
        production_json = json.dumps(production_list)

        # Process credits data
        cast_json_str = credits_row.get('cast', '[]') if credits_row is not None else '[]'
        crew_json_str = credits_row.get('crew', '[]') if credits_row is not None else '[]'

        # Extract director from crew
        director = self.processor.extract_director(crew_json_str)

        # Extract cast names
        cast_names = self.processor.extract_cast_names(cast_json_str)
        cast_json = json.dumps(cast_names)

        # Extract crew names
        crew_names = self.processor.extract_crew_names(crew_json_str)
        crew_json = json.dumps(crew_names)

        # Build movie dict
        movie_dict = {
            'id': movie_id,
            'title': title,
            'year': year,
            'director': director,
            'overview': overview,
            'rating': movie_row.get('vote_average'),
            'genres': genres_json,
            'cast': cast_json,
            'crew': crew_json,
            'keywords': keywords_json,
            'production': production_json,
            'budget': int(movie_row.get('budget', 0)),
            'revenue': int(movie_row.get('revenue', 0)),
            'runtime': movie_row.get('runtime'),
            'popularity': movie_row.get('popularity'),
            'vote_count': int(movie_row.get('vote_count', 0)),
            'release_date': movie_row.get('release_date'),
            'original_language': movie_row.get('original_language')
        }

        return movie_dict

    def run(self):
        """
        Execute complete ingestion pipeline.

        Steps:
            1. Read CSV files
            2. Merge movies and credits by ID
            3. Process and transform data
            4. Create SQL database
            5. Populate database
        """
        logger.info("="*70)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("="*70)

        # Step 1: Read CSV files
        logger.info("Step 1: Reading CSV files...")
        movies_df, credits_df = self.reader.read_all()
        self.reader.validate_datasets(movies_df, credits_df)

        # Step 2: Merge datasets
        logger.info("Step 2: Merging datasets...")
        # Rename 'id' in credits to 'movie_id' for clarity
        credits_df = credits_df.rename(columns={'movie_id': 'id'})

        # Merge on 'id'
        merged_df = movies_df.merge(
            credits_df[['id', 'cast', 'crew']], 
            on='id', 
            how='left'
        )
        logger.info(f"Merged dataset: {len(merged_df)} records")

        # Step 3: Process all records
        logger.info("Step 3: Processing movie records...")
        processed_movies = []

        for idx, (_, row) in enumerate(merged_df.iterrows(), start=1):
            try:
                # Create a mock credits row from merged data
                credits_data = pd.Series({
                    'cast': row.get('cast', '[]'),
                    'crew': row.get('crew', '[]')
                })

                movie_dict = self.process_movie_record(row, credits_data)
                processed_movies.append(movie_dict)

                if (idx + 1) % 500 == 0:
                    logger.info(f"Processed {idx + 1}/{len(merged_df)} movies")

            except Exception as e:
                logger.error(f"Error processing movie ID {row.get('id')}: {str(e)}")
                continue

        logger.info(f"Processed {len(processed_movies)} movies successfully")

        # Step 4: Create and populate database
        logger.info("Step 4: Creating SQL database...")
        with SQLiteMovieDB(SQLITE_DB) as db:
            db.create_schema()
            db.bulk_insert_movies(processed_movies)
            db.verify_data()

        logger.info("" + "="*70)
        logger.info("DATA INGESTION COMPLETE")
        logger.info(f"Database created at: {SQLITE_DB}")
        logger.info("="*70)


def main():
    """Main entry point for ingestion."""
    pipeline = IngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
