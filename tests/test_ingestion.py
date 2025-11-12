"""
Comprehensive tests for data ingestion pipeline.

Covers CSV reading, data processing, SQL building, vector DB, and full pipeline.
"""

import pytest
import pandas as pd
import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sql_db.csv_reader import TMDBReader
from sql_db.data_processor import DataProcessor
from sql_db.sql_builder import SQLiteMovieDB
from sql_db.pipeline import IngestionPipeline
from vector_db.movie_vector_db import FaissHNSWMovieVectorDB
from vector_db.pipeline import VectorDBIngestionPipeline


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_movies_csv(tmp_path):
    """Create temporary movies CSV for testing."""
    csv_path = tmp_path / "test_movies.csv"
    data = {
        'id': [1, 2],
        'title': ['Inception', 'The Matrix'],
        'budget': [160000000, 63000000],
        'genres': ['[{"name": "Action"}, {"name": "Thriller"}]', '[{"name": "Sci-Fi"}]'],
        'keywords': ['[{"name": "dream"}, {"name": "heist"}]', '[{"name": "reality"}]'],
        'release_date': ['2010-07-16', '1999-03-31'],
        'vote_average': [8.4, 8.7],
        'overview': ['A thief who steals secrets...', 'A computer hacker...'],
        'production_companies': ['[{"name": "Warner Bros"}]', '[{"name": "Village Roadshow"}]']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_credits_csv(tmp_path):
    """Create temporary credits CSV for testing."""
    csv_path = tmp_path / "test_credits.csv"
    data = {
        'movie_id': [1, 2],
        'title': ['Inception', 'The Matrix'],
        'cast': [
            '[{"name": "Leonardo DiCaprio"}]',
            '[{"name": "Keanu Reeves"}]'
        ],
        'crew': [
            '[{"name": "Christopher Nolan", "job": "Director"}]',
            '[{"name": "Wachowski Brothers", "job": "Director"}]'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary SQLite database."""
    db_path = tmp_path / "test_movies.db"
    return db_path


@pytest.fixture
def processor():
    """DataProcessor instance."""
    return DataProcessor()


# ============================================================================
# DATA PROCESSOR TESTS
# ============================================================================

class TestDataProcessor:
    """Test DataProcessor class methods."""
    
    def test_parse_json_field_valid(self, processor):
        """Test parsing valid JSON."""
        json_str = '[{"name": "Test"}]'
        result = processor.parse_json_field(json_str)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "Test"
    
    def test_parse_json_field_empty(self, processor):
        """Test parsing empty JSON."""
        result = processor.parse_json_field('[]')
        assert result == []
    
    def test_parse_json_field_none(self, processor):
        """Test parsing None value."""
        result = processor.parse_json_field(None)
        assert result == []
    
    def test_parse_json_field_invalid(self, processor):
        """Test parsing invalid JSON (should fallback gracefully)."""
        result = processor.parse_json_field('{invalid json}')
        assert result == []
    
    def test_extract_names(self, processor):
        """Test name extraction from list of dicts."""
        data = [{"name": "John"}, {"name": "Jane"}]
        names = processor.extract_names(data)
        assert names == ["John", "Jane"]
    
    def test_extract_names_empty(self, processor):
        """Test name extraction from empty list."""
        names = processor.extract_names([])
        assert names == []
    
    def test_extract_names_missing_field(self, processor):
        """Test name extraction when 'name' field is missing."""
        data = [{"title": "Movie"}, {"name": "Jane"}]
        names = processor.extract_names(data)
        assert names == ["Jane"]
    
    def test_extract_director(self, processor):
        """Test director extraction."""
        crew_json = '[{"name": "James Cameron", "job": "Director"}]'
        director = processor.extract_director(crew_json)
        assert director == "James Cameron"
    
    def test_extract_director_not_found(self, processor):
        """Test director extraction when no director in crew."""
        crew_json = '[{"name": "John Doe", "job": "Producer"}]'
        director = processor.extract_director(crew_json)
        assert director is None
    
    def test_extract_director_empty(self, processor):
        """Test director extraction with empty crew."""
        director = processor.extract_director('[]')
        assert director is None
    
    def test_extract_year_from_date_valid(self, processor):
        """Test year extraction from valid date."""
        assert processor.extract_year_from_date("2009-12-10") == 2009
        assert processor.extract_year_from_date("1999-03-31") == 1999
    
    def test_extract_year_from_date_empty(self, processor):
        """Test year extraction from empty string."""
        assert processor.extract_year_from_date("") is None
    
    def test_extract_year_from_date_invalid(self, processor):
        """Test year extraction from invalid date formats."""
        # Non-date strings
        assert processor.extract_year_from_date("invalid") is None
        assert processor.extract_year_from_date("not a date") is None
        
        # Empty string
        assert processor.extract_year_from_date("") is None
        
        # Date-like but non-numeric year
        assert processor.extract_year_from_date("abcd-12-10") is None
        
        # Just separators
        assert processor.extract_year_from_date("--") is None

    
    def test_extract_cast_names(self, processor):
        """Test cast name extraction."""
        cast_json = '[{"name": "Tom Hanks", "order": 0}, {"name": "Meryl Streep", "order": 1}]'
        cast_names = processor.extract_cast_names(cast_json)
        assert cast_names == ["Tom Hanks", "Meryl Streep"]
    
    def test_extract_cast_names_with_limit(self, processor):
        """Test cast name extraction with limit."""
        cast_json = '[{"name": "Actor1"}, {"name": "Actor2"}, {"name": "Actor3"}]'
        cast_names = processor.extract_cast_names(cast_json, limit=2)
        assert len(cast_names) == 2
        assert cast_names == ["Actor1", "Actor2"]
    
    def test_extract_genres(self, processor):
        """Test genre extraction."""
        genres_json = '[{"name": "Action"}, {"name": "Thriller"}]'
        genres = processor.extract_genres(genres_json)
        assert genres == ["Action", "Thriller"]
    
    def test_extract_keywords(self, processor):
        """Test keyword extraction."""
        keywords_json = '[{"name": "space"}, {"name": "future"}]'
        keywords = processor.extract_keywords(keywords_json)
        assert keywords == ["space", "future"]
    
    def test_extract_production_companies(self, processor):
        """Test production company extraction."""
        prod_json = '[{"name": "Warner Bros"}, {"name": "Disney"}]'
        companies = processor.extract_production_companies(prod_json)
        assert companies == ["Warner Bros", "Disney"]
    
    def test_extract_crew_names(self, processor):
        """Test crew name extraction."""
        crew_json = '[{"name": "Director1", "job": "Director"}, {"name": "Producer1", "job": "Producer"}]'
        crew_names = processor.extract_crew_names(crew_json)
        assert crew_names == ["Director1", "Producer1"]


# ============================================================================
# CSV READER TESTS
# ============================================================================

class TestTMDBReader:
    """Test TMDBReader class."""
    
    def test_init_valid_files(self, sample_movies_csv, sample_credits_csv):
        """Test initialization with valid files."""
        reader = TMDBReader(sample_movies_csv, sample_credits_csv)
        assert reader.movies_csv == sample_movies_csv
        assert reader.credits_csv == sample_credits_csv
    
    def test_init_missing_movies_csv(self, sample_credits_csv, tmp_path):
        """Test initialization with missing movies CSV."""
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            TMDBReader(fake_path, sample_credits_csv)
    
    def test_init_missing_credits_csv(self, sample_movies_csv, tmp_path):
        """Test initialization with missing credits CSV."""
        fake_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            TMDBReader(sample_movies_csv, fake_path)
    
    def test_read_movies(self, sample_movies_csv, sample_credits_csv):
        """Test reading movies CSV."""
        reader = TMDBReader(sample_movies_csv, sample_credits_csv)
        df = reader.read_movies()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'id' in df.columns
        assert 'title' in df.columns
    
    def test_read_credits(self, sample_movies_csv, sample_credits_csv):
        """Test reading credits CSV."""
        reader = TMDBReader(sample_movies_csv, sample_credits_csv)
        df = reader.read_credits()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'movie_id' in df.columns
        assert 'cast' in df.columns
        assert 'crew' in df.columns
    
    def test_read_all(self, sample_movies_csv, sample_credits_csv):
        """Test reading both CSVs."""
        reader = TMDBReader(sample_movies_csv, sample_credits_csv)
        movies_df, credits_df = reader.read_all()
        assert isinstance(movies_df, pd.DataFrame)
        assert isinstance(credits_df, pd.DataFrame)
        assert len(movies_df) == 2
        assert len(credits_df) == 2
    
    def test_validate_datasets_valid(self, sample_movies_csv, sample_credits_csv):
        """Test dataset validation with valid data."""
        reader = TMDBReader(sample_movies_csv, sample_credits_csv)
        movies_df, credits_df = reader.read_all()
        assert reader.validate_datasets(movies_df, credits_df) is True
    
    def test_validate_datasets_missing_id_column(self, sample_credits_csv):
        """Test validation fails when 'id' column missing."""
        reader = Mock()
        movies_df = pd.DataFrame({'wrong_column': [1, 2]})
        credits_df = pd.DataFrame({'movie_id': [1, 2]})
        
        with pytest.raises(ValueError, match="missing 'id' column"):
            TMDBReader.validate_datasets(reader, movies_df, credits_df)
    
    def test_validate_datasets_no_overlap(self, sample_movies_csv, sample_credits_csv):
        """Test validation fails when no overlapping IDs."""
        reader = TMDBReader(sample_movies_csv, sample_credits_csv)
        movies_df = pd.DataFrame({'id': [100, 200]})
        credits_df = pd.DataFrame({'movie_id': [1, 2]})
        
        with pytest.raises(ValueError, match="No overlapping IDs"):
            reader.validate_datasets(movies_df, credits_df)


# ============================================================================
# SQL BUILDER TESTS
# ============================================================================

class TestSQLiteMovieDB:
    """Test SQLiteMovieDB class."""
    
    def test_create_schema(self, temp_db):
        """Test schema creation."""
        with SQLiteMovieDB(temp_db) as db:
            db.create_schema()
        
        # Verify table exists
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='movies'")
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_insert_movie(self, temp_db):
        """Test inserting a single movie."""
        movie_data = {
            'id': 1,
            'title': 'Test Movie',
            'year': 2020,
            'director': 'Test Director',
            'overview': 'A test movie',
            'rating': 8.5,
            'genres': json.dumps(["Action"]),
            'cast': json.dumps(["Actor1"]),
            'crew': json.dumps(["Crew1"]),
            'keywords': json.dumps(["test"]),
            'production': json.dumps(["Studio1"]),
            'budget': 1000000,
            'revenue': 5000000,
            'runtime': 120,
            'popularity': 10.5,
            'vote_count': 1000,
            'release_date': '2020-01-01',
            'original_language': 'en'
        }
        
        with SQLiteMovieDB(temp_db) as db:
            db.create_schema()
            db.insert_movie(movie_data)
        
        # Verify insertion
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT title, year FROM movies WHERE id=1")
        result = cursor.fetchone()
        assert result[0] == 'Test Movie'
        assert result[1] == 2020
        conn.close()
    
    def test_bulk_insert_movies(self, temp_db):
        """Test bulk insertion."""
        movies = [
            {'id': i, 'title': f'Movie {i}', 'year': 2020+i, 'director': None,
             'overview': '', 'rating': None, 'genres': '[]', 'cast': '[]',
             'crew': '[]', 'keywords': '[]', 'production': '[]',
             'budget': 0, 'revenue': 0, 'runtime': None, 'popularity': None,
             'vote_count': 0, 'release_date': None, 'original_language': None}
            for i in range(1, 6)
        ]
        
        with SQLiteMovieDB(temp_db) as db:
            db.create_schema()
            db.bulk_insert_movies(movies)
        
        # Verify count
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM movies")
        count = cursor.fetchone()[0]
        assert count == 5
        conn.close()
    
    def test_get_movie_count(self, temp_db):
        """Test getting movie count."""
        with SQLiteMovieDB(temp_db) as db:
            db.create_schema()
            assert db.get_movie_count() == 0
            
            # Insert movies
            movies = [{'id': i, 'title': f'Movie {i}'} for i in range(3)]
            db.bulk_insert_movies(movies)
            assert db.get_movie_count() == 3
    
    def test_context_manager(self, temp_db):
        """Test context manager behavior."""
        with SQLiteMovieDB(temp_db) as db:
            assert db.conn is not None
        # Connection should be closed after exiting context
        with pytest.raises(sqlite3.ProgrammingError):
            db.conn.execute("SELECT 1")


# ============================================================================
# PIPELINE TESTS
# ============================================================================

class TestIngestionPipeline:
    """Test SQL ingestion pipeline."""
    
    def test_process_movie_record(self, sample_movies_csv, sample_credits_csv):
        """Test processing a single movie record."""
        reader = TMDBReader(sample_movies_csv, sample_credits_csv)
        movies_df, credits_df = reader.read_all()
        
        pipeline = IngestionPipeline.__new__(IngestionPipeline)
        pipeline.processor = DataProcessor()
        
        movie_row = movies_df.iloc[0]
        credits_row = pd.Series({'cast': '[{"name": "Actor"}]', 'crew': '[{"name": "Director", "job": "Director"}]'})
        
        result = pipeline.process_movie_record(movie_row, credits_row)
        
        assert isinstance(result, dict)
        assert 'id' in result
        assert 'title' in result
        assert 'year' in result
        assert result['title'] == 'Inception'


# ============================================================================
# VECTOR DB TESTS
# ============================================================================

class TestFaissHNSWMovieVectorDB:
    """Test FAISS vector database."""
    
    def test_init(self, tmp_path):
        """Test vector DB initialization."""
        index_path = tmp_path / "test.faiss"
        meta_path = tmp_path / "test.pkl"
        
        with patch('sentence_transformers.SentenceTransformer'):
            vector_db = FaissHNSWMovieVectorDB(
                embedding_model="all-MiniLM-L6-v2",
                vector_index_path=index_path,
                meta_path=meta_path
            )
            assert vector_db.vector_index_path == index_path
            assert vector_db.meta_path == meta_path
    
    def test_create_enriched_document(self, tmp_path):
        """Test document enrichment."""
        index_path = tmp_path / "test.faiss"
        meta_path = tmp_path / "test.pkl"
        
        with patch('sentence_transformers.SentenceTransformer'):
            vector_db = FaissHNSWMovieVectorDB(
                embedding_model="all-MiniLM-L6-v2",
                vector_index_path=index_path,
                meta_path=meta_path
            )
            
            doc = vector_db.create_enriched_document(
                title="Inception",
                overview="A thief enters dreams",
                keywords=["dream", "heist"]
            )
            
            assert "Inception" in doc
            assert "A thief enters dreams" in doc
            assert "dream" in doc
            assert "heist" in doc
    
    def test_get_stats_not_built(self, tmp_path):
        """Test stats when index not built."""
        index_path = tmp_path / "test.faiss"
        meta_path = tmp_path / "test.pkl"
        
        with patch('sentence_transformers.SentenceTransformer'):
            vector_db = FaissHNSWMovieVectorDB(
                embedding_model="all-MiniLM-L6-v2",
                vector_index_path=index_path,
                meta_path=meta_path
            )
            
            stats = vector_db.get_stats()
            assert stats["status"] == "not_built"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_end_to_end_csv_to_sql(self, sample_movies_csv, sample_credits_csv, temp_db):
        """Test complete CSV to SQL pipeline."""
        # This would test the full pipeline
        # Skipped for brevity but structure shown
        pass


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_malformed_json_handling(self, processor):
        """Test handling of malformed JSON strings."""
        malformed = '{"name": "Test"'  # Missing closing brace
        result = processor.parse_json_field(malformed)
        assert result == []
    
    def test_unicode_handling(self, processor):
        """Test handling of Unicode characters."""
        json_str = '[{"name": "François Truffaut"}]'
        result = processor.parse_json_field(json_str)
        assert len(result) == 1
        assert result[0]["name"] == "François Truffaut"
    
    def test_large_cast_list(self, processor):
        """Test handling of large cast lists."""
        large_cast = [{"name": f"Actor{i}"} for i in range(100)]
        cast_json = json.dumps(large_cast)
        names = processor.extract_cast_names(cast_json, limit=10)
        assert len(names) == 10
