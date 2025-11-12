"""
MCP Server Integration Tests.

Tests all MCP server endpoints with various parameter combinations.

Run:
    pytest tests/test_mcp_server.py -v
"""

import pytest
from mcp_server.database.sqlite_client import SQLiteClient
from mcp_server.database.vector_client import VectorDBClient
from config.settings import SQLITE_DB, VECTOR_DB_CONFIG


@pytest.fixture(scope="module")
def sql_client():
    """SQLite client fixture."""
    return SQLiteClient(SQLITE_DB)


@pytest.fixture(scope="module")
def vector_client():
    """Vector DB client fixture."""
    return VectorDBClient(
        VECTOR_DB_CONFIG["index_path"],
        VECTOR_DB_CONFIG["meta_path"],
        VECTOR_DB_CONFIG["embedding_model"]
    )


class TestSQLQueries:
    """Test SQL database queries with various parameters."""

    def test_single_genre(self, sql_client):
        """Test query with single genre."""
        results = sql_client.query_movies(
            genre="Romance",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=10,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        assert len(results) <= 10
        if results:
            # Check structure of first result
            assert "title" in results[0]

    def test_multiple_genres_comma_separated(self, sql_client):
        """Test query with multiple genres (comma-separated)."""
        results = sql_client.query_movies(
            genre="Romance, Thriller",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=10,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_year_exact_filter(self, sql_client):
        """Test exact year filter."""
        results = sql_client.query_movies(
            genre=None,
            year=2015,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=10,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        if results:
            assert results[0].get("year") == 2015

    def test_year_range_filter(self, sql_client):
        """Test year range filter."""
        results = sql_client.query_movies(
            genre=None,
            year=None,
            year_min=2010,
            year_max=2020,
            cast=None,
            director=None,
            title=None,
            limit=10,
            offset=0,
            order_by="year",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        if results:
            for movie in results:
                year = movie.get("year")
                if year:
                    assert 2010 <= year <= 2020

    def test_cast_filter(self, sql_client):
        """Test cast member filter."""
        results = sql_client.query_movies(
            genre=None,
            year=None,
            year_min=None,
            year_max=None,
            cast="Tom Hanks",
            director=None,
            title=None,
            limit=10,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        if results:
            # Cast should contain Tom Hanks
            first_result = results[0]
            assert "cast" in first_result

    def test_director_filter(self, sql_client):
        """Test director filter."""
        results = sql_client.query_movies(
            genre=None,
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director="Christopher Nolan",
            title=None,
            limit=10,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        if results:
            assert "director" in results[0]

    def test_title_search(self, sql_client):
        """Test title partial match."""
        results = sql_client.query_movies(
            genre=None,
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title="Inception",
            limit=10,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        if results:
            # Title should contain "Inception"
            assert "Inception" in results[0]["title"]

    def test_combined_filters(self, sql_client):
        """Test multiple filters combined."""
        results = sql_client.query_movies(
            genre="Action",
            year=None,
            year_min=2010,
            year_max=2020,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="rating",
            order_dir="DESC"
        )
        assert isinstance(results, list)
        assert len(results) <= 5

    def test_pagination(self, sql_client):
        """Test pagination with offset."""
        page1 = sql_client.query_movies(
            genre="Action",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        page2 = sql_client.query_movies(
            genre="Action",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=5,
            order_by="title",
            order_dir="ASC"
        )

        assert isinstance(page1, list)
        assert isinstance(page2, list)
        
        # Pages should be different if there are enough results
        if len(page1) == 5 and len(page2) > 0:
            assert page1[0]["title"] != page2[0]["title"]

    def test_order_by_rating(self, sql_client):
        """Test ordering by rating."""
        results = sql_client.query_movies(
            genre="Action",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="rating",
            order_dir="DESC"
        )
        assert isinstance(results, list)
        
        # Check ratings are in descending order
        if len(results) >= 2:
            ratings = [r.get("rating", 0) for r in results]
            assert ratings == sorted(ratings, reverse=True)

    def test_no_results(self, sql_client):
        """Test query with no matching results."""
        results = sql_client.query_movies(
            genre="NonexistentGenre",
            year=1800,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=10,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
        assert len(results) == 0

    def test_limit_constraint(self, sql_client):
        """Test limit parameter is respected."""
        results = sql_client.query_movies(
            genre="Drama",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=3,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert len(results) <= 3


class TestVectorSearch:
    """Test vector-based semantic search."""

    def test_basic_semantic_search(self, vector_client):
        """Test basic semantic search."""
        results = vector_client.search(
            query_text="someone comes back to life",
            top_k=10
        )
        assert isinstance(results, list)
        assert len(results) <= 10
        
        # Results are tuples: (metadata_dict, similarity_score)
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2
            metadata, score = result
            assert isinstance(metadata, dict)
            assert isinstance(score, float)

    def test_theme_search(self, vector_client):
        """Test thematic search."""
        results = vector_client.search(
            query_text="redemption and second chances",
            top_k=10
        )
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_plot_description_search(self, vector_client):
        """Test plot-based search."""
        results = vector_client.search(
            query_text="a heist where things go wrong",
            top_k=10
        )
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_emotional_vibe_search(self, vector_client):
        """Test emotional concept search."""
        results = vector_client.search(
            query_text="dark and gritty crime story",
            top_k=10
        )
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_action_scene_search(self, vector_client):
        """Test action-specific search."""
        results = vector_client.search(
            query_text="intense car chases and explosions",
            top_k=10
        )
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_similarity_score_range(self, vector_client):
        """Test similarity scores are in valid range."""
        results = vector_client.search(
            query_text="space exploration",
            top_k=10
        )
        for metadata, score in results:
            # Similarity scores should be between 0 and 1
            assert 0 <= score <= 1

    def test_different_top_k_values(self, vector_client):
        """Test different top_k values."""
        results_5 = vector_client.search(query_text="love story", top_k=5)
        results_10 = vector_client.search(query_text="love story", top_k=10)
        
        assert len(results_5) <= 5
        assert len(results_10) <= 10

    def test_very_specific_query(self, vector_client):
        """Test highly specific query."""
        results = vector_client.search(
            query_text="artificial intelligence becomes sentient",
            top_k=10
        )
        assert isinstance(results, list)

    def test_emotional_concept(self, vector_client):
        """Test abstract emotional concepts."""
        results = vector_client.search(
            query_text="loss and grief",
            top_k=10
        )
        assert isinstance(results, list)

    def test_genre_theme_combo(self, vector_client):
        """Test genre + theme combination."""
        results = vector_client.search(
            query_text="sci-fi comedy with aliens",
            top_k=10
        )
        assert isinstance(results, list)


class TestCaching:
    """Test caching behavior."""

    def test_sql_cache_hit(self, sql_client):
        """Test SQL query caching."""
        # First query (cache miss)
        results1 = sql_client.query_movies(
            genre="Drama",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        
        # Second identical query (should be cache hit)
        results2 = sql_client.query_movies(
            genre="Drama",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        
        assert results1 == results2

    def test_vector_cache_hit(self, vector_client):
        """Test vector search caching."""
        results1 = vector_client.search("action movie", top_k=5)
        results2 = vector_client.search("action movie", top_k=5)
        
        # Should return identical results
        assert len(results1) == len(results2)
        if results1:
            # Compare first result
            assert results1[0][1] == results2[0][1]  # Same similarity score

    def test_cache_different_params(self, sql_client):
        """Test cache doesn't mix different parameters."""
        results1 = sql_client.query_movies(
            genre="Action",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        results2 = sql_client.query_movies(
            genre="Comedy",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        
        # Different genres should give different results
        if results1 and results2:
            assert results1 != results2

    def test_cache_clear(self, sql_client):
        """Test cache clearing."""
        sql_client.clear_cache()
        results = sql_client.query_movies(
            genre="Comedy",
            year=None,
            year_min=None,
            year_max=None,
            cast=None,
            director=None,
            title=None,
            limit=5,
            offset=0,
            order_by="title",
            order_dir="ASC"
        )
        assert isinstance(results, list)
