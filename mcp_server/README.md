# Movie Recommender MCP Server

A production-grade Model Context Protocol (MCP) server for semantic and structured movie recommendations.

## Overview

This MCP server exposes two main tools:

### 1. `query_sql_db` - Structured Database Queries
Query the movie database with flexible filters:
- Filter by genre, year range, cast, director, title
- Multiple filtering options
- Configurable response format

Example:
```json
{
  "genre": "Action",
  "year_min": 2015,
  "year_max": 2020,
  "director": "Christopher Nolan"
}
```

### 2. `query_vector_db` - Semantic Search
Find movies matching scene descriptions or themes:
- "Movies where someone comes back to life"
- "Films about environmental destruction"
- "Stories with cultural conflicts"

Example:
```json
{
  "query_text": "someone comes back to life",
  "top_k": 5
}
```

## Overall Architecture

```
movie_recommender
├── api
│   ├── __init__.py
│   └── README.md
├── chat_assistant
│   ├── __init__.py
│   └── README.md
├── config
│   └── settings.py
├── data
│   ├── processed
│   │   ├── movie_vectors_hnsw.faiss
│   │   ├── movie_vectors_meta.pkl
│   │   └── movies.db
│   ├── tmdb_5000_credits.csv
│   └── tmdb_5000_movies.csv
├── logging_config
│   ├── __init__.py
│   └── logger.py
├── mcp_server
│   ├── __init__.py
│   ├── movie_recommender_mcp
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── config.py
│   │   ├── database
│   │   │   ├── __init__.py
│   │   │   ├── sqlite_client.py
│   │   │   └── vector_client.py
│   │   ├── handlers
│   │   │   ├── __init__.py
│   │   │   ├── sql_handler.py
│   │   │   └── vector_handler.py
│   │   ├── server copy.py
│   │   ├── server_stdio.py
│   │   ├── server.py
│   │   ├── tools
│   │   │   ├── __init__.py
│   │   │   ├── sql_tool.py
│   │   │   └── vector_tool.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── query_builder.py
│   │       └── validators.py
│   ├── README.md
│   └── requirements.txt
├── README.md
├── requirements.txt
├── scripts
│   ├── run_ingesion.py
│   ├── run_sql_ingestion.py
│   └── run_vector_ingestion.py
├── sql_db
│   ├── __init__.py
│   ├── csv_reader.py
│   ├── data_processor.py
│   ├── pipeline.py
│   └── sql_builder.py
├── tests
│   ├── __init__.py
│   └── test_ingestion.py
└── vector_db
    ├── __init__.py
    ├── movie_vector_db.py
    └── pipeline.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python -m movie_recommender_mcp.server
```

Or with logging:
```bash
VERBOSE=true python -m movie_recommender_mcp.server
```

## Tool Specifications

### query_sql_db

**Parameters:**
- `genre` (string, optional): Movie genre
- `year` (integer, optional): Specific year
- `year_min` (integer, optional): Minimum year
- `year_max` (integer, optional): Maximum year
- `cast` (string, optional): Actor/actress name
- `director` (string, optional): Director name
- `title` (string, optional): Movie title
- `limit` (integer, optional, default=10): Max results
- `response_format` (string, optional): "summary", "detailed", or "info"

**Response Format:**
```json
{
  "success": true,
  "count": 5,
  "movies": [
    {
      "title": "The Dark Knight",
      "release_date": "2008-07-18",
      "director": "Christopher Nolan",
      "cast": ["Christian Bale", "Heath Ledger", "Aaron Eckhart"],
      "rating": 9.0
    }
  ],
  "query_filters": {...}
}
```

### query_vector_db

**Parameters:**
- `query_text` (string, required): Scene description or query
- `top_k` (integer, optional, default=5): Number of results (1-50)

**Response Format:**
```json
{
  "success": true,
  "count": 3,
  "movies": [
    {
      "title": "Interstellar",
      "director": "Christopher Nolan",
      "cast": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
      "release_date": "2014-11-07",
      "keywords": ["space", "time", "love", "survival"],
      "rating": 8.6,
      "similarity_score": 0.85
    }
  ],
  "query": "someone comes back to life",
  "search_type": "semantic"
}
```

## Integration with Client

This server communicates via STDIO using JSON-RPC 2.0 protocol.

See Part 3 for Ollama + Llama3.2 client implementation.
