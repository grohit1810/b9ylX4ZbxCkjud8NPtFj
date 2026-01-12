# Movie Recommendation Chatbot

A conversational AI system that recommends movies by combining structured database queries and semantic search. It uses a ReAct agent powered by Ollama LLM that maintains multi-turn conversations and intelligently decides when to query the movie database or perform semantic search.

## Overview

The system has three core flows:
1. **Data Ingestion**: Load TMDB 5000 dataset into SQL and vector databases
2. **Agent Reasoning**: LLM decides which tools to use based on user queries
3. **API Serving**: FastAPI endpoints manage user sessions and chat conversations

---

## Quick Start

### Prerequisites
- **Python**: 3.11+ required
- **Redis**: `brew install redis` (Mac) or `apt-get install redis` (Linux)
- **Ollama**: [Install from ollama.ai](https://ollama.ai)

### One-Time Setup
```bash
# 1. Clone repo
git clone https://github.com/movie-recommender-agent.git
cd movie-recommender-agent

# 2. Setup Python environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download LLM model
ollama pull llama3.2

# 4. Ingest data (one-time, ~2 minutes)
python -m data_ingestion_scripts.run_data_ingestion
```

### Running the System
**Terminal 1: Start Redis**
```bash
redis-server
```

**Terminal 2: Start Ollama**
```bash
ollama serve
```

**Terminal 3: Start MCP Server**
```bash
python -m mcp_server.server
# Verify: http://localhost:8765/health
```

**Terminal 4: Start API**
```bash
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
# API Docs: http://localhost:8000/docs
```

---

## Usage Examples

### Example 1: Simple Recommendation
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend some action movies"}'
```

**Response**:
```json
{
  "user_id": "09395cc2-9e06-46ee-8a17-7d0080dbf1ee",
  "chat_id": "d81849b1-936f-478b-8167-53f80b0f9ed1",
  "message": "Recommend some action movies",
  "reply": "Here are some highly-rated action movies: 1. Mad Max: Fury Road (2015)...",
  "is_new_user": true,
  "is_new_chat": true
}
```

**What happens**: Agent uses SQL tool to query `genre="Action"`, returns top 10 by rating.

---

### Example 2: Semantic Search
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Movies about time travel and alternate realities"}'
```

**What happens**: Agent uses Vector tool (semantic search) to find movies with similar plot descriptions.

---

### Example 3: Multi-turn Conversation
```bash
# First request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Movies directed by Christopher Nolan"}'

# Returns user_id and chat_id in response

# Follow-up (uses conversation history)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "09395cc2-9e06-46ee-8a17-7d0080dbf1ee",
    "chat_id": "d81849b1-936f-478b-8167-53f80b0f9ed1",
    "message": "Tell me more about the third one"
  }'
```

**What happens**: Agent loads conversation checkpoint, understands "the third one" refers to the 3rd movie from previous response.

---

## API Endpoints

### POST /chat
Chat with the agent

**Request**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "optional-uuid",
    "chat_id": "optional-uuid",
    "message": "Recommend action movies from 2015"
  }'
```

**Response**:
```json
{
  "user_id": "uuid",
  "chat_id": "uuid", 
  "message": "user query",
  "reply": "AI response",
  "is_new_user": false,
  "is_new_chat": false
}
```

---

### POST /chat/history
Get conversation history

**Request**:
```bash
curl -X POST http://localhost:8000/chat/history \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "uuid",
    "chat_id": "uuid"
  }'
```

**Response**:
```json
{
  "user_id": "uuid",
  "chat_id": "uuid",
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

---

### POST /chat/archive
Archive a chat conversation

```bash
curl -X POST http://localhost:8000/chat/archive \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "uuid",
    "chat_id": "uuid"
  }'
```

---

### POST /chat/unarchive
Restore an archived chat

```bash
curl -X POST http://localhost:8000/chat/unarchive \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "uuid",
    "chat_id": "uuid"
  }'
```

---

### GET /health
Health check endpoint

```bash
curl http://localhost:8000/health
```

**Response**: `{"status":"healthy","database":"connected","cache":"connected","agent":"initialized"}`

---

### GET /metrics
Prometheus metrics endpoint

```bash
curl http://localhost:8000/metrics
```
---

## Architecture Deep Dive

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Layer                           │
│                     (HTTP/cURL/Browser)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (Port 8000)              │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  Session Mgmt    │  │  Validation     │  │  Metrics       │  │
│  │  (Redis Cache)   │  │  (Pydantic)     │  │  (Prometheus)  │  │
│  └──────────────────┘  └─────────────────┘  └────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MovieAgent (ReAct Pattern)                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Ollama LLM (llama3.2)                       │   │
│  │              Model Context Protocol Tools                │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LangGraph State Manager (AsyncSqliteSaver)              │   │
│  │  • Conversation checkpoints                              │   │
│  │  • Multi-turn context                                    │   │
│  │  Storage: data/application/llm_memory.db                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (Port 8765)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server (Tool Provider)                   │
│  ┌─────────────────────┐          ┌────────────────────────┐    │
│  │   query_sql_db      │          │  query_vector_db       │    │
│  │   • Filters         │          │  • Semantic search     │    │
│  │   • Pagination      │          │  • Embeddings          │    │
│  │   • LRU Cache       │          │  • LRU Cache           │    │
│  └─────────┬───────────┘          └────────┬───────────────┘    │
│            │                               │                    │
│  ┌─────────▼───────────────────────────────▼────────────────┐   │
│  │           LangChain Middleware Pipeline                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────┬────────────────────────────┬──────────────────────┘
              │                            │
              ▼                            ▼
┌──────────────────────┐      ┌─────────────────────────────────┐
│   SQLite Database    │      │      FAISS Vector Index         │
│   movies.db          │      │  movie_vectors_hnsw.faiss       │
│                      │      │  movie_vectors_meta.pkl         │
│  • 4,800 movies      │      │                                 │
│  • Structured data   │      │  • 384-dim embeddings           │
│  • JSON fields       │      │  • HNSW algorithm               │
│  • Indexed queries   │      │  • Plot descriptions            │
└──────────────────────┘      └─────────────────────────────────┘
```

---

### Data Layer

#### SQLite Database Schema
**Table: movies**
```sql
CREATE TABLE movies (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  year INTEGER,
  director TEXT,
  overview TEXT,
  rating FLOAT,
  genres TEXT, -- JSON array: ["Action", "Thriller"]
  cast TEXT, -- JSON array: ["Tom Cruise", ...]
  crew TEXT, -- JSON array: crew names
  keywords TEXT, -- JSON array: ["space", "culture clash", ...]
  production TEXT, -- JSON array: ["Disney", "Warner Bros"]
  budget INTEGER,
  revenue INTEGER,
  runtime INTEGER,
  popularity FLOAT,
  vote_count INTEGER,
  release_date TEXT,
  original_language TEXT
)
```

---

#### Vector Index (FAISS)
**Specifications**:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Dataset**: ~4,800 movie plot descriptions
- **Memory**: ~200MB RAM (index + metadata)
- **Query Time**: <50ms for top-K=5

---

### Agent Layer

#### ReAct Agent with LangGraph
The agent uses the **ReAct (Reasoning + Acting)** pattern:
1. **Reason**: Analyze user query, decide which tool(s) to use
2. **Act**: Execute tools via MCP server
3. **Observe**: Process tool outputs
4. **Respond**: Generate natural language response

#### Conversation Memory (LangGraph Checkpoints)
```python
# Checkpoint structure
{
    "thread_id": "chat_id",              # Maps to chat session
    "checkpoint_id": "timestamp",         # Incremental version
    "state": {
        "messages": [                     # Full conversation history
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "tool_calls": [...],              # Track tool usage
        "metadata": {...}
    }
}
```

**Storage**: `data/application/llm_memory.db` (SQLite)

**Multi-turn Flow**:
1. User sends message with `chat_id`
2. Agent loads checkpoint from `llm_memory.db`
3. Full conversation history injected into LLM context
4. Agent can reference: *"Earlier you asked about Nolan films..."*
5. New checkpoint saved after each turn

---

### Tool Layer (MCP Server)

The MCP (Model Context Protocol) server exposes two tools:

#### Tool 1: `query_sql_db`
**Use case**: Structured queries with filters
```python
async def query_sql_db(
    genre: Annotated[Optional[str], Field(title="genre(s)", description="Movie genre(s) (e.g., 'Action', 'Drama')")] = None,
    year: Annotated[Optional[int], Field(title="year", description="Specific year")] = None,
    year_min: Annotated[Optional[int], Field(title="year_min", description="Minimum release year")] = None,
    year_max: Annotated[Optional[int], Field(title="year_max", description="Maximum release year")] = None,
    cast: Annotated[Optional[str], Field(title="cast(s)", description="Actor/actress name(s)")] = None,
    director: Annotated[Optional[str], Field(title="director", description="Director name")] = None,
    title: Annotated[Optional[str], Field(title="title", description="Movie title (exact or partial)")] = None,
    limit: Annotated[int, Field(title="limit", description="Number of results (default: 10)")] = MCP_SQL_TOOL_CONFIG.get("default_sql_limit", 10),
    offset: Annotated[int, Field(title="offset", description="Pagination offset (default: 0)")] = MCP_SQL_TOOL_CONFIG.get("default_offset", 0),
    order_by: Annotated[Optional[OrderBy], Field(title="order_by", description="Sort field (default: rating)") ] = MCP_SQL_TOOL_CONFIG.get("default_order_by", "rating"),
    order_dir: Annotated[Literal["ASC", "DESC"], Field(title="order_dir", description="Sort direction (default: DESC)")] = MCP_SQL_TOOL_CONFIG.get("default_order_dir", "DESC"),
    response_format: Annotated[Optional[Literal["summary", "detailed"]], Field(title="response_format", description="Response format type")] = MCP_SQL_TOOL_CONFIG.get("default_response_format", "summary"),
) -> CallToolResult
    """
    Returns: Movies matching filters, ranked by rating
    """
```

---

#### Tool 2: `query_vector_db`
**Use case**: Semantic/thematic queries
```python
async def query_vector_db(
    query_text: Annotated[str, Field(title="query_text", description="Scene description or theme (e.g., 'someone comes back to life')")],
    limit: Annotated[int, Field(title="limit", description="Number of results (default: 5)")] = MCP_VECTOR_TOOL_CONFIG.get("default_top_k", 5),
) -> CallToolResult
    """
    1. Embed query_text using MiniLM-L6-v2
    2. Search FAISS HNSW index
    3. Return top-K movies with similarity scores
    """
```

---

### Caching Strategy

#### Redis (Session Validation)
**Purpose**: Fast validation of `(user_id, chat_id)` ownership without DB hits


#### LRU Caches (Query Results)
**Per-tool caches**:
- `sql_query_cache`: 128 entries (SQL queries) - can be configured in config
- `vector_query_cache`: 128 entries (semantic searches) - can be configured in config


### API Layer (FastAPI)

#### Request Validation
**Pydantic Models**:
```python
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    user_id: Optional[str] = Field(
        None,
        description="User identifier. If not provided, a new user will be created.",
    )
    chat_id: Optional[str] = Field(
        None,
        description="Chat session identifier. If not provided, a new chat will be created.",
    )
    message: str = Field(
        ...,
        description="User's movie-related query"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Chat session identifier")
    message: str = Field(..., description="User's original message")
    reply: str = Field(..., description="Agent's response")
    is_new_user: bool = Field(..., description="Whether a new user was created")
    is_new_chat: bool = Field(..., description="Whether a new chat was created")
```

**Validation Flow**:
1. Pydantic validates request schema
2. Redis checks session ownership
3. FastAPI passes validated data to agent
4. Agent returns structured response

---

#### Why MCP Server Architecture?
- **Decoupling**: Agent logic separate from data access
- **Tool reusability**: MCP tools can be used by other agents
- **Independent scaling**: Scale MCP server separately from API
- **Testability**: Mock MCP responses for agent testing

---

## Data Ingestion

### Flow Diagram
```
TMDB CSV files (movies, credits)
        ↓
Parse JSON fields (genres, cast, crew, keywords)
        ↓
Split into two parallel paths:
├─ SQL Path: Create SQLite DB with movie metadata
└─ Vector Path: Embed movie descriptions, build FAISS index
```

### Configuration
Edit `config/settings.py`:
- **CSV locations**: `MOVIES_CSV`, `CREDITS_CSV`
- **Database paths**: `SQLITE_DB`, `VECTOR_DB_PATH`
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`

### Run Ingestion
```bash
python -m data_ingestion_scripts.run_data_ingestion
```

**Output**:
- `data/processed/movies.db` - SQLite database
- `data/processed/movie_vectors_hnsw.faiss` - Vector index
- `data/processed/movie_vectors_meta.pkl` - Movie metadata for vectors

**Duration**: ~2 minutes on standard hardware


---

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v --cov=. --cov-report=html
```

### Run Specific Test Suites
```bash
# Data ingestion only
python -m pytest tests/test_ingestion.py -v

# MCP server tools
python -m pytest tests/test_mcp_server.py -v

# Movie Agent 
python -m pytest tests/test_movie_agent.py -v

# API endpoints
python -m pytest tests/test_api.py -v
```

---

## Troubleshooting

### Redis connection errors
```bash
# Check if Redis is running
redis-cli ping  # Should return "PONG"

# If not running
redis-server --daemonize yes
```

### Ollama model not found
```bash
# Verify model is downloaded
ollama list

# Re-download if missing
ollama pull llama3.2
```

### Port already in use
```bash
# Find process using port 8000
lsof -i :8000
kill -9 <PID>

# Or change port in config/settings.py
```

### Data ingestion fails
- Ensure CSV files exist in `data/raw/`
- Check file paths in `config/settings.py`
- Verify `data/processed/` directory is writable

---

## Summary

**Architecture Highlights**:
- Clean separation: Data → Tools → Agent → API
- Hybrid retrieval: SQL (structured) + Vector (semantic)
- Stateful conversations: LangGraph checkpoints enable multi-turn context

**Key Innovations**:
- ReAct agent intelligently selects tools based on query intent
- MCP protocol enables tool reusability across agents
- Dual caching strategy: Redis (sessions) + LRU (queries)
- Full conversation history preserved via LangGraph state

**Tech Stack**:
- **LLM**: Ollama (llama3.2)
- **Framework**: FastAPI, LangChain, LangGraph
- **Databases**: SQLite, FAISS
- **Caching**: Redis
- **Testing**: pytest
