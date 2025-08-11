# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Setup and Dependencies
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# Create environment file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Development Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Main API endpoint: POST /api/query
- Course analytics: GET /api/courses

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** that enables semantic search and AI-powered Q&A over course materials using ChromaDB and Anthropic's Claude.

### Core Architecture Pattern

The system follows a **multi-tier RAG architecture**:

1. **Document Processing Pipeline**: Structured parsing of course documents into hierarchical Course → Lesson → Chunk relationships
2. **Dual Vector Storage**: ChromaDB with separate collections for course metadata (`course_catalog`) and content chunks (`course_content`)
3. **Tool-Based AI Integration**: Claude uses a search tool to decide when to query the vector store vs. use general knowledge
4. **Session-Aware Conversations**: Maintains conversation history for contextual follow-up questions

### Key Components Interaction

- **RAGSystem** (`rag_system.py`) - Central orchestrator that coordinates all components
- **DocumentProcessor** (`document_processor.py`) - Parses structured course files with metadata extraction and smart text chunking
- **VectorStore** (`vector_store.py`) - ChromaDB wrapper with semantic course name resolution and filtered search
- **AIGenerator** (`ai_generator.py`) - Claude API client with tool execution handling
- **CourseSearchTool** (`search_tools.py`) - Implements the search interface that Claude calls
- **SessionManager** (`session_manager.py`) - Tracks conversation history per user session

### Document Processing Logic

Course documents must follow this structure:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 0: Introduction
[lesson content]

Lesson 1: Next Topic
Lesson Link: [optional lesson url]
[lesson content]
```

The processor creates overlapping text chunks (800 chars, 100 overlap) with contextual prefixes like "Course [title] Lesson [#] content: [chunk]" to improve search relevance.

### Search Flow Architecture

1. **Query Reception**: FastAPI endpoint validates request and manages sessions
2. **AI Decision**: Claude's system prompt guides when to search vs. use general knowledge
3. **Semantic Resolution**: Course names are resolved using vector similarity (e.g., "MCP" → "MCP Introduction Course")
4. **Filtered Search**: ChromaDB queries with course/lesson filters applied
5. **Response Synthesis**: Claude combines search results into natural language answers
6. **Source Tracking**: UI displays which courses/lessons informed the response

### Configuration

Key settings in `backend/config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Overlap between chunks for context preservation
- `MAX_RESULTS: 5` - Maximum search results per query
- `MAX_HISTORY: 2` - Conversation messages to retain for context
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - Sentence transformer model

### Data Models

- `Course`: Contains title, instructor, lessons list
- `Lesson`: lesson_number, title, optional lesson_link  
- `CourseChunk`: content, course_title, lesson_number, chunk_index
- Uses course title as unique identifier across the system

### Development Notes

- The system automatically loads documents from `docs/` folder on startup
- ChromaDB data persists in `./chroma_db` directory
- Conversation sessions are memory-only (not persisted)
- Course deduplication prevents reprocessing existing materials
- Tool execution follows Anthropic's function calling protocol