"""
Pytest configuration and shared fixtures for RAG system tests
"""
import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Add the backend directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from config import Config


@pytest.fixture
def test_config():
    """Create a test configuration with proper MAX_RESULTS setting"""
    config = Config()
    # Override the problematic MAX_RESULTS setting
    config.MAX_RESULTS = 5
    config.CHROMA_PATH = tempfile.mkdtemp()  # Use temp directory for tests
    return config


@pytest.fixture
def temp_chroma_path():
    """Create and cleanup temporary ChromaDB path"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to RAG", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Vector Databases", lesson_link="https://example.com/lesson2"),
        Lesson(lesson_number=3, title="Advanced Techniques", lesson_link=None)
    ]
    
    return Course(
        title="RAG System Fundamentals",
        instructor="Dr. AI Expert",
        course_link="https://example.com/course",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is an introduction to RAG systems. RAG stands for Retrieval-Augmented Generation.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Vector databases are essential for semantic search in RAG systems.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced RAG techniques include query expansion and result reranking.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Configure default return values
    mock_store.search.return_value = SearchResults(
        documents=["Sample content about RAG systems"],
        metadata=[{"course_title": "RAG System Fundamentals", "lesson_number": 1}],
        distances=[0.1]
    )
    
    mock_store.get_course_outline.return_value = {
        "title": "RAG System Fundamentals",
        "instructor": "Dr. AI Expert",
        "course_link": "https://example.com/course",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Introduction to RAG", "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 2, "lesson_title": "Vector Databases", "lesson_link": "https://example.com/lesson2"}
        ]
    }
    
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    
    return mock_store


@pytest.fixture
def mock_empty_vector_store():
    """Create a mock vector store that returns empty results"""
    mock_store = Mock(spec=VectorStore)
    
    # Configure to return empty results
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )
    
    mock_store.get_course_outline.return_value = None
    mock_store.get_lesson_link.return_value = None
    
    return mock_store


@pytest.fixture
def mock_error_vector_store():
    """Create a mock vector store that returns errors"""
    mock_store = Mock(spec=VectorStore)
    
    # Configure to return errors
    mock_store.search.return_value = SearchResults.empty("Search error: Vector store unavailable")
    mock_store.get_course_outline.return_value = None
    
    return mock_store


@pytest.fixture
def course_search_tool_with_mock(mock_vector_store):
    """Create CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool_with_mock(mock_vector_store):
    """Create CourseOutlineTool with mock vector store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager_with_mocks(course_search_tool_with_mock, course_outline_tool_with_mock):
    """Create ToolManager with mock tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool_with_mock)
    manager.register_tool(course_outline_tool_with_mock)
    return manager


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful response without tool use
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response")]
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use():
    """Create a mock Anthropic client that uses tools"""
    mock_client = Mock()
    
    # Mock initial response with tool use
    mock_tool_use_response = Mock()
    mock_tool_use_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "test_tool_id"
    mock_tool_block.input = {"query": "test query"}
    
    mock_tool_use_response.content = [mock_tool_block]
    
    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.content = [Mock(text="Response after using tools")]
    
    # Configure the client to return different responses on subsequent calls
    mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
    
    return mock_client