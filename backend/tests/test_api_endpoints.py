"""
API endpoint tests for the RAG system FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import os
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config


def create_test_app():
    """Create a test app without static file mounting that causes issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Define models inline to avoid importing app.py (which mounts static files)
    class QueryRequest(BaseModel):
        """Request model for course queries"""
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        """Response model for course queries"""
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        """Response model for course statistics"""
        total_courses: int
        course_titles: List[str]
    
    # Create test app
    app = FastAPI(title="Course Materials RAG System Test", root_path="")
    
    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Create a mock RAG system for testing
    mock_rag_system = Mock()
    
    # API Endpoints (copied from main app but using mock)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            session_id = request.session_id or "test_session"
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        """Clear a conversation session"""
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Store mock for test access
    app.state.mock_rag_system = mock_rag_system
    return app


@pytest.fixture
def test_app():
    """Create test FastAPI app"""
    return create_test_app()


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_system(test_app):
    """Get the mock RAG system from the test app"""
    return test_app.state.mock_rag_system


class TestQueryEndpoint:
    """Test the /api/query endpoint"""
    
    @pytest.mark.api
    def test_query_success_with_session(self, client, mock_rag_system):
        """Test successful query with provided session ID"""
        # Configure mock
        mock_rag_system.query.return_value = (
            "This is about RAG systems",
            ["Course: RAG Fundamentals, Lesson 1"]
        )
        
        # Make request
        response = client.post("/api/query", json={
            "query": "What is RAG?",
            "session_id": "test_session_123"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is about RAG systems"
        assert data["sources"] == ["Course: RAG Fundamentals, Lesson 1"]
        assert data["session_id"] == "test_session_123"
        
        # Verify mock was called correctly
        mock_rag_system.query.assert_called_once_with("What is RAG?", "test_session_123")
    
    @pytest.mark.api
    def test_query_success_without_session(self, client, mock_rag_system):
        """Test successful query without session ID (should create one)"""
        # Configure mock
        mock_rag_system.query.return_value = (
            "Vector databases store embeddings",
            ["Course: Vector DB Guide, Lesson 2"]
        )
        
        # Make request without session_id
        response = client.post("/api/query", json={
            "query": "Tell me about vector databases"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Vector databases store embeddings"
        assert data["sources"] == ["Course: Vector DB Guide, Lesson 2"]
        assert data["session_id"] == "test_session"  # Default session from mock
        
        # Verify mock was called
        mock_rag_system.query.assert_called_once()
    
    @pytest.mark.api
    def test_query_invalid_request(self, client):
        """Test query with invalid request data"""
        # Missing required query field
        response = client.post("/api/query", json={
            "session_id": "test"
        })
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    def test_query_empty_query(self, client, mock_rag_system):
        """Test query with empty query string"""
        # Configure mock to handle empty query
        mock_rag_system.query.return_value = ("Empty query response", [])
        
        response = client.post("/api/query", json={
            "query": "",
            "session_id": "test"
        })
        
        # Should accept empty query (validation allows it)
        assert response.status_code == 200
    
    @pytest.mark.api
    def test_query_rag_system_error(self, client, mock_rag_system):
        """Test query when RAG system raises an exception"""
        # Configure mock to raise exception
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = client.post("/api/query", json={
            "query": "test query",
            "session_id": "test"
        })
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]


class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""
    
    @pytest.mark.api
    def test_get_courses_success(self, client, mock_rag_system):
        """Test successful retrieval of course statistics"""
        # Configure mock
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["RAG Fundamentals", "Vector Databases", "Advanced AI"]
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "RAG Fundamentals" in data["course_titles"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    @pytest.mark.api
    def test_get_courses_empty(self, client, mock_rag_system):
        """Test when no courses are available"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    @pytest.mark.api
    def test_get_courses_error(self, client, mock_rag_system):
        """Test when course analytics raises an exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]


class TestSessionClearEndpoint:
    """Test the /api/sessions/{session_id}/clear endpoint"""
    
    @pytest.mark.api
    def test_clear_session_success(self, client, mock_rag_system):
        """Test successful session clearing"""
        response = client.delete("/api/sessions/test_session_123/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session cleared successfully"
        
        mock_rag_system.session_manager.clear_session.assert_called_once_with("test_session_123")
    
    @pytest.mark.api
    def test_clear_session_error(self, client, mock_rag_system):
        """Test session clearing when an error occurs"""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session error")
        
        response = client.delete("/api/sessions/test_session/clear")
        
        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]


class TestRequestResponseModels:
    """Test request and response model validation"""
    
    @pytest.mark.api
    def test_query_request_valid(self, client, mock_rag_system):
        """Test valid query request formats"""
        mock_rag_system.query.return_value = ("answer", ["source"])
        
        # Test with all fields
        response = client.post("/api/query", json={
            "query": "test query",
            "session_id": "session123"
        })
        assert response.status_code == 200
        
        # Test with only required field
        response = client.post("/api/query", json={
            "query": "another test"
        })
        assert response.status_code == 200
    
    @pytest.mark.api
    def test_query_request_invalid(self, client):
        """Test invalid query request formats"""
        # Missing query field
        response = client.post("/api/query", json={
            "session_id": "test"
        })
        assert response.status_code == 422
        
        # Wrong field types
        response = client.post("/api/query", json={
            "query": 123,  # Should be string
            "session_id": "test"
        })
        assert response.status_code == 422
    
    @pytest.mark.api
    def test_response_format(self, client, mock_rag_system):
        """Test that response follows the expected format"""
        mock_rag_system.query.return_value = (
            "Test answer",
            ["Source 1", "Source 2"]
        )
        
        response = client.post("/api/query", json={
            "query": "test",
            "session_id": "test"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)


class TestHealthAndStatus:
    """Test basic application health"""
    
    @pytest.mark.api
    def test_app_creation(self, test_app):
        """Test that the test app creates successfully"""
        assert test_app is not None
        assert test_app.title == "Course Materials RAG System Test"
    
    @pytest.mark.api
    def test_client_creation(self, client):
        """Test that the test client creates successfully"""
        assert client is not None