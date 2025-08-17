"""
Integration tests for RAG system functionality
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager


class TestRAGSystemInitialization:
    """Test RAG system initialization and component integration"""

    def test_rag_system_initialization(self, test_config):
        """Test that RAG system initializes all components correctly"""
        rag = RAGSystem(test_config)

        # Verify all components are initialized
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None

        # Verify tools are registered
        tool_definitions = rag.tool_manager.get_tool_definitions()
        tool_names = [tool["name"] for tool in tool_definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_rag_system_with_problematic_config(self):
        """Test RAG system initialization with MAX_RESULTS=0 config"""
        from config import Config

        # Create config with the problematic setting
        problematic_config = Config()
        problematic_config.MAX_RESULTS = 0  # This is the problematic setting

        # Should still initialize, but will have search issues
        rag = RAGSystem(problematic_config)
        assert rag is not None
        assert rag.vector_store.max_results == 0  # This will cause search failures


class TestRAGSystemToolRegistration:
    """Test tool registration and management"""

    def test_tool_manager_has_correct_tools(self, test_config):
        """Test that tool manager has both search and outline tools"""
        rag = RAGSystem(test_config)

        # Check tools are registered
        assert "search_course_content" in rag.tool_manager.tools
        assert "get_course_outline" in rag.tool_manager.tools

        # Verify tools are correct instances
        search_tool = rag.tool_manager.tools["search_course_content"]
        outline_tool = rag.tool_manager.tools["get_course_outline"]

        assert isinstance(search_tool, CourseSearchTool)
        assert isinstance(outline_tool, CourseOutlineTool)

        # Verify tools share the same vector store
        assert search_tool.store is rag.vector_store
        assert outline_tool.store is rag.vector_store

    def test_tool_definitions_format(self, test_config):
        """Test that tool definitions are correctly formatted for Anthropic API"""
        rag = RAGSystem(test_config)

        tool_definitions = rag.tool_manager.get_tool_definitions()

        assert len(tool_definitions) == 2

        for tool_def in tool_definitions:
            # Each tool should have required fields
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def

            # Input schema should be properly structured
            schema = tool_def["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


class TestRAGSystemQuery:
    """Test RAG system query processing"""

    @patch("anthropic.Anthropic")
    def test_query_without_tools(self, mock_anthropic_class, test_config):
        """Test query that doesn't require tools"""
        # Setup mock for AI response without tool use
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="This is a general knowledge response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        rag = RAGSystem(test_config)

        # Query that shouldn't trigger tools
        response, sources = rag.query("What is 2+2?")

        assert response == "This is a general knowledge response"
        assert sources == []  # No sources for general knowledge

        # Verify AI was called
        mock_client.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_query_with_tool_use(
        self, mock_anthropic_class, test_config, sample_course, sample_course_chunks
    ):
        """Test query that uses search tool"""
        # Setup mock for AI response with tool use
        mock_client = Mock()

        # Mock initial response with tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"

        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "test_id"
        mock_tool_block.input = {"query": "machine learning"}

        mock_tool_response.content = [mock_tool_block]

        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Machine learning is covered in lesson 2")
        ]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]
        mock_anthropic_class.return_value = mock_client

        # Create RAG system and add test data
        rag = RAGSystem(test_config)
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_course_chunks)

        # Query that should trigger tool use
        response, sources = rag.query("Tell me about machine learning in the courses")

        assert response == "Machine learning is covered in lesson 2"
        # Should have sources from tool execution
        assert len(sources) > 0

        # Verify AI was called twice (initial + follow-up)
        assert mock_client.messages.create.call_count == 2

    def test_query_with_session_management(self, test_config):
        """Test query with session management"""
        # This test uses mocks to avoid actual API calls
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_response.content = [Mock(text="Response with session")]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_class.return_value = mock_client

            rag = RAGSystem(test_config)

            # First query in session
            response1, _ = rag.query("First question", session_id="test_session")

            # Second query in same session
            response2, _ = rag.query("Follow-up question", session_id="test_session")

            # Both should succeed
            assert response1 == "Response with session"
            assert response2 == "Response with session"

            # Check that conversation history was used in second call
            second_call_args = mock_client.messages.create.call_args_list[1][1]
            assert "Previous conversation" in second_call_args["system"]


class TestRAGSystemErrorHandling:
    """Test RAG system error handling"""

    @patch("anthropic.Anthropic")
    def test_query_with_api_error(self, mock_anthropic_class, test_config):
        """Test handling of Anthropic API errors"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client

        rag = RAGSystem(test_config)

        # Should handle API error gracefully or raise appropriate exception
        with pytest.raises(Exception):
            rag.query("Test query")

    def test_query_with_empty_vector_store(self, test_config):
        """Test query when vector store has no data"""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            # Setup mock for tool use that finds no results
            mock_client = Mock()

            mock_tool_response = Mock()
            mock_tool_response.stop_reason = "tool_use"

            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "test_id"
            mock_tool_block.input = {"query": "test"}

            mock_tool_response.content = [mock_tool_block]

            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="No relevant content found")]

            mock_client.messages.create.side_effect = [
                mock_tool_response,
                mock_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            rag = RAGSystem(test_config)

            # Query should work even with empty vector store
            response, sources = rag.query("Tell me about machine learning")

            assert response == "No relevant content found"
            assert sources == []  # No sources when nothing found


class TestRAGSystemAnalytics:
    """Test RAG system analytics functionality"""

    def test_get_course_analytics_empty(self, test_config):
        """Test course analytics with empty system"""
        rag = RAGSystem(test_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []

    def test_get_course_analytics_with_data(self, test_config, sample_course):
        """Test course analytics with course data"""
        rag = RAGSystem(test_config)
        rag.vector_store.add_course_metadata(sample_course)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 1
        assert sample_course.title in analytics["course_titles"]


class TestRAGSystemDataManagement:
    """Test RAG system data loading and management"""

    def test_add_course_document_success(self, test_config, tmp_path):
        """Test successful course document addition"""
        # Create a temporary course document
        course_file = tmp_path / "test_course.txt"
        course_content = """Course Title: Test Course
Course Link: https://test.com
Course Instructor: Test Instructor

Lesson 0: Introduction
This is the introduction to the test course.

Lesson 1: Advanced Topics
This covers advanced topics in the test course.
"""
        course_file.write_text(course_content)

        rag = RAGSystem(test_config)

        # Should successfully add the course
        course, chunk_count = rag.add_course_document(str(course_file))

        assert course is not None
        assert course.title == "Test Course"
        assert chunk_count > 0

        # Verify course was added to vector store
        analytics = rag.get_course_analytics()
        assert analytics["total_courses"] == 1

    def test_add_course_document_failure(self, test_config):
        """Test handling of failed course document addition"""
        rag = RAGSystem(test_config)

        # Try to add non-existent file
        course, chunk_count = rag.add_course_document("/nonexistent/file.txt")

        assert course is None
        assert chunk_count == 0

    def test_add_course_folder_empty(self, test_config, tmp_path):
        """Test adding courses from empty folder"""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        rag = RAGSystem(test_config)

        total_courses, total_chunks = rag.add_course_folder(str(empty_folder))

        assert total_courses == 0
        assert total_chunks == 0

    def test_add_course_folder_nonexistent(self, test_config):
        """Test adding courses from non-existent folder"""
        rag = RAGSystem(test_config)

        total_courses, total_chunks = rag.add_course_folder("/nonexistent/folder")

        assert total_courses == 0
        assert total_chunks == 0


class TestRAGSystemConfigurationIssues:
    """Test RAG system with various configuration issues"""

    def test_max_results_zero_integration(self, sample_course, sample_course_chunks):
        """Test complete RAG system flow with MAX_RESULTS=0 issue"""
        from config import Config

        # Create config with problematic MAX_RESULTS=0
        problematic_config = Config()
        problematic_config.MAX_RESULTS = 0
        problematic_config.CHROMA_PATH = "/tmp/test_chroma"

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            # Setup mock for tool use
            mock_client = Mock()

            mock_tool_response = Mock()
            mock_tool_response.stop_reason = "tool_use"

            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.id = "test_id"
            mock_tool_block.input = {"query": "machine learning"}

            mock_tool_response.content = [mock_tool_block]

            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="No relevant content found")]

            mock_client.messages.create.side_effect = [
                mock_tool_response,
                mock_final_response,
            ]
            mock_anthropic_class.return_value = mock_client

            # Create RAG system with problematic config
            rag = RAGSystem(problematic_config)

            # Add course data
            rag.vector_store.add_course_metadata(sample_course)
            rag.vector_store.add_course_content(sample_course_chunks)

            # Query should fail to find results due to MAX_RESULTS=0
            response, sources = rag.query("Tell me about machine learning")

            # Even with relevant data, should return "no content found" due to config issue
            assert "No relevant content found" in response
            assert len(sources) == 0

    def test_missing_api_key(self):
        """Test RAG system behavior with missing API key"""
        from config import Config

        config = Config()
        config.ANTHROPIC_API_KEY = ""  # Missing API key

        # Should still initialize (error will occur during actual API calls)
        rag = RAGSystem(config)
        assert rag is not None

        # Actual query would fail with API key error, but initialization succeeds
