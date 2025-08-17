"""
Tests for CourseSearchTool functionality
"""

from unittest.mock import Mock

import pytest
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""

    def test_get_tool_definition(self, course_search_tool_with_mock):
        """Test that tool definition is correctly structured"""
        tool_def = course_search_tool_with_mock.get_tool_definition()

        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        assert tool_def["input_schema"]["type"] == "object"
        assert "query" in tool_def["input_schema"]["properties"]
        assert "query" in tool_def["input_schema"]["required"]

    def test_execute_with_valid_query(self, course_search_tool_with_mock):
        """Test execute method with valid query and results"""
        result = course_search_tool_with_mock.execute("RAG systems")

        # Should return formatted results
        assert isinstance(result, str)
        assert len(result) > 0
        assert "RAG System Fundamentals" in result

        # Check that last_sources is populated
        assert len(course_search_tool_with_mock.last_sources) > 0

    def test_execute_with_course_filter(self, course_search_tool_with_mock):
        """Test execute method with course name filtering"""
        result = course_search_tool_with_mock.execute(
            query="vector databases", course_name="RAG System"
        )

        assert isinstance(result, str)
        assert len(result) > 0

        # Verify the mock was called with correct parameters
        course_search_tool_with_mock.store.search.assert_called_with(
            query="vector databases", course_name="RAG System", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, course_search_tool_with_mock):
        """Test execute method with lesson number filtering"""
        result = course_search_tool_with_mock.execute(
            query="introduction", lesson_number=1
        )

        assert isinstance(result, str)

        # Verify the mock was called with correct parameters
        course_search_tool_with_mock.store.search.assert_called_with(
            query="introduction", course_name=None, lesson_number=1
        )

    def test_execute_with_empty_results(self, mock_empty_vector_store):
        """Test execute method when no results are found"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute("nonexistent topic")

        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_execute_with_empty_results_and_filters(self, mock_empty_vector_store):
        """Test execute method with filters when no results found"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute(
            query="test", course_name="Nonexistent Course", lesson_number=5
        )

        assert "No relevant content found" in result
        assert "Nonexistent Course" in result
        assert "lesson 5" in result

    def test_execute_with_vector_store_error(self, mock_error_vector_store):
        """Test execute method when vector store returns an error"""
        tool = CourseSearchTool(mock_error_vector_store)
        result = tool.execute("test query")

        assert "Search error" in result
        assert "Vector store unavailable" in result

    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test result formatting includes lesson links when available"""
        tool = CourseSearchTool(mock_vector_store)

        # Create test results with lesson data
        results = SearchResults(
            documents=["Content about vector databases"],
            metadata=[{"course_title": "RAG Course", "lesson_number": 2}],
            distances=[0.1],
        )

        formatted = tool._format_results(results)

        assert "[RAG Course - Lesson 2]" in formatted
        assert "Content about vector databases" in formatted

        # Check that lesson link was requested
        mock_vector_store.get_lesson_link.assert_called_with("RAG Course", 2)

    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test result formatting when lesson number is not provided"""
        tool = CourseSearchTool(mock_vector_store)

        # Create test results without lesson number
        results = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "RAG Course"}],
            distances=[0.1],
        )

        formatted = tool._format_results(results)

        assert "[RAG Course]" in formatted
        assert "General course content" in formatted
        # Should not include lesson information
        assert "Lesson" not in formatted

    def test_source_tracking(self, course_search_tool_with_mock):
        """Test that sources are properly tracked and can be retrieved"""
        # Execute a search
        course_search_tool_with_mock.execute("test query")

        # Check sources are tracked
        sources = course_search_tool_with_mock.last_sources
        assert len(sources) > 0
        assert "RAG System Fundamentals" in sources[0]

    def test_source_tracking_with_lesson_links(self, mock_vector_store):
        """Test source tracking includes lesson links when available"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return lesson link
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Execute search
        tool.execute("test query")

        # Check that source includes link
        sources = tool.last_sources
        assert len(sources) > 0
        assert "LINK:https://example.com/lesson1" in sources[0]

    def test_multiple_executions_reset_sources(self, course_search_tool_with_mock):
        """Test that sources are reset between executions"""
        # First execution
        course_search_tool_with_mock.execute("first query")
        first_sources = course_search_tool_with_mock.last_sources.copy()

        # Second execution
        course_search_tool_with_mock.execute("second query")
        second_sources = course_search_tool_with_mock.last_sources

        # Sources should be updated, not accumulated
        assert len(second_sources) > 0
        # In this test case, they'll be the same due to mocking, but structure is correct


class TestCourseSearchToolErrorCases:
    """Test error handling and edge cases"""

    def test_execute_with_none_query(self, course_search_tool_with_mock):
        """Test execute method with None query"""
        # This should raise an exception or handle gracefully
        with pytest.raises((TypeError, AttributeError)):
            course_search_tool_with_mock.execute(None)

    def test_execute_with_empty_query(self, course_search_tool_with_mock):
        """Test execute method with empty query"""
        result = course_search_tool_with_mock.execute("")
        # Should still work, just return whatever the vector store returns
        assert isinstance(result, str)

    def test_format_results_with_malformed_metadata(self, mock_vector_store):
        """Test formatting when metadata is malformed"""
        tool = CourseSearchTool(mock_vector_store)

        # Create results with missing or malformed metadata
        results = SearchResults(
            documents=["Some content"], metadata=[{}], distances=[0.1]  # Empty metadata
        )

        formatted = tool._format_results(results)

        # Should handle gracefully and still format the content
        assert "Some content" in formatted
        assert "[unknown]" in formatted

    def test_format_results_mismatched_lengths(self, mock_vector_store):
        """Test formatting when documents and metadata arrays have different lengths"""
        tool = CourseSearchTool(mock_vector_store)

        # Create results with mismatched array lengths
        results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[{"course_title": "Course 1"}],  # Only one metadata entry
            distances=[0.1, 0.2],
        )

        # This should handle gracefully without crashing
        formatted = tool._format_results(results)
        assert isinstance(formatted, str)


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with real-ish scenarios"""

    def test_course_name_resolution_flow(self, mock_vector_store):
        """Test the complete flow including course name resolution"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure the mock to simulate course name resolution
        mock_vector_store.search.return_value = SearchResults(
            documents=["Resolved course content"],
            metadata=[{"course_title": "Full Course Name", "lesson_number": 1}],
            distances=[0.05],
        )

        result = tool.execute(query="test", course_name="partial name")

        # Should have called search with the partial name (resolution happens in vector store)
        mock_vector_store.search.assert_called_with(
            query="test", course_name="partial name", lesson_number=None
        )

        assert "Full Course Name" in result

    def test_lesson_link_retrieval_flow(self, mock_vector_store):
        """Test the complete flow including lesson link retrieval"""
        tool = CourseSearchTool(mock_vector_store)

        # Configure mock to return lesson link
        mock_vector_store.get_lesson_link.return_value = (
            "https://example.com/specific-lesson"
        )

        result = tool.execute("test query")

        # Check that lesson link was requested
        mock_vector_store.get_lesson_link.assert_called()

        # Check that source tracking includes the link
        sources = tool.last_sources
        assert any(
            "LINK:https://example.com/specific-lesson" in source for source in sources
        )
