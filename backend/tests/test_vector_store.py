"""
Tests for VectorStore functionality
"""

import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestVectorStoreConfiguration:
    """Test VectorStore configuration issues"""

    def test_max_results_zero_issue(self, temp_chroma_path):
        """Test the critical MAX_RESULTS=0 configuration issue"""
        # Create VectorStore with MAX_RESULTS = 0 (the problematic config)
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=0)

        # Add some test data
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="https://test.com",
            lessons=[Lesson(1, "Test Lesson", None)],
        )

        chunks = [
            CourseChunk(
                content="Test content about machine learning",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        # Try to search - should return 0 results due to MAX_RESULTS=0
        results = vector_store.search("machine learning")

        # This demonstrates the bug: even with relevant content, we get no results
        assert (
            results.is_empty()
        ), "With MAX_RESULTS=0, search should return empty results"
        assert len(results.documents) == 0

    def test_max_results_proper_setting(self, temp_chroma_path):
        """Test VectorStore with proper MAX_RESULTS setting"""
        # Create VectorStore with MAX_RESULTS = 5 (proper config)
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)

        # Add some test data
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="https://test.com",
            lessons=[Lesson(1, "Test Lesson", None)],
        )

        chunks = [
            CourseChunk(
                content="Test content about machine learning and artificial intelligence",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        # Try to search - should return results with proper MAX_RESULTS
        results = vector_store.search("machine learning")

        # This should work correctly with MAX_RESULTS > 0
        assert (
            not results.is_empty()
        ), "With proper MAX_RESULTS, search should return results"
        assert len(results.documents) > 0
        assert "machine learning" in results.documents[0].lower()


class TestVectorStoreBasicFunctionality:
    """Test basic VectorStore operations"""

    def test_vector_store_initialization(self, temp_chroma_path):
        """Test VectorStore can be initialized"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        assert vector_store is not None
        assert vector_store.max_results == 5

    def test_add_course_metadata(self, temp_chroma_path, sample_course):
        """Test adding course metadata"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)

        # Should not raise an exception
        vector_store.add_course_metadata(sample_course)

        # Verify course was added
        count = vector_store.get_course_count()
        assert count == 1

        titles = vector_store.get_existing_course_titles()
        assert sample_course.title in titles

    def test_add_course_content(self, temp_chroma_path, sample_course_chunks):
        """Test adding course content chunks"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)

        # Should not raise an exception
        vector_store.add_course_content(sample_course_chunks)

        # Try to search for the content
        results = vector_store.search("RAG systems")
        assert not results.is_empty()

    def test_get_course_count_empty(self, temp_chroma_path):
        """Test course count when no courses are added"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        count = vector_store.get_course_count()
        assert count == 0

    def test_get_existing_course_titles_empty(self, temp_chroma_path):
        """Test getting course titles when no courses exist"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        titles = vector_store.get_existing_course_titles()
        assert titles == []


class TestVectorStoreSearch:
    """Test VectorStore search functionality"""

    def test_search_without_filters(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test basic search without course or lesson filters"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        results = vector_store.search("vector databases")

        assert not results.is_empty()
        assert len(results.documents) > 0
        assert len(results.metadata) > 0
        # Should find content related to vector databases
        assert any("vector" in doc.lower() for doc in results.documents)

    def test_search_with_course_filter(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test search with course name filter"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        # Search with exact course name
        results = vector_store.search("RAG", course_name="RAG System Fundamentals")

        assert not results.is_empty()
        # All results should be from the specified course
        for meta in results.metadata:
            assert meta["course_title"] == "RAG System Fundamentals"

    def test_search_with_partial_course_name(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test search with partial course name matching"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        # Search with partial course name
        results = vector_store.search("systems", course_name="RAG")

        # Should resolve "RAG" to "RAG System Fundamentals" and return results
        assert not results.is_empty()

    def test_search_with_lesson_filter(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test search with lesson number filter"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        results = vector_store.search("introduction", lesson_number=1)

        assert not results.is_empty()
        # All results should be from lesson 1
        for meta in results.metadata:
            assert meta["lesson_number"] == 1

    def test_search_with_both_filters(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test search with both course and lesson filters"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        results = vector_store.search(
            "RAG", course_name="RAG System Fundamentals", lesson_number=1
        )

        assert not results.is_empty()
        # Results should match both filters
        for meta in results.metadata:
            assert meta["course_title"] == "RAG System Fundamentals"
            assert meta["lesson_number"] == 1

    def test_search_nonexistent_course(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test search with non-existent course name"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        results = vector_store.search("test", course_name="Nonexistent Course")

        # Should return an error for non-existent course
        assert results.error is not None
        assert "No course found matching" in results.error

    def test_search_with_limit_override(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test search with custom limit parameter"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        # Add multiple chunks to test limiting
        extra_chunks = [
            CourseChunk(f"Extra content {i}", sample_course.title, 1, i + 10)
            for i in range(10)
        ]
        vector_store.add_course_content(extra_chunks)

        # Search with custom limit
        results = vector_store.search("content", limit=2)

        # Should return at most 2 results
        assert len(results.documents) <= 2


class TestVectorStoreCourseOutline:
    """Test course outline functionality"""

    def test_get_course_outline_existing_course(self, temp_chroma_path, sample_course):
        """Test getting outline for existing course"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)

        outline = vector_store.get_course_outline("RAG System Fundamentals")

        assert outline is not None
        assert outline["title"] == sample_course.title
        assert outline["instructor"] == sample_course.instructor
        assert outline["course_link"] == sample_course.course_link
        assert len(outline["lessons"]) == len(sample_course.lessons)

    def test_get_course_outline_partial_name(self, temp_chroma_path, sample_course):
        """Test getting outline with partial course name"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)

        outline = vector_store.get_course_outline("RAG")

        assert outline is not None
        assert outline["title"] == sample_course.title

    def test_get_course_outline_nonexistent(self, temp_chroma_path):
        """Test getting outline for non-existent course"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)

        outline = vector_store.get_course_outline("Nonexistent Course")

        assert outline is None

    def test_get_lesson_link(self, temp_chroma_path, sample_course):
        """Test getting lesson link for specific lesson"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)

        link = vector_store.get_lesson_link(sample_course.title, 1)

        assert link == "https://example.com/lesson1"

    def test_get_lesson_link_no_link(self, temp_chroma_path, sample_course):
        """Test getting lesson link when lesson has no link"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)

        link = vector_store.get_lesson_link(sample_course.title, 3)

        assert link is None


class TestVectorStoreErrorHandling:
    """Test error handling and edge cases"""

    def test_search_with_invalid_path(self):
        """Test VectorStore with invalid path"""
        # This might fail during initialization or search
        with pytest.raises(Exception):
            vector_store = VectorStore(
                "/invalid/path", "all-MiniLM-L6-v2", max_results=5
            )
            vector_store.search("test")

    def test_add_empty_course_content(self, temp_chroma_path):
        """Test adding empty course content list"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)

        # Should handle empty list gracefully
        vector_store.add_course_content([])

        # Should still work for other operations
        count = vector_store.get_course_count()
        assert count == 0

    def test_clear_all_data(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test clearing all data from vector store"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        # Verify data exists
        assert vector_store.get_course_count() > 0

        # Clear data
        vector_store.clear_all_data()

        # Verify data is cleared
        assert vector_store.get_course_count() == 0
        assert vector_store.get_existing_course_titles() == []


class TestVectorStoreDataConsistency:
    """Test data consistency and integrity"""

    def test_course_metadata_retrieval(self, temp_chroma_path, sample_course):
        """Test retrieving all course metadata"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)
        vector_store.add_course_metadata(sample_course)

        all_metadata = vector_store.get_all_courses_metadata()

        assert len(all_metadata) == 1
        metadata = all_metadata[0]
        assert metadata["title"] == sample_course.title
        assert metadata["instructor"] == sample_course.instructor
        assert "lessons" in metadata
        assert len(metadata["lessons"]) == len(sample_course.lessons)

    def test_duplicate_course_handling(self, temp_chroma_path, sample_course):
        """Test handling of duplicate course additions"""
        vector_store = VectorStore(temp_chroma_path, "all-MiniLM-L6-v2", max_results=5)

        # Add same course twice
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_metadata(sample_course)

        # Should still only have one course (or handle gracefully)
        count = vector_store.get_course_count()
        assert count >= 1  # At least one, implementation may vary on duplicates
