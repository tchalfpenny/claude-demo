"""
Data validation tests for RAG system
"""
import pytest
import os
from rag_system import RAGSystem


class TestDataValidation:
    """Test data validation and system state"""
    
    def test_vector_store_has_course_data(self, test_config):
        """Test if the vector store actually contains course data"""
        rag = RAGSystem(test_config)
        
        # Check if there are any courses loaded
        analytics = rag.get_course_analytics()
        course_count = analytics["total_courses"]
        course_titles = analytics["course_titles"]
        
        print(f"Found {course_count} courses in vector store")
        print(f"Course titles: {course_titles}")
        
        # This test will reveal if data is missing
        if course_count == 0:
            pytest.fail("No course data found in vector store - this could explain query failures")
        
        assert course_count > 0, "Vector store should contain course data"
        assert len(course_titles) > 0, "Should have course titles"
    
    def test_vector_store_search_with_real_data(self, test_config):
        """Test vector store search with actual data"""
        rag = RAGSystem(test_config)
        
        # Try a basic search to see if anything is returned
        results = rag.vector_store.search("course")
        
        print(f"Search results: {len(results.documents)} documents found")
        print(f"Has error: {results.error}")
        print(f"Is empty: {results.is_empty()}")
        
        if results.error:
            pytest.fail(f"Vector store search returned error: {results.error}")
        
        if results.is_empty():
            pytest.fail("Vector store search returned no results - check if data is loaded and MAX_RESULTS config")
    
    def test_course_search_tool_with_real_data(self, test_config):
        """Test CourseSearchTool with actual system data"""
        rag = RAGSystem(test_config)
        
        # Get the search tool
        search_tool = rag.search_tool
        
        # Try executing a search
        result = search_tool.execute("introduction")
        
        print(f"Search tool result: {result}")
        print(f"Last sources: {search_tool.last_sources}")
        
        if "No relevant content found" in result:
            pytest.fail("Search tool found no content - check data loading and configuration")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_docs_folder_exists(self):
        """Test if the docs folder exists and contains course documents"""
        docs_path = "/Users/tylerhalfpenny/Desktop/repos/claude-demo/docs"
        
        print(f"Checking docs folder at: {docs_path}")
        
        if not os.path.exists(docs_path):
            pytest.fail(f"Docs folder not found at {docs_path}")
        
        # List files in docs folder
        files = os.listdir(docs_path)
        course_files = [f for f in files if f.lower().endswith(('.txt', '.pdf', '.docx'))]
        
        print(f"Found {len(course_files)} course files: {course_files}")
        
        if len(course_files) == 0:
            pytest.fail("No course documents found in docs folder")
        
        assert len(course_files) > 0, "Should have course documents in docs folder"
    
    def test_config_values(self, test_config):
        """Test configuration values that might cause issues"""
        print(f"MAX_RESULTS: {test_config.MAX_RESULTS}")
        print(f"CHROMA_PATH: {test_config.CHROMA_PATH}")
        print(f"ANTHROPIC_API_KEY present: {'Yes' if test_config.ANTHROPIC_API_KEY else 'No'}")
        print(f"EMBEDDING_MODEL: {test_config.EMBEDDING_MODEL}")
        
        # Check for the critical MAX_RESULTS=0 issue
        if test_config.MAX_RESULTS == 0:
            pytest.fail("MAX_RESULTS is set to 0 - this will cause all searches to return empty results!")
        
        assert test_config.MAX_RESULTS > 0, "MAX_RESULTS must be greater than 0"
        
        # Check API key
        if not test_config.ANTHROPIC_API_KEY:
            pytest.fail("ANTHROPIC_API_KEY is missing - this will cause API errors")
    
    def test_chroma_db_exists(self, test_config):
        """Test if ChromaDB database exists and contains data"""
        chroma_path = test_config.CHROMA_PATH
        
        print(f"Checking ChromaDB at: {chroma_path}")
        
        if not os.path.exists(chroma_path):
            pytest.fail(f"ChromaDB not found at {chroma_path} - database may not be initialized")
        
        # Check if there are database files
        db_files = os.listdir(chroma_path)
        print(f"ChromaDB files: {db_files}")
        
        if len(db_files) == 0:
            pytest.fail("ChromaDB directory is empty - no data has been loaded")
        
        assert len(db_files) > 0, "ChromaDB should contain data files"


class TestSystemIntegration:
    """Test system integration with real components"""
    
    def test_end_to_end_query_flow(self, test_config):
        """Test complete query flow without mocking"""
        # Note: This test will use real API if ANTHROPIC_API_KEY is set
        # It will fail if API key is missing, which is expected
        
        rag = RAGSystem(test_config)
        
        # Check if we have data first
        analytics = rag.get_course_analytics()
        if analytics["total_courses"] == 0:
            pytest.skip("No course data available for integration test")
        
        # Try a simple query that shouldn't need tools
        try:
            response, sources = rag.query("What is 2+2?")
            
            print(f"Response: {response}")
            print(f"Sources: {sources}")
            
            assert isinstance(response, str)
            assert len(response) > 0
            
        except Exception as e:
            print(f"Query failed with error: {e}")
            pytest.fail(f"End-to-end query failed: {e}")
    
    def test_tool_execution_flow(self, test_config):
        """Test tool execution without full API integration"""
        rag = RAGSystem(test_config)
        
        # Test tool manager directly
        tool_manager = rag.tool_manager
        
        # Get tool definitions
        tool_defs = tool_manager.get_tool_definitions()
        print(f"Available tools: {[tool['name'] for tool in tool_defs]}")
        
        # Test direct tool execution
        try:
            result = tool_manager.execute_tool("search_course_content", query="introduction")
            print(f"Direct tool execution result: {result}")
            
            assert isinstance(result, str)
            
            if "No relevant content found" in result:
                pytest.fail("Tool execution found no content - check data and configuration")
                
        except Exception as e:
            print(f"Tool execution failed: {e}")
            pytest.fail(f"Tool execution failed: {e}")


class TestConfigurationValidation:
    """Test configuration-related issues"""
    
    def test_original_config_issues(self):
        """Test the original configuration from config.py"""
        from config import config as original_config
        
        print("Original configuration values:")
        print(f"MAX_RESULTS: {original_config.MAX_RESULTS}")
        print(f"CHROMA_PATH: {original_config.CHROMA_PATH}")
        print(f"CHUNK_SIZE: {original_config.CHUNK_SIZE}")
        print(f"CHUNK_OVERLAP: {original_config.CHUNK_OVERLAP}")
        
        # Check for the critical issue
        if original_config.MAX_RESULTS == 0:
            pytest.fail(
                "CRITICAL ISSUE FOUND: MAX_RESULTS=0 in original config! "
                "This will cause all searches to return 0 results, explaining the 'query failed' errors."
            )
    
    def test_vector_store_with_zero_max_results(self):
        """Test vector store behavior with MAX_RESULTS=0"""
        from vector_store import VectorStore
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create vector store with MAX_RESULTS=0 (problematic config)
            vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2", max_results=0)
            
            # Add some test data
            from models import Course, Lesson, CourseChunk
            
            course = Course(
                title="Test Course",
                instructor="Test Instructor",
                course_link="https://test.com",
                lessons=[Lesson(1, "Test Lesson", None)]
            )
            
            chunks = [CourseChunk(
                content="Test content about machine learning",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )]
            
            vector_store.add_course_metadata(course)
            vector_store.add_course_content(chunks)
            
            # Try to search - this should demonstrate the bug
            results = vector_store.search("machine learning")
            
            print(f"Search with MAX_RESULTS=0: {len(results.documents)} results")
            
            if results.is_empty():
                pytest.fail(
                    "CONFIRMED: MAX_RESULTS=0 causes empty search results even with relevant data! "
                    "This is the root cause of 'query failed' errors."
                )
    
    def test_vector_store_with_proper_max_results(self):
        """Test vector store behavior with proper MAX_RESULTS"""
        from vector_store import VectorStore
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create vector store with proper MAX_RESULTS
            vector_store = VectorStore(temp_dir, "all-MiniLM-L6-v2", max_results=5)
            
            # Add some test data
            from models import Course, Lesson, CourseChunk
            
            course = Course(
                title="Test Course",
                instructor="Test Instructor",
                course_link="https://test.com",
                lessons=[Lesson(1, "Test Lesson", None)]
            )
            
            chunks = [CourseChunk(
                content="Test content about machine learning and artificial intelligence",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )]
            
            vector_store.add_course_metadata(course)
            vector_store.add_course_content(chunks)
            
            # Try to search - this should work correctly
            results = vector_store.search("machine learning")
            
            print(f"Search with MAX_RESULTS=5: {len(results.documents)} results")
            
            if not results.is_empty():
                print("SUCCESS: Proper MAX_RESULTS allows search to return results")
            else:
                pytest.fail("Even with proper MAX_RESULTS, no results returned - other issues present")
            
            assert not results.is_empty(), "With proper MAX_RESULTS, search should return results"