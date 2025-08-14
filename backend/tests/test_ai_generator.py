"""
Tests for AIGenerator functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGeneratorBasic:
    """Test basic AIGenerator functionality"""
    
    def test_ai_generator_initialization(self):
        """Test AIGenerator can be initialized"""
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        assert generator is not None
        assert generator.model == "claude-sonnet-4-20250514"
        assert generator.base_params["model"] == "claude-sonnet-4-20250514"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected tool usage guidance"""
        prompt = AIGenerator.SYSTEM_PROMPT
        
        # Check for key tool usage instructions
        assert "Content search tool" in prompt
        assert "Course outline tool" in prompt
        assert "One tool call per query maximum" in prompt
        
        # Check for response protocol
        assert "General knowledge questions" in prompt
        assert "Course content questions" in prompt
        assert "Course outline/structure questions" in prompt


class TestAIGeneratorWithoutTools:
    """Test AIGenerator without tool usage"""
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test generating response without tools"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Setup mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="This is a test response")]
        mock_client.messages.create.return_value = mock_response
        
        # Create generator and generate response
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = generator.generate_response("What is machine learning?")
        
        # Verify response
        assert result == "This is a test response"
        
        # Verify API was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert "What is machine learning?" in call_args["messages"][0]["content"]
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test generating response with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with history")]
        mock_client.messages.create.return_value = mock_response
        
        # Create generator
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Generate response with history
        history = "Previous: What is AI? Response: AI is artificial intelligence."
        result = generator.generate_response("Tell me more", conversation_history=history)
        
        assert result == "Response with history"
        
        # Check that history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous conversation" in call_args["system"]
        assert history in call_args["system"]


class TestAIGeneratorWithTools:
    """Test AIGenerator with tool usage"""
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic_class):
        """Test response generation when tools are available but not used"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"  # No tool use
        mock_response.content = [Mock(text="Direct response without tools")]
        mock_client.messages.create.return_value = mock_response
        
        # Create generator
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Mock tools and tool manager
        mock_tools = [{"name": "test_tool", "description": "Test tool"}]
        mock_tool_manager = Mock()
        
        result = generator.generate_response(
            "What is 2+2?", 
            tools=mock_tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct response without tools"
        
        # Verify tools were provided to API
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == mock_tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # Verify tool manager was not used
        mock_tool_manager.execute_tool.assert_not_called()
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic_class):
        """Test response generation when tools are used"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock initial response with tool use
        mock_tool_use_response = Mock()
        mock_tool_use_response.stop_reason = "tool_use"
        
        # Mock tool use content block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "test_tool_id"
        mock_tool_block.input = {"query": "machine learning"}
        
        mock_tool_use_response.content = [mock_tool_block]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Response after using tools")]
        
        # Configure the client to return different responses on subsequent calls
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Create generator
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Mock tools and tool manager
        mock_tools = [{"name": "search_course_content", "description": "Search course content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        result = generator.generate_response(
            "What is machine learning in courses?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Response after using tools"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )
        
        # Verify two API calls were made (initial + follow-up)
        assert mock_client.messages.create.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool use response
        mock_tool_use_response = Mock()
        mock_tool_use_response.stop_reason = "tool_use"
        
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "test_tool_id"
        mock_tool_block.input = {"query": "test"}
        
        mock_tool_use_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Error handled response")]
        
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Create generator
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Mock tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool error: Search failed"
        
        result = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should still return a response even if tool fails
        assert result == "Error handled response"
        assert mock_tool_manager.execute_tool.called


class TestAIGeneratorToolExecution:
    """Test AIGenerator tool execution logic"""
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response with multiple tool uses
        mock_tool_use_response = Mock()
        mock_tool_use_response.stop_reason = "tool_use"
        
        # Create multiple tool blocks
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_id_1"
        mock_tool_block1.input = {"query": "test1"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.id = "tool_id_2"
        mock_tool_block2.input = {"course_name": "test course"}
        
        mock_tool_use_response.content = [mock_tool_block1, mock_tool_block2]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Multi-tool response")]
        
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Create generator
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        result = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )
        
        assert result == "Multi-tool response"
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="test1")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="test course")
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_message_structure(self, mock_anthropic_class):
        """Test correct message structure in tool execution flow"""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool use response
        mock_tool_use_response = Mock()
        mock_tool_use_response.stop_reason = "tool_use"
        
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "test_tool_id"
        mock_tool_block.input = {"query": "test"}
        
        mock_tool_use_response.content = [mock_tool_block]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response")]
        
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Create generator
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        result = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify final API call had correct message structure
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = final_call_args["messages"]
        
        # Should have: user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"  # Original query
        assert messages[1]["role"] == "assistant"  # Tool use response
        assert messages[2]["role"] == "user"  # Tool results
        
        # Check tool results structure
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "test_tool_id"
        assert tool_results[0]["content"] == "Tool result"


class TestAIGeneratorErrorCases:
    """Test AIGenerator error handling"""
    
    @patch('anthropic.Anthropic')
    def test_anthropic_api_error(self, mock_anthropic_class):
        """Test handling of Anthropic API errors"""
        # Setup mock client to raise an exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Should raise the exception (or handle it gracefully depending on implementation)
        with pytest.raises(Exception):
            generator.generate_response("Test query")
    
    @patch('anthropic.Anthropic')
    def test_missing_tool_manager_with_tools(self, mock_anthropic_class):
        """Test behavior when tools are provided but tool_manager is None"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool use response
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [Mock(type="tool_use", name="test", id="1", input={})]
        mock_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # Should handle missing tool_manager gracefully
        result = generator.generate_response(
            "Test query",
            tools=[{"name": "test_tool"}],
            tool_manager=None  # Missing tool manager
        )
        
        # Implementation should handle this case (exact behavior may vary)
        assert isinstance(result, str)
    
    def test_invalid_api_key(self):
        """Test behavior with invalid API key"""
        # This test may not be feasible without actual API calls
        # But we can test that generator initializes with any string
        generator = AIGenerator("", "claude-sonnet-4-20250514")
        assert generator is not None