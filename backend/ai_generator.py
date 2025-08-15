import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Content search tool**: Use for questions about specific course content or detailed educational materials
- **Course outline tool**: Use for questions about course structure, lesson lists, or course overviews
- **Sequential tool usage**: You can make multiple tool calls across up to 2 rounds to gather comprehensive information
- **Strategic planning**: Use first tool results to inform subsequent tool calls
- **Tool combination**: You may use different tools in sequence to build comprehensive answers
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use content search tool, optionally followed by outline tool for context
- **Course outline/structure questions**: Use outline tool, optionally followed by content search for details
- **Complex questions**: May require multiple tool calls to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "according to the outline tool"

For outline queries, ensure responses include:
- Course title and instructor
- Course link (if available)  
- Complete lesson list with lesson numbers and titles
- Lesson links (if available)

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_tool_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling across up to max_tool_rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of tool rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation messages
        messages = [{"role": "user", "content": query}]
        
        # Sequential tool execution loop
        for round_num in range(max_tool_rounds):
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content
            }
            
            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Check termination conditions
            if response.stop_reason != "tool_use":
                # Claude didn't use tools - we're done
                return response.content[0].text
            
            if not tool_manager:
                # No tool manager available - return tool use response
                return response.content[0].text if response.content else "No response"
            
            # Execute tools and prepare for next round
            messages.append({"role": "assistant", "content": response.content})
            tool_results = self._execute_tools(response, tool_manager)
            
            if not tool_results:
                # Tool execution failed - return error
                return "Tool execution failed"
            
            messages.append({"role": "user", "content": tool_results})
        
        # Max rounds reached - make final call without tools
        return self._make_final_response(messages, system_content)
    
    def _execute_tools(self, response, tool_manager) -> List[Dict]:
        """
        Execute all tool calls from a response and return results.
        
        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools
            
        Returns:
            List of tool results or empty list if execution fails
        """
        tool_results = []
        
        try:
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
        except Exception as e:
            # Log error and return empty results
            print(f"Tool execution error: {str(e)}")
            return []
        
        return tool_results
    
    def _make_final_response(self, messages: List[Dict], system_content: str) -> str:
        """
        Make final API call without tools when max rounds reached.
        
        Args:
            messages: Complete conversation history
            system_content: System prompt content
            
        Returns:
            Final response text
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
            # Explicitly no tools
        }
        
        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception as e:
            return f"Error generating final response: {str(e)}"
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Legacy method for backward compatibility with single-round tool execution.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute tools using new method
        tool_results = self._execute_tools(initial_response, tool_manager)
        
        if not tool_results:
            return "Tool execution failed"
        
        # Add tool results as single message
        messages.append({"role": "user", "content": tool_results})
        
        # Use new final response method
        return self._make_final_response(messages, base_params["system"])