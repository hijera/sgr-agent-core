"""Context window management for infinite agent execution."""

import tiktoken
from typing import Any


class ContextWindowManager:
    """Manages conversation history to stay within token limits."""

    def __init__(
        self,
        max_tokens: int = 70000,
        system_prompt_reserve: int = 2000,
        response_reserve: int = 12000,
        model: str = "gpt-4",
    ):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum context window size
            system_prompt_reserve: Tokens reserved for system prompt
            response_reserve: Tokens reserved for model response
            model: Model name for tokenizer
        """
        self.max_tokens = max_tokens
        self.system_prompt_reserve = system_prompt_reserve
        self.response_reserve = response_reserve
        self.available_tokens = max_tokens - system_prompt_reserve - response_reserve
        
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """
        Count tokens in message list.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total token count
        """
        total_tokens = 0
        for message in messages:
            # Count role tokens
            total_tokens += 4  # Every message has role overhead
            
            # Count content tokens
            if message.get("content"):
                total_tokens += len(self.encoding.encode(str(message["content"])))
            
            # Count tool_calls tokens
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    if tool_call.get("function"):
                        total_tokens += len(
                            self.encoding.encode(
                                tool_call["function"].get("name", "")
                                + tool_call["function"].get("arguments", "")
                            )
                        )
        
        return total_tokens

    def truncate_conversation(
        self,
        conversation: list[dict[str, Any]],
        keep_first_n: int = 2,
        keep_last_n: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Truncate conversation to fit within token limits.
        
        Strategy:
        1. Always keep first N messages (initial task)
        2. Always keep last N messages (recent context)
        3. Remove middle messages if needed
        4. Add summary message about removed content
        
        Args:
            conversation: Full conversation history
            keep_first_n: Number of initial messages to keep
            keep_last_n: Number of recent messages to keep
            
        Returns:
            Truncated conversation
        """
        if len(conversation) <= keep_first_n + keep_last_n:
            return conversation
        
        current_tokens = self.count_tokens(conversation)
        
        if current_tokens <= self.available_tokens:
            return conversation
        
        # Keep first and last messages
        first_messages = conversation[:keep_first_n]
        last_messages = conversation[-keep_last_n:]
        removed_count = len(conversation) - keep_first_n - keep_last_n
        
        # Create summary message
        summary_message = {
            "role": "system",
            "content": f"[Context truncated: {removed_count} intermediate messages removed to manage context window]",
        }
        
        truncated = first_messages + [summary_message] + last_messages
        
        # If still too large, remove more aggressively
        if self.count_tokens(truncated) > self.available_tokens:
            # Keep only essential messages
            truncated = [
                conversation[0],  # Initial task
                summary_message,
                *conversation[-5:],  # Last 5 messages
            ]
        
        return truncated

    def sliding_window_truncate(
        self,
        conversation: list[dict[str, Any]],
        window_size: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Use sliding window approach - keep only recent messages.
        
        Args:
            conversation: Full conversation history
            window_size: Number of recent messages to keep
            
        Returns:
            Truncated conversation with sliding window
        """
        if len(conversation) <= window_size:
            return conversation
        
        # Always keep first message (initial task)
        first_message = conversation[0]
        recent_messages = conversation[-window_size:]
        
        # Add summary
        removed_count = len(conversation) - window_size - 1
        summary_message = {
            "role": "system",
            "content": f"[Sliding window: {removed_count} older messages removed. Keeping initial task and last {window_size} messages]",
        }
        
        return [first_message, summary_message] + recent_messages

    def smart_truncate(
        self,
        conversation: list[dict[str, Any]],
        preserve_tool_results: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Smart truncation that preserves important messages.
        
        Keeps:
        - Initial task (first message)
        - Tool execution results (if preserve_tool_results=True)
        - Recent messages
        - Final answers and reasoning
        
        Args:
            conversation: Full conversation history
            preserve_tool_results: Whether to keep tool execution results
            
        Returns:
            Smartly truncated conversation
        """
        current_tokens = self.count_tokens(conversation)
        
        if current_tokens <= self.available_tokens:
            return conversation
        
        # Categorize messages
        initial_task = [conversation[0]] if conversation else []
        important_messages = []
        recent_messages = []
        
        # Keep last 8 messages as recent
        recent_threshold = max(1, len(conversation) - 8)
        
        for idx, msg in enumerate(conversation[1:], start=1):
            if idx >= recent_threshold:
                recent_messages.append(msg)
            elif preserve_tool_results and msg.get("role") == "tool":
                # Keep tool results as they contain search/extraction data
                important_messages.append(msg)
            elif msg.get("tool_calls"):
                # Keep messages with tool calls
                tool_names = [
                    tc.get("function", {}).get("name", "")
                    for tc in msg.get("tool_calls", [])
                ]
                # Keep important tools: FinalAnswer, CreateReport, WebSearch
                if any(
                    name in ["finalanswertool", "createreporttool", "websearchtool"]
                    for name in tool_names
                ):
                    important_messages.append(msg)
        
        # Combine and check tokens
        truncated = initial_task + important_messages + recent_messages
        
        # If still too large, keep only initial + recent
        if self.count_tokens(truncated) > self.available_tokens:
            removed_count = len(important_messages)
            summary = {
                "role": "system",
                "content": f"[Context optimized: {removed_count} intermediate messages removed]",
            }
            truncated = initial_task + [summary] + recent_messages[-6:]
        
        return truncated

