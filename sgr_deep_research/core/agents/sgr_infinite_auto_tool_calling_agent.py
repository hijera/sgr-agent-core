"""Infinite SGR Auto Tool Calling Agent with context management."""

from typing import Literal, Type

from sgr_deep_research.core.agents.sgr_tools_agent import SGRToolCallingResearchAgent
from sgr_deep_research.core.context_manager import ContextWindowManager
from sgr_deep_research.core.models import AgentStatesEnum
from sgr_deep_research.core.prompts import PromptLoader
from sgr_deep_research.core.tools import BaseTool, FinalAnswerTool
from sgr_deep_research.settings import get_config

config = get_config()


class SGRInfiniteAutoToolCallingAgent(SGRToolCallingResearchAgent):
    """
    Infinite SGR Auto Tool Calling Research Agent.
    
    Features:
    - Never auto-completes (infinite execution)
    - Automatic context window management (70k tokens)
    - Can continue conversation by agent ID
    - Uses tool_choice="auto" for flexible responses
    """

    name: str = "sgr_infinite_auto_tool_calling_agent"

    def __init__(
        self,
        task: str,
        toolkit: list[Type[BaseTool]] | None = None,
        max_clarifications: int = 10,  # Increased for infinite mode
        max_searches: int = 20,  # Increased for infinite mode
        max_iterations: int = 1000,  # Very high limit for infinite mode
        context_window_size: int = 70000,  # 70k tokens
    ):
        super().__init__(task, toolkit, max_clarifications, max_searches, max_iterations)
        self.tool_choice: Literal["auto"] = "auto"
        
        # Initialize context manager
        self.context_manager = ContextWindowManager(
            max_tokens=context_window_size,
            system_prompt_reserve=2000,
            response_reserve=config.openai.max_tokens,
            model=config.openai.model,
        )
        
        # Track if agent should continue
        self.is_infinite = True
        self.user_requested_stop = False

    async def _prepare_context(self) -> list[dict]:
        """
        Prepare conversation context with automatic truncation.
        
        Manages context window to stay within 70k tokens by:
        1. Keeping initial task
        2. Keeping recent messages
        3. Preserving important tool results
        4. Removing old intermediate messages
        """
        # Get infinite mode system prompt
        system_prompt_content = PromptLoader.get_infinite_system_prompt(self.toolkit)
        
        # Apply smart truncation to conversation
        truncated_conversation = self.context_manager.smart_truncate(
            self.conversation,
            preserve_tool_results=True,
        )
        
        # Log if truncation occurred
        if len(truncated_conversation) < len(self.conversation):
            removed = len(self.conversation) - len(truncated_conversation)
            self.logger.info(
                f"🔄 Context truncated: removed {removed} messages, "
                f"keeping {len(truncated_conversation)} messages"
            )
            token_count = self.context_manager.count_tokens(truncated_conversation)
            self.logger.info(f"📊 Current context tokens: {token_count}/{self.context_manager.available_tokens}")
        
        # Rebuild context with system prompt
        result = [{"role": "system", "content": system_prompt_content}]
        result.extend(truncated_conversation)
        
        return result

    async def _select_action_phase(self, reasoning) -> BaseTool:
        """
        Select action with infinite mode handling.
        
        Overrides parent to:
        1. Handle content-only responses
        2. Never auto-complete unless user explicitly requests
        3. Convert FinalAnswerTool to continuation in infinite mode
        """
        tool = await super()._select_action_phase(reasoning)
        
        # In infinite mode, intercept FinalAnswerTool
        if isinstance(tool, FinalAnswerTool) and self.is_infinite and not self.user_requested_stop:
            # Check if this is a real user request to stop
            if tool.status == AgentStatesEnum.COMPLETED:
                # Log but don't actually complete
                self.logger.info(
                    "🔄 Infinite mode: Agent suggested completion but will continue. "
                    "Send 'stop' or 'finish' to end conversation."
                )
                # Change state back to researching
                self._context.state = AgentStatesEnum.RESEARCHING
        
        return tool

    async def continue_conversation(self, user_message: str):
        """
        Continue existing conversation with new user input.
        
        Args:
            user_message: New message from user
        """
        # Check if user wants to stop
        if user_message.lower().strip() in ["stop", "finish", "end", "завершить", "стоп", "закончить"]:
            self.user_requested_stop = True
            self._context.state = AgentStatesEnum.COMPLETED
            self.logger.info("🛑 User requested stop - completing agent")
            
            # Add stop message to conversation
            self.conversation.append({
                "role": "user",
                "content": "User requested to stop the conversation.",
            })
            return
        
        # Add user message to conversation
        self.conversation.append({
            "role": "user",
            "content": user_message,
        })
        
        # Reset state to researching if it was completed
        if self._context.state in AgentStatesEnum.FINISH_STATES.value:
            self._context.state = AgentStatesEnum.RESEARCHING
        
        self.logger.info(f"💬 Continuing conversation: {user_message[:100]}...")

    async def execute(self):
        """
        Execute infinite agent loop.
        
        Overrides parent to:
        1. Never auto-complete
        2. Wait for user input instead of finishing
        3. Manage context window automatically
        """
        self.logger.info(f"🚀 Starting INFINITE agent for task: '{self.task}'")
        self.conversation.extend([
            {
                "role": "user",
                "content": f"Task: {self.task}\n\nNote: This is an infinite conversation mode. "
                           "I will continue helping until you explicitly say 'stop' or 'finish'.",
            }
        ])
        
        try:
            while not self.user_requested_stop:
                self._context.iteration += 1
                self.logger.info(f"🔄 Infinite Step {self._context.iteration} started")
                
                # Check context size and log
                token_count = self.context_manager.count_tokens(self.conversation)
                self.logger.info(f"📊 Context tokens: {token_count}/{self.context_manager.available_tokens}")
                
                reasoning = await self._reasoning_phase()
                self._context.current_step_reasoning = reasoning
                action_tool = await self._select_action_phase(reasoning)
                await self._action_phase(action_tool)
                
                # Handle clarifications
                if action_tool.__class__.__name__ == "ClarificationTool":
                    self.logger.info("\n⏸️  Research paused - please answer questions")
                    self._context.state = AgentStatesEnum.WAITING_FOR_CLARIFICATION
                    self._context.clarification_received.clear()
                    await self._context.clarification_received.wait()
                    continue
                
                # In infinite mode, if agent tries to complete, wait for user input
                if isinstance(action_tool, FinalAnswerTool) and not self.user_requested_stop:
                    self.logger.info(
                        "\n⏸️  Agent provided answer but staying active. "
                        "Send new message to continue or 'stop' to finish."
                    )
                    self._context.state = AgentStatesEnum.WAITING_FOR_CLARIFICATION
                    self._context.clarification_received.clear()
                    await self._context.clarification_received.wait()
                    continue
                
        except Exception as e:
            self.logger.error(f"❌ Agent execution error: {str(e)}")
            self._context.state = AgentStatesEnum.FAILED
            import traceback
            traceback.print_exc()
        finally:
            if self.streaming_generator is not None:
                self.streaming_generator.finish()
            self._save_agent_log()
            self.logger.info("🏁 Infinite agent finished")

