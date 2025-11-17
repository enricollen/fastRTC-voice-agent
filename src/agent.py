"""
Agent module to coordinate conversation between user and AI.
"""
import os
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage

from .llm_service import LLMService
#from .chat_history import ChatHistory
from .tools import get_weather

load_dotenv()

class Agent:
    """
    AI Agent that coordinates conversation flow between the user and LLM.
    Manages system prompts, chat history, and LLM interactions.
    Uses LlamaIndex ReAct agent for tool use capabilities.
    """

    def __init__(self):
        """Initialize the agent with necessary services and configuration."""
        # get system prompt from yaml
        config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            prompts_config = yaml.safe_load(f)
            self.system_prompt = prompts_config['system_prompts']['weather_expert']
        
        self.llm_service = LLMService()
        
        # initialize tools
        self.tools = [
            FunctionTool.from_defaults(
                fn=get_weather,
                name="get_weather",
                description="get current weather in a location, ONLY use this when explicitly asked about weather"
            )
        ]
        
        # LLamaIndex memory with token limits
        self.memory = Memory.from_defaults(
            token_limit=4000, 
            chat_history_token_ratio=0.8  # 80% of token limit for chat history
        )
        
        # add system prompt to memory
        self.memory.put_messages([
            ChatMessage(role="system", content=self.system_prompt)
        ])
        
        # llamaindex react agent
        self._init_agent()
        
        logger.debug("Agent initialized with system prompt, memory and services")

    def _init_agent(self):
        """Initialize the LlamaIndex ReAct agent with tools."""
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # get the LiteLLM instance from LLMService
        llm = self.llm_service.llm
        
        # create react agent with tools
        self.react_agent = ReActAgent(
            #name="Assistant",
            #description="An AI assistant that can help with various tasks",
            tools=self.tools,
            llm=llm,
            verbose=True,
            system_prompt=self.system_prompt
        )
        
        logger.info(f"react agent initialized with llm: {llm_provider} using model: {self.llm_service.model}")

    def _strip_think_tags(self, text: str) -> str:
        """Remove content between <think> tags from the response."""
        import re
        return re.sub(r'<think>.*?</think>\s*\n?', '', text, flags=re.DOTALL).strip()

    async def _run_agent_async(self, input_text: str):
        """async helper to run the agent with context."""
        # create context on first use to store conversation history
        if self.ctx is None:
            self.ctx = Context(self.react_agent)
        
        # run agent and await response
        handler = self.react_agent.run(user_msg=input_text, ctx=self.ctx)
        return await handler

    def invoke(self, input_text: str, config: dict = None):
        """
        Process user input and generate a response using the LLM.
        Will use tools if needed based on the input.
        
        Args:
            input_text: The user's input text
            config: Optional configuration dictionary. Defaults to None.
            
        Returns:
            dict: Response containing the messages
        """
        if config is None:
            config = {"configurable": {"thread_id": "default_user"}}
        
        logger.info(f'💭 Thinking about: "{input_text}"')
        
        try:
            # run the async agent in sync context
            import asyncio
            response = asyncio.run(self._run_agent_async(input_text))
            
            # removing <think> tags
            assistant_response = self._strip_think_tags(str(response))
            
            #logger.info(f'💬 agent response: "{assistant_response}"')
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            assistant_response = "Mi dispiace, ma ho un problema di connessione. Potresti ripetere la tua domanda?"
            # add the failed exchange to memory
            self.memory.put_messages([
                ChatMessage(role="user", content=input_text),
                ChatMessage(role="assistant", content=assistant_response)
            ])
        
        # return formatted response
        return {
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": assistant_response}
            ]
        }

    def clear_history(self):
        """
        Clear the chat history.
        
        Returns:
            bool: True if history was cleared successfully
        """
        self.memory = Memory.from_defaults(
            token_limit=4000,
            chat_history_token_ratio=0.8
        )
        # re-add system prompt to memory
        self.memory.put_messages([
            ChatMessage(role="system", content=self.system_prompt)
        ])
        return True