"""
Agent module to coordinate conversation between user and AI.
"""
import os
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from .llm_service import LLMService
from .chat_history import ChatHistory

load_dotenv()

class Agent:
    """
    AI Agent that coordinates conversation flow between the user and LLM.
    Manages system prompts, chat history, and LLM interactions.
    """

    def __init__(self):
        """Initialize the agent with necessary services and configuration."""
        # get system prompt from yaml
        config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            prompts_config = yaml.safe_load(f)
            self.system_prompt = prompts_config['system_prompts']['chef_assistant']
        
        # initialize chat history and LLM
        self.chat_history = ChatHistory(self.system_prompt)
        self.llm_service = LLMService()
        
        logger.debug("Agent initialized with system prompt and services")

    def invoke(self, input_text: str, config: dict = None):
        """
        Process user input and generate a response using the LLM.
        
        Args:
            input_text: The user's input text
            config: Optional configuration dictionary. Defaults to None.
            
        Returns:
            dict: Response containing the messages
        """
        if config is None:
            config = {"configurable": {"thread_id": "default_user"}}
        
        logger.info(f'💭 Thinking about: "{input_text}"')
        
        # get the current chat history
        messages = self.chat_history.get_messages()
        
        # add the new user message for the LLM (not adding to history yet)
        messages.append({"role": "user", "content": input_text})
        
        logger.debug(f"Context length: {len(messages)} messages")
        
        # generate response using the LLM
        try:
            assistant_response = self.llm_service.generate_response(messages)
            
            # update chat history with this last exchange
            self.chat_history.add_exchange(input_text, assistant_response)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            assistant_response = "Mi dispiace, ma ho un problema di connessione. Potresti ripetere la tua domanda?"
            self.chat_history.add_exchange(input_text, assistant_response)
        
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
        return self.chat_history.clear()