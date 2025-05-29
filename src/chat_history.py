"""
Chat history management module.
"""
import os
from loguru import logger


class ChatHistory:
    """
    Manages the conversation history for the voice agent.
    Maintains a list of messages in memory for the duration of the session.
    """

    def __init__(self, system_prompt):
        """
        Initialize the chat history with a system prompt.
        
        Args:
            system_prompt (str): The system prompt to set the AI's behavior.
        """
        self.system_prompt = system_prompt
        self.max_history_pairs = int(os.getenv("MAX_HISTORY_MESSAGES", "5"))
        self._messages = [{"role": "system", "content": system_prompt}]
        logger.debug(f"Chat history initialized with max {self.max_history_pairs} message pairs")

    def get_messages(self):
        """
        Get the current chat history.
        
        Returns:
            list: A copy of the message history to send to the LLM.
        """
        # check if system prompt is at the beginning
        if not self._messages or self._messages[0].get("role") != "system":
            self._messages.insert(0, {"role": "system", "content": self.system_prompt})
            
        return self._messages.copy()

    def add_exchange(self, user_message, assistant_message):
        """
        Add a user-assistant message pair to the history.
        
        Args:
            user_message (str): The user's message.
            assistant_message (str): The assistant's response.
        """
        # check if system prompt is at the beginning
        if not self._messages or self._messages[0].get("role") != "system":
            self._messages = [{"role": "system", "content": self.system_prompt}]
            
        # add the new exchange
        self._messages.append({"role": "user", "content": user_message})
        self._messages.append({"role": "assistant", "content": assistant_message})
        
        # cut off history if it exceeds the maximum size
        max_messages = (self.max_history_pairs * 2) + 1  # +1 for system message
        if len(self._messages) > max_messages:
            # keep system message and cut off the rest
            self._messages = [self._messages[0]] + self._messages[-(max_messages-1):]
            
        logger.debug(f"Chat history updated, now contains {len(self._messages)} messages")

    def clear(self):
        """
        Reset the chat history to just the system message.
        
        Returns:
            bool: True if the history was cleared successfully.
        """
        self._messages = [{"role": "system", "content": self.system_prompt}]
        logger.info("Chat history cleared")
        return True

    def current_length(self):
        """
        Get the current number of messages in the history.
        
        Returns:
            int: The number of messages.
        """
        return len(self._messages) 