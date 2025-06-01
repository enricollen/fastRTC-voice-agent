"""
LLM service module for handling interactions with language models.
"""
import os
from loguru import logger
import litellm


class LLMService:
    """
    Service for interacting with Language Models through LiteLLM.
    Supports different providers like OpenAI, Gemini, Ollama, OpenRouter, and Groq.
    """

    def __init__(self):
        """Initialize the LLM service with configuration from environment variables."""
        self.llm_mode = os.getenv("LLM_MODE", "cloud")
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))
        
        # model mappings
        self.model_mapping = {
            "openai": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "ollama": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            "gemini": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            "openrouter": os.getenv("OPENROUTER_MODEL", "qwen/qwq-32b:free"),
            "groq": os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
        }
        
        # api base URLs
        self.api_base_mapping = {
            "openai": os.getenv("OPENAI_API_BASE", None),
            "ollama": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            "gemini": os.getenv("GEMINI_API_BASE", None),
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": os.getenv("GROQ_API_BASE", None),
        }
        
        # parse fallback models from environment
        self.fallback_models = os.getenv("LLM_FALLBACKS", "").split(",")
        
        # set current model based on configuration
        self.model = self._get_model_name()
        
        logger.debug(f"LLM Service initialized with provider: {self.llm_provider}, model: {self.model}")

    def _get_model_name(self):
        """
        Get the correct model name based on LLM provider and mode.
        
        Returns:
            str: The model name for LiteLLM
        """
        provider = self.llm_provider.lower()
        
        if self.llm_mode == "local":
            provider = "ollama"
        
        # get the model from mapping
        model = self.model_mapping.get(provider)

        # add provider prefixes as needed
        if provider == "ollama":
            model = f"ollama/{model}"
        elif provider == "openrouter":
            model = f"openrouter/{model}"
        elif provider == "groq":
            model = f"groq/{model}"
        
        return model

    def generate_response(self, messages):
        """
        Generate a response from the LLM based on the given messages.
        
        Args:
            messages (list): List of message dictionaries with role and content
            
        Returns:
            str: The generated response text
            
        Raises:
            Exception: If all model attempts fail
        """
        provider = self.llm_provider.lower()
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        # add API base if configured
        api_base = self.api_base_mapping.get(provider)
        if api_base:
            kwargs["api_base"] = api_base
        
        # add API key if provider is OpenRouter or Groq
        if provider == "openrouter":
            kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY")
            # dummy OpenAI API key is required by OpenRouter
            os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy_value")
        elif provider == "groq":
            # groq uses GROQ_API_KEY environment variable by default in litellm
            if not os.getenv("GROQ_API_KEY"):
                logger.warning("GROQ_API_KEY environment variable not set")
        
        logger.debug(f"Generating response with model: {self.model}")
        
        # try primary model
        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with primary model: {str(e)}")
            logger.error(f"Provider List: https://docs.litellm.ai/docs/providers")
            
            # try fallback models
            for fallback_model in self.fallback_models:
                if not fallback_model.strip():
                    continue
                    
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    kwargs["model"] = fallback_model
                    response = litellm.completion(**kwargs)
                    return response.choices[0].message.content
                except Exception as fallback_error:
                    logger.error(f"Fallback model {fallback_model} failed: {str(fallback_error)}")
            
            # if all attempts fail ==> error message
            return "mi dispiace, ma ho un problema di connessione. potresti ripetere la tua domanda?" 