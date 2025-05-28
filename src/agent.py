import os
from loguru import logger
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import yaml
from pathlib import Path

load_dotenv()

# get system prompt from yaml
config_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    prompts_config = yaml.safe_load(f)
    system_prompt = prompts_config['system_prompts']['chef_assistant']

llm_mode = os.getenv("LLM_MODE", "cloud")

if llm_mode == "cloud":
    # dummy OpenAI API key is required for OpenRouter to work
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "dummy_value"

    llm = OpenRouter(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=4096,
        context_window=32768,
        model=os.getenv("CLOUD_LLM_MODEL", "qwen/qwq-32b:free") 
    )
else:
    llm = ChatOllama(
        model=os.getenv("LOCAL_LLM_MODEL", "llama3.1:8b"),
        temperature=0.5,
    )

def invoke(input_text: str, config: dict = None):
    """
    Simple chat agent that processes the input and returns a response.
    Uses either OpenRouter (cloud) or Ollama (local) for model inference.
    
    Args:
        input_text: The user's input text
        config: Optional configuration dictionary. Defaults to None.
    
    Returns:
        dict: Response containing the messages
    """
    if config is None:
        config = {"configurable": {"thread_id": "default_user"}}

    logger.info(f'💭 Thinking about: "{input_text}"')

    if llm_mode == "cloud":
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=input_text)
        ]
        response = llm.chat(messages)
        assistant_response = response.message.content
    else:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input_text)
        ]
        response = llm.invoke(messages)
        assistant_response = response.content

    return {
        "messages": [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": assistant_response}
        ]
    }