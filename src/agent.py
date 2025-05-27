from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

model = ChatOllama(
    model="llama3.1:8b", 
    temperature=0.5,
)

system_prompt = """Sei Fernando, un'assistente chef di cucina amichevole e disponibile.
Le tue risposte verranno convertite in audio quindi evita di usare caratteri o simboli speciali.
Mantieni le tue risposte amichevoli e conversazionali in italiano.

Per le ricette, spiega gli ingredienti in modo chiaro e semplice."""

def invoke(input_text: str, config: dict = None):
    """
    Simple chat agent that processes the input and returns a response.
    
    Args:
        input_text: The user's input text
        config: Optional configuration dictionary
    
    Returns:
        dict: Response containing the messages
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_text)
    ]
    
    response = model.invoke(messages)
    logger.info(f'💭 Thinking about: "{input_text}"')
    
    return {
        "messages": [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": response.content}
        ]
    }

agent_config = {"configurable": {"thread_id": "default_user"}}
