# FastRTC Voice Agent

A voice-enabled AI assistant that can engage in natural conversations. The project combines FastRTC for real-time communication, ElevenLabs for speech synthesis/recognition, and LiteLLM for LLM-agnostic model support.

## Features

- Real-time voice conversations in multiple languages
- Speech-to-Text (ElevenLabs, Groq, OpenAI)
- Text-to-Speech (ElevenLabs)
- Web interface or phone number access (Gradio)
- Fully customizable assistant persona
- LLM-agnostic: easily switch between OpenAI, Gemini, Ollama, and OpenRouter
- Automatic fallback to alternative models if primary model fails
- Session-based chat history for context-aware conversations
- Clean, modular class-based architecture

## Prerequisites
- ElevenLabs API key
- API key for your preferred LLM provider (OpenAI, Gemini, or OpenRouter)
- Optional: Groq or OpenAI API key for speech-to-text
- Microphone
- For local models: Ollama installed and running

## How to use

1. Clone the repository and navigate to the project directory

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Unix/MacOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env-example .env
   # Edit .env and add your API keys and model settings
   ```

## Architecture

The project follows a modular class-based design:

- **Agent**: Coordinates conversation flow and manages the interaction between components
- **LLMService**: Handles communication with different LLM providers through LiteLLM
- **ChatHistory**: Manages conversation history and context
- **SpeechService**: Handles both text-to-speech and speech-to-text using ElevenLabs

## LLM Configuration

This project uses LiteLLM to support multiple LLM providers. Configure your preferred model in the `.env` file:

### Local Models with Ollama

```
LLM_MODE=local
OLLAMA_MODEL=llama3.1:8b
OLLAMA_API_BASE=http://localhost:11434
```

### Cloud Models

For OpenAI:
```
LLM_MODE=cloud
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_LLM_MODEL=gpt-3.5-turbo
```

For Google Gemini:
```
LLM_MODE=cloud
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-pro
```

For OpenRouter (access to many models):
```
LLM_MODE=cloud
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=qwen/qwq-32b:free
```

### Model Fallbacks

You can specify fallback models in case your primary model fails:

```
LLM_FALLBACKS=gpt-3.5-turbo,ollama/llama3.1:8b
```

### Chat History

The system maintains conversation context by storing recent messages in session memory. 
You can configure how many messages to keep in the context:

```
MAX_HISTORY_MESSAGES=10
```

To clear the chat history during a conversation, just say "clear history", "reset chat", or "nuova conversazione".

## Speech-to-Text Configuration

The project supports multiple STT providers:

### ElevenLabs (default)
```
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_STT_LANGUAGE=en
```

### Groq
```
GROQ_API_KEY=your_groq_api_key
GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_STT_LANGUAGE=en
```

### OpenAI
```
OPENAI_API_KEY=your_openai_api_key
OPENAI_STT_MODEL=gpt-4o-transcribe
OPENAI_STT_LANGUAGE=en
```

To select your preferred STT provider, set:
```
DEFAULT_STT_PROVIDER=elevenlabs  # or groq or openai
```

## Running the Application

Start the application with the web interface:
```bash
python main.py
```

Or get a temporary phone number for voice calls:
```bash
python main.py --phone
```

## How it Works

1. The system captures your voice input
2. Converts speech to text using ElevenLabs
3. Sends the text to your configured LLM via LiteLLM
4. Converts the response to speech using ElevenLabs
5. Plays back the audio response

## Extending the Project

The modular design makes it easy to extend the project:

- Add new LLM providers by extending the LLMService class
- Support additional speech services by creating alternate implementations of SpeechService
- Create custom agents with different behaviors by extending the Agent class