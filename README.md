# 🎙️ FastRTC Voice Agent

A model-agnostic voice-enabled AI assistant that can engage in natural conversations. The project combines FastRTC for real-time communication, multiple speech services, and LiteLLM for flexible model support.

## ✨ Features

- Real-time voice conversations in multiple languages
- **Model-agnostic architecture**: Mix and match different providers for each component
- Speech-to-Text: ElevenLabs, Groq, OpenAI
- Text-to-Speech: ElevenLabs, Kokoro
- Web interface or phone number access (Gradio)
- Fully customizable assistant persona
- LLM-agnostic: easily switch between OpenAI, Gemini, Groq, Ollama, and OpenRouter
- Automatic fallback to alternative models if primary model fails
- Session-based chat history for context-aware conversations
- Clean, modular class-based architecture
- Environment variables and command-line arguments for flexible configuration

## 🔧 Prerequisites
- API keys for your preferred providers
- Microphone
- For local models: Ollama installed and running

## 🚀 How to use

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
   cp .env_example .env
   # Edit .env and add your API keys and model settings
   ```

5. Run the application:
   ```bash
   # Default configuration from .env
   python main.py
   
   # Override with command-line arguments
   python main.py --tts elevenlabs --stt openai --voice custom_voice_id
   ```

## 🏗️ Architecture

The project follows a modular class-based design:

- **Agent**: Coordinates conversation flow and manages the interaction between components
- **LLMService**: Handles communication with different LLM providers through LiteLLM
- **ChatHistory**: Manages conversation history and context
- **SpeechService**: Handles both text-to-speech and speech-to-text using multiple providers

## 🧩 Provider Configuration Examples

### 🌐 Full Cloud Setup (Default)
Using cloud services for all components:

```
# Speech service
TTS_PROVIDER=elevenlabs
STT_PROVIDER=elevenlabs

# LLM
LLM_PROVIDER=openai
OPENAI_LLM_MODEL=gpt-3.5-turbo
```

### 💻 Full Local Setup
Running everything locally:

```
# Speech service
TTS_PROVIDER=kokoro
STT_PROVIDER=openai  # Currently no local STT option

# LLM
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_API_BASE=http://localhost:11434
```

### 🔄 Hybrid Setup (Local LLM, Cloud Speech)
Local LLM with cloud speech services:

```
# Speech service
TTS_PROVIDER=elevenlabs
STT_PROVIDER=elevenlabs

# LLM
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

### 🔄 Hybrid Setup (Cloud LLM, Mixed Speech)
Cloud LLM with mixed speech services:

```
# Speech service
TTS_PROVIDER=kokoro  # Local TTS
STT_PROVIDER=openai  # Cloud STT

# LLM
LLM_PROVIDER=openai
OPENAI_LLM_MODEL=gpt-3.5-turbo
```

## 🔄 Speech Service Configuration

The system supports multiple providers for both speech-to-text and text-to-speech:

### 🎤 Speech-to-Text Providers

#### ElevenLabs
```
STT_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_STT_MODEL=scribe_v1
ELEVENLABS_STT_LANGUAGE=ita
```

#### Groq
```
STT_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_STT_MODEL=whisper-large-v3-turbo
GROQ_STT_LANGUAGE=it
```

#### OpenAI
```
STT_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_STT_MODEL=gpt-4o-transcribe
OPENAI_STT_LANGUAGE=it
```

### 🔊 Text-to-Speech Providers

#### ElevenLabs
```
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
ELEVENLABS_TTS_MODEL=eleven_multilingual_v2
ELEVENLABS_LANGUAGE=it
```

#### Kokoro
```
TTS_PROVIDER=kokoro
KOKORO_VOICE=im_nicola
KOKORO_LANGUAGE=i
TTS_SPEED=1.0  # Adjust speech speed
```

## 🧠 LLM Configuration

This project uses LiteLLM to support multiple LLM providers. Configure your preferred model in the `.env` file:

### Local Models with Ollama

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_API_BASE=http://localhost:11434
```

### Cloud Models

#### OpenAI
```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_LLM_MODEL=gpt-3.5-turbo
```

#### Google Gemini
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-flash
```

#### Groq
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_LLM_MODEL=llama-3.1-8b-instant
```

#### OpenRouter (access to many models)
```
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
MAX_HISTORY_MESSAGES=5
```

To clear the chat history during a conversation, just say "clear history", "reset chat", or "nuova conversazione".

## 🚀 Running the Application

Start the application with the web interface:
```bash
python main.py
```

Or get a temporary phone number for voice calls:
```bash
python main.py --phone
```

Override environment settings with command-line arguments:
```bash
# Use Kokoro TTS with a specific voice and speed
python main.py --tts kokoro --voice im_nicola --speed 1.0

# Use OpenAI for speech recognition
python main.py --stt openai --tts elevenlabs
```

## ⚙️ How it Works

1. The system captures your voice input
2. Converts speech to text using your configured STT provider
3. Sends the text to your configured LLM via LiteLLM
4. Converts the response to speech using your configured TTS provider
5. Plays back the audio response

## 🔌 Extending the Project

The modular design makes it easy to extend the project:

- Add new LLM providers by extending the LLMService class
- Support additional speech services by creating alternate implementations
- Create custom agents with different behaviors by extending the Agent class