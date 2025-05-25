# FastRTC Voice Agent

A voice-enabled AI assistant that can engage in natural conversations. The project combines FastRTC for real-time communication, ElevenLabs for speech synthesis/recognition, and Ollama for local LLM-powered agent.

## Features

- Real-time voice conversations in multiple languages
- Speech-to-Text (ElevenLabs)
- Text-to-Speech (ElevenLabs)
- Web interface or phone number access (Gradio)
- Fully customizable assistant persona

## Prerequisites
- ElevenLabs API key
- Microphone

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
   cp .env.example .env
   # Edit .env and add your ELEVENLABS_API_KEY
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
3. Generates a textual response with your local LLM using Ollama
4. Converts the response to speech using ElevenLabs
5. Plays back the audio response