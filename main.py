import argparse
from typing import Generator, Tuple
import numpy as np
import os
from loguru import logger
from dotenv import load_dotenv
from src.speech import SpeechService
from src.agent import Agent
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
)

# load environment variables
load_dotenv()

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# initialize services with defaults from environment variables
speech_service = None  
agent = Agent()

# get defaults from environment variables
default_tts_provider = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
default_stt_provider = os.getenv("STT_PROVIDER", "elevenlabs").lower()

# default voice based on provider
default_voice_id = None
if default_tts_provider == "elevenlabs":
    default_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
elif default_tts_provider == "kokoro":
    default_voice_id = os.getenv("KOKORO_VOICE", "im_nicola")

# default speed (only relevant for kokoro)
default_speed = float(os.getenv("TTS_SPEED", "1.0"))

def response(
    audio: tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input, transcribe it, generate a response using the agent, and deliver TTS audio.

    Args:
        audio: Tuple containing sample rate and audio data

    Yields:
        Tuples of (sample_rate, audio_array) for audio playback
    """
    logger.info("🎙️ Received audio input")

    logger.debug("🔄 Transcribing audio...")
    # set STT parameters based on the active provider
    stt_kwargs = {}
    if stt_provider in ["groq", "openai"]:
        stt_kwargs["response_format"] = "text"
        
    transcript = speech_service.speech_to_text(audio, **stt_kwargs)
    logger.info(f'👂 Transcribed: "{transcript}"')
    
    logger.debug("🧠 Running agent...")
    agent_response = agent.invoke(transcript)
    response_text = agent_response["messages"][-1]["content"]
    logger.info(f'💬 Response: "{response_text}"')

    logger.debug("🔊 Generating speech...")
    # set TTS parameters based on the active provider
    tts_kwargs = {}
    if voice_id:
        tts_kwargs["voice_id"] = voice_id
    
    # add speed parameter only for kokoro
    if tts_provider == "kokoro" and speed != 1.0:
        tts_kwargs["speed"] = speed
    
    yield from speech_service.text_to_speech(response_text, **tts_kwargs)


def create_stream() -> Stream:
    """
    Create and configure a Stream instance with audio capabilities.

    Returns:
        Stream: Configured FastRTC Stream instance
    """
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.2,
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastRTC Voice Agent")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (automatically provides a temporary phone number)",
    )
    parser.add_argument(
        "--tts", 
        choices=["elevenlabs", "kokoro"], 
        default=default_tts_provider,
        help="TTS provider to use (elevenlabs or kokoro)",
    )
    parser.add_argument(
        "--stt", 
        choices=["elevenlabs", "groq", "openai"], 
        default=default_stt_provider,
        help="STT provider to use (elevenlabs, groq, or openai)",
    )
    parser.add_argument(
        "--voice", 
        type=str, 
        help="Voice ID/name to use (provider-specific, defaults to provider's default)",
        default=default_voice_id,
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=default_speed,
        help="Speech speed multiplier (only applicable for Kokoro TTS)",
    )
    args = parser.parse_args()

    # configuration in global variables
    tts_provider = args.tts
    stt_provider = args.stt
    voice_id = args.voice
    speed = args.speed
    
    speech_service = SpeechService(
        tts_provider=tts_provider, 
        stt_provider=stt_provider
    )
    
    # info about the configuration
    logger.info(f"🔊 Initialized speech service with {tts_provider} TTS provider")
    logger.info(f"🎤 Initialized speech service with {stt_provider} STT provider")
    if tts_provider == "kokoro" and speed != 1.0:
        logger.info(f"Speech speed: {speed}x")
    logger.info(f"tts model preloaded during startup")
    
    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
        logger.info("🚀 Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("🚀 Launching with Gradio UI...")
        stream.ui.launch()