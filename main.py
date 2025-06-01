import argparse
from typing import Generator, Tuple
import numpy as np
from loguru import logger
from src.speech import SpeechService
from src.agent import Agent
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
)

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# Initialize services
speech_service = None  
agent = Agent()
voice_id = None  
speed = 1.0  # just for kokoro
tts_provider = "elevenlabs"  # default tts provider
stt_provider = "elevenlabs"  # default stt provider

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
        default="elevenlabs",
        help="TTS provider to use (elevenlabs or kokoro)",
    )
    parser.add_argument(
        "--stt", 
        choices=["elevenlabs", "groq", "openai"], 
        default="elevenlabs",
        help="STT provider to use (elevenlabs, groq, or openai)",
    )
    parser.add_argument(
        "--voice", 
        type=str, 
        help="Voice ID/name to use (provider-specific, defaults to provider's default)",
        default=None,
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (only applicable for Kokoro TTS)",
    )
    args = parser.parse_args()

    # configuration in global variables
    tts_provider = args.tts
    stt_provider = args.stt
    
    # Set default voice ID based on the TTS provider if not specified
    if args.voice is None:
        if args.tts == "elevenlabs":
            voice_id = "JBFqnCBsd6RMkjVDRZzb"  # default elevenlabs voice id
        else:  # kokoro
            voice_id = "im_nicola"  # default kokoro voice
    else:
        voice_id = args.voice
        
    speed = args.speed
    
    speech_service = SpeechService(tts_provider=tts_provider, stt_provider=stt_provider)
    
    # info about the configuration
    logger.info(f"🔊 Initialized speech service with {tts_provider} TTS provider")
    logger.info(f"🎤 Initialized speech service with {stt_provider} STT provider")
    if voice_id:
        logger.info(f"Using voice: {voice_id}")
    if tts_provider == "kokoro" and speed != 1.0:
        logger.info(f"Speech speed: {speed}x")
    
    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
        logger.info("🚀 Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("🚀 Launching with Gradio UI...")
        stream.ui.launch()