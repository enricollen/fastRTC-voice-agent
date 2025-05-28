import argparse
from typing import Generator, Tuple
import numpy as np
from loguru import logger
from src.process_tts import process_elevenlabs_tts, process_elevenlabs_stt
from src.agent import invoke as agent_invoke
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
    transcript = process_elevenlabs_stt(audio)
    logger.info(f'👂 Transcribed: "{transcript}"')    
    logger.debug("🧠 Running agent...")
    agent_response = agent_invoke(transcript)
    response_text = agent_response["messages"][-1]["content"]
    logger.info(f'💬 Response: "{response_text}"')

    logger.debug("🔊 Generating speech...")
    yield from process_elevenlabs_tts(response_text)


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
    args = parser.parse_args()

    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
        logger.info("🚀 Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("🚀 Launching with Gradio UI...")
        stream.ui.launch()