import os
import tempfile
import numpy as np
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from typing import Generator, Tuple

def process_elevenlabs_tts(
    text: str,
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    language: str = "it",
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process ElevenLabs TTS response into a complete audio segment in Italian.

    Args:
        text: Text to synthesize
        voice_id: ElevenLabs voice ID
        model_id: ElevenLabs model ID
        output_format: Output audio format
        language: Language code (default 'it' for Italian)

    Yields:
        A single tuple of (sample_rate, audio_array) for audio playback
    """
    load_dotenv()
    elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    
    # get the bytes directly from the generator
    audio = b"".join(elevenlabs.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    ))
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_file.write(audio)
    temp_file_path = temp_file.name
    temp_file.close()
    try:
        import soundfile as sf
        data, sample_rate = sf.read(temp_file_path, dtype="int16")
        audio_array = np.array(data).reshape(1, -1)
        yield (sample_rate, audio_array)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def process_elevenlabs_stt(
    audio: tuple[int, np.ndarray],
    model_id: str = "scribe_v1",
    language_code: str = "ita",
    diarize: bool = False,
    tag_audio_events: bool = False,
) -> str:
    """
    Process audio input using ElevenLabs Speech-to-Text API in Italian.

    Args:
        audio: Tuple containing sample rate and audio data
        model_id: ElevenLabs STT model ID
        language_code: Language code (default 'ita' for Italian)
        diarize: Whether to annotate who is speaking
        tag_audio_events: Tag audio events like laughter, applause, etc.

    Returns:
        Transcribed text
    """
    import io
    import soundfile as sf

    load_dotenv()
    elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    # write audio to a BytesIO buffer as wav
    buf = io.BytesIO()
    sf.write(buf, audio[1].flatten(), audio[0], format="WAV")
    buf.seek(0)
    transcription = elevenlabs.speech_to_text.convert(
        file=buf,
        model_id=model_id,
        tag_audio_events=tag_audio_events,
        language_code=language_code,
        diarize=diarize,
    )
    # return just the text part of the transcription
    return transcription.text if hasattr(transcription, 'text') else transcription