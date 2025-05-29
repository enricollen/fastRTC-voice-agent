"""
Speech service module for TTS and STT operations.
"""
import os
import tempfile
import numpy as np
from typing import Generator, Tuple
from elevenlabs.client import ElevenLabs
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class SpeechService:
    """
    Handles all speech-related operations including text-to-speech and speech-to-text.
    Uses ElevenLabs for both TTS and STT functionality.
    """

    def __init__(self):
        """Initialize the speech service with ElevenLabs client."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not found in environment variables")
            
        self.elevenlabs_client = ElevenLabs(api_key=self.api_key)
        
        # default TTS settings
        self.default_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
        self.default_model_id = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
        self.default_language = os.getenv("ELEVENLABS_LANGUAGE", "it")
        
        # default STT settings
        self.stt_model_id = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
        self.stt_language = os.getenv("ELEVENLABS_STT_LANGUAGE", "ita")
        
        logger.debug("Speech service initialized with ElevenLabs")

    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = None,
        output_format: str = "mp3_44100_128",
        language: str = None,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Convert text to speech using ElevenLabs.
        
        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID (defaults to environment setting)
            model_id: ElevenLabs model ID (defaults to environment setting)
            output_format: Output audio format
            language: Language code (defaults to environment setting)
            
        Yields:
            A tuple of (sample_rate, audio_array) for audio playback
        """
        if not text:
            logger.warning("Empty text provided to text_to_speech")
            return
            
        # use default values if not provided
        voice_id = voice_id or self.default_voice_id
        model_id = model_id or self.default_model_id
        language = language or self.default_language
        
        logger.debug(f"Converting text to speech, length: {len(text)}")
        
        try:
            # get the bytes directly from the generator
            audio = b"".join(self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
            ))
            
            # write to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_file_path = temp_file.name
            temp_file.write(audio)
            temp_file.close()
            
            try:
                # read the audio file
                import soundfile as sf
                data, sample_rate = sf.read(temp_file_path, dtype="int16")
                audio_array = np.array(data).reshape(1, -1)
                
                yield (sample_rate, audio_array)
            finally:
                # always clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error in text_to_speech: {str(e)}")
            raise

    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        diarize: bool = False,
        tag_audio_events: bool = False,
    ) -> str:
        """
        Convert speech to text using ElevenLabs.
        
        Args:
            audio: Tuple containing sample rate and audio data
            model_id: ElevenLabs STT model ID (defaults to environment setting)
            language_code: Language code (defaults to environment setting)
            diarize: Whether to annotate who is speaking
            tag_audio_events: Tag audio events like laughter, applause, etc.
            
        Returns:
            Transcribed text
        """
        if not audio or len(audio) != 2:
            logger.warning("Invalid audio provided to speech_to_text")
            return ""
            
        # use default values if not provided
        model_id = model_id or self.stt_model_id
        language_code = language_code or self.stt_language
        
        logger.debug(f"Converting speech to text, audio length: {audio[1].shape[1]} samples")
        
        try:
            import io
            import soundfile as sf
            
            # write audio to BytesIO buffer as wav
            buf = io.BytesIO()
            sf.write(buf, audio[1].flatten(), audio[0], format="WAV")
            buf.seek(0)
            
            # call ElevenLabs STT API
            transcription = self.elevenlabs_client.speech_to_text.convert(
                file=buf,
                model_id=model_id,
                tag_audio_events=tag_audio_events,
                language_code=language_code,
                diarize=diarize,
            )
            
            # return just the text part of the transcription
            return transcription.text if hasattr(transcription, 'text') else transcription
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {str(e)}")
            return "" 