"""
elevenlabs implementation of the tts provider.
"""
import os
import tempfile
import numpy as np
from typing import Generator, Tuple
from elevenlabs.client import ElevenLabs
from loguru import logger
import soundfile as sf

from .provider import ProviderTTS


class ElevenLabsTTS(ProviderTTS):
    """elevenlabs implementation of ttsprovider."""
    
    def __init__(self):
        """initialize elevenlabs provider."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = None
        self.default_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
        self.default_model_id = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
        self.default_language = os.getenv("ELEVENLABS_LANGUAGE", "it")
        self.initialized = False
        
    def initialize(self) -> None:
        """initialize the elevenlabs client."""
        if not self.api_key:
            logger.warning("elevenlabs_api_key not found in environment variables")
        
        self.client = ElevenLabs(api_key=self.api_key)
        logger.debug("elevenlabs provider initialized")
        self.initialized = True
    
    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = None,
        output_format: str = "mp3_44100_128",
        language: str = None,
        **kwargs
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        convert text to speech using elevenlabs.
        
        args:
            text: text to synthesize
            voice_id: elevenlabs voice id (defaults to environment setting)
            model_id: elevenlabs model id (defaults to environment setting)
            output_format: output audio format
            language: language code (defaults to environment setting)
            
        yields:
            a tuple of (sample_rate, audio_array) for audio playback
        """
        if not text:
            logger.warning("empty text provided to elevenlabs text_to_speech")
            return
        
        if not self.client:
            self.initialize()
        
        # use default values if not provided
        voice_id = voice_id or self.default_voice_id
        model_id = model_id or self.default_model_id
        language = language or self.default_language
        
        logger.debug(f"converting text to speech with elevenlabs, length: {len(text)}")
        
        try:
            # get the bytes directly from the generator
            audio = b"".join(self.client.text_to_speech.convert(
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
                data, sample_rate = sf.read(temp_file_path, dtype="int16")
                audio_array = np.array(data).reshape(1, -1)
                
                yield (sample_rate, audio_array)
            finally:
                # always clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            logger.error(f"error in elevenlabs text_to_speech: {str(e)}")
            raise 