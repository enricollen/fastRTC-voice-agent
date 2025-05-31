"""
elevenlabs implementation of the stt provider.
"""
import os
import io
import numpy as np
from typing import Optional, Dict, Any
from elevenlabs.client import ElevenLabs
from loguru import logger
import soundfile as sf


class ElevenLabsSTT:
    """elevenlabs implementation of stt provider."""
    
    def __init__(self):
        """initialize elevenlabs stt provider."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = None
        self.default_model_id = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
        self.default_language = os.getenv("ELEVENLABS_STT_LANGUAGE", "ita")
        self.initialized = False
        
    def initialize(self) -> None:
        """initialize the elevenlabs client."""
        if not self.api_key:
            logger.warning("elevenlabs_api_key not found in environment variables")
        
        self.client = ElevenLabs(api_key=self.api_key)
        logger.debug("elevenlabs stt provider initialized")
        self.initialized = True
    
    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        diarize: bool = False,
        tag_audio_events: bool = False,
    ) -> str:
        """
        convert speech to text using elevenlabs.
        
        args:
            audio: tuple containing sample rate and audio data
            model_id: elevenlabs stt model id (defaults to environment setting)
            language_code: language code (defaults to environment setting)
            diarize: whether to annotate who is speaking
            tag_audio_events: tag audio events like laughter, applause, etc.
            
        returns:
            transcribed text
        """
        if not audio or len(audio) != 2:
            logger.warning("invalid audio provided to speech_to_text")
            return ""
            
        if not self.client:
            self.initialize()
            
        # use default values if not provided
        model_id = model_id or self.default_model_id
        language_code = language_code or self.default_language
        
        logger.debug(f"converting speech to text, audio length: {audio[1].shape[1]} samples")
        
        try:
            # write audio to bytesio buffer as wav
            buf = io.BytesIO()
            sf.write(buf, audio[1].flatten(), audio[0], format="WAV")
            buf.seek(0)
            
            # call elevenlabs stt api
            transcription = self.client.speech_to_text.convert(
                file=buf,
                model_id=model_id,
                tag_audio_events=tag_audio_events,
                language_code=language_code,
                diarize=diarize,
            )
            
            # return just the text part of the transcription
            return transcription.text if hasattr(transcription, 'text') else transcription
            
        except Exception as e:
            logger.error(f"error in speech_to_text: {str(e)}")
            return "" 