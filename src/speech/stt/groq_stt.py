"""
groq implementation of the stt provider using groq client directly.
"""
import os
import io
import numpy as np
from typing import Optional, Dict, Any
from groq import Groq
from loguru import logger
import soundfile as sf
from fastrtc import audio_to_bytes
from .provider import ProviderSTT


class GroqSTT(ProviderSTT):
    """groq implementation of stt provider using groq client directly."""
    
    def __init__(self):
        """initialize groq stt provider."""
        self.api_key = os.getenv("GROQ_API_KEY")
        self.default_model_id = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
        self.default_language = os.getenv("GROQ_STT_LANGUAGE", "it")
        self.client = None
        self.initialized = False
        
    def initialize(self) -> None:
        """initialize the groq client."""
        if not self.api_key:
            logger.warning("groq_api_key not found in environment variables")
        
        # initialize groq client
        self.client = Groq(api_key=self.api_key)
        logger.debug("groq stt provider initialized")
        self.initialized = True
    
    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        prompt: str = None,
        temperature: float = 0,
        response_format: str = "text",
        **kwargs
    ) -> str:
        """
        convert speech to text using groq client directly.
        
        args:
            audio: tuple containing sample rate and audio data
            model_id: groq model id (defaults to environment setting)
            language_code: language code (defaults to environment setting)
            prompt: optional prompt for context or spelling
            temperature: sampling temperature (0-1)
            response_format: output format ("text" or "json")
            
        returns:
            transcribed text
        """
        if not audio or len(audio) != 2:
            logger.warning("invalid audio provided to speech_to_text")
            return ""
            
        if not self.initialized:
            self.initialize()
            
        # use default values if not provided
        model_id = model_id or self.default_model_id
        
        logger.debug(f"converting speech to text using groq, audio length: {audio[1].shape[1]} samples")
        
        try:
            # convert audio to bytes using fastrtc utility
            audio_bytes = audio_to_bytes(audio)
            
            # call groq stt api directly
            transcript = self.client.audio.transcriptions.create(
                file=("audio-file.mp3", audio_bytes),
                model=model_id,
                response_format=response_format,
                prompt=prompt,
                temperature=temperature,
                language=language_code
            )
            
            return transcript
            
        except Exception as e:
            logger.error(f"error in speech_to_text with groq: {str(e)}")
            return "" 