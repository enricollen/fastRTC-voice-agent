"""
openai implementation of the stt provider. (https://platform.openai.com/docs/api-reference/audio/createTranscription)
"""
import os
import io
import numpy as np
from typing import Optional, Dict, Any
from openai import OpenAI
from loguru import logger
import soundfile as sf
from fastrtc import audio_to_bytes
from .provider import ProviderSTT


class OpenAISTT(ProviderSTT):
    """openai implementation of stt provider."""
    
    def __init__(self):
        """initialize openai stt provider."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.default_model_id = os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe") # gpt-4o-mini-transcribe / whisper-1
        self.default_language = os.getenv("OPENAI_STT_LANGUAGE", "it")
        self.client = None
        self.initialized = False
        
    def initialize(self) -> None:
        """initialize the openai client."""
        if not self.api_key:
            logger.warning("openai_api_key not found in environment variables")
        
        # initialize openai client
        self.client = OpenAI(api_key=self.api_key)
        logger.debug("openai stt provider initialized")
        self.initialized = True
    
    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        prompt: str = None,
        temperature: float = 0,
        response_format: str = "json",
        **kwargs
    ) -> str:
        """
        convert speech to text using openai.
        
        args:
            audio: tuple containing sample rate and audio data
            model_id: openai model id (defaults to environment setting)
            language_code: language code (defaults to environment setting)
            prompt: optional prompt to guide the model's style
            temperature: sampling temperature (0-1)
            response_format: output format (json is the only supported format for gpt-4o models)
            
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
        language = language_code or self.default_language
        
        logger.debug(f"converting speech to text using openai, audio length: {audio[1].shape[1]} samples")
        
        try:
            # convert audio to bytes using fastrtc utility
            audio_bytes = audio_to_bytes(audio)
            
            # create a file-like object
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio-file.mp3"  # openai needs a filename
            
            # call openai stt api
            transcript = self.client.audio.transcriptions.create(
                model=model_id,
                file=audio_file,
                response_format=response_format,
                prompt=prompt,
                temperature=temperature,
                language=language
            )
            
            # extract text from response
            if response_format == "json":
                return transcript.text
            return transcript
            
        except Exception as e:
            logger.error(f"error in speech_to_text with openai: {str(e)}")
            return "" 