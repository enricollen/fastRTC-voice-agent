"""
speech service module for tts and stt operations.
"""
import os
from typing import Generator, Tuple, Dict, Any
import numpy as np
from loguru import logger
from dotenv import load_dotenv

from .tts import ProviderTTS, ElevenLabsTTS, KokoroTTS
from .stt import ProviderSTT, ElevenLabsSTT, GroqSTT, OpenAISTT

load_dotenv()


class SpeechService:
    """
    handles all speech-related operations including text-to-speech and speech-to-text.
    supports multiple tts providers including elevenlabs and kokoro tts.
    supports multiple stt providers including elevenlabs, groq, and openai.
    """

    def __init__(self, tts_provider: str = "elevenlabs", stt_provider: str = "elevenlabs"):
        """
        initialize the speech service.
        
        args:
            tts_provider: tts provider to use ("elevenlabs" or "kokoro")
            stt_provider: stt provider to use ("elevenlabs", "groq", or "openai")
        """
        # initialize tts providers
        self.tts_providers: Dict[str, ProviderTTS] = {
            "elevenlabs": ElevenLabsTTS(),
            "kokoro": KokoroTTS()
        }
        
        # set active tts provider
        self.tts_provider = tts_provider.lower()
        if self.tts_provider not in self.tts_providers:
            logger.warning(f"unknown tts provider '{tts_provider}', falling back to elevenlabs")
            self.tts_provider = "elevenlabs"
            
        logger.debug(f"speech service initialized with {self.tts_provider} tts provider")
        
        # initialize stt providers
        self.stt_providers: Dict[str, ProviderSTT] = {
            "elevenlabs": ElevenLabsSTT(),
            "groq": GroqSTT(),
            "openai": OpenAISTT()
        }
        
        # set active stt provider
        self.stt_provider = stt_provider.lower()
        if self.stt_provider not in self.stt_providers:
            logger.warning(f"unknown stt provider '{stt_provider}', falling back to elevenlabs")
            self.stt_provider = "elevenlabs"
            
        logger.debug(f"speech service initialized with {self.stt_provider} stt provider")
        
        # always preload tts model to reduce initial latency
        self.preload_tts()

    def preload_tts(self) -> None:
        """
        preload the active tts provider to reduce latency on first use.
        """
        provider = self.tts_providers[self.tts_provider]
        if not hasattr(provider, 'initialized') or not provider.initialized:
            logger.info(f"preloading {self.tts_provider} tts model...")
            provider.initialize()
            provider.initialized = True
            logger.info(f"{self.tts_provider} tts model preloaded successfully")

    def set_tts_provider(self, provider_name: str) -> None:
        """
        change the active tts provider.
        
        args:
            provider_name: name of the provider to use ("elevenlabs" or "kokoro")
        """
        if provider_name.lower() not in self.tts_providers:
            logger.warning(f"unknown tts provider '{provider_name}', ignoring request")
            return
            
        self.tts_provider = provider_name.lower()
        logger.debug(f"changed tts provider to {self.tts_provider}")
        
        # preload the new provider
        self.preload_tts()

    def set_stt_provider(self, provider_name: str) -> None:
        """
        change the active stt provider.
        
        args:
            provider_name: name of the provider to use ("elevenlabs", "groq", or "openai")
        """
        if provider_name.lower() not in self.stt_providers:
            logger.warning(f"unknown stt provider '{provider_name}', ignoring request")
            return
            
        self.stt_provider = provider_name.lower()
        logger.debug(f"changed stt provider to {self.stt_provider}")

    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = None,
        output_format: str = "mp3_44100_128",
        language: str = None,
        speed: float = 1.0,
        **kwargs
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        convert text to speech using the active tts provider.
        
        args:
            text: text to synthesize
            voice_id: voice id or name (provider-specific)
            model_id: model id (provider-specific)
            output_format: output audio format (provider-specific)
            language: language code (provider-specific)
            speed: speech speed multiplier (only for kokoro)
            
        yields:
            a tuple of (sample_rate, audio_array) for audio playback
        """
        if not text:
            logger.warning("empty text provided to text_to_speech")
            return
            
        provider = self.tts_providers[self.tts_provider]
        
        # model should already be initialized, but check just in case
        if not hasattr(provider, 'initialized') or not provider.initialized:
            provider.initialize()
            provider.initialized = True
            
        logger.debug(f"converting text to speech using {self.tts_provider}, length: {len(text)}")
        
        yield from provider.text_to_speech(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            language=language,
            speed=speed,
            **kwargs
        )

    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        prompt: str = None,
        temperature: float = 0,
        response_format: str = "text",
        diarize: bool = False,
        tag_audio_events: bool = False,
        **kwargs
    ) -> str:
        """
        convert speech to text using the active stt provider.
        
        args:
            audio: tuple containing sample rate and audio data
            model_id: model id (provider-specific)
            language_code: language code (provider-specific)
            prompt: optional prompt for context or spelling (groq and openai only)
            temperature: sampling temperature (groq and openai only)
            response_format: output format (groq and openai only)
            diarize: whether to annotate who is speaking (elevenlabs only)
            tag_audio_events: tag audio events like laughter, applause, etc. (elevenlabs only)
            
        returns:
            transcribed text
        """
        if not audio or len(audio) != 2:
            logger.warning("invalid audio provided to speech_to_text")
            return ""
            
        provider = self.stt_providers[self.stt_provider]
        
        # lazy initialization of provider
        if not hasattr(provider, 'initialized') or not provider.initialized:
            provider.initialize()
            provider.initialized = True
            
        # provider-specific parameters
        provider_kwargs = kwargs.copy()
        if self.stt_provider == "elevenlabs":
            provider_kwargs.update({
                "diarize": diarize,
                "tag_audio_events": tag_audio_events
            })
        elif self.stt_provider in ["groq", "openai"]:
            provider_kwargs.update({
                "prompt": prompt,
                "temperature": temperature,
                "response_format": response_format
            })
            
        return provider.speech_to_text(
            audio=audio,
            model_id=model_id,
            language_code=language_code,
            **provider_kwargs
        ) 