"""
Whisper local implementation of the stt provider using faster-whisper.
"""
import os
import numpy as np
from loguru import logger
from faster_whisper import WhisperModel
import torch
from .provider import ProviderSTT


class WhisperSTT(ProviderSTT):
    """Whisper local implementation of stt provider."""
    
    def __init__(self):
        """Initialize whisper stt provider."""
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
        self.default_language = os.getenv("WHISPER_LANGUAGE", "it")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = None
        self.initialized = False
        
    def initialize(self) -> None:
        """Initialize the whisper model."""
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.debug(f"whisper stt provider initialized with model {self.model_size} on {self.device}")
            self.initialized = True
        except Exception as e:
            logger.error(f"error initializing whisper model: {str(e)}")
            self.initialized = False
    
    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        beam_size: int = 5,
        **kwargs
    ) -> str:
        """
        Convert speech to text using whisper locally.
        
        Args:
            audio: tuple containing sample rate and audio data
            model_id: not used for whisper local (model is loaded at initialization)
            language_code: language code (defaults to environment setting)
            beam_size: beam size for decoding (default: 5)
            
        Returns:
            transcribed text
        """
        if not audio or len(audio) != 2:
            logger.warning("invalid audio provided to speech_to_text")
            return ""
            
        if not self.initialized:
            self.initialize()
            
        if not self.initialized:
            logger.error("whisper model failed to initialize")
            return ""
            
        # use default language if not provided
        language = language_code or self.default_language
        
        logger.debug(f"converting speech to text using whisper local, audio length: {audio[1].shape[1]} samples")
        
        try:            # convert numpy array to temporary file
            import tempfile
            import soundfile as sf
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            segments = None
            info = None
            
            try:
                # save audio to temporary file
                sf.write(temp_file.name, audio[1].T, audio[0])
                temp_file.close()
                
                # transcribe
                segments, info = self.model.transcribe(
                    temp_file.name,
                    language=language,
                    without_timestamps=True,
                    beam_size=beam_size
                )
            except Exception as e:
                logger.error(f"error in transcription: {str(e)}")
                return ""
            finally:
                # clean up the temp file
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")
            
            # process results after file cleanup
            if segments and info:
                # combine all segments into one text
                text = " ".join(segment.text for segment in segments)
                logger.debug(f"detected language: {info.language} with probability: {info.language_probability}")
                return text.strip()
            
            return ""
                
        except Exception as e:
            logger.error(f"error in speech_to_text with whisper: {str(e)}")
            return ""