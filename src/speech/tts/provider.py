"""
base abstract class for tts providers.
"""
import abc
import numpy as np
from typing import Generator, Tuple, Optional


class ProviderTTS(abc.ABC):
    """base abstract class for tts providers."""
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """initialize the tts provider."""
        pass
    
    @abc.abstractmethod
    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = None,
        output_format: str = None,
        language: str = None,
        **kwargs
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """convert text to speech."""
        pass 