"""
base abstract class for stt providers.
"""
import abc
import numpy as np
from typing import Optional, Dict, Any


class ProviderSTT(abc.ABC):
    """base abstract class for stt providers."""
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """initialize the stt provider."""
        pass
    
    @abc.abstractmethod
    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        **kwargs
    ) -> str:
        """convert speech to text."""
        pass 