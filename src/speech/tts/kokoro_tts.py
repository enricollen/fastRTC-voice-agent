"""
kokoro implementation of the tts provider.
"""
import os
import numpy as np
from typing import Generator, Tuple
from loguru import logger
import torch

from .provider import ProviderTTS


class KokoroTTS(ProviderTTS):
    """kokoro tts implementation of ttsprovider."""
    
    def __init__(self):
        """initialize kokoro provider."""
        self.model = None
        self.default_voice = os.getenv("KOKORO_VOICE", "im_nicola") # if_sara
        self.default_language = os.getenv("KOKORO_LANGUAGE", "i") 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000  # kokoro uses 24khz
        self.initialized = False
        
    def initialize(self) -> None:
        """initialize the kokoro model."""
        try:
            # import locally to avoid startup dependency if not using kokoro
            from kokoro import KPipeline
            
            # initialize the pipeline with default language
            self.model = KPipeline(lang_code=self.default_language)
            logger.debug(f"kokoro provider initialized on {self.device}")
            
            # pre-load the default voice
            try:
                self._load_voice(self.default_voice)
            except FileNotFoundError:
                # if the default voice fails, try to use a fallback voice
                logger.warning(f"failed to load default voice {self.default_voice}, trying fallback voices")
                fallback_voices = ["im_marcello", "im_roberto", "im_matteo", "af_bella"]
                
                for fallback in fallback_voices:
                    try:
                        logger.info(f"trying fallback voice: {fallback}")
                        self._load_voice(fallback)
                        self.default_voice = fallback  # update default voice
                        logger.info(f"using fallback voice: {fallback}")
                        break
                    except Exception:
                        continue
                else:
                    # if all fallbacks fail, raise the original error
                    logger.error("all fallback voices failed, please download voices manually")
                    raise
            
            self.initialized = True
            
        except ImportError:
            logger.error("kokoro tts not available. install with: pip install kokorotts")
            raise
        except Exception as e:
            logger.error(f"error initializing kokoro tts: {str(e)}")
            raise
    
    def _load_voice(self, voice_name: str) -> None:
        """
        load a kokoro voice model.
        
        args:
            voice_name: name of the voice to load (with or without .pt extension)
        """
        if not self.model:
            self.initialize()
            
        # format voice name correctly - strip .pt if it was included
        voice_name = voice_name.replace('.pt', '')
        voice_path = os.path.abspath(os.path.join("voices", f"{voice_name}.pt"))
        
        if not os.path.exists(voice_path):
            logger.warning(f"voice file not found: {voice_path}")
            # attempt to auto-download the missing voice
            self._download_voice(voice_name)
            # check again after download attempt
            if not os.path.exists(voice_path):
                logger.info("available voices can be downloaded with: import kokorotts; kokorotts.download_voice_files()")
                raise FileNotFoundError(f"voice file not found: {voice_path}")
        
        # check if voice is already loaded
        if hasattr(self.model, 'voices') and voice_name in self.model.voices:
            return
            
        # load voice if not already loaded
        try:
            self.model.load_voice(voice_path)
            logger.debug(f"loaded kokoro voice: {voice_name}")
        except Exception as e:
            logger.error(f"error loading kokoro voice {voice_name}: {str(e)}")
            raise
            
    def _download_voice(self, voice_name: str) -> bool:
        """
        attempt to download a specific voice file.
        
        args:
            voice_name: name of the voice to download
            
        returns:
            true if successful, false otherwise
        """
        try:
            logger.info(f"attempting to download missing voice: {voice_name}")
            
            voices_dir = os.path.join(os.path.abspath("."), "voices")
            if not os.path.exists(voices_dir):
                os.makedirs(voices_dir)
            
            # italian voice files
            VOICE_FILES = [
                "im_nicola.pt", 
                "if_sara.pt",  
            ]
            
            from pathlib import Path
            import shutil
            
            def download_voice_files(voice_files=None, repo_version="main", required_count=1):
                """download voice files from hugging face.

                args:
                    voice_files: optional list of voice files to download. if none, download all voice_files.
                    repo_version: version/tag of the repository to use (default: "main")
                    required_count: minimum number of voices required (default: 1)

                returns:
                    list of successfully downloaded voice files

                raises:
                    valueerror: if fewer than required_count voices could be downloaded
                """
                voices_dir = Path(os.path.abspath("voices"))
                voices_dir.mkdir(exist_ok=True)

                try:
                    from huggingface_hub import hf_hub_download
                except ImportError:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                    from huggingface_hub import hf_hub_download
                    
                downloaded_voices = []
                failed_voices = []

                # if specific voice files are requested, use those else use all.
                files_to_download = voice_files if voice_files is not None else VOICE_FILES
                total_files = len(files_to_download)

                print(f"\ndownloading voice files... ({total_files} total files)")

                # check for existing voice first
                existing_files = []
                for voice_file in files_to_download:
                    voice_path = voices_dir / voice_file
                    if voice_path.exists():
                        print(f"voice file {voice_file} already exists")
                        downloaded_voices.append(voice_file)
                        existing_files.append(voice_file)

                # remove existing files from the download list
                files_to_download = [f for f in files_to_download if f not in existing_files]
                if not files_to_download and downloaded_voices:
                    print(f"all required voice files already exist ({len(downloaded_voices)} files)")
                    return downloaded_voices

                # proceed with downloading missing files
                retry_count = 3
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        for voice_file in files_to_download:
                            # path where the voice file should be
                            voice_path = voices_dir / voice_file

                            # try with retries
                            for attempt in range(retry_count):
                                try:
                                    print(f"downloading {voice_file}... (attempt {attempt+1}/{retry_count})")
                                    # download to a temporary location
                                    temp_path = hf_hub_download(
                                        repo_id="hexgrad/Kokoro-82M",
                                        filename=f"voices/{voice_file}",
                                        local_dir=temp_dir,
                                        force_download=True,
                                        revision=repo_version
                                    )

                                    # move the file to the correct location
                                    os.makedirs(os.path.dirname(str(voice_path)), exist_ok=True)
                                    shutil.copy2(temp_path, str(voice_path))  # use copy2 instead of move

                                    # check file integrity
                                    if os.path.getsize(str(voice_path)) > 0:
                                        downloaded_voices.append(voice_file)
                                        print(f"successfully downloaded {voice_file}")
                                        break  # success, exit retry loop
                                    else:
                                        print(f"warning: downloaded file {voice_file} has zero size, retrying...")
                                        os.remove(str(voice_path))  # remove invalid file
                                        if attempt == retry_count - 1:
                                            failed_voices.append(voice_file)
                                except (IOError, OSError, ValueError, FileNotFoundError, ConnectionError) as e:
                                    print(f"warning: failed to download {voice_file} (attempt {attempt+1}): {e}")
                                    if attempt == retry_count - 1:
                                        failed_voices.append(voice_file)
                                        print(f"error: failed all {retry_count} attempts to download {voice_file}")
                except Exception as e:
                    print(f"error during voice download process: {e}")
                    import traceback
                    traceback.print_exc()

                # results
                if failed_voices:
                    print(f"warning: failed to download {len(failed_voices)} voice files: {', '.join(failed_voices)}")

                if not downloaded_voices:
                    error_msg = "no voice files could be downloaded. please check your internet connection."
                    print(f"error: {error_msg}")
                    raise ValueError(error_msg)
                elif len(downloaded_voices) < required_count:
                    error_msg = f"only {len(downloaded_voices)} voice files could be downloaded, but {required_count} were required."
                    print(f"error: {error_msg}")
                    raise ValueError(error_msg)
                else:
                    print(f"successfully processed {len(downloaded_voices)} voice files")

                return downloaded_voices
            
            # try to download the specific voice
            try:
                voice_filename = f"{voice_name}.pt"
                logger.info(f"downloading voice file: {voice_filename}")
                downloaded = download_voice_files(
                    voice_files=[voice_filename], 
                    required_count=1
                )
                
                # check if we successfully downloaded the voice
                if voice_filename in downloaded:
                    logger.info(f"successfully downloaded voice: {voice_name}")
                    return True
                
                # if we didn't get the requested voice but downloaded others, update default voice
                voice_path = os.path.join(voices_dir, voice_filename)
                if not os.path.exists(voice_path) and downloaded:
                    # get first available italian voice
                    italian_voices = [v.replace('.pt', '') for v in downloaded if v.startswith('i')]
                    if italian_voices:
                        new_voice = italian_voices[0]
                        logger.info(f"voice {voice_name} not found but downloaded {new_voice}")
                        self.default_voice = new_voice
                        logger.info(f"updated default voice to: {new_voice}")
                        return True
                    else:
                        # use any voice as fallback
                        new_voice = downloaded[0].replace('.pt', '')
                        logger.info(f"no italian voices found, using {new_voice}")
                        self.default_voice = new_voice
                        logger.info(f"updated default voice to: {new_voice}")
                        return True
                
                return False
                
            except Exception as e:
                logger.error(f"error downloading voice: {str(e)}")
                
                # try downloading all voices since the specific one failed
                try:
                    logger.info("attempting to download all voices...")
                    downloaded = download_voice_files(required_count=1)
                    
                    # check available voices
                    if downloaded:
                        # get first available italian voice
                        italian_voices = [v.replace('.pt', '') for v in downloaded if v.startswith('i')]
                        if italian_voices:
                            new_voice = italian_voices[0]
                            logger.info(f"downloaded italian voice: {new_voice}")
                            self.default_voice = new_voice
                            logger.info(f"updated default voice to: {new_voice}")
                            return True
                        else:
                            # use any voice as fallback
                            new_voice = downloaded[0].replace('.pt', '')
                            logger.info(f"no italian voices found, using {new_voice}")
                            self.default_voice = new_voice
                            logger.info(f"updated default voice to: {new_voice}")
                            return True
                            
                    return False
                except Exception as e:
                    logger.error(f"error downloading all voices: {str(e)}")
                    return False
                
        except Exception as e:
            logger.error(f"error in voice download process: {str(e)}")
            return False
    
    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = None,  # ignored for kokoro
        output_format: str = None,  # ignored for kokoro
        language: str = None,
        speed: float = 1.0,
        **kwargs
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        convert text to speech using kokoro tts.
        
        args:
            text: text to synthesize
            voice_id: voice name (e.g., 'af_bella')
            language: language code ('a' for american english, 'b' for british english)
            speed: speech speed multiplier (default: 1.0)
            
        yields:
            a tuple of (sample_rate, audio_array) for audio playback
        """
        if not text:
            logger.warning("empty text provided to kokoro text_to_speech")
            return
            
        if not self.model:
            self.initialize()
            
        # use default values if not provided
        voice = voice_id or self.default_voice
        lang = language or self.default_language
        
        logger.debug(f"converting text to speech with kokoro, length: {len(text)}")
        
        try:
            # voice name and path
            voice_name = voice.replace('.pt', '')
            voice_path = os.path.abspath(os.path.join("voices", f"{voice_name}.pt"))
            
            # check if voice file exists
            if not os.path.exists(voice_path):
                logger.error(f"voice file not found: {voice_path}")
                raise FileNotFoundError(f"voice file not found: {voice_path}")
                
            # ensure voice is loaded
            self._load_voice(voice_name)
            
            # generate speech
            generator = self.model(
                text,
                voice=voice_path,
                speed=speed,
                split_pattern=r'\n+'
            )
            
            # process generated audio
            for _, _, audio in generator:
                if audio is not None:
                    # convert to tensor if numpy array
                    if isinstance(audio, np.ndarray):
                        audio_tensor = torch.from_numpy(audio).float()
                    else:
                        audio_tensor = audio
                        
                    # convert to numpy for consistency with other tts providers
                    audio_array = audio_tensor.cpu().numpy().reshape(1, -1)
                    yield (self.sample_rate, audio_array)
                    
        except Exception as e:
            logger.error(f"error in kokoro text_to_speech: {str(e)}")
            raise 