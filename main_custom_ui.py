import os
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from dotenv import load_dotenv
from typing import Generator, Tuple, List, Dict, Any
import numpy as np
from src.speech import SpeechService
from src.agent import Agent
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
)

load_dotenv()

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# initialize services with defaults from environment variables
tts_provider = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
stt_provider = os.getenv("STT_PROVIDER", "elevenlabs").lower()
voice_id = None
if tts_provider == "elevenlabs":
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
elif tts_provider == "kokoro":
    voice_id = os.getenv("KOKORO_VOICE", "im_nicola")
speed = float(os.getenv("TTS_SPEED", "1.0"))

speech_service = SpeechService(
    tts_provider=tts_provider,
    stt_provider=stt_provider
)
agent = Agent()

# create fastapi app
app = FastAPI(title="FastRTC Voice Agent")

# create templates directory for html templates
templates = Jinja2Templates(directory="templates")

# mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# websocket connections
active_connections: List[WebSocket] = []

# websocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"websocket client connected, total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"websocket client disconnected, remaining connections: {len(self.active_connections)}")

    async def broadcast_message(self, message: Dict[str, Any]):
        async with self._lock:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"error broadcasting message: {e}")
                    self.active_connections.remove(connection)

manager = ConnectionManager()

# global event loop for async tasks
loop = asyncio.get_event_loop()

# modified response function that sends messages to websocket clients
def response(
    audio: tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input, transcribe it, generate a response using the agent, and deliver TTS audio.

    Args:
        audio: Tuple containing sample rate and audio data

    Yields:
        Tuples of (sample_rate, audio_array) for audio playback
    """
    logger.info(f"🎙️ received audio input - sample rate: {audio[0]}, shape: {audio[1].shape}")

    # debug audio characteristics
    """if len(audio[1]) > 0:
        logger.info(f"audio stats - min: {audio[1].min()}, max: {audio[1].max()}, mean: {audio[1].mean()}")
    else:
        logger.warning("received empty audio array")
        return"""

    logger.debug("🔄 transcribing audio...")
    # set STT parameters based on the active provider
    stt_kwargs = {}
    if stt_provider in ["groq", "openai"]:
        stt_kwargs["response_format"] = "text"
        
    transcript = speech_service.speech_to_text(audio, **stt_kwargs)
    logger.info(f'👂 transcribed: "{transcript}"')
    
    # send transcript to websocket clients
    if transcript.strip():
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_message({
                "type": "transcript",
                "text": transcript
            }),
            loop
        )
    else:
        logger.warning("empty transcript, no speech detected")
        return
    
    logger.debug("🧠 running agent...")
    agent_response = agent.invoke(transcript)
    response_text = agent_response["messages"][-1]["content"]
    logger.info(f'💬 response: "{response_text}"')
    
    # send response to websocket clients
    if response_text.strip():
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_message({
                "type": "response",
                "text": response_text
            }),
            loop
        )

    logger.debug("🔊 generating speech...")
    # set TTS parameters based on the active provider
    tts_kwargs = {}
    if voice_id:
        tts_kwargs["voice_id"] = voice_id
    
    # add speed parameter only for kokoro
    if tts_provider == "kokoro" and speed != 1.0:
        tts_kwargs["speed"] = speed
    
    yield from speech_service.text_to_speech(response_text, **tts_kwargs)

# create stream
def create_stream():
    """
    Create and configure a Stream instance with audio capabilities.
    """
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.2,  # higher threshold to be less sensitive
                audio_chunk_duration=0.8,  # longer chunks to reduce false detections
                started_talking_threshold=0.3  # less sensitive to detecting speech start
            ),
            can_interrupt=True  # disable interruption while the model is responding
        ),
        # disable echo cancellation for better audio quality
        track_constraints={
            "echoCancellation": False,
            "noiseSuppression": {"exact": True},
            "autoGainControl": {"exact": True},
            "sampleRate": {"ideal": 24000},
            "sampleSize": {"ideal": 16},
            "channelCount": {"exact": 1},
        }
    )

# create and mount stream
stream = create_stream()
stream.mount(app)

# websocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # keep the connection alive
            data = await websocket.receive_text()
            # handle client messages here
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """
    Serve the main HTML page
    """
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 starting fastrtc voice agent with custom ui...")
    uvicorn.run("main_custom_ui:app", host="localhost", port=8000, reload=True)
