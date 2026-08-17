"""
Text-to-Speech REST & WebSocket Streaming Routes.
"""

import asyncio
import io
import json

import soundfile as sf
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import Response

from api.schemas.requests import TTSRequest
from itsme.inference.engine import CosyVoiceInferenceEngine
from itsme.inference.streaming import CosyVoiceStreamer
from itsme.utils.logging import get_logger

logger = get_logger("itsme.api.tts")
router = APIRouter(tags=["TTS"])

# Shared engine instance
engine = CosyVoiceInferenceEngine()
streamer = CosyVoiceStreamer(engine)

@router.post("/api/v1/tts")
async def generate_tts(request: TTSRequest):
    """
    REST Endpoint: Synthesize speech from text and return WAV audio stream.
    """
    try:
        res = engine.synthesize(
            text=request.text,
            speed=request.speed,
            temperature=request.temperature
        )
        audio_np = res["audio_np"]
        sr = res["sample_rate"]
        
        # Encode WAV to bytes buffer
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format='WAV', subtype='PCM_16')
        buf.seek(0)
        
        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="itsme_speech.wav"'}
        )
    except Exception as e:
        logger.error(f"REST TTS generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {e!s}"
        )

@router.websocket("/api/v1/tts/stream")
async def stream_tts_websocket(websocket: WebSocket):
    """
    WebSocket Endpoint: Stream PCM 16-bit 24kHz audio chunks for incoming text requests.
    """
    await websocket.accept()
    logger.info("WebSocket client connected to /api/v1/tts/stream")
    
    try:
        while True:
            data_str = await websocket.receive_text()
            try:
                data = json.loads(data_str)
                text = data.get("text", "")
                if not text:
                    await websocket.send_json({"error": "Empty text provided."})
                    continue
                    
                logger.info(f"WebSocket processing text streaming: '{text[:40]}...'")
                
                # Stream PCM byte chunks
                for chunk in streamer.stream_pcm_chunks(text):
                    await websocket.send_bytes(chunk)
                    await asyncio.sleep(0.005)
                    
                # Signal completion
                await websocket.send_json({"status": "EOS"})
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format."})
            except Exception as ex:
                logger.error(f"Error during WebSocket streaming: {ex}")
                await websocket.send_json({"error": f"Streaming error: {ex!s}"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected gracefully.")
