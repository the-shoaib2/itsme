"""
System & Health API Routes for ItsMe.
"""

import time

from fastapi import APIRouter, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api.schemas.requests import HealthCheckResponse, ModelInfoResponse
from itsme.config.config import get_config
from itsme.utils.hardware import detect_device

router = APIRouter(tags=["System"])

START_TIME = time.time()
config = get_config()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = config.get("api.key", os.getenv("API_KEY", "secret-itsme-key-change-me"))
    # If API key configured, check header
    if expected_key and api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key."
        )
    return api_key

import os


@router.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """Get service health status."""
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=round(time.time() - START_TIME, 2),
        device=detect_device()
    )

@router.get("/api/v1/model", response_model=ModelInfoResponse)
async def get_model_info():
    """Get active TTS model information."""
    model_name = config.get("model.name", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    sample_rate = config.get("model.sample_rate", 24000)
    
    return ModelInfoResponse(
        model_name=model_name,
        device=detect_device(),
        sample_rate=sample_rate,
        status="loaded",
        supported_voices=["itsme"]
    )
