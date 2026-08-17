"""
Pydantic Schemas for ItsMe API.
"""

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=1000, description="Text to synthesize")
    voice: str = Field(default="itsme", description="Voice profile ID")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed rate")
    temperature: float = Field(default=0.7, ge=0.1, le=1.5, description="Sampling temperature")

class VoiceInfoResponse(BaseModel):
    id: str
    name: str
    description: str
    language: str
    sample_rate: int
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    device: str
    sample_rate: int
    status: str
    supported_voices: list[str]

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    device: str
