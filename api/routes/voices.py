"""
Voice Profiles API Routes for ItsMe.
"""


from fastapi import APIRouter, HTTPException, status

from api.schemas.requests import VoiceInfoResponse

router = APIRouter(tags=["Voices"])

VOICES_DB = {
    "itsme": {
        "id": "itsme",
        "name": "ItsMe",
        "description": "The Voice of Me - Authorised Personal Voice Profile",
        "language": "en",
        "sample_rate": 24000,
        "status": "active"
    }
}

@router.get("/api/v1/voices", response_model=list[VoiceInfoResponse])
async def list_voices():
    """List all available voice profiles."""
    return [VoiceInfoResponse(**v) for v in VOICES_DB.values()]

@router.get("/api/v1/voices/{voice_id}", response_model=VoiceInfoResponse)
async def get_voice_info(voice_id: str):
    """Get detailed information for specific voice profile."""
    if voice_id not in VOICES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice profile '{voice_id}' not found."
        )
    return VoiceInfoResponse(**VOICES_DB[voice_id])
