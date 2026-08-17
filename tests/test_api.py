"""
Unit Tests for FastAPI Endpoints.
"""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_get_model_info():
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()
    assert data["sample_rate"] == 24000
    assert "itsme" in data["supported_voices"]

def test_list_voices():
    response = client.get("/api/v1/voices")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["id"] == "itsme"

def test_get_voice_detail():
    response = client.get("/api/v1/voices/itsme")
    assert response.status_code == 200
    assert response.json()["id"] == "itsme"

def test_tts_endpoint():
    response = client.post("/api/v1/tts", json={"text": "Hello world from ItsMe testing.", "voice": "itsme"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 100
