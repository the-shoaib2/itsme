"""
FastAPI Server Application Entry Point for ItsMe.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import system, tts, voices

app = FastAPI(
    title="ItsMe — The Voice of Me API",
    description="Production-quality personal voice cloning and voice fine-tuning REST & WebSocket API",
    version="0.1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(system.router)
app.include_router(voices.router)
app.include_router(tts.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
