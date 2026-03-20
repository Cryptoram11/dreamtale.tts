from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

class TTSRequest(BaseModel):
    text: str
    language: str = "en-US"

@app.get("/")
def root():
    return {"status": "DreamTale TTS server is running"}

@app.post("/tts")
def text_to_speech(req: TTSRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    voice_map = {
        "en-US": {"languageCode": "en-US", "name": "en-US-Journey-F"},
        "ar":    {"languageCode": "ar-XA", "name": "ar-XA-Wavenet-A"},
        "fr":    {"languageCode": "fr-FR", "name": "fr-FR-Journey-F"},
        "es":    {"languageCode": "es-ES", "name": "es-ES-Journey-F"},
    }

    voice = voice_map.get(req.language, voice_map["en-US"])

    payload = {
        "input": {"text": req.text.strip()},
        "voice": voice,
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 0.85,
        }
    }

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Google TTS error: {response.text}")

    audio_base64 = response.json().get("audioContent", "")
    if not audio_base64:
        raise HTTPException(status_code=500, detail="No audio returned")

    return {"audio": audio_base64}
