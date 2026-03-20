from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import base64
import os
import io

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
    language: str = "en"

@app.get("/")
def root():
    return {"status": "DreamTale TTS server is running"}

@app.post("/tts-stream")
def tts_stream(req: TTSRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    voice_map = {
        "en":  {"languageCode": "en-US",  "name": "en-US-Journey-F"},
        "ar":  {"languageCode": "ar-XA",  "name": "ar-XA-Journey-F"},
        "fr":  {"languageCode": "fr-FR",  "name": "fr-FR-Journey-F"},
        "es":  {"languageCode": "es-ES",  "name": "es-ES-Journey-F"},
        "pt":  {"languageCode": "pt-BR",  "name": "pt-BR-Journey-F"},
        "de":  {"languageCode": "de-DE",  "name": "de-DE-Journey-F"},
        "zh":  {"languageCode": "cmn-CN", "name": "cmn-CN-Wavenet-D"},
        "hi":  {"languageCode": "hi-IN",  "name": "hi-IN-Journey-F"},
        "tr":  {"languageCode": "tr-TR",  "name": "tr-TR-Journey-F"},
        "id":  {"languageCode": "id-ID",  "name": "id-ID-Journey-F"},
        "ru":  {"languageCode": "ru-RU",  "name": "ru-RU-Wavenet-E"},
        "ja":  {"languageCode": "ja-JP",  "name": "ja-JP-Journey-F"},
        "ko":  {"languageCode": "ko-KR",  "name": "ko-KR-Journey-F"},
        "it":  {"languageCode": "it-IT",  "name": "it-IT-Journey-F"},
        "nl":  {"languageCode": "nl-NL",  "name": "nl-NL-Journey-F"},
    }

    voice = voice_map.get(req.language, voice_map["en"])

    payload = {
        "input": {"text": req.text.strip()},
        "voice": voice,
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": 0.85
        }
    }

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Google TTS error: {response.text}")

    audio_bytes = base64.b64decode(response.json().get("audioContent", ""))

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline",
            "Accept-Ranges": "bytes"
        }
    )
