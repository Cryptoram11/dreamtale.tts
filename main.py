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
FAL_API_KEY = os.environ.get("FAL_API_KEY", "")

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

class AvatarRequest(BaseModel):
    image_url: str
    character_name: str

class IllustrationRequest(BaseModel):
    image_url: str
    character_name: str
    scene: str

@app.get("/")
def root():
    return {"status": "DreamTale server is running"}

@app.get("/music/{track_name}")
def get_music(track_name: str):
    music_urls = {
        "ocean":   "https://cdn.pixabay.com/audio/2022/03/10/audio_270f798b14.mp3",
        "forest":  "https://cdn.pixabay.com/audio/2022/03/15/audio_8cb5b8b3b4.mp3",
        "rain":    "https://cdn.pixabay.com/audio/2022/01/18/audio_d0c6ff1c32.mp3",
        "lullaby": "https://cdn.pixabay.com/audio/2022/10/30/audio_8ff4e6b5de.mp3"
    }
    url = music_urls.get(track_name)
    if not url:
        raise HTTPException(status_code=404, detail="Track not found")
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch audio")
    return StreamingResponse(
        io.BytesIO(response.content),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline", "Accept-Ranges": "bytes", "Cache-Control": "public, max-age=86400"}
    )

@app.post("/tts-stream")
def tts_stream(req: TTSRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    voice_map = {
        "en":  {"languageCode": "en-US",  "name": "en-US-Journey-F"},
        "ar":  {"languageCode": "ar-XA",  "name": "ar-XA-Wavenet-A"},
        "fr":  {"languageCode": "fr-FR",  "name": "fr-FR-Journey-F"},
        "es":  {"languageCode": "es-ES",  "name": "es-ES-Journey-F"},
        "pt":  {"languageCode": "pt-BR",  "name": "pt-BR-Wavenet-A"},
        "de":  {"languageCode": "de-DE",  "name": "de-DE-Journey-F"},
        "zh":  {"languageCode": "cmn-CN", "name": "cmn-CN-Wavenet-A"},
        "hi":  {"languageCode": "hi-IN",  "name": "hi-IN-Wavenet-A"},
        "tr":  {"languageCode": "tr-TR",  "name": "tr-TR-Wavenet-E"},
        "id":  {"languageCode": "id-ID",  "name": "id-ID-Wavenet-A"},
        "ru":  {"languageCode": "ru-RU",  "name": "ru-RU-Wavenet-A"},
        "ja":  {"languageCode": "ja-JP",  "name": "ja-JP-Wavenet-A"},
        "ko":  {"languageCode": "ko-KR",  "name": "ko-KR-Wavenet-A"},
        "it":  {"languageCode": "it-IT",  "name": "it-IT-Journey-F"},
        "nl":  {"languageCode": "nl-NL",  "name": "nl-NL-Wavenet-D"},
    }

    voice = voice_map.get(req.language, voice_map["en"])
    payload = {
        "input": {"text": req.text.strip()},
        "voice": voice,
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": 0.85}
    }

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Google TTS error: {response.text}")

    audio_bytes = base64.b64decode(response.json().get("audioContent", ""))
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline", "Accept-Ranges": "bytes"}
    )

@app.post("/create-avatar")
def create_avatar(req: AvatarRequest):
    if not FAL_API_KEY:
        raise HTTPException(status_code=500, detail="FAL API key not configured")

    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt": f"A cute child named {req.character_name} in a Pixar style children's book illustration. Warm colors, friendly expression, soft lighting, high quality, no text, no watermark.",
        "reference_image_urls": [req.image_url],
        "style": "AUTO",
        "magic_prompt_option": "OFF"
    }

    response = requests.post("https://fal.run/fal-ai/ideogram/character", headers=headers, json=payload, timeout=90)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"FAL error: {response.text}")

    images = response.json().get("images", [])
    if not images:
        raise HTTPException(status_code=500, detail="No image returned")
    return {"avatar_url": images[0].get("url", "")}

@app.post("/create-illustration")
def create_illustration(req: IllustrationRequest):
    if not FAL_API_KEY:
        raise HTTPException(status_code=500, detail="FAL API key not configured")

    headers = {"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "prompt": f"Pixar style children's storybook illustration. The child from the reference photo is the main character standing in the center of the scene. The child is {req.scene}. Full body visible, child is prominent in the image, magical colorful background, warm soft lighting, no text, no watermark.",
        "reference_image_urls": [req.image_url],
        "style": "AUTO",
        "magic_prompt_option": "OFF"
    }

    response = requests.post("https://fal.run/fal-ai/ideogram/character", headers=headers, json=payload, timeout=90)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"FAL error: {response.text}")

    images = response.json().get("images", [])
    if not images:
        raise HTTPException(status_code=500, detail="No image returned")
    return {"illustration_url": images[0].get("url", "")}
