from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import requests
import base64
import os
import io
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
FAL_API_KEY = os.environ.get("FAL_API_KEY", "")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

photo_store = {}

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

class IllustrationRequest(BaseModel):
    image_url: str = ""
    character_name: str
    scene: str
    age: int = 6
    character_description: str = ""
    reference_image_id: str = ""

class DescribeChildRequest(BaseModel):
    photo_url: str = ""
    photo_base64: str = ""
    age: int = 6

@app.get("/")
def root():
    return {"status": "DreamTale server is running"}

@app.post("/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    contents = await file.read()
    photo_id = str(uuid.uuid4())
    photo_store[photo_id] = {
        "data": contents,
        "content_type": file.content_type or "image/jpeg"
    }
    photo_url = f"https://dreamtale-tts.onrender.com/photo/{photo_id}"
    photo_b64 = base64.b64encode(contents).decode("utf-8")
    content_type = file.content_type or "image/jpeg"
    data_uri = f"data:{content_type};base64,{photo_b64}"
    print(f"[UPLOAD] Photo stored: {photo_id}")
    return {
        "photo_url": photo_url,
        "photo_id": photo_id,
        "photo_base64": data_uri
    }

@app.get("/photo/{photo_id}")
def get_photo(photo_id: str):
    if photo_id not in photo_store:
        raise HTTPException(status_code=404, detail="Photo not found")
    photo = photo_store[photo_id]
    return StreamingResponse(
        io.BytesIO(photo["data"]),
        media_type=photo["content_type"]
    )

@app.post("/describe-child")
def describe_child(req: DescribeChildRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    image_content = None
    if req.photo_base64 and req.photo_base64.strip():
        image_content = {
            "type": "image_url",
            "image_url": {"url": req.photo_base64.strip()}
        }
        print("[DESCRIBE] Using base64 image data")
    elif req.photo_url and req.photo_url.strip():
        image_content = {
            "type": "image_url",
            "image_url": {"url": req.photo_url.strip()}
        }
        print(f"[DESCRIBE] Using photo URL: {req.photo_url[:80]}...")
    else:
        raise HTTPException(status_code=400, detail="No photo provided")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": 150,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_content,
                    {
                        "type": "text",
                        "text": f"Describe this child for a cartoon illustrator who needs to draw them accurately. The child is {req.age} years old. Format EXACTLY like this example: 'a 5 year old boy with short black curly hair, dark brown eyes, dark brown skin, round face, wearing a white shirt'. IMPORTANT: You MUST mention the skin color/tone explicitly (e.g. dark brown skin, light skin, olive skin, tan skin, pale skin). IMPORTANT: You MUST mention hair color, hair style, eye color, skin tone, and face shape. Under 40 words. No name. No extra sentences. Just the description."
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code != 200:
            print(f"[DESCRIBE ERROR] OpenAI returned {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"OpenAI error: {response.text}")
        description = response.json()["choices"][0]["message"]["content"].strip()
        print(f"[DESCRIBE] Result: {description}")
        return {"character_description": description}
    except requests.exceptions.Timeout:
        print("[DESCRIBE ERROR] OpenAI request timed out")
        raise HTTPException(status_code=500, detail="OpenAI request timed out")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[DESCRIBE ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts-stream")
def tts_stream(req: TTSRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    voice_map = {
        "en":  {"languageCode": "en-US",  "name": "en-US-Journey-F"},
        "ar":  {"languageCode": "ar-XA",  "name": "ar-XA-Wavenet-D"},
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

@app.post("/create-illustration")
def create_illustration(req: IllustrationRequest):
    if not SILICONFLOW_API_KEY:
        raise HTTPException(status_code=500, detail="SiliconFlow API key not configured")

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    if req.character_description and req.character_description.strip():
        character_desc = req.character_description.strip()
    else:
        character_desc = f"a {req.age} year old child with big expressive eyes, round face, soft cheeks, cheerful smile"

    print(f"[ILLUSTRATION] Character description: {character_desc}")

    prompt = f"A wide establishing shot of a children's storybook scene, seen from far away. {req.scene}. Small figures of {character_desc} can be seen in the distance. Detailed environment fills the entire frame, warm golden lighting, vibrant watercolor storybook colors."

    print(f"[ILLUSTRATION] Prompt: {prompt[:300]}...")

    payload = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt,
        "image_size": "768x1024",
        "num_inference_steps": 4,
        "n": 1
    }

    try:
        response = requests.post(
            "https://api.siliconflow.com/v1/images/generations",
            headers=headers,
            json=payload,
            timeout=120
        )
        if response.status_code != 200:
            print(f"[ILLUSTRATION ERROR] SiliconFlow returned {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"SiliconFlow error: {response.text}")
        result = response.json()
        images = result.get("images", [])
        if not images:
            raise HTTPException(status_code=500, detail="No image returned")
        siliconflow_url = images[0].get("url", "")
        if not siliconflow_url:
            raise HTTPException(status_code=500, detail="No image URL returned")
        img_response = requests.get(siliconflow_url, timeout=30)
        if img_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch image from SiliconFlow")
        img_id = str(uuid.uuid4())
        photo_store[img_id] = {
            "data": img_response.content,
            "content_type": "image/jpeg"
        }
        proxied_url = f"https://dreamtale-tts.onrender.com/photo/{img_id}"
        print(f"[ILLUSTRATION] Success: {proxied_url}")
        return {"illustration_url": proxied_url}
    except requests.exceptions.Timeout:
        print("[ILLUSTRATION ERROR] SiliconFlow request timed out")
        raise HTTPException(status_code=500, detail="Image generation timed out")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ILLUSTRATION ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
