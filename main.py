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
import re

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
    scene: str =""
    age: int = 6
    character_description: str = ""
    reference_image_id: str = ""
    shot_type: str = "wide"
    image_prompt: str = ""

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

    # character_description is intentionally ignored for wide shots.
    # Option 1 strategy: no appearance anchor in the text prompt. Character consistency
    # will come from Kontext-pro reference image wiring on pages 2+ (future work).
    print(f"[ILLUSTRATION] Character description received (ignored for wide shots): {req.character_description[:80] if req.character_description else 'none'}")
    # NEW PATH: if image_prompt is provided, use it directly and skip all the old logic.
    # This bypasses regex, template branching, shot_type logic — everything.
    if req.image_prompt and req.image_prompt.strip():
        prompt = req.image_prompt.strip()
        print(f"[ILLUSTRATION] Using direct image_prompt: {prompt[:200]}...")

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

    # OLD PATH: if no image_prompt, use the existing scene + template + regex logic below.
    scene_text = req.scene or ""

    # Sentence-level filter: drop any sentence that mentions a person or person-action.
    # This preserves grammar on the surviving sentences — FLUX needs readable English,
    # not shredded fragments from word-level stripping.
    person_markers = re.compile(
        r'\b('
        r'\d+[-\s]?year[-\s]?old|'
        r'year\s*old|'
        r'boy|boys|girl|girls|child|children|kid|kids|toddler|toddlers|baby|babies|'
        r'person|people|figure|figures|hero|heroes|'
        r'he|she|they|him|her|them|his|hers|their|theirs|'
        r'wearing|hair|eyes|skin|face'
        r')\b',
        flags=re.IGNORECASE
    )

    sentences = re.split(r'(?<=[.!?])\s+', scene_text)
    clean_sentences = [s for s in sentences if s.strip() and not person_markers.search(s)]
    scene_text = ' '.join(clean_sentences).strip()

    # Safety fallback — if GPT's scene was entirely character-focused and nothing
    # survived the filter, use a generic landscape cue so FLUX doesn't hallucinate wildly.
    if len(scene_text) < 20:
        scene_text = "a vast detailed storybook landscape with rich environmental detail"

    # Hardcoded wide-shot template. GPT's cleaned scene only fills the Environment slot.
    # No character_description injection — generic "small child figure" only.
    if req.shot_type == "hero":
        prompt = (
            f"Hand-painted watercolor children's storybook illustration, hero shot. "
            f"A young child in a natural action pose fitting the scene — looking around with curiosity, reaching toward something, turning to look, gesturing, or caught mid-motion — never standing stiff and centered facing the camera. Framed from the knees up, taking up the middle third of the frame vertically, the child's head in the upper portion of the frame, shoulders and upper body visible, plenty of environment visible above the head and to the sides. "            f"The environment is visible behind the child, soft and slightly out of focus, supporting the character. "
            f"Hand-painted watercolor in classic children's picture book style, soft brushstrokes, painted textures, warm cozy colors, gentle lighting. "
            f"NOT photo-realistic, NOT 3D rendered, NOT CGI. "
            f"Like a published children's picture book illustration."
        )
    else:
        prompt = (
            f"Hand-painted watercolor children's storybook illustration, wide shot. "
            f"Environment: {scene_text}. "
            f"A small child figure is visible in the scene, about one fifth of the frame height, full body from head to feet, shown from a distance so the whole setting is clearly visible around them. "
            f"The environment fills most of the frame with rich detail. "
            f"Hand-painted watercolor in classic children's picture book style, soft brushstrokes, painted textures, warm colors, gentle lighting. "
            f"NOT photo-realistic, NOT 3D rendered, NOT CGI. "
            f"Eye-level wide angle, camera pulled back, character small in a big scene."
        )
    print(f"[ILLUSTRATION] Final prompt: {prompt}")

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
