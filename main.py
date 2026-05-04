from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import requests
import base64
import os
import io
import uuid
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

photo_store = {}

class TTSRequest(BaseModel):
    text: str
    language: str = "en"

class IllustrationRequest(BaseModel):
    image_prompt: str = ""

class StoryRequest(BaseModel):
    child_name: str
    age: str
    theme: str
    length: str
    language: str
    story_type: str
    character_description: str = ""

class ThemeRequest(BaseModel):
    language: str
    recent_themes: list = []
    child_age: int = None

@app.get("/")
def root():
    return {"status": "DreamTale server is running"}

@app.get("/photo/{photo_id}")
def get_photo(photo_id: str):
    if photo_id not in photo_store:
        raise HTTPException(status_code=404, detail="Photo not found")
    photo = photo_store[photo_id]
    return StreamingResponse(
        io.BytesIO(photo["data"]),
        media_type=photo["content_type"]
    )

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
        headers={
            "Content-Disposition": "inline",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(audio_bytes))
        }
    )

@app.post("/create-illustration")
def create_illustration(req: IllustrationRequest):
    if not SILICONFLOW_API_KEY:
        raise HTTPException(status_code=500, detail="SiliconFlow API key not configured")

    if not req.image_prompt or not req.image_prompt.strip():
        raise HTTPException(status_code=400, detail="image_prompt is required")

    prompt = req.image_prompt.strip()
    print(f"[ILLUSTRATION] Prompt: {prompt[:200]}...")

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt,
        "image_size": "768x1024",
        "num_inference_steps": 12,
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

@app.post("/generate-story")
def generate_story(req: StoryRequest):
    import openai

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    description = req.character_description if req.character_description else f"a {req.age} year old child"
    characters_block = f"- {req.child_name}: {description}"

    system_prompt = (
        "You are a children's book author who also writes image prompts for each page.\n\n"
        "Output ONLY valid JSON:\n"
        "{\n"
        '  "title": "...",\n'
        '  "pages": [{"text": "...", "shot_type": "hero", "image_prompt": "..."}],\n'
        '  "moral": "..."\n'
        "}\n\n"
        f"All text/title/moral fields in {req.language}. All image_prompt fields in English.\n\n"
        "STORY RULES:\n"
        "- Never mention character appearance in story text\n"
        "- Action-driven opening, not waking up\n"
        "- Add dialogue and sensory details\n"
        "- short=3 pages, medium=5 pages, long=7 pages\n\n"
        "IMAGE PROMPT RULES:\n"
        "Every image_prompt must follow this exact structure:\n"
        "\"Children's picture book illustration. [CAMERA AND DISTANCE]. [CHARACTER DESCRIPTION VERBATIM] is [DESCRIBE THE ACTION IN PHYSICAL DETAIL: body position, limbs, facial expression, what their hands are doing]. [DESCRIBE WHAT IS HAPPENING IN THE SCENE AROUND THEM: where the creature/object is, what it is doing, how it relates to the character]. In the background: [AT LEAST 3 SPECIFIC ENVIRONMENT DETAILS]. The character is small relative to the scene. Whimsical storybook art style. No text, no words, no watermarks.\"\n\n"
        "CAMERA AND DISTANCE rules:\n"
        "- Page 1: 'Wide shot, full body visible, low camera angle looking slightly up, character face visible'\n"
        "- Last page: 'Wide establishing shot, character small in frame, ground level camera'\n"
        "- Middle pages: alternate between 'Wide shot, full body visible' and 'Medium wide shot, character from knees up'\n"
        "- NEVER crop closer than knees. NEVER use high angle or bird's eye view.\n\n"
        "CHARACTER rules:\n"
        "- Paste character description VERBATIM into every prompt\n"
        "- When animals/creatures appear, state their position: 'standing 2 meters to the left of the character'\n\n"
        "CRITICAL: The image must show the specific action happening in the story text. Not just the setting."
    )
    user_message = (
        f"CHARACTERS:\n{characters_block}\n\n"
        f"Write a {req.length} {req.story_type} story in {req.language} about: {req.theme}\n\n"
        f"The main character is {req.child_name}."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        story_json_str = response.choices[0].message.content
        story_data = json.loads(story_json_str)
        return {"story": story_data}

    except Exception as e:
        print(f"[STORY ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-theme")
def generate_theme(req: ThemeRequest):
    import openai
    import random

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    fallbacks = [
        "A tiny dragon who is afraid of fire and wants to be a baker",
        "A race across the ocean on the backs of friendly whales",
        "A pet goldfish who secretly runs the household at night",
    ]

    recent_block = "None yet." if not req.recent_themes else "\n".join([f"- {t}" for t in req.recent_themes[:10]])
    seed = random.randint(0, 99999)
    age_block = f"AGE: {req.child_age} years old. Fun adventures appropriate for this age."

       system_content = (
        "You are a children's story theme generator. "
        "Output ONLY a JSON object: {\"theme\": \"...\"}\n"
        "The theme must be ONE SHORT SENTENCE, 8-16 words, in the requested language.\n\n"
        "RULES:\n"
        "- Look at the recent themes and detect what category the user prefers: fantasy, realistic, animals, adventure, school, family, silly, nature, etc.\n"
        "- Generate a new theme in the SAME category they seem to enjoy, but with a fresh scenario.\n"
        "- If no recent themes exist, pick based on age: under 4 = simple animals and home; 4-6 = magical creatures and short adventures; 7-10 = school, sports, friendships, mild fantasy; 11+ = real-life challenges, friendships, discovering talents.\n"
        "- Always match difficulty and concept complexity to the child's age.\n"
        "- Include both fantasy AND real-life themes in rotation — never stay only in one genre.\n"
        "- NEVER repeat a recent theme. NEVER use overused tropes like 'a dragon who' or 'a princess who'.\n"
        "- Theme must have ONE clear setting and ONE concrete situation."
    )

    user_content = (
        f"Give me ONE fresh story theme.\n\n"
        f"LANGUAGE: {req.language}\n"
        f"{age_block}\n\n"
        f"RECENT THEMES TO AVOID:\n{recent_block}\n\n"
        f"CREATIVITY SEED: {seed}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=1.1,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        )

        content = json.loads(response.choices[0].message.content)
        theme = content.get("theme", "").strip()

        if theme:
            return {"theme": theme}

        return {"theme": random.choice(fallbacks)}

    except Exception as e:
        print(f"[THEME ERROR] {str(e)}")
        return {"theme": random.choice(fallbacks)}
