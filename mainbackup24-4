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
        "num_inference_steps": 8,
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
        "You are a world-class children's book author AND an expert at writing FLUX image generation prompts. "
        "You write beautiful original stories AND ready-to-use image prompts for each page.\n\n"
        "TWO SEPARATE JOBS, NEVER MIX THEM:\n\n"
        "JOB 1 — STORY TEXT (\"text\" field):\n"
        "- Write the story. Dialogue, action, emotions, plot.\n"
        "- NEVER describe what characters look like in the story text. Illustrations handle that.\n\n"
        "JOB 2 — IMAGE PROMPT (\"image_prompt\" field):\n"
        "- Write a ready-to-use FLUX prompt that creates a beautiful children's picture book illustration for the page.\n"
        "- Always in English regardless of story language.\n"
        "- Follow the STRICT TEMPLATE SYSTEM below exactly.\n"
        "- CRITICAL: You will be given character descriptions under CHARACTERS. Embed each character's full description EXACTLY as given into every image_prompt. Do not paraphrase, do not shorten, do not change anything. Use the descriptions verbatim.\n"
        "- CRITICAL: Every image_prompt must depict the KEY ACTION or MOMENT described in that page's story text. If the story says a puppy is carrying a muddy paper, the image must show the puppy carrying the muddy paper. NEVER illustrate just the setting — illustrate the SPECIFIC SCENE HAPPENING in the text.\n"
        "- CRITICAL: Characters must be shown DOING something — mid-action, mid-motion, mid-discovery. NEVER show characters simply standing still.\n\n"
        "Output ONLY as a JSON object in this exact format:\n"
        "{\n"
        '  "title": "Story title here",\n'
        '  "pages": [\n'
        "    {\n"
        '      "text": "Page text in the story\'s language",\n'
        '      "shot_type": "hero",\n'
        '      "image_prompt": "A ready-to-use FLUX image generation prompt in English"\n'
        "    }\n"
        "  ],\n"
        '  "moral": "Today\'s lesson: moral here"\n'
        "}\n\n"
        f"All \"text\", \"title\", and \"moral\" fields must be in {req.language}. All \"image_prompt\" fields must be in English. The \"shot_type\" field must always be \"hero\".\n\n"
        "HOW TO WRITE THE image_prompt FIELD — STRICT TEMPLATE SYSTEM:\n\n"
        "You MUST pick ONE of the 6 SHOT RECIPES below for each page. Fill in the blanks and that is your image_prompt.\n\n"
        "SHOT RECIPE A — OVER-THE-SHOULDER:\n"
        "\"Children's picture book illustration. [CHARACTER DESCRIPTION VERBATIM] shown from behind over their shoulder, looking at [SPECIFIC OBJECT/SCENE]. The character takes up the left or right third of the frame. Focus is on [THE THING THEY ARE LOOKING AT] which fills the rest of the image. Whimsical storybook art style, soft lighting, detailed background.\"\n\n"
        "SHOT RECIPE B — SIDE PROFILE WALKING/MOVING:\n"
        "\"Children's picture book illustration. Side profile view of [CHARACTER DESCRIPTION VERBATIM] walking/running/moving [DIRECTION] across the frame. Character positioned in left or right third. Background shows [ENVIRONMENT]. Whimsical storybook art style, sense of motion.\"\n\n"
        "SHOT RECIPE C — HIGH ANGLE LOOKING DOWN:\n"
        "\"Children's picture book illustration. High angle view looking down at [CHARACTER DESCRIPTION VERBATIM] who is [ACTION] in [LOCATION]. Character appears small in frame, surrounded by [ENVIRONMENT DETAILS]. Whimsical storybook art style, bird's eye perspective.\"\n\n"
        "SHOT RECIPE D — LOW ANGLE LOOKING UP:\n"
        "\"Children's picture book illustration. Low angle view looking up at [CHARACTER DESCRIPTION VERBATIM] who is [ACTION]. Character towers in frame against [SKY/CEILING/BACKGROUND]. Whimsical storybook art style, heroic perspective.\"\n\n"
        "SHOT RECIPE E — THREE-QUARTER BACK VIEW ENTERING (MANDATORY FOR PAGE 1):\n"
        "\"Children's picture book illustration. Three-quarter back view of [CHARACTER DESCRIPTION VERBATIM] stepping into/entering [NEW LOCATION]. Character positioned in foreground on left or right side. The new environment opens up before them in the background. Whimsical storybook art style, sense of discovery.\"\n\n"
        "SHOT RECIPE F — SMALL IN BIG SCENE, BACK VIEW (MANDATORY FOR LAST PAGE):\n"
        "\"Children's picture book illustration. [CHARACTER DESCRIPTION VERBATIM] shown small from behind, standing in a vast [ENVIRONMENT]. Character takes up only small portion of bottom of frame. Epic landscape/scene fills most of image. Whimsical storybook art style, sense of wonder and scale.\"\n\n"
        "SHOT TYPE RULES:\n"
        "- Page 1 MUST use Recipe E\n"
        "- Last page MUST use Recipe F\n"
        "- Middle pages: freely mix Recipes A, B, C, D — never use the same recipe twice in a row\n"
        "- NEVER write a centered front-facing portrait composition\n\n"
        "CRITICAL RULES FOR CHARACTER DESCRIPTIONS:\n"
        "1. Copy character descriptions EXACTLY into every image_prompt where those characters appear.\n"
        "2. NEVER paraphrase character descriptions.\n"
        "3. NEVER add clothing or features not in the original description.\n"
        "4. NEVER omit details from the original description.\n\n"
        "STORY LENGTH TARGETS:\n"
        "- short: 3 pages\n"
        "- medium: 5 pages\n"
        "- long: 7 pages\n\n"
        "STORY QUALITY RULES:\n"
        "- Action-driven openings (character doing something interesting, not waking up)\n"
        "- Surprising twists, creative solutions, meaningful choices\n"
        "- Dialogue that reveals character\n"
        "- Sensory details (sounds, textures, smells)\n"
        "- NEVER mention character appearance in story text\n"
        "- Stories MUST match the requested theme and tone\n\n"
        "THEME-SPECIFIC TONE GUIDANCE:\n"
        "For REALISTIC themes (school, friends, sports, family): keep magic minimal, ground in real-world settings.\n"
        "For FANTASY/ADVENTURE themes (dragons, space, pirates, magic): embrace imagination and wonder.\n\n"
        "AVOID:\n"
        "- Waking up openings\n"
        "- Describing character appearance in story text\n"
        "- Moralizing or preachy conclusions\n"
        "- Overly complex vocabulary"
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
        "You are a creative children's story idea generator. "
        "Output ONLY a JSON object: {\"theme\": \"...\"}\n"
        "The theme must be ONE SHORT SENTENCE, 8-16 words, in the requested language.\n"
        "Mix genres: magical, realistic, silly, adventurous.\n"
        "NEVER repeat recent themes. NEVER use overused tropes.\n"
        "Pick themes with ONE clear physical setting and CONCRETE visual subjects."
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
