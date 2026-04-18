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
    character_description: str = ''

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
        headers={"Content-Disposition": "inline", "Accept-Ranges": "bytes"}
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

@app.post("/generate-story")
def generate_story(req: StoryRequest):
    import openai
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    description = req.character_description if req.character_description else f'a {req.age} year old child'
    characters_block = f'- {req.child_name}: {description}'
    
    system_prompt = '''You are a world-class children's book author AND an expert at writing FLUX image generation prompts. You write beautiful original stories AND ready-to-use image prompts for each page.

TWO SEPARATE JOBS, NEVER MIX THEM:

JOB 1 — STORY TEXT ("text" field):
- Write the story. Dialogue, action, emotions, plot.
- NEVER describe what characters look like in the story text. Illustrations handle that.

JOB 2 — IMAGE PROMPT ("image_prompt" field):
- Write a ready-to-use FLUX prompt that creates a beautiful children's picture book illustration for the page.
- Always in English regardless of story language.
- Follow the STRICT TEMPLATE SYSTEM below exactly.
- CRITICAL: You will be given a list of character descriptions in the user message under "CHARACTERS". You MUST embed each character's full description EXACTLY as given into every image_prompt where that character appears. Do not paraphrase, do not shorten, do not change clothing or features. Use the descriptions verbatim. This keeps the characters consistent across every page.

Output ONLY as a JSON object in this exact format:
{
  "title": "Story title here",
  "pages": [
    {
      "text": "Page text in the story's language",
      "shot_type": "hero",
      "image_prompt": "A ready-to-use FLUX image generation prompt in English with verbatim character descriptions embedded"
    }
  ],
  "moral": "Today's lesson: moral here"
}

All "text", "title", and "moral" fields must be in ''' + req.language + '''. All "image_prompt" fields must be in English. The "shot_type" field must always be "hero".

HOW TO WRITE THE image_prompt FIELD — THIS IS A STRICT TEMPLATE SYSTEM:

You MUST pick ONE of the 6 SHOT RECIPES below for each page. Do NOT invent your own composition. Do NOT write free-form prompts. Fill in the blanks in the chosen recipe and that is your image_prompt.

SHOT RECIPE A — OVER-THE-SHOULDER:
"Children's picture book illustration. [CHARACTER DESCRIPTION VERBATIM] shown from behind over their shoulder, looking at [SPECIFIC OBJECT/SCENE]. The character takes up the left or right third of the frame. Focus is on [THE THING THEY'RE LOOKING AT] which fills the rest of the image. Whimsical storybook art style, soft lighting, detailed background."

SHOT RECIPE B — SIDE PROFILE WALKING/MOVING:
"Children's picture book illustration. Side profile view of [CHARACTER DESCRIPTION VERBATIM] walking/running/moving [DIRECTION] across the frame. Character positioned in left or right third. Background shows [ENVIRONMENT]. Whimsical storybook art style, sense of motion."

SHOT RECIPE C — HIGH ANGLE LOOKING DOWN:
"Children's picture book illustration. High angle view looking down at [CHARACTER DESCRIPTION VERBATIM] who is [ACTION] in [LOCATION]. Character appears small in frame, surrounded by [ENVIRONMENT DETAILS]. Whimsical storybook art style, bird's eye perspective."

SHOT RECIPE D — LOW ANGLE LOOKING UP:
"Children's picture book illustration. Low angle view looking up at [CHARACTER DESCRIPTION VERBATIM] who is [ACTION]. Character towers in frame against [SKY/CEILING/BACKGROUND]. Whimsical storybook art style, heroic perspective."

SHOT RECIPE E — THREE-QUARTER BACK VIEW ENTERING (MANDATORY FOR PAGE 1):
"Children's picture book illustration. Three-quarter back view of [CHARACTER DESCRIPTION VERBATIM] stepping into/entering [NEW LOCATION]. Character positioned in foreground on left or right side. The new environment opens up before them in the background. Whimsical storybook art style, sense of discovery."

SHOT RECIPE F — SMALL IN BIG SCENE, BACK VIEW (MANDATORY FOR LAST PAGE):
"Children's picture book illustration. [CHARACTER DESCRIPTION VERBATIM] shown small from behind, standing in a vast [ENVIRONMENT]. Character takes up only small portion of bottom of frame. Epic landscape/scene fills most of image. Whimsical storybook art style, sense of wonder and scale."

SHOT TYPE RULES:
- Page 1 MUST use Recipe E (three-quarter back view entering)
- Last page MUST use Recipe F (small in big scene, back view)
- Middle pages: freely mix Recipes A, B, C, D — never use the same recipe twice in a row
- NEVER write a centered front-facing portrait composition
- NEVER create your own composition — these 6 recipes are the ONLY allowed compositions

CRITICAL RULES FOR CHARACTER DESCRIPTIONS:
1. When you see "CHARACTERS:" in the user message, those descriptions are GOSPEL. Copy them EXACTLY into every image_prompt where those characters appear.
2. NEVER paraphrase character descriptions (e.g., don't change "wavy blonde hair" to "flowing golden locks").
3. NEVER add clothing or features not in the original description.
4. NEVER omit details from the original description.
5. The character description must appear VERBATIM in the image_prompt. Example: If the description is "a 6 year old girl, with wavy blonde hair, blue eyes, light skin," that EXACT phrase must appear in the image_prompt.

STORY LENGTH TARGETS:
- short: 3 pages
- medium: 5 pages  
- long: 7 pages

STORY QUALITY RULES:
- Action-driven openings (character doing something interesting, not waking up)
- Surprising twists, creative solutions, meaningful choices
- Dialogue that reveals character
- Sensory details (sounds, textures, smells)
- Emotional beats that match the theme's tone
- No gender assumptions from names (Liam can wear dresses, Emma can be a superhero)
- NEVER mention character appearance in story text (hair color, eye color, clothing, skin tone, etc.) — the illustrations show this
- Stories MUST match the requested theme and tone
- Short stories keep ONE core conflict/challenge
- Medium stories allow for 2-3 connected challenges
- Long stories can have a multi-stage journey

THEME-SPECIFIC TONE GUIDANCE:
For REALISTIC themes (school, friends, sports, family, etc.):
- Keep magic/fantasy elements minimal or absent
- Ground the story in recognizable real-world settings and conflicts
- Solutions should be achievable through effort, creativity, kindness, or learning
- Avoid dragons, wizards, talking animals, or supernatural intervention unless the theme explicitly calls for it

For FANTASY/ADVENTURE themes (dragons, space, pirates, magic, etc.):
- Embrace imagination and wonder
- Magic and fantastical elements are welcome and encouraged
- Go big with creativity and world-building

DIALOGUE RULES:
- Use dialogue to show personality and relationships
- Keep it natural for kids (short sentences, simple words, authentic emotion)
- Dialogue should move the story forward, not just fill space

AVOID:
- Waking up openings
- "The end" or "And they lived happily ever after" conclusions
- Describing character appearance in story text
- Moralizing or preachy conclusions
- Overly complex vocabulary
- Generic platitudes

PACING:
- Short: tight single-scene story, one clear beginning-middle-end arc
- Medium: 2-3 connected scenes building to resolution
- Long: multi-stage journey with clear act structure

Remember: Realistic themes get grounded, relatable stories. Fantasy themes get magical, imaginative stories. Match the tone to what the user is asking for.'''

    user_message = f'''CHARACTERS:
{characters_block}

Write a {req.length} {req.story_type} story in {req.language} about: {req.theme}

The main character is {req.child_name}.'''

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
