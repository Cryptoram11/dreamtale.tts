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
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    # Per-language speed tuning: nova reads English fast, Arabic slow
    speed_map = {
        "en": 0.9,
        "ar": 1.15,
    }
    speed = speed_map.get(req.language, 1.0)

    try:
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini-tts",
                "voice": "nova",
                "input": req.text.strip(),
                "speed": speed,
                "instructions": "You are a loving parent reading a bedtime story to your own small child. Speak in a low, soft, intimate voice, full of warmth and tenderness, almost a whisper. Smile through your voice. Slow down at the end of sentences and let them land gently. Pause between sentences like you're letting the child picture the scene.",
                "response_format": "mp3",
            },
            timeout=120,
        )
        if response.status_code != 200:
            print(f"[TTS ERROR] OpenAI returned {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"OpenAI TTS error: {response.text}")

        audio_bytes = response.content
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(audio_bytes))
            }
        )
    except requests.exceptions.Timeout:
        print("[TTS ERROR] OpenAI TTS request timed out")
        raise HTTPException(status_code=500, detail="TTS timed out")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TTS ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    import random

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    description = req.character_description if req.character_description else f"a {req.age} year old child"
    characters_block = description if "\n" in description else f"- {req.child_name}: {description}"

    # Age-based pages and words per page
    try:
        age_num = int(req.age.split(" ")[0].split(" and ")[0])
    except (ValueError, IndexError):
        age_num = 5

    length_key = req.length.lower()
    pages = {"short": 3, "medium": 5}.get(length_key, 7)

    if age_num <= 3:
        words = {"short": "20-30", "medium": "25-35"}.get(length_key, "30-40")
        vocab_rule = "Age 2-3: only very simple everyday words, max 3 syllables. Short sentences. Repeat one key phrase on every page — toddlers love repetition."
    elif age_num <= 5:
        words = {"short": "40-55", "medium": "50-65"}.get(length_key, "55-70")
        vocab_rule = "Age 4-5: simple sentences, easy dialogue, familiar words. No abstract words like 'reassured' or 'determined'."
    elif age_num <= 7:
        words = {"short": "60-80", "medium": "80-100"}.get(length_key, "100-120")
        vocab_rule = "Age 6-7: fun adventurous language, simple dialogue, mild suspense allowed."
    elif age_num <= 10:
        words = {"short": "80-110", "medium": "100-130"}.get(length_key, "120-160")
        vocab_rule = "Age 8-10: richer vocabulary, deeper emotions, more complex plot allowed."
    else:
        words = {"short": "100-130", "medium": "120-160"}.get(length_key, "150-200")
        vocab_rule = "Age 11+: sophisticated language, nuanced themes, real-world complexity."

    seed = random.randint(0, 99999)
    skeletons = [
        "PROBLEM-SOLVING: the hero faces a concrete problem and solves it through their own idea",
        "MISUNDERSTANDING: something seems scary or unfair but turns out to be a misunderstanding",
        "HELPING SOMEONE ELSE: the hero notices someone in trouble and helps them",
        "OVERCOMING A FEAR: the hero is afraid of something specific and faces it step by step",
        "UNEXPECTED FRIENDSHIP: the hero meets someone very different and they find common ground",
    ]
    skeleton = random.choice(skeletons)

    system_prompt = (
        "You are a children's book author who also writes image prompts for each page.\n\n"
        "Output ONLY valid JSON:\n"
        "{\n"
        '  "title": "...",\n'
        '  "pages": [{"text": "...", "shot_type": "hero", "image_prompt": "..."}],\n'
        '  "moral": "..."\n'
        "}\n\n"
        f"All text/title/moral fields in {req.language}. All image_prompt fields in English.\n"
        "ARABIC RULE: If the story language is Arabic, ALL story text, title, and moral MUST be written with FULL tashkeel (حركات) on every word — fatha, damma, kasra, sukun, shadda, tanwin. Like a traditional children's book. No word may be left unvocalized.\n"
        "In image_prompt fields, NEVER use Arabic script — transliterate character names to Latin letters (لِيُونَا becomes Leona). Every character of every image_prompt must be English/Latin.\n\n"
        "STEP 1 — CLASSIFY THE THEME. Before writing, silently classify the user's theme into exactly one category:\n"
        "A) EVERYDAY/REALISTIC (school, family, friends, home, sports, pets)\n"
        "B) EDUCATIONAL (science, nature, numbers, how things work)\n"
        "C) VALUES/FEELINGS (sharing, honesty, fear, jealousy, kindness)\n"
        "D) ADVENTURE (exploration, journeys, discovery — real world)\n"
        "E) MAGICAL (fantasy, magic, mythical creatures)\n\n"
        "TONE CONTRACT — follow the rules of the classified category strictly:\n"
        "- A EVERYDAY: ZERO magic. No talking animals, no enchanted objects, no fantasy. Conflicts are real-kid-sized: getting lost, a broken toy, a disagreement. The wonder comes from the child's real world.\n"
        "- B EDUCATIONAL: weave 2-3 TRUE facts naturally into the plot. No magic unless the theme explicitly asks. The facts must be accurate.\n"
        "- C VALUES: the emotional arc IS the plot. Real situations, real feelings. No magic unless the theme asks.\n"
        "- D ADVENTURE: exciting but real-world. No magic unless the theme explicitly includes it.\n"
        "- E MAGICAL: full fantasy allowed.\n"
        "NEVER drift a realistic theme into fantasy. If the parent typed 'first day at school', there are no magical hallways.\n\n"
        f"PLOT SKELETON for this story: {skeleton}\n"
        "Use this skeleton as the story's structure. Fill it with fresh, specific details from the theme.\n\n"
        f"LENGTH: Write EXACTLY {pages} pages. Each page {words} words. Not more pages, not fewer.\n\n"
        f"VOCABULARY: {vocab_rule}\n\n"
        "STORY RULES:\n"
        "- Never mention character appearance in story text\n"
        "- Action-driven opening, not waking up, never 'Once upon a time'\n"
        "- At least 40% dialogue with real kid voices: 'Whoa!' not 'How wonderful!'\n"
        "- One sensory detail per page (sound, smell, texture, temperature)\n"
        "- Story must have a clear problem, build-up, and satisfying resolution\n"
        "- LAST PAGE MUST WIND DOWN: quiet, warm, safe, sleepy. Yawning, cozy light, calm. "
        "This is a bedtime story — a high-energy ending like 'ready for anything!' is a failure. "
        "The final sentences should make a child's eyes heavy. "
        "MANDATORY: the last page must contain at least one physical sleepiness cue — a yawn, heavy eyes, curling up, dimming light, or settling into quiet — in every language, no exceptions. Laughing or cheering on the last page is forbidden.\n\n"
        "MORAL RULES:\n"
        "- The moral must name the SPECIFIC thing the hero did in this story, not a generic virtue.\n"
        "- Bad: 'Bravery grows with a helping hand.' Good: 'Leona asked the teacher for help instead of hiding — asking is brave.'\n"
        "- The moral must connect to the theme the parent chose.\n\n"
        "- The action named in the moral MUST actually happen in the story text. Never invent an action for the moral that the hero did not do in the pages.\n\n"

        "CHARACTER DESCRIPTION RULES:\n"
        "- The CHARACTERS block below contains each character's exact description including their outfit.\n"
        "- Paste each description into every image_prompt CHARACTER-FOR-CHARACTER. Never reword, shorten, hyphenate differently, or split it. It is one atomic block.\n"
        "- The name alone is NEVER enough. Every image_prompt must contain the FULL physical description (age, skin, face, hair, eyes, outfit) even though it repeats on every page. Writing just 'Leona is...' without the full description is a failure.\n""- OUTFIT ADAPTATION: if the story setting demands different clothing (winter, rain, swimming, bedtime), adapt the garment ONCE for the whole story but KEEP THE SAME COLOR (red t-shirt becomes red winter coat, red pajamas). Then use that adapted outfit identically on every page. Never change clothing mid-story.\n"
        "- Format: 'NAME, DESCRIPTION, is ACTION' — for example: 'Leona, a 3 year old girl with olive skin, a round face, braided light brown hair, brown eyes, is holding her mother's hand'. Note the commas — never write 'is' twice.\n"
        "- MULTI-CHARACTER: assign each character a fixed position ('on the left, [full description A]; on the right, [full description B]') and keep those positions on every page. Never merge characters. Never drop a character from a page where the story text includes them.\n\n"
        "IMAGE PROMPT RULES:\n"
        "Every image_prompt must follow this exact structure:\n"
        "\"Children's picture book illustration. [CAMERA AND DISTANCE]. [CHARACTER DESCRIPTION VERBATIM] is [DESCRIBE THE ACTION IN PHYSICAL DETAIL: body position, limbs, facial expression, what their hands are doing]. [DESCRIBE WHAT IS HAPPENING IN THE SCENE AROUND THEM: where the creature/object is, what it is doing, how it relates to the character]. In the background: [AT LEAST 3 SPECIFIC ENVIRONMENT DETAILS FROM THE STORY]. The character is small relative to the scene. Whimsical storybook art style. No text, no words, no watermarks.\"\n\n"
        "CAMERA AND DISTANCE rules:\n"
        "- Page 1: 'Wide shot, full body visible, low camera angle looking slightly up, character face and expression visible'\n"
        "- Last page: 'Wide establishing shot, character small in frame, ground level camera'\n"
        "- Middle pages: alternate between 'Wide shot, full body visible' and 'Medium wide shot, character from knees up'\n"
        "- NEVER crop closer than knees. NEVER use high angle or bird's eye view.\n\n"
        "- When animals or creatures appear, state their exact position relative to the character: 'standing 2 meters to the left', 'perched on a branch above', 'running ahead of the character'\n"
        "- NEVER merge a character and an animal into one figure\n\n"
        "CRITICAL: The image must illustrate the specific moment happening in the story text. "
        "A viewer who has not read the story should understand what is happening just by looking at the image."
    )

    user_message = (
        f"CREATIVITY SEED: {seed}\n"
        "This story must be COMPLETELY DIFFERENT from any other story with the same theme — different setting details, different conflict, different resolution.\n\n"
        f"CHARACTERS:\n{characters_block}\n\n"
        f"Write a {req.length} story in {req.language} about: {req.theme}\n\n"
        f"The hero(es): {req.child_name}."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.85,
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
        "A child who forgot their homework finds a creative solution",
        "Two friends disagree on the playground and learn to listen",
        "A kid discovers a hidden talent during a school talent show",
    ]

    recent_block = "None yet." if not req.recent_themes else "\n".join([f"- {t}" for t in req.recent_themes[:10]])
    seed = random.randint(0, 99999)
    age_block = f"AGE: {req.child_age} years old."

    system_content = (
        "You are a children's story theme generator. "
        "Output ONLY a JSON object: {\"theme\": \"...\"}\n"
        "The theme must be ONE SHORT SENTENCE, 8-16 words, in the requested language.\n\n"
        "RULES:\n"
        "- Look at the recent themes and detect what category the user prefers: fantasy, realistic, animals, adventure, school, family, silly, nature, etc.\n"
        "- Generate a new theme in the SAME category they seem to enjoy, but with a completely fresh scenario.\n"
        "- If no recent themes exist, pick based on age: under 4 = simple animals and home situations; 4-6 = animals, simple adventures, family; 7-10 = school, sports, friendships, mild fantasy; 11+ = real-life challenges, discovering talents, friendships.\n"
        "- Always match difficulty and concept complexity to the child's age.\n"
        "- Rotate between fantasy AND real-life themes — never stay in one genre more than 2 themes in a row.\n"
        "- NEVER use the word magical. NEVER use overused tropes like 'a dragon who' or 'a princess who'.\n"
        "- NEVER repeat or closely resemble a recent theme.\n"
        "- Theme must have ONE clear setting and ONE concrete situation."
    )

    user_content = (
        f"Give me ONE fresh story theme.\n\n"
        f"LANGUAGE: {req.language}\n"
        f"{age_block}\n\n"
        f"RECENT THEMES TO AVOID:\n{recent_block}\n\n"
        f"PATTERN DETECTED: Look at the recent themes above and identify what category they belong to. Generate a theme in that SAME category.\n\n"
        f"CREATIVITY SEED: {seed}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.7,
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
