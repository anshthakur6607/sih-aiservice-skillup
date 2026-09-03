"""
WebRTC Live Tutor Router - Full-duplex audio via Sarvam STT + Gemini + Chroma RAG
Supports:
  - Audio input (webm/opus from browser) → STT via Sarvam saaras:v3 (primary) → Gemini fallback
  - Text input (typed) → direct LLM call
  - TTS response (gemini-2.5-flash-preview-tts, 24kHz PCM)
  - Barge-in (interrupt mid-speech)
  - 10 Indian languages
  - Course RAG: Supabase course_materials (20) + Chroma mospi_knowledge_base per question
"""
import os, json, logging, base64, asyncio, io
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter()

LANG_VOICE = {
    "en": "Kore", "hi": "Kore", "bn": "Kore", "ta": "Kore", "te": "Kore",
    "mr": "Kore", "gu": "Kore", "kn": "Kore", "ml": "Kore", "pa": "Kore", "or": "Kore",
}

LANG_FULL_NAME = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia",
}

# Sarvam BCP-47 mapping: https://docs.sarvam.ai/api-reference/speech-to-text/transcribe
LANG_TO_SARVAM = {
    "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
    "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN", "pa": "pa-IN", "or": "od-IN",
    "as": "as-IN", "ur": "ur-IN",
}

def _client():
    from google import genai
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

async def synth_tts(text: str, language: str = "en") -> Optional[str]:
    """Synthesize speech using Gemini TTS. Returns base64-encoded PCM 16-bit 24kHz mono audio."""
    if not text or not os.getenv("GOOGLE_API_KEY"):
        return None
    try:
        from google.genai import types
        client = _client()
        voice = LANG_VOICE.get(language, "Kore")
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            )
        )
        try:
            parts = response.candidates[0].content.parts
            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    raw = inline.data
                    if isinstance(raw, str):
                        raw = base64.b64decode(raw)
                    return base64.b64encode(raw).decode("ascii")
        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(f"TTS parse failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Gemini TTS failed: {e}")
        return None

# ========== SARVAM STT (Primary) ==========
async def transcribe_with_sarvam(audio_bytes: bytes, language: str = "en", mime: str = "audio/webm") -> tuple[Optional[str], Optional[str]]:
    """Sarvam saaras:v3 REST STT. Returns (transcript, detected_language_code). Docs: https://docs.sarvam.ai/api-reference/speech-to-text/transcribe"""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key or not audio_bytes or len(audio_bytes) < 500:
        return None, None
    # Try explicit lang first, retry unknown auto-detect if empty (fixes Hindi spoken while selector on en)
    candidates = []
    explicit = LANG_TO_SARVAM.get(language, "unknown")
    if explicit != "unknown": candidates = [explicit, "unknown"]
    else: candidates = ["unknown"]
    for sarvam_lang in candidates:
        # Determine filename / content-type from mime
        # Browser sends audio/webm;codecs=opus -> strip params
        base_mime = mime.split(";")[0].strip() or "audio/webm"
        ext = "webm"
        if "mp3" in base_mime: ext = "mp3"
        elif "wav" in base_mime: ext = "wav"
        elif "ogg" in base_mime: ext = "ogg"
        elif "mp4" in base_mime or "m4a" in base_mime: ext = "m4a"
        elif "flac" in base_mime: ext = "flac"
        filename = f"audio.{ext}"
        try:
            import httpx
            files = {"file": (filename, audio_bytes, base_mime)}
            data = {"language_code": sarvam_lang, "model": "saaras:v3"}
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.sarvam.ai/speech-to-text",
                    headers={"api-subscription-key": api_key},
                    files=files,
                    data=data,
                )
                if resp.status_code != 200:
                    logger.warning(f"Sarvam STT {resp.status_code} lang={sarvam_lang}: {resp.text[:300]}")
                    continue
                j = resp.json()
                transcript = (j.get("transcript") or "").strip()
                detected = j.get("language_code")
                if transcript:
                    logger.info(f"Sarvam STT ok lang={detected or sarvam_lang} len={len(transcript)} text='{transcript[:80]}'")
                    return transcript, detected
                logger.info(f"Sarvam empty with {sarvam_lang}, trying next candidate")
        except Exception as e:
            logger.warning(f"Sarvam STT exception lang={sarvam_lang}: {e}")
            continue
    return None, None

async def transcribe_with_gemini(audio_bytes: bytes, mime: str = "audio/webm", language: str = "en") -> str:
    """Fallback: Gemini multimodal STT"""
    if not audio_bytes or not os.getenv("GOOGLE_API_KEY"):
        return ""
    try:
        from google.genai import types
        client = _client()
        lang_name = LANG_FULL_NAME.get(language, "English")
        prompt = f"Listen to this audio and transcribe exactly what the user said. Reply with ONLY the transcribed text in {lang_name}, no commentary, no labels."
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime.split(';')[0].strip()),
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=parts)],
        )
        return (response.text or "").strip()
    except Exception as e:
        logger.error(f"Gemini STT failed: {e}")
        return ""

async def transcribe_audio(audio_bytes: bytes, mime: str = "audio/webm", language: str = "en") -> tuple[str, Optional[str]]:
    """Primary: Sarvam saaras:v3 → fallback Gemini 2.5 Flash. Returns (text, detected_lang_code)."""
    if not audio_bytes or len(audio_bytes) < 500:
        return "", None
    sarvam_text, detected = await transcribe_with_sarvam(audio_bytes, language=language, mime=mime)
    if sarvam_text and len(sarvam_text.strip()) >= 2:
        return sarvam_text.strip(), detected
    logger.info("Sarvam empty/failed, falling back to Gemini STT")
    gem = await transcribe_with_gemini(audio_bytes, mime=mime, language=language)
    return gem.strip(), None

async def chat_reply(prompt: str, language: str = "en") -> str:
    """Generate a tutor reply with RAG context."""
    if not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not configured."
    try:
        client = _client()
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return (response.text or "").strip()[:900]
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return ""

# ========== RAG HELPERS (Supabase + Chroma) ==========
def _build_supabase_rag(course_id: str, course_title_hint: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Fetch course + study materials from Supabase. Returns (rag_context, course_title)."""
    rag = ""
    title = course_title_hint
    try:
        from supabase import create_client
        supa = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
        if course_id and course_id not in ("general", "demo"):
            c = supa.table("courses").select("title, description, provider").eq("id", course_id).maybe_single().execute()
            if c and c.data:
                title = title or c.data.get("title")
                rag += f"Course: {c.data.get('title','')} by {c.data.get('provider','')} - {(c.data.get('description') or '')[:800]}\n"
            # Fetch up to 20 materials: PDFs, notes, captions, media metadata
            m = supa.table("course_materials").select("title, content_text, type, url, storage_path").eq("course_id", course_id).limit(20).execute()
            if m and m.data:
                for mm in m.data:
                    txt = (mm.get("content_text") or "")[:3000]
                    meta = f"Material: {mm.get('title','')} ({mm.get('type','')}) Source: {mm.get('url') or mm.get('storage_path') or 'course record'}"
                    if txt:
                        rag += f"\n{meta}\n{txt}\n"
                    else:
                        rag += f"\n{meta}\n"
                logger.info(f"Supabase RAG loaded {len(m.data)} materials for {course_id}: {len(rag)} chars")
        if title:
            rag = f"Course focus: {title}\n" + rag
    except Exception as e:
        logger.warning(f"Supabase RAG fetch failed: {e}")
    return rag, title

def _query_chroma(course_id: str, query_text: str, n_results: int = 5) -> str:
    """Query Chroma mospi_knowledge_base for course-filtered chunks relevant to query_text."""
    if not query_text or not course_id or course_id in ("general", "demo"):
        return ""
    try:
        import chromadb
        from chromadb.config import Settings
        persist = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        col_name = os.getenv("CHROMA_COLLECTION_NAME", "mospi_knowledge_base")
        # Use get() fallback if embedding fn missing; else query()
        chroma = chromadb.Client(Settings(persist_directory=persist, anonymized_telemetry=False))
        try:
            col = chroma.get_collection(col_name)
        except Exception:
            col = chroma.get_or_create_collection(col_name)
            return ""
        # Prefer filtered query
        try:
            res = col.query(query_texts=[query_text[:4000]], n_results=n_results, where={"course_id": course_id})
            docs = (res.get("documents") or [[]])[0]
            if docs:
                joined = "\n---\n".join(d[:1200] for d in docs if d)
                logger.info(f"Chroma hit {len(docs)} chunks for course={course_id} query='{query_text[:40]}'")
                return joined
        except Exception as qe:
            logger.warning(f"Chroma query where=course_id failed: {qe}, trying get()")
        # Fallback: list course chunks
        try:
            listed = col.get(where={"course_id": course_id}, limit=n_results)
            docs = listed.get("documents") or []
            if docs:
                return "\n---\n".join(d[:1200] for d in docs[:n_results] if d)
        except Exception as ge:
            logger.warning(f"Chroma get fallback failed: {ge}")
    except Exception as e:
        logger.warning(f"Chroma unavailable: {e}")
    return ""

@router.websocket("/ws/live-tutor")
async def live_tutor_ws(websocket: WebSocket, token: Optional[str] = None):
    await websocket.accept()
    logger.info(f"Live tutor WS connected (token={'yes' if token else 'no'}) sarvam={'yes' if os.getenv('SARVAM_API_KEY') else 'no'}")

    course_id = "general"
    module_id = "live_tutor"
    language = "en"
    course_title = None
    rag_context = ""
    conversation_history = []  # for multi-turn context
    # Buffer for small WS chunks to avoid per-250ms STT spam with Sarvam (which needs >0.5s audio)
    audio_buffer = bytearray()
    last_audio_mime = "audio/webm"

    try:
        # 1) Receive init message
        init_raw = await websocket.receive_text()
        init = json.loads(init_raw)
        course_id = init.get("course_id") or "general"
        module_id = init.get("module_id") or "live_tutor"
        language = init.get("language") or "en"
        if init.get("course_title"):
            course_title = init.get("course_title")
        logger.info(f"init course={course_id} module={module_id} lang={language} title={course_title}")

        # 2) Build RAG context: Supabase (20 materials) + Chroma initial dump
        rag_context, course_title = _build_supabase_rag(course_id, course_title)
        chroma_init = _query_chroma(course_id, f"overview introduction key concepts of {course_title or course_id}", n_results=3)
        if chroma_init:
            rag_context += f"\n\n--- Chroma course knowledge ---\n{chroma_init}\n"

        # 3) Welcome message
        lang_name = LANG_FULL_NAME.get(language, "English")
        welcome_text = (
            f"Welcome to {course_title or course_id}! I'm your live AI tutor. "
            f"I'll answer in {lang_name}. You can speak or type your question, "
            f"and interrupt me anytime by speaking. What would you like to learn?"
        )
        await websocket.send_text(json.dumps({"type": "welcome", "message": welcome_text}))
        await websocket.send_text(json.dumps({"type": "transcript", "role": "ai", "text": welcome_text, "timestamp": 0}))

        # Synthesize welcome TTS
        tts_b64 = await synth_tts(welcome_text, language)
        if tts_b64:
            await websocket.send_text(json.dumps({
                "type": "audio", "mime": "audio/pcm;rate=24000",
                "data": tts_b64, "sample_rate": 24000
            }))

        # 4) Main loop
        while True:
            msg = await websocket.receive()

            # === TEXT MESSAGE (typed input or control) ===
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                mtype = data.get("type")

                if mtype == "interrupt":
                    await websocket.send_text(json.dumps({"type": "interrupt_acknowledged"}))
                    audio_buffer.clear()
                    continue

                if mtype == "language_change":
                    language = data.get("language", language)
                    lang_name = LANG_FULL_NAME.get(language, "English")
                    ack = f"Switching to {lang_name}. Ask me anything."
                    await websocket.send_text(json.dumps({"type": "transcript", "role": "ai", "text": ack, "timestamp": 0}))
                    tts_b64 = await synth_tts(ack, language)
                    if tts_b64:
                        await websocket.send_text(json.dumps({
                            "type": "audio", "mime": "audio/pcm;rate=24000",
                            "data": tts_b64, "sample_rate": 24000
                        }))
                    continue

                if mtype == "pause":
                    logger.info(f"pause at {data.get('video_timestamp')}")
                    await websocket.send_text(json.dumps({"type": "session_saved", "summary": "Paused"}))
                    continue

                if mtype == "user_text":
                    user_text = (data.get("text") or "").strip()
                    if not user_text:
                        continue
                    # If client also sent language override with text
                    if data.get("language"):
                        language = data.get("language")
                    await websocket.send_text(json.dumps({
                        "type": "transcript", "role": "user", "text": user_text, "timestamp": 0
                    }))
                    await handle_user_question(
                        websocket, user_text, language, course_title, course_id, module_id, rag_context,
                        conversation_history,
                    )
                    continue

                # Optional: client signals end of utterance → flush buffer
                if mtype == "utterance_end":
                    if len(audio_buffer) > 800:
                        buf = bytes(audio_buffer)
                        audio_buffer.clear()
                        mime = last_audio_mime
                        await websocket.send_text(json.dumps({
                            "type": "transcript", "role": "user",
                            "text": "(voice received — transcribing with Sarvam...)", "timestamp": 0
                        }))
                        user_text, detected = await transcribe_audio(buf, mime=mime, language=language)
                        # Auto-switch to detected language (e.g., hi-IN when user spoke Hindi while selector was en)
                        if detected:
                            m = {"en-IN":"en","hi-IN":"hi","bn-IN":"bn","ta-IN":"ta","te-IN":"te","mr-IN":"mr","gu-IN":"gu","kn-IN":"kn","ml-IN":"ml","od-IN":"or","pa-IN":"pa"}.get(detected)
                            if m and m in LANG_FULL_NAME: language = m
                        if not user_text or len(user_text.strip()) < 2:
                            await websocket.send_text(json.dumps({
                                "type": "transcript", "role": "ai",
                                "text": "I didn't catch that. Could you try again?", "timestamp": 0
                            }))
                            tts_b64 = await synth_tts("I didn't catch that. Could you try again?", language)
                            if tts_b64:
                                await websocket.send_text(json.dumps({
                                    "type": "audio", "mime": "audio/pcm;rate=24000",
                                    "data": tts_b64, "sample_rate": 24000
                                }))
                            await websocket.send_text(json.dumps({"type": "ai_speaking_end"}))
                            continue
                        await websocket.send_text(json.dumps({
                            "type": "transcript", "role": "user",
                            "text": user_text, "timestamp": 0
                        }))
                        # Tell frontend the effective language so UI updates
                        await websocket.send_text(json.dumps({"type": "language_detected", "language": language, "detected": detected}))
                        await handle_user_question(
                            websocket, user_text, language, course_title, course_id, module_id, rag_context,
                            conversation_history,
                        )
                    else:
                        audio_buffer.clear()
                    continue

            # === BINARY MESSAGE (audio bytes from user) ===
            if "bytes" in msg and msg["bytes"]:
                audio_bytes = msg["bytes"]
                # Heuristic mime from browser; default webm
                mime = "audio/webm"
                last_audio_mime = mime
                # Buffer small chunks until utterance likely complete; Sarvam needs >30KB for good accuracy
                # If frontend sends 1s chunks (~20-40KB webm/opus), transcribe immediately; else buffer
                if len(audio_bytes) < 12000:
                    audio_buffer.extend(audio_bytes)
                    # Debounce: wait a bit for more chunks before transcribing; avoid per-250ms spam
                    # Transcribe when buffer exceeds ~30KB or after a short silence timeout is handled via utterance_end
                    if len(audio_buffer) < 28000:
                        # Send a lightweight ack so user sees activity, but don't STT yet
                        continue
                    # Buffer full enough -> flush
                    to_transcribe = bytes(audio_buffer)
                    audio_buffer.clear()
                else:
                    # Large chunk: transcribe directly (plus any buffered prior)
                    if len(audio_buffer) > 0:
                        to_transcribe = bytes(audio_buffer) + audio_bytes
                        audio_buffer.clear()
                    else:
                        to_transcribe = audio_bytes

                await websocket.send_text(json.dumps({
                    "type": "transcript", "role": "user",
                    "text": "(voice received — transcribing with Sarvam...)", "timestamp": 0
                }))

                user_text, detected = await transcribe_audio(to_transcribe, mime=mime, language=language)
                if detected:
                    m = {"en-IN":"en","hi-IN":"hi","bn-IN":"bn","ta-IN":"ta","te-IN":"te","mr-IN":"mr","gu-IN":"gu","kn-IN":"kn","ml-IN":"ml","od-IN":"or","pa-IN":"pa"}.get(detected)
                    if m and m in LANG_FULL_NAME: language = m
                if not user_text or len(user_text.strip()) < 2:
                    await websocket.send_text(json.dumps({
                        "type": "transcript", "role": "ai",
                        "text": "I didn't catch that. Could you try again?", "timestamp": 0
                    }))
                    tts_b64 = await synth_tts("I didn't catch that. Could you try again?", language)
                    if tts_b64:
                        await websocket.send_text(json.dumps({
                            "type": "audio", "mime": "audio/pcm;rate=24000",
                            "data": tts_b64, "sample_rate": 24000
                        }))
                    await websocket.send_text(json.dumps({"type": "ai_speaking_end"}))
                    continue

                await websocket.send_text(json.dumps({
                    "type": "transcript", "role": "user",
                    "text": user_text, "timestamp": 0
                }))
                if detected:
                    await websocket.send_text(json.dumps({"type": "language_detected", "language": language, "detected": detected}))
                await handle_user_question(
                    websocket, user_text, language, course_title, course_id, module_id, rag_context,
                    conversation_history,
                )

    except WebSocketDisconnect:
        logger.info("Live tutor WS disconnected")
    except Exception as e:
        logger.error(f"Live tutor error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def handle_user_question(websocket, user_text, language, course_title, course_id, module_id, rag_context, conversation_history):
    """Generate a tutor reply (text + TTS) for the user's question with per-question Chroma RAG."""
    lang_name = LANG_FULL_NAME.get(language, "English")
    # Per-question Chroma retrieval grounded to selected course
    chroma_q = _query_chroma(course_id, user_text, n_results=5)
    context_snippet = rag_context
    if chroma_q:
        context_snippet = f"{rag_context}\n\n--- Retrieved for this question (Chroma) ---\n{chroma_q}\n"
    context_snippet = context_snippet[:8000] if context_snippet else f"Course: {course_title or course_id}"

    # Build conversation context (last 4 turns)
    history_text = ""
    for h in conversation_history[-4:]:
        history_text += f"{h['role'].upper()}: {h['text']}\n"
    conversation_history.append({"role": "user", "text": user_text})

    prompt = (
        f"You are a live AI tutor for NSSTA/MoSPI government officials. "
        f"Course: {course_title or course_id} | Module: {module_id} | Language: {language}\n"
        f"Context (RAG grounded - use ONLY this when answering course questions, cite study material when relevant):\n{context_snippet}\n\n"
        f"Recent conversation:\n{history_text}\n"
        f"USER (in {lang_name}): {user_text}\n\n"
        f"Respond in {lang_name}. Be interactive: answer from the provided Context/study material first; if info missing say so. "
        f"Ask a follow-up, give a tiny practice exercise, keep under 5 sentences, mention the course title once. Interrupt-friendly."
    )

    txt = await chat_reply(prompt, language)
    if not txt:
        txt = f"Nice — let's explore {course_title or 'this topic'} in {lang_name}. What part would you like to start with?"
    conversation_history.append({"role": "assistant", "text": txt})

    await websocket.send_text(json.dumps({
        "type": "transcript", "role": "ai", "text": txt, "timestamp": 0
    }))

    # TTS the reply
    tts_b64 = await synth_tts(txt, language)
    if tts_b64:
        await websocket.send_text(json.dumps({
            "type": "audio", "mime": "audio/pcm;rate=24000",
            "data": tts_b64, "sample_rate": 24000
        }))
        logger.info(f"TTS sent: {len(tts_b64)} chars b64 for '{txt[:50]}...'")
    else:
        logger.warning(f"TTS returned None for: {txt[:80]}")

    await websocket.send_text(json.dumps({"type": "ai_speaking_end"}))
