"""
WebRTC Live Tutor Router - Full-duplex audio via Gemini Multimodal
Supports:
  - Audio input (webm/opus from browser) → STT via Gemini multimodal
  - Text input (typed) → direct LLM call
  - TTS response (gemini-2.5-flash-preview-tts, 24kHz PCM)
  - Barge-in (interrupt mid-speech)
  - 10 Indian languages
  - Course RAG context from Supabase
"""
import os, json, logging, base64, asyncio
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

TEXT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
TTS_MODELS = ["gemini-2.5-flash-preview-tts", "gemini-2.5-flash"]

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
        last_error = None
        for model_name in TTS_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
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
                break
            except Exception as exc:
                last_error = exc
                logger.warning(f"TTS model {model_name} failed: {exc}")
        else:
            if last_error:
                raise last_error
            return None
        # Extract audio from response.candidates[0].content.parts[*].inline_data.data
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
            logger.warning(f"TTS parse failed: {e}; trying text fallback")
        return None
    except Exception as e:
        logger.warning(f"Gemini TTS failed: {e}")
        return None

async def transcribe_audio(audio_bytes: bytes, mime: str = "audio/webm", language: str = "en") -> str:
    """Transcribe audio bytes using Gemini multimodal. Returns transcribed text."""
    if not audio_bytes or not os.getenv("GOOGLE_API_KEY"):
        return ""
    try:
        from google.genai import types
        client = _client()
        lang_name = LANG_FULL_NAME.get(language, "English")
        prompt = f"Listen to this audio and transcribe exactly what the user said. Reply with ONLY the transcribed text in {lang_name}, no commentary, no labels."
        # Build parts: text prompt + audio
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime),
        ]
        for model_name in TEXT_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[types.Content(role="user", parts=parts)],
                )
                return (response.text or "").strip()
            except Exception as exc:
                logger.warning(f"STT model {model_name} failed: {exc}")
        return ""
        return (response.text or "").strip()
    except Exception as e:
        logger.error(f"STT failed: {e}")
        return ""

async def chat_reply(prompt: str, language: str = "en") -> str:
    """Generate a tutor reply with RAG context."""
    if not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not configured."
    try:
        from google.genai import types
        client = _client()
        last_error = None
        for model_name in TEXT_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(),
                )
                return (response.text or "").strip()[:700]
            except Exception as exc:
                last_error = exc
                logger.warning(f"Tutor model {model_name} failed: {exc}")
        if last_error:
            raise last_error
        return ""
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return ""

@router.websocket("/ws/live-tutor")
async def live_tutor_ws(websocket: WebSocket, token: Optional[str] = None):
    await websocket.accept()
    logger.info(f"Live tutor WS connected (token={'yes' if token else 'no'})")

    course_id = "general"
    module_id = "live_tutor"
    language = "en"
    course_title = None
    rag_context = ""
    conversation_history = []  # for multi-turn context

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

        # 2) Fetch RAG context from Supabase
        try:
            from supabase import create_client
            supa = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
            if course_id and course_id != "general":
                c = supa.table("courses").select("title, description").eq("id", course_id).maybe_single().execute()
                if c and c.data:
                    course_title = course_title or c.data.get("title")
                    rag_context += f"Course: {c.data.get('title','')} - {(c.data.get('description') or '')[:600]}\n"
                m = supa.table("course_materials").select("title, content_text").eq("course_id", course_id).limit(2).execute()
                if m and m.data:
                    for mm in m.data:
                        txt = (mm.get("content_text") or "")[:700]
                        if txt: rag_context += f"\nMaterial: {mm.get('title','')}\n{txt}\n"
            if course_title:
                rag_context = f"Course: {course_title}\n" + rag_context
        except Exception as e:
            logger.warning(f"RAG fetch failed: {e}")

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
                    # Typed user question
                    user_text = (data.get("text") or "").strip()
                    if not user_text:
                        continue
                    await websocket.send_text(json.dumps({
                        "type": "transcript", "role": "user", "text": user_text, "timestamp": 0
                    }))
                    await handle_user_question(
                        websocket, user_text, language, course_title, course_id, module_id, rag_context,
                        conversation_history,
                    )
                    continue

            # === BINARY MESSAGE (audio bytes from user) ===
            if "bytes" in msg and msg["bytes"]:
                audio_bytes = msg["bytes"]
                # Determine mime from size/header (browser sends webm/opus)
                mime = "audio/webm"

                await websocket.send_text(json.dumps({
                    "type": "transcript", "role": "user",
                    "text": "(voice received — transcribing...)", "timestamp": 0
                }))

                # STT: transcribe user audio
                user_text = await transcribe_audio(audio_bytes, mime=mime, language=language)
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
    """Generate a tutor reply (text + TTS) for the user's question."""
    lang_name = LANG_FULL_NAME.get(language, "English")
    context_snippet = rag_context[:2000] if rag_context else f"Course: {course_title or course_id}"

    # Build conversation context (last 4 turns)
    history_text = ""
    for h in conversation_history[-4:]:
        history_text += f"{h['role'].upper()}: {h['text']}\n"
    conversation_history.append({"role": "user", "text": user_text})

    prompt = (
        f"You are a live AI tutor for NSSTA/MoSPI government officials. "
        f"Course: {course_title or course_id} | Module: {module_id} | Language: {language}\n"
        f"Context:\n{context_snippet}\n\n"
        f"Recent conversation:\n{history_text}\n"
        f"USER (in {lang_name}): {user_text}\n\n"
        f"Respond in {lang_name}. Be interactive: ask a follow-up, give a tiny practice exercise, "
        f"keep under 4 sentences, mention the course title once. Interrupt-friendly."
    )

    txt = await chat_reply(prompt, language)
    if not txt:
        error_msg = "The AI tutor is temporarily unavailable. Please check the AI service configuration or try again in a moment."
        await websocket.send_text(json.dumps({"type": "error", "message": error_msg, "timestamp": 0}))
        return
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
