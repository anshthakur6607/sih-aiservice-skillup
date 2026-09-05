"""
Multilingual RAG Chat Router - ChromaDB + Gemini + Sarvam/Bhashini fallback
"""
import os, logging
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
logger = logging.getLogger(__name__)
router = APIRouter()

AI_KEY = os.getenv("AI_SERVICE_API_KEY")


def is_gemini_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    status_values = [
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "code", None),
    ]
    if any(value == 429 for value in status_values if value is not None):
        return True
    return any(marker in text for marker in [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "resource exhausted",
        "quota exceeded",
        "quota",
        "exceeded",
    ])


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != AI_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class ChatRequest(BaseModel):
    message: str
    course_id: Optional[str] = None
    user_id: str
    voice_mode: bool = False

class ChatResponse(BaseModel):
    answer: str
    language: str
    sources: List[dict]
    audio_url: Optional[str] = None

def detect_language(text: str) -> str:
    # naive detect: if devanagari present -> hi
    if any('\u0900' <= ch <= '\u097F' for ch in text):
        return "hi"
    if any('\u0B80' <= ch <= '\u0BFF' for ch in text):
        return "ta"
    if any('\u0B00' <= ch <= '\u0B0F' for ch in text):
        return "or"
    if any('\u0C00' <= ch <= '\u0C7F' for ch in text):
        return "te"
    if any('\u0D00' <= ch <= '\u0D7F' for ch in text):
        return "ml"
    return "en"

@router.post("/api/ai/chat/multilingual", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def multilingual_chat(req: ChatRequest):
    lang = detect_language(req.message)
    # Try course-specific context if course_id provided
    context = ""
    if req.course_id and req.course_id != "demo" and req.course_id != "general":
        try:
            from supabase import create_client
            supa = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
            # Fetch course details
            course_res = supa.table("courses").select("title, description, provider").eq("id", req.course_id).maybe_single().execute()
            course = course_res.data if course_res else None
            if course:
                context += f"Course: {course.get('title','')} - {course.get('description','')[:500]}\n"
            # Fetch up to 3 materials' extracted text
            mat_res = supa.table("course_materials").select("title, content_text, type, url").eq("course_id", req.course_id).limit(3).execute()
            if mat_res and mat_res.data:
                for m in mat_res.data[:2]:
                    txt = (m.get("content_text") or "")[:800]
                    if txt:
                        context += f"\nMaterial: {m.get('title','')} ({m.get('type','')})\n{txt}\n"
            if context:
                logger.info(f"Loaded course context for {req.course_id}: {len(context)} chars")
        except Exception as e:
            logger.warning(f"Course context fetch failed: {e}")

    # Fallback to ChromaDB retrieval if no course context
    if not context:
        context = "MoSPI statistical manuals context"
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.Client(Settings(persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY","./chroma_data"), anonymized_telemetry=False))
            col = client.get_collection("mospi_knowledge_base")
            # Embed query would go here; fallback to keyword
            res = col.query(query_texts=[req.message], n_results=2)
            if res and res.get("documents"):
                context = "\n".join(res["documents"][0][:2])
        except Exception as e:
            logger.warning(f"Chroma retrieval fallback: {e}")

    # Gemini generation (google-genai) with model fallback chain
    answer_en = f"Based on course material: {context[:300]}... Answer to '{req.message}'"
    gemini_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    gemini_ok = False
    for model_name in gemini_models:
        try:
            from google import genai
            from google.genai import types
            if os.getenv("GOOGLE_API_KEY"):
                client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                prompt = f"User question: {req.message}\nContext: {context}\nAnswer concisely, preserve technical terms:"
                resp = client.models.generate_content(model=model_name, contents=prompt, config=types.GenerateContentConfig())
                answer_en = (resp.text or "")[:800]
                gemini_ok = True
                break
        except Exception as e:
            if is_gemini_rate_limit_error(e):
                logger.warning(f"Gemini {model_name} hit rate limit; switching to Sarvam AI fallback")
                break
            logger.warning(f"Gemini {model_name} fallback: {e}")
            continue
    
    # Sarvam AI fallback for chat when all Gemini models fail
    if not gemini_ok and os.getenv("SARVAM_API_KEY"):
        try:
            import httpx
            sarvam_payload = {
                "model": "sarvam-2b-v0.5",
                "messages": [
                    {"role": "system", "content": "You are a helpful government training tutor. Answer concisely."},
                    {"role": "user", "content": f"Context: {context[:500]}\nQuestion: {req.message}"}
                ],
                "temperature": 0.7,
                "max_tokens": 800,
            }
            async with httpx.AsyncClient() as http_client:
                r = await http_client.post(
                    "https://api.sarvam.ai/chat/completions",
                    json=sarvam_payload,
                    headers={"Authorization": f"Bearer {os.getenv('SARVAM_API_KEY')}", "Content-Type": "application/json"},
                    timeout=15.0,
                )
                r.raise_for_status()
                answer_en = r.json()["choices"][0]["message"]["content"][:800]
                logger.info("Chat response generated via Sarvam AI fallback")
        except Exception as e:
            logger.warning(f"Sarvam AI chat fallback failed: {e}")

    # Translate if needed via Sarvam AI / Bhashini
    answer = answer_en
    if lang != "en":
        try:
            from services.sarvam_bhashini import translate_text
            answer = await translate_text(answer_en, source_lang="en", target_lang=lang)
        except Exception as e:
            logger.warning(f"Translation failed for {lang}: {e}")
            # Fallback: return English with language tag if translation service unavailable
            if not os.getenv("SARVAM_API_KEY") and not os.getenv("BHASHINI_API_KEY"):
                raise HTTPException(status_code=503, detail=f"Translation to {lang} unavailable. Configure SARVAM_API_KEY or BHASHINI_API_KEY")

    return ChatResponse(answer=answer, language=lang, sources=[{"course_id": req.course_id or "demo", "chunk_index": 0, "preview": context[:150]}])

@router.post("/api/ai/rag/ingest", dependencies=[Depends(verify_api_key)])
async def rag_ingest(payload: dict):
    # Simple ingest endpoint for testing
    course_id = payload.get("course_id", "demo")
    content = payload.get("content", "")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(content)
        import chromadb
        from chromadb.config import Settings
        client = chromadb.Client(Settings(persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY","./chroma_data"), anonymized_telemetry=False))
        col = client.get_or_create_collection("mospi_knowledge_base")
        ids = [f"{course_id}_{i}" for i in range(len(chunks))]
        col.add(ids=ids, documents=chunks, metadatas=[{"course_id": course_id} for _ in chunks])
        return {"success": True, "chunks_stored": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
