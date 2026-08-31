"""
Quiz Generation Router - S3 ingestion + Bloom taxonomy + difficulty tagging
"""
import os, logging, json, re, io
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
logger = logging.getLogger(__name__)
router = APIRouter()

AI_KEY = os.getenv("AI_SERVICE_API_KEY")
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != AI_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class IngestRequest(BaseModel):
    s3_key: str
    course_id: str
    question_count: int = 20
    bloom_levels: List[str] = ["understand","apply","analyze"]
    language: str = "en"

@router.post("/api/ai/quiz/ingest-from-s3", dependencies=[Depends(verify_api_key)])
async def ingest_from_s3(req: IngestRequest):
    # Download from S3 - requires AWS credentials; if not configured, try Supabase Storage fallback
    text_content = ""
    s3_configured = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    if s3_configured:
        import boto3
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION","ap-south-1"),
                           aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                           aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
        obj = s3.get_object(Bucket=os.getenv("AWS_S3_BUCKET","mospi-course-materials"), Key=req.s3_key)
        data = obj["Body"].read()
        if req.s3_key.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            text_content = "\n".join([p.extract_text() or "" for p in reader.pages])
        elif req.s3_key.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(data))
            text_content = "\n".join([p.text for p in doc.paragraphs])
        else:
            text_content = data.decode("utf-8", errors="ignore")
        if not text_content.strip():
            raise HTTPException(status_code=422, detail="S3 file empty or unreadable")
    else:
        # Supabase Storage fallback - fetch via public URL if s3_key is Supabase path
        try:
            from supabase import create_client
            supa = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
            # Attempt to download from Supabase storage bucket "course-materials"
            bucket = supa.storage.from_("course-materials")
            data = bucket.download(req.s3_key)
            text_content = data.decode("utf-8", errors="ignore") if isinstance(data, (bytes, bytearray)) else str(data)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"S3 not configured and Supabase Storage fallback failed: {str(e)}. Set AWS_ACCESS_KEY_ID/SECRET or upload via Supabase Storage")

    # Generate via Gemini - requires GOOGLE_API_KEY (google-genai SDK)
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured. Set GOOGLE_API_KEY in ai-service/.env to generate quizzes")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_name = "gemini-3.5-flash"
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text_content)[:5]
    questions = []
    for chunk in chunks:
        prompt = f"Generate {max(1, req.question_count//len(chunks))} MCQs from:\n{chunk[:1500]}\nBloom: {req.bloom_levels}. Return JSON array with text,options(4),correct_answer(0-3),bloom_level,difficulty_estimate in [-3,3],explanation. Output pure JSON."
        resp = client.models.generate_content(model=model_name, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
        txt = resp.text
        if "```" in txt:
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL)
            if m:
                txt = m.group(1)
        parsed = json.loads(txt)
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed["questions"]
        if isinstance(parsed, list):
            questions.extend(parsed)
        if len(questions) >= req.question_count:
            break
    if not questions:
        raise HTTPException(status_code=500, detail="Gemini returned no questions. Try with different Bloom levels or check API quota")

    # Store in Supabase if configured
    try:
        from supabase import create_client
        if os.getenv("SUPABASE_URL"):
            supa = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
            for q in questions[:req.question_count]:
                supa.table("questions").insert({
                    "course_id": req.course_id,
                    "text": q["text"],
                    "options": q["options"],
                    "correct_answer": q["correct_answer"],
                    "bloom_level": q["bloom_level"],
                    "difficulty_beta": q.get("difficulty_estimate",0),
                    "explanation": q.get("explanation",""),
                    "language": req.language,
                    "source_s3_key": req.s3_key
                }).execute()
    except Exception as e:
        logger.warning(f"Supabase store fallback: {e}")

    return {"success": True, "questions_generated": len(questions[:req.question_count]), "question_ids": [f"q-{i}" for i in range(len(questions[:req.question_count]))]}
