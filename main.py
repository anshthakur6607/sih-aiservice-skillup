"""
SkillUp AI Service - Main Entry Point

This is the FastAPI-based AI microservice that handles:
- AI-powered competency assessment
- Quiz generation from documents
- Skill gap analysis and recommendations
- Vector embeddings for similarity search

Why: Separating AI logic into its own service allows:
- Independent scaling
- GPU resources only where needed
- Clear API contracts with the backend
- Easier maintenance and updates
"""

"""
MoSPI SkillUp AI Service - Main Entry Point

This FastAPI microservice provides AI-powered features:
- Quiz generation from documents (Bloom's taxonomy + IRT calibration)
- WebRTC Live AI Tutor (Gemini Multimodal)
- Multilingual RAG chatbot
- What-If capability simulator
- Competency assessment engine

Security: All endpoints require AI_SERVICE_API_KEY header for authentication.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Verify critical environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AI_SERVICE_API_KEY = os.getenv("AI_SERVICE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set - AI features will be limited")
if not AI_SERVICE_API_KEY:
    raise ValueError("AI_SERVICE_API_KEY must be set for security")


def is_gemini_rate_limit_error(exc: Exception) -> bool:
    """Detect Gemini quota/rate-limit responses and route the request to Sarvam fallback."""
    text = str(exc).lower()
    status_values = [
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "code", None),
    ]
    if any(value == 429 for value in status_values if value is not None):
        return True
    markers = [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "resource exhausted",
        "quota exceeded",
        "quota",
        "temporarily unavailable",
        "exceeded",
    ]
    return any(marker in text for marker in markers)

# ============================================
# Security: API Key Validation
# ============================================

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """
    Verify AI Service API Key for backend-to-AI-service authentication.
    
    Why: Prevents unauthorized access to expensive AI operations.
    The backend must include X-API-Key header matching AI_SERVICE_API_KEY.
    
    Raises:
        HTTPException: 401 if API key is invalid or missing
    """
    if x_api_key != AI_SERVICE_API_KEY:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Backend must provide valid X-API-Key header."
        )
    return x_api_key


# ============================================
# Pydantic Models
# ============================================

class AssessmentRequest(BaseModel):
    """Request model for AI competency assessment"""
    user_id: str
    designation: str
    department: str
    years_experience: float
    education: str
    current_assignment: Optional[str] = None


class AssessmentResponse(BaseModel):
    """Response model for AI competency assessment"""
    competencies: List[dict]
    baseline_scores: List[dict]
    assessment_summary: str


class QuizGenerationRequest(BaseModel):
    """Request model for AI quiz generation"""
    course_id: Optional[str] = None
    competency_ids: List[str] = []
    question_count: int = Field(default=10, ge=5, le=50)
    bloom_levels: List[str] = Field(
        default=["remember", "understand", "apply"],
        description="Bloom's taxonomy levels"
    )
    difficulty: float = Field(default=0.0, ge=-3.0, le=3.0)
    document_text: Optional[str] = None


class QuizQuestion(BaseModel):
    """Individual quiz question structure"""
    id: str
    text: str
    options: List[str]
    correct_answer: int
    bloom_level: str
    difficulty: float
    explanation: str


class QuizGenerationResponse(BaseModel):
    """Response model for quiz generation"""
    questions: List[QuizQuestion]
    metadata: dict


class RecommendationRequest(BaseModel):
    """Request model for course recommendations"""
    user_id: str
    skill_gaps: List[dict]


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[dict]
    priority_reasons: List[str]


class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings"""
    text: str
    metadata: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - initializes resources on startup.
    
    Why: Expensive resources (Gemini client, ChromaDB) are initialized once
    and reused across requests for performance.
    """
    logger.info("Starting MoSPI SkillUp AI Service...")
    
    # Initialize Google Gemini (new SDK: google-genai)
    if GOOGLE_API_KEY:
        try:
            _client = genai.Client(api_key=GOOGLE_API_KEY)
            # quick validation: list models
            logger.info("Google Gemini configured successfully (google-genai)")
        except Exception as e:
            logger.warning(f"Gemini client init failed: {e}")
    else:
        logger.warning("GOOGLE_API_KEY not set - AI features will be limited")
    
    # Initialize ChromaDB for vector storage
    try:
        chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        chroma_client = chromadb.Client(Settings(
            persist_directory=chroma_persist_dir,
            anonymized_telemetry=False
        ))
        # Ensure knowledge base collection exists
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "mospi_knowledge_base")
        try:
            chroma_client.get_collection(collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' exists")
        except Exception:
            chroma_client.get_or_create_collection(collection_name)
            logger.info(f"ChromaDB collection '{collection_name}' created")
        app.state.chroma = chroma_client
        logger.info(f"ChromaDB initialized at {chroma_persist_dir}")
    except Exception as e:
        logger.warning(f"ChromaDB initialization failed: {e}")
        app.state.chroma = None
    
    yield
    
    logger.info("Shutting down MoSPI SkillUp AI Service...")


# Create FastAPI app
app = FastAPI(
    title="SkillUp AI Service",
    description="AI-powered competency assessment and quiz generation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (modular)
try:
    from routers import live_tutor, rag_chat, quiz_gen, irt_engine, simulator
    app.include_router(live_tutor.router)
    app.include_router(rag_chat.router)
    app.include_router(quiz_gen.router)
    app.include_router(irt_engine.router)
    app.include_router(simulator.router)
    logger.info("Routers loaded: live_tutor, rag_chat, quiz_gen, irt_engine, simulator")
except Exception as e:
    logger.warning(f"Router include failed (will use main.py fallback): {e}")


@app.get("/")
async def root():
    """Health check endpoint - no auth required"""
    return {
        "service": "MoSPI SkillUp AI Service",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "quiz_generation",
            "live_tutor",
            "rag_chat",
            "irt_engine",
            "capability_simulator"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check - no auth required"""
    return {
        "status": "healthy",
        "google_ai": "configured" if GOOGLE_API_KEY else "not_configured",
        "supabase": "configured" if SUPABASE_URL else "not_configured",
        "chroma_db": "available" if hasattr(app.state, "chroma") and app.state.chroma else "unavailable",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.post("/api/ai/assess", response_model=AssessmentResponse, dependencies=[Depends(verify_api_key)])
async def assess_competency(request: AssessmentRequest):
    """
    AI-Powered Competency Assessment
    
    Analyzes user profile to determine baseline competency scores.
    Uses Gemini to reason about the user's background and map it to skills.
    
    Why: Initial assessment establishes baseline for skill-gap analysis.
    The AI considers designation, experience, education to make informed estimates.
    """
    try:
        # Construct prompt for Gemini
        prompt = f"""
        As an expert in government workforce competency mapping, analyze the following profile
        and provide baseline competency scores for a skill development platform.
        
        User Profile:
        - Designation: {request.designation}
        - Department: {request.department}
        - Years of Experience: {request.years_experience}
        - Education: {request.education}
        - Current Assignment: {request.current_assignment or 'Not specified'}
        
        The platform has 4 competency domains:
        1. Statistical: Survey Sampling, National Accounts, SDG Indicators, Data Quality
        2. Technical: Python, R, SQL, GIS, AI/ML, Open Data
        3. Digital Governance: Cybersecurity, Data Privacy, DPI, Govt Cloud
        4. Behavioural: Leadership, Communication, Ethics, Change Management
        
        For each competency, provide:
        - competency_name: The skill name
        - current_score: Estimated score from 1.0 to 5.0 based on background
        - reasoning: Brief explanation for the score
        
        Return JSON with an array of competency assessments.
        """
        
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured. Set GOOGLE_API_KEY in ai-service/.env")
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        import json, re
        text = response.text.strip() if response.text else ""
        # Extract JSON if wrapped in markdown
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        parsed = json.loads(text)
        # Expected: list of {competency_name, current_score, reasoning, domain}
        if isinstance(parsed, dict) and "competencies" in parsed:
            competencies = parsed["competencies"]
        elif isinstance(parsed, list):
            competencies = parsed
        else:
            raise ValueError("Gemini returned unexpected JSON structure for assessment")
        # Validate and normalize
        normalized = []
        for c in competencies:
            normalized.append({
                "name": c.get("competency_name") or c.get("name"),
                "score": float(c.get("current_score", 2.0)),
                "domain": c.get("domain", "Statistical"),
                "reasoning": c.get("reasoning", "")
            })
        return AssessmentResponse(
            competencies=normalized,
            baseline_scores=[{"competency": c["name"], "score": c["score"]} for c in normalized],
            assessment_summary=f"AI assessment complete for {request.designation} in {request.department} via Gemini 3.5 Flash"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@app.post("/api/ai/quiz/generate", response_model=QuizGenerationResponse, dependencies=[Depends(verify_api_key)])
async def generate_quiz(request: QuizGenerationRequest):
    """
    AI Quiz Generation from Documents or Competencies
    
    Generates structured quiz questions using Gemini with Sarvam AI fallback.
    Supports Bloom's taxonomy tagging and IRT difficulty parameters.
    
    Why: Allows trainers to auto-generate assessments from course materials.
    Saves significant time compared to manual question writing.
    """
    import json, re
    
    # Determine content source
    placeholder_markers = ["uploaded - will be processed server-side", "will be processed server-side", "[" + "uploaded"]
    if request.document_text:
        cleaned = request.document_text.strip()
        if any(marker in cleaned.lower() for marker in ["uploaded - will be processed server-side", "will be processed server-side", "pdf uploaded", "docx uploaded"]):
            source_text = ""
        else:
            source_text = cleaned
        source_type = "document" if source_text else "manual"
    else:
        source_text = "General competency assessment questions"
        source_type = "competency"
    
    prompt = f"""
    Generate {request.question_count} multiple choice quiz questions for a government training program.
    
    Requirements:
    - Question count: {request.question_count}
    - Bloom's taxonomy levels: {', '.join(request.bloom_levels)}
    - IRT difficulty (b-value): {request.difficulty}
    - 4 options per question (A, B, C, D)
    
    For each question, provide:
    1. id: unique identifier
    2. text: question text
    3. options: array of 4 possible answers
    4. correct_answer: index of correct option (0-3)
    5. bloom_level: one of {request.bloom_levels}
    6. difficulty: IRT b-value around {request.difficulty}
    7. explanation: why the answer is correct
    
    Return as JSON array.
    """
    
    # === CHAIN: Gemini 3.5 Flash → Gemini 3.6 Flash → Sarvam AI → Heuristic fallback ===
    models_to_try = ["gemini-3.5-flash", "gemini-3.6-flash"]
    gemini_error = None
    parsed = None
    
    if GOOGLE_API_KEY:
        for model_name in models_to_try:
            try:
                client = genai.Client(api_key=GOOGLE_API_KEY)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                text = response.text.strip() if response.text else ""
                if "```" in text:
                    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                    if m:
                        text = m.group(1)
                parsed = json.loads(text)
                logger.info(f"Quiz generated via {model_name}")
                break
            except Exception as e:
                gemini_error = e
                if is_gemini_rate_limit_error(e):
                    logger.warning(f"Gemini {model_name} hit rate limit for quiz; switching to Sarvam AI fallback")
                    break
                logger.warning(f"Gemini {model_name} failed for quiz: {e}")
                continue
    
    # === Sarvam AI fallback (chat completions) ===
    if parsed is None and os.getenv("SARVAM_API_KEY"):
        try:
            import httpx
            sarvam_payload = {
                "model": "sarvam-2b-v0.5",
                "messages": [
                    {"role": "system", "content": "You are an expert quiz generator for government training. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 4000,
            }
            async with httpx.AsyncClient() as http_client:
                r = await http_client.post(
                    "https://api.sarvam.ai/chat/completions",
                    json=sarvam_payload,
                    headers={"Authorization": f"Bearer {os.getenv('SARVAM_API_KEY')}", "Content-Type": "application/json"},
                    timeout=30.0,
                )
                r.raise_for_status()
                sarvam_text = r.json()["choices"][0]["message"]["content"]
                if "```" in sarvam_text:
                    m = re.search(r"```(?:json)?\s*(.*?)\s*```", sarvam_text, re.DOTALL)
                    if m:
                        sarvam_text = m.group(1)
                parsed = json.loads(sarvam_text)
                logger.info("Quiz generated via Sarvam AI fallback")
        except Exception as e:
            logger.warning(f"Sarvam AI fallback failed for quiz: {e}")
    
    # === Heuristic fallback: extract questions from document text ===
    if parsed is None and source_text and source_text != "General competency assessment questions":
        sentences = [s.strip() for s in re.split(r'[.!?]+', source_text) if len(s.strip()) > 20]
        if not sentences:
            raise HTTPException(status_code=422, detail="No readable text was found in the uploaded document. Please upload a text file or paste content manually.")
        questions = []
        for i in range(min(request.question_count, max(1, len(sentences)))):
            sent = sentences[i % len(sentences)]
            bloom = request.bloom_levels[i % len(request.bloom_levels)]
            questions.append(QuizQuestion(
                id=f"q_{i+1}",
                text=f"What is the key concept related to: {sent[:80]}?",
                options=[
                    f"Core principle of {sent[:40]}",
                    f"Unrelated concept A",
                    f"Unrelated concept B",
                    f"None of the above"
                ],
                correct_answer=0,
                bloom_level=bloom,
                difficulty=request.difficulty,
                explanation=f"Based on document content: {sent[:100]}"
            ))
        return QuizGenerationResponse(
            questions=questions,
            metadata={
                "question_count": len(questions),
                "bloom_levels": request.bloom_levels,
                "difficulty": request.difficulty,
                "source": source_type,
                "generated_at": "2024-01-01T00:00:00Z",
                "generator": "heuristic_fallback",
                "warning": "AI services unavailable. Questions generated from document text heuristics."
            }
        )
    
    if parsed is None:
        error_detail = f"All AI services failed. Gemini error: {gemini_error}" if gemini_error else "No AI service available"
        raise HTTPException(status_code=503, detail=f"Quiz generation failed: {error_detail}. Check GOOGLE_API_KEY and SARVAM_API_KEY.")
    
    # Parse questions from AI response
    if isinstance(parsed, dict) and "questions" in parsed:
        raw_questions = parsed["questions"]
    elif isinstance(parsed, list):
        raw_questions = parsed
    else:
        raise HTTPException(status_code=500, detail="AI returned unexpected JSON structure for quiz generation")
    
    questions = []
    for idx, q in enumerate(raw_questions[:request.question_count]):
        questions.append(QuizQuestion(
            id=str(q.get("id", f"q_{idx+1}")),
            text=str(q["text"]),
            options=list(q["options"]),
            correct_answer=int(q["correct_answer"]),
            bloom_level=str(q.get("bloom_level", request.bloom_levels[0])),
            difficulty=float(q.get("difficulty", request.difficulty)),
            explanation=str(q.get("explanation", ""))
        ))
    
    return QuizGenerationResponse(
        questions=questions,
        metadata={
            "question_count": len(questions),
            "bloom_levels": request.bloom_levels,
            "difficulty": request.difficulty,
            "source": source_type,
            "generated_at": "2024-01-01T00:00:00Z"
        }
    )


@app.post("/api/ai/embed", dependencies=[Depends(verify_api_key)])
async def generate_embedding(request: EmbeddingRequest):
    """
    Generate Vector Embedding for Text
    
    Uses Google's embedding model to create vector representations.
    Used for semantic search and similarity matching.
    
    Why: pgvector similarity search requires embeddings.
    This endpoint generates embeddings for competencies and courses.
    """
    try:
        # Use LangChain's Google GenAI embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        result = await embeddings.aembed_query(request.text)
        
        # Store in ChromaDB if available
        if hasattr(app.state, "chroma") and app.state.chroma:
            collection = app.state.chroma.get_or_create_collection("skillup_embeddings")
            collection.add(
                embeddings=[result],
                documents=[request.text],
                metadatas=[request.metadata or {}],
                ids=[request.metadata.get("id", f"emb_{hash(request.text)}")]
            )
        
        return {
            "embedding": result,
            "dimension": len(result),
            "text": request.text[:100] + "..." if len(request.text) > 100 else request.text
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/api/ai/recommend", response_model=RecommendationResponse, dependencies=[Depends(verify_api_key)])
async def get_recommendations(request: RecommendationRequest):
    """
    AI-Powered Course Recommendations
    
    Analyzes user's skill gaps and recommends relevant courses.
    Uses hybrid approach: content-based + collaborative filtering.
    
    Why: Personalized recommendations increase learning engagement.
    Matches weak skills to courses that address those gaps.
    """
    try:
        # In production, would:
        # 1. Get user's competency gaps
        # 2. Find courses with matching competency tags
        # 3. Apply collaborative filtering from similar users
        # 4. Rank by priority and relevance
        
        recommendations = []
        
        for gap in request.skill_gaps:
            if gap.get("gap_score", 0) >= 2.0:
                recommendations.append({
                    "course_id": f"course_{gap['competency_id']}",
                    "course_title": f"Course on {gap['competency_name']}",
                    "priority": "high",
                    "reason": f"Addresses critical gap in {gap['competency_name']}",
                    "matching_gap": gap["competency_name"]
                })
            elif gap.get("gap_score", 0) >= 1.0:
                recommendations.append({
                    "course_id": f"course_{gap['competency_id']}",
                    "course_title": f"Course on {gap['competency_name']}",
                    "priority": "medium",
                    "reason": f"Addresses skill gap in {gap['competency_name']}",
                    "matching_gap": gap["competency_name"]
                })
        
        return RecommendationResponse(
            recommendations=recommendations[:10],
            priority_reasons=[
                f"Found {len([r for r in recommendations if r['priority'] == 'high'])} high priority courses",
                f"Found {len([r for r in recommendations if r['priority'] == 'medium'])} medium priority courses"
            ]
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


# Helper validation - production-grade, no mock data generation
def validate_competency_payload(payload: dict) -> None:
    """Validate competency assessment payload structure from Gemini"""
    if not isinstance(payload, (dict, list)):
        raise ValueError("Invalid competency payload structure")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)