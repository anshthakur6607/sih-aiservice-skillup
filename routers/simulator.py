"""
What-If Capability Simulator - Gemini 1.5 Pro forecasting
"""
import os, json, logging
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
logger = logging.getLogger(__name__)
router = APIRouter()
AI_KEY = os.getenv("AI_SERVICE_API_KEY")
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != AI_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class SimulateRequest(BaseModel):
    scenario: str
    snapshot: Optional[List[Dict]] = None

@router.post("/api/ai/simulate", dependencies=[Depends(verify_api_key)])
async def simulate(req: SimulateRequest):
    snapshot = req.snapshot or [{"department":"NSSO","domain":"Statistical","avg_score":2.8,"count":45}]
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured. Set GOOGLE_API_KEY in ai-service/.env to enable What-If simulation")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    model_name = "gemini-3.5-flash"
    prompt = f"""
You are MoSPI workforce planner. Current snapshot: {json.dumps(snapshot)}
Scenario: "{req.scenario}"
Task: predict averages after training. Increase relevant competency by +0.6 for affected dept.
Return JSON: {{"predicted_averages":{{"Statistical":3.4}}, "improvement":{{"Statistical":0.6}}, "affected_users": 20, "reasoning":"..."}}
"""
    resp = client.models.generate_content(model=model_name, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
    data = json.loads(resp.text)
    return {"success": True, "data": data}
