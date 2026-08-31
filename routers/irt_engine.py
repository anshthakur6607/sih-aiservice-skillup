"""
IRT Engine Router - 1PL Rasch Model scoring
P = exp(theta - beta) / (1 + exp(theta - beta))
"""
import math
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os, logging
logger = logging.getLogger(__name__)
router = APIRouter()
AI_KEY = os.getenv("AI_SERVICE_API_KEY")
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != AI_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class IRTRequest(BaseModel):
    responses: List[dict]  # {question_id, is_correct, beta}
    # is_correct bool, beta float

class IRTResponse(BaseModel):
    theta: float
    standard_error: float
    percentile: int
    raw_score: int
    total: int

def normal_cdf(x: float) -> float:
    t = 1/(1+0.2316419*abs(x))
    d = 0.3989423*math.exp(-x*x/2)
    prob = d*t*(0.3193815 + t*(-0.3565638 + t*(1.781478 + t*(-1.821256 + t*1.330274))))
    return 1-prob if x>0 else prob

@router.post("/api/ai/irt/score", response_model=IRTResponse, dependencies=[Depends(verify_api_key)])
async def irt_score(req: IRTRequest):
    n = len(req.responses)
    if n == 0:
        raise HTTPException(status_code=400, detail="No responses")
    raw = sum(1 for r in req.responses if r.get("is_correct"))
    if raw == 0:
        return IRTResponse(theta=-3.0, standard_error=1.5, percentile=1, raw_score=0, total=n)
    if raw == n:
        return IRTResponse(theta=3.0, standard_error=1.5, percentile=99, raw_score=n, total=n)
    theta = 0.0
    for _ in range(20):
        fd = 0.0
        sd = 0.0
        for r in req.responses:
            beta = float(r.get("beta", 0))
            is_correct = 1 if r.get("is_correct") else 0
            exp_term = math.exp(theta - beta)
            p = exp_term/(1+exp_term)
            fd += is_correct - p
            sd -= p*(1-p)
        if sd == 0:
            break
        delta = -fd/sd
        theta += delta
        if abs(delta) < 0.01:
            break
    info = 0.0
    for r in req.responses:
        beta = float(r.get("beta",0))
        p = math.exp(theta - beta)/(1+math.exp(theta - beta))
        info += p*(1-p)
    se = 1/math.sqrt(info) if info>0 else 1.5
    perc = int(normal_cdf(theta)*100)
    return IRTResponse(theta=round(theta,2), standard_error=round(se,2), percentile=perc, raw_score=raw, total=n)
