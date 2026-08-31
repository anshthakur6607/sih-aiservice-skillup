"""
Sarvam AI / Bhashini translation wrapper - production implementation
Fallback chain: Sarvam -> Bhashini -> error (no mock)
"""
import os, logging, httpx
logger = logging.getLogger(__name__)

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    if os.getenv("SARVAM_API_KEY"):
        try:
            return await translate_with_sarvam(text, source_lang, target_lang)
        except Exception as e:
            logger.warning(f"Sarvam failed: {e}")
    if os.getenv("BHASHINI_API_KEY"):
        try:
            return await translate_with_bhashini(text, source_lang, target_lang)
        except Exception as e:
            logger.warning(f"Bhashini failed: {e}")
    raise RuntimeError("No translation service configured. Set SARVAM_API_KEY or BHASHINI_API_KEY")

async def translate_with_sarvam(text: str, source: str, target: str) -> str:
    url = "https://api.sarvam.ai/translate"
    headers = {"Authorization": f"Bearer {os.getenv('SARVAM_API_KEY')}", "Content-Type": "application/json"}
    payload = {"input": text, "source_language_code": source, "target_language_code": target, "speaker_gender": "female", "mode": "formal", "model": "mayura:v1", "enable_preprocessing": True}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers, timeout=10.0)
        r.raise_for_status()
        return r.json()["translated_text"]

async def translate_with_bhashini(text: str, source: str, target: str) -> str:
    url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    headers = {"Authorization": f"Bearer {os.getenv('BHASHINI_API_KEY')}", "Content-Type": "application/json"}
    payload = {"pipelineTasks": [{"taskType": "translation", "config": {"language": {"sourceLanguage": source, "targetLanguage": target}}}], "inputData": {"input": [{"source": text}]}}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=payload, headers=headers, timeout=10.0)
        r.raise_for_status()
        return r.json()["pipelineResponse"][0]["output"][0]["target"]
