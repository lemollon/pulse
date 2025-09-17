from fastapi import FastAPI, Request
from pydantic import BaseModel
import hmac, hashlib, json
import os

app = FastAPI()

# ---- Secrets ----
ARS_SECRET = os.getenv("ARS_SECRET", "ars_secret_2c5d6a3b7a9f4d0c8e1f5a7b3c9d2e4f").encode()

# ---- Models ----
class Lead(BaseModel):
    name: str
    contact: str
    channel_pref: str = "email"
    notes: str = ""
    last_interaction: str = None

class Context(BaseModel):
    today: str
    hour_local: int
    weekend: bool
    holiday_flag: bool = False
    avg_sentiment_7d: float = 0.0
    complaint_wait_time: float = 0.0
    recency_days: int = 0
    prior_reply_rate: float = 0.0

class PlanRequest(BaseModel):
    cohort: str
    lead: dict
    context: dict

# ---- Helpers ----
def verify_signature(signature: str, body: bytes) -> bool:
    expected = hmac.new(ARS_SECRET, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

# ---- Routes ----
@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "ARS", "version": "0.1.0"}

@app.post("/ars/plan")
async def ars_plan(request: Request):
    # Validate signature
    signature = request.headers.get("x-signature", "")
    body = await request.body()
    if not verify_signature(signature, body):
        return {"error": "Invalid signature"}

    data = json.loads(body)

    lead = data.get("lead", {})
    context = data.get("context", {})

    # ---- Dummy logic: generate a simple 3-step plan ----
    plan = [
        {
            "send_dt": context.get("today"),
            "channel": "email",
            "subject": "Sweet to Connect!",
            "body": f"Hi {lead.get('name','friend')}, great to connect! 🍩 Let’s keep the conversation going."
        },
        {
            "send_dt": "in 2 days",
            "channel": "sms",
            "body": f"Hey {lead.get('name','friend')}! Just checking in to see if you had any questions."
        },
        {
            "send_dt": "in 1 week",
            "channel": "email",
            "subject": "Let’s Take the Next Step!",
            "body": f"Hi {lead.get('name','friend')}, following up on our chat. Excited to bring sweetness to your day!"
        }
    ]

    return {"arm": "default", "score": 0.8, "steps": plan}
