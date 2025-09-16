# ars_api.py
import os, hmac, hashlib, json, datetime as dt
from fastapi import FastAPI, Header, HTTPException
from ars_engine import AdaptiveRevenueSequencer, TemplateArm

# Shared secret (override in env on Render)
DEFAULT_ARS_SECRET = "ars_secret_2c5d6a3b7a9f4d0c8e1f5a7b3c9d2e4f"
SECRET = os.getenv("ARS_SECRET", DEFAULT_ARS_SECRET).encode()

def verify(sig: str, body: bytes) -> bool:
    mac = hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    return bool(sig) and hmac.compare_digest(mac, sig)

app = FastAPI(title="Pulse ARS API")

ARMS = [TemplateArm("day0_email_day2_sms_day7_email")]
PRIORS = {"day0_email_day2_sms_day7_email": 0.1}
ARS = AdaptiveRevenueSequencer(ARMS, PRIORS)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ars/plan")
def plan(payload: dict, x_signature: str = Header(default=None)):
    body = json.dumps(payload).encode()
    if not verify(x_signature or "", body):
        raise HTTPException(status_code=401, detail="Invalid signature")
    cohort  = payload.get("cohort", "donut_shop")
    lead    = payload["lead"]
    context = payload["context"]

    context.setdefault("today", str(dt.date.today()))
    context.setdefault("hour_local", dt.datetime.now().hour)
    context.setdefault("weekend", dt.date.today().weekday() >= 5)
    context.setdefault("holiday_flag", False)

    weights, stats = {}, {}  # plug DB here later if you want
    seq, arm = ARS.plan(lead, context, cohort_weights=weights, arm_stats=stats)
    score = "Warm" if lead.get("channel_pref") in ("email","sms") else "Cold"
    return {"score": score, "reason": "Heuristic demo score.", "arm": arm, "steps": seq}
