# app.py — Pulse v1.8.1 (all features + price-level fix + polished UX)
# - Google Places (New) competitor search w/ Nearby fallback
# - Folium map + heat glow
# - KPIs, Top-10 bar, Rating-vs-Reviews scatter
# - AI Market Summary, Opportunity Finder + AI Actions, Owner Playbook
# - Insights export (Markdown), Polished follow-ups export (TXT/DOCX)
# - "Prepare Follow-Up Engine" button (non-jargony ARS warmup)
# - ARS client w/ retries + local fallback
# - Theme toggle (forced), safe rerun, friendly cards
# - Price level formatted ($/$$/…); no huge metric text

import os, io, re, json, hmac, hashlib, textwrap, time
import datetime as dt
from typing import Dict, List
from urllib.parse import urlparse, urlunparse

import requests
import pandas as pd
import altair as alt
import streamlit as st

# ---------- Optional DOCX export ----------
DOCX_AVAILABLE = True
try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except Exception:
    DOCX_AVAILABLE = False

# ---------- Maps ----------
import folium
from streamlit_folium import st_folium

# ---------- OpenAI (optional, for AI summaries) ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------------- Page / Meta ----------------------
VERSION = "1.8.1"
PRIMARY = "#6B5BFF"
ACCENT = "#00C29A"

st.set_page_config(page_title="Pulse — Win Your Neighborhood", page_icon="🧭", layout="wide")

# ---------------------- Safe rerun helper ----------------------
def safe_rerun():
    """Works on both new & old Streamlit versions."""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---------------------- Secrets ----------------------
def _secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

GOOGLE_PLACES_API_KEY = _secret("GOOGLE_PLACES_API_KEY", "")
OPENAI_API_KEY        = _secret("OPENAI_API_KEY", "")
ARS_URL               = _secret("ARS_URL", "http://localhost:8080/ars/plan")
ARS_SECRET_RAW        = _secret("ARS_SECRET", "ars_secret_2c5d6a3b7a9f4d0c8e1f5a7b3c9d2e4f")
ARS_SECRET            = ARS_SECRET_RAW.encode() if isinstance(ARS_SECRET_RAW, str) else ARS_SECRET_RAW

# ---------------------- Theme Toggle (forced) ----------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default = dark (best contrast)

with st.sidebar:
    st.markdown("### 🎚️ Theme")
    choice = st.radio("Appearance", ["Dark", "Light"],
                      index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.theme = "dark" if choice.lower() == "dark" else "light"

THEME_DARK = st.session_state.theme == "dark"

def inject_css(theme_dark: bool):
    if theme_dark:
        card_bg, card_text, card_border = "#0b1220", "#e5e7eb", "#1f2937"
        btn_download_text = "#0b1220"
    else:
        card_bg, card_text, card_border = "#f8fafc", "#0f172a", "#e5e7eb"
        btn_download_text = "#0b1220"

    st.markdown(f"""
    <style>
    :root {{
      --primary:{PRIMARY};
      --accent:{ACCENT};
      --card-bg:{card_bg};
      --card-text:{card_text};
      --card-border:{card_border};
    }}
    .stButton>button {{
      background: var(--primary) !important; color:white !important;
      border-radius:10px; border:none;
    }}
    .stDownloadButton>button {{
      background: var(--accent) !important; color:{btn_download_text} !important;
      border-radius:10px; border:none; font-weight:700;
    }}
    .kpi-card {{
      background:var(--card-bg); color:var(--card-text);
      border:1px solid var(--card-border);
      border-radius:12px; padding:12px 16px;
    }}
    .kpi-label {{ font-size:.85rem; opacity:.85; }}
    .kpi-value {{ font-weight:800; font-size:1.4rem; margin-top:2px; }}
    .section-title {{ font-weight:700; font-size:1.05rem; margin:1rem 0 .25rem; }}
    .small-note {{ color:#6b7280; font-size:.9rem; }}
    </style>
    """, unsafe_allow_html=True)

inject_css(THEME_DARK)

# ---------------------- UI helpers (beautiful, non-jargony) ----------------------
def info_card(title: str, body_md: str):
    st.markdown(f"""
    <div style="
      border:1px solid var(--card-border);
      background:var(--card-bg);
      color:var(--card-text);
      border-radius:12px;
      padding:14px 16px;
      margin:6px 0;">
      <div style="font-weight:700;margin-bottom:6px">{title}</div>
      <div style="opacity:.95">{body_md}</div>
    </div>
    """, unsafe_allow_html=True)

def tiny_badge(text: str, color="#10b981"):
    st.markdown(
        f"<span style='padding:.15rem .45rem;border:1px solid {color};"
        f"border-radius:999px;color:{color};font-size:.85rem;margin-right:.35rem;'>"
        f"{text}</span>", unsafe_allow_html=True
    )

# --- Price level formatting (FIX) ---
PRICE_MAP = {
    "PRICE_LEVEL_FREE": "Free ($0)",
    "PRICE_LEVEL_INEXPENSIVE": "$ (Inexpensive)",
    "PRICE_LEVEL_MODERATE": "$$ (Moderate)",
    "PRICE_LEVEL_EXPENSIVE": "$$$ (Expensive)",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$ (Very expensive)",
}
def pretty_price(level) -> str:
    if not level:
        return "—"
    s = str(level).upper()
    if s in PRICE_MAP:
        return PRICE_MAP[s]
    return s.replace("PRICE_LEVEL_", "").replace("_", " ").title()

# ---------------------- Hero Title + Feature Badges ----------------------
APP_TITLE = "Pulse — Competitor Insights + Follow-Up Plans"
st.markdown(f"""
<h1 style="margin-bottom:0">{APP_TITLE}</h1>
<p style="margin-top:4px;opacity:.9">
  <span style="padding:.2rem .5rem;border:1px solid #3b82f6;border-radius:999px;color:#3b82f6">Google Places (New)</span>
  <span style="padding:.2rem .5rem;border:1px solid #10b981;border-radius:999px;color:#10b981">AI Market Summary</span>
  <span style="padding:.2rem .5rem;border:1px solid #f59e0b;border-radius:999px;color:#f59e0b">Opportunity Score</span>
  <span style="padding:.2rem .5rem;border:1px solid #8b5cf6;border-radius:999px;color:#8b5cf6">Ready-to-Send Follow-Ups</span>
</p>
""", unsafe_allow_html=True)
st.caption(f"Pulse v{VERSION} — Theme: {'Dark' if THEME_DARK else 'Light'}")

info_card("What this does",
          "Type a business category and city. Pulse finds nearby competitors, highlights where you can win, "
          "and generates a *ready-to-send* follow-up sequence for your leads. No jargon—just moves you can ship this week.")

with st.expander("How it works (60 seconds)"):
    st.markdown("""
- **Find & compare:** We use **Google Places (New)** to find similar businesses and show **rating + review volume**.
- **Opportunity Score:** Lower ratings, big review counts, and gaps on the map = chances to win market share fast.
- **Playbook & promos:** AI turns the data into a 14-day **Owner Playbook** and **Steal-Share Plays**.
- **Follow-up engine:** Click **Prepare Follow-Up Engine** (in the sidebar) once per day for instant plan generation.
""")

# ---------------------- OpenAI helpers ----------------------
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None
def llm_ok() -> bool: return client is not None

def llm(prompt: str, system: str, temp: float = 0.35) -> str:
    if not llm_ok():
        return ""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=temp,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(AI unavailable: {e})"

# ---------------------- Text helpers ----------------------
def strip_code_fences(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if s.startswith("```"):
        s = s[3:]
        s = re.sub(r'^(json|JSON|python|txt)\s*\n', '', s, count=1)
        if "```" in s:
            s = s.split("```", 1)[0]
    return s.strip()

def steps_to_markdown(steps: List[Dict]) -> str:
    lines = []
    for step in steps:
        dtv = step.get("send_dt", "")
        ch = (step.get("channel", "") or "").title()
        subject = step.get("subject", "")
        body = (step.get("body", "") or "").strip()
        header = f"📅 **{dtv} — {ch}**"
        if ch.lower() == "email" and subject:
            lines += [header, f"**Subject:** {subject}", body, ""]
        else:
            lines += [header, body, ""]
    return "\n".join(lines).strip()

def coerce_polished_to_markdown(raw_text: str, fallback_steps: List[Dict]) -> str:
    cleaned = strip_code_fences(raw_text or "")
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return steps_to_markdown(obj)
        if isinstance(obj, dict) and "steps" in obj and isinstance(obj["steps"], list):
            return steps_to_markdown(obj["steps"])
    except Exception:
        pass
    return cleaned or steps_to_markdown(fallback_steps)

def build_txt_bytes(title: str, body_md: str) -> bytes:
    text = f"{title}\n\n{body_md}"
    return textwrap.dedent(text).encode("utf-8")

def build_docx_bytes(title: str, body_md: str) -> bytes:
    if not DOCX_AVAILABLE: return b""
    doc = Document()
    doc.add_heading(title, level=1).alignment = WD_ALIGN_PARAGRAPH.LEFT
    for block in body_md.split("\n\n"):
        if not block.strip(): continue
        if block.strip().startswith("📅") or block.strip().startswith("**Subject:**"):
            p = doc.add_paragraph()
            run = p.add_run(block.replace("**Subject:**", "Subject:").strip()); run.bold = True
        else:
            doc.add_paragraph(block.strip())
    buf = io.BytesIO(); doc.save(buf); buf.seek(0); return buf.read()

# ---------------------- Google Places (New) ----------------------
def gp_headers(field_mask: str):
    if not GOOGLE_PLACES_API_KEY:
        raise RuntimeError("GOOGLE_PLACES_API_KEY not set.")
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": field_mask,
    }

def geocode_location(city: str, state: str, zip_code: str):
    parts = []
    if city.strip(): parts.append(city.strip())
    if state.strip(): parts.append(state.strip())
    address = ", ".join(parts)
    if zip_code.strip():
        address = f"{address} {zip_code.strip()}" if address else zip_code.strip()
    if not address:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address": address, "key": GOOGLE_PLACES_API_KEY}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return None
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])

def search_competitors(term: str, city: str, state: str, zip_code: str, limit: int = 12) -> pd.DataFrame:
    err_msg = ""
    # Text Search
    text_url = "https://places.googleapis.com/v1/places:searchText"
    loc = ", ".join([x for x in [city.strip(), state.strip()] if x])
    loc = f"{loc} {zip_code.strip()}" if zip_code.strip() else loc
    q = f"{term.strip()} in {loc}" if loc else (term.strip() or "donut shop")
    body = {"textQuery": q, "maxResultCount": limit}
    fields = ",".join([
        "places.id","places.displayName","places.formattedAddress","places.location",
        "places.rating","places.userRatingCount","places.priceLevel",
    ])
    rt = requests.post(text_url, headers=gp_headers(fields), json=body, timeout=20)
    rt.raise_for_status()
    jt = rt.json()
    status = jt.get("status", "OK")
    if "error" in jt: err_msg = jt["error"].get("message", "")
    places = jt.get("places", []) or []

    # Nearby fallback
    if not places:
        geo = geocode_location(city, state, zip_code)
        if geo:
            lat, lng = geo
            nb_url = "https://places.googleapis.com/v1/places:searchNearby"
            nb_body = {
                "maxResultCount": limit,
                "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": 15000.0}},
                "rankPreference": "RELEVANCE",
                "includedTypes": ["bakery","cafe"],
                "keyword": term or "donut doughnut",
            }
            rn = requests.post(nb_url, headers=gp_headers(fields), json=nb_body, timeout=20)
            rn.raise_for_status()
            jn = rn.json()
            if "error" in jn and not err_msg:
                err_msg = jn["error"].get("message","")
            status = f"{status} -> Nearby:{'OK' if 'error' not in jn else 'ERROR'}"
            places = jn.get("places", []) or []

    out = []
    for p in places[:limit]:
        name = ((p.get("displayName") or {}).get("text")) or p.get("name","").split("/")[-1]
        locd = p.get("location") or {}
        out.append({
            "Name": name,
            "Rating": float(p.get("rating", 0) or 0),
            "Reviews": int(p.get("userRatingCount", 0) or 0),
            "Address": p.get("formattedAddress", ""),
            "PriceLevel": p.get("priceLevel", None),
            "Lat": locd.get("latitude", None),
            "Lng": locd.get("longitude", None),
            "PlaceID": p.get("id", p.get("name","").split("/")[-1]),
        })
    df = pd.DataFrame(out)
    df._search_status = status
    df._error_message = err_msg
    return df

# ---------------------- Follow-Up Engine (ARS) ----------------------
def _normalize_ars_steps(obj):
    arm, score, steps = None, None, []
    if isinstance(obj, dict):
        arm = obj.get("arm"); score = obj.get("score")
        raw = obj.get("steps", obj.get("plan", obj.get("data", [])))
        if isinstance(raw, list):
            steps = raw
        elif isinstance(raw, dict) and "steps" in raw and isinstance(raw["steps"], list):
            steps = raw["steps"]
    elif isinstance(obj, list):
        steps = obj
    normed = []
    for s in steps:
        if not isinstance(s, dict): continue
        send_dt = s.get("send_dt") or s.get("date") or s.get("when") or ""
        channel = (s.get("channel") or "").lower() or "email"
        subject = s.get("subject","") if channel=="email" else ""
        body = s.get("body","")
        if channel=="email" and not subject: subject = "Quick follow-up"
        normed.append({"send_dt":str(send_dt), "channel":channel, "subject":subject, "body":body})
    return arm, score, normed

def _fallback_local_plan(lead: Dict, context: Dict, reason: str = "") -> Dict:
    """Local ready-to-send plan so the UI never blocks."""
    today = dt.date.today()
    name = lead.get("name", "there")
    prefer_sms = (lead.get("channel_pref") or "").lower() == "sms"
    steps = [
        {
            "send_dt": str(today),
            "channel": "sms" if prefer_sms else "email",
            "subject": "Quick hello 👋" if not prefer_sms else "",
            "body": f"Hi {name.split()[0]}, great chatting! Would love to continue the conversation. Any questions I can answer?",
        },
        {
            "send_dt": str(today + dt.timedelta(days=2)),
            "channel": "sms" if prefer_sms else "email",
            "subject": "A sweet next step?" if not prefer_sms else "",
            "body": "Just checking in—can I hold a spot for you this week? We can set up a quick taste test or pre-order.",
        },
        {
            "send_dt": str(today + dt.timedelta(days=7)),
            "channel": "email",
            "subject": "Ready when you are 🍩",
            "body": "Following up with a friendly nudge—happy to help with an office order or weekend pickup. What works best?",
        },
    ]
    return {"arm": "fallback", "score": None, "steps": steps, "_raw": {"error": reason}}

def plan_with_ars(lead: Dict, context: Dict, cohort="donut_shop", retries: int = 3, timeout: int = 45) -> Dict:
    """Robust client: retries on 5xx/timeouts; on failure, returns local ready-to-send plan."""
    payload = {"cohort": cohort, "lead": lead, "context": context}
    body = json.dumps(payload).encode()
    sig = hmac.new(ARS_SECRET, body, hashlib.sha256).hexdigest()

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                ARS_URL,
                headers={"x-signature": sig, "Content-Type": "application/json"},
                data=body,
                timeout=timeout,
            )
            txt = r.text
            if r.status_code >= 500:
                last_err = f"ARS {r.status_code}: {txt[:200]}"
                time.sleep(min(1.5 * attempt, 6))
                continue
            if not r.ok:
                raise RuntimeError(f"ARS HTTP {r.status_code}: {txt[:300]}")
            try:
                data = r.json()
            except Exception:
                data = json.loads(txt)
            arm, score, steps = _normalize_ars_steps(data)
            return {"arm": arm, "score": score, "steps": steps, "_raw": data}
        except requests.exceptions.Timeout as e:
            last_err = f"Timeout: {e}"
            time.sleep(min(1.5 * attempt, 6))
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(min(1.5 * attempt, 6))
    return _fallback_local_plan(lead, context, reason=last_err or "Unknown error")

def warm_ars(pings: int = 3, pause: float = 3.0) -> str:
    """
    Hit /healthz and /ars/plan (tiny payload) a few times to wake the engine (Render free tier).
    Returns a human-readable status string.
    """
    if not ARS_URL:
        return "Follow-Up Engine URL is missing."
    try:
        u = urlparse(ARS_URL)
        health = urlunparse((u.scheme, u.netloc, "/healthz", "", "", ""))
        last = ""
        for i in range(pings):
            try:
                r = requests.get(health, timeout=8)
                last = f"Health {r.status_code}"
                tiny = {"cohort":"diagnostic","lead":{"name":"ping"},"context":{"today":"1970-01-01"}}
                r2 = requests.post(ARS_URL, json=tiny, timeout=8)
                last = f"Warm attempt {i+1}/{pings}: {r2.status_code}"
            except Exception as e:
                last = f"Warm attempt {i+1}/{pings} error: {e}"
            time.sleep(pause)
        return last
    except Exception as e:
        return f"Warmup error: {e}"

# ---------------------- Analytics / Viz helpers ----------------------
def opportunity_score(row):
    """Transparent scoring for buyers (explained in UI)."""
    rating = float(row["Rating"] or 0)
    reviews = int(row["Reviews"] or 0)
    base = max(0.0, 5.0 - rating)  # lower rating => more opportunity
    big_player = 1.0 if (reviews >= 300 and rating < 4.2) else 0.0
    sleeper    = 1.0 if (reviews < 60 and rating >= 4.5) else 0.0
    return round(base + 1.5*big_player + 0.8*sleeper, 2)

def render_folium_map(df: pd.DataFrame, heatmap: bool = True):
    dfc = df.dropna(subset
