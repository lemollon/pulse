# app.py — Pulse v1.6 "Must-Have" Edition
# Google Places (New) + Folium + Opportunity Heatmap + Gap Radar
# ARS follow-up planner (explainability) + AI promos + TXT/DOCX exports
# Guardrails: feature registry, startup self-test, fail-soft for deps

import os, io, re, json, time, hmac, hashlib, textwrap, math
import datetime as dt
import requests
import pandas as pd
import altair as alt
import streamlit as st
from typing import Dict, List, Tuple

# ----- Optional docs export -----
DOCX_AVAILABLE = True
try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except Exception:
    DOCX_AVAILABLE = False

# ----- Maps -----
import folium
from streamlit_folium import st_folium

# ----- OpenAI (optional) -----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------------- THEME & STYLE ----------------------
st.set_page_config(page_title="Pulse — Win Your Neighborhood", page_icon="🧭", layout="wide")
PRIMARY = "#6B5BFF"
ACCENT = "#00C29A"

st.markdown(f"""
<style>
/* Clean card look */
.reportview-container .main .block-container {{padding-top: 1rem;}}
div.stButton>button {{background:{PRIMARY}; color:white; border-radius:8px; border:none;}}
div.stDownloadButton>button {{background:{ACCENT}; color:white; border-radius:8px; border:none;}}
.kpi {{background:#f8f9fb;border:1px solid #eef1f6;border-radius:12px;padding:10px 14px;margin-bottom:8px;}}
.section-title {{font-weight:700;font-size:1.2rem;margin-top:1rem;}}
.small-note {{color:#6b7280;font-size:0.9rem;}}
hr {{border-top:1px solid #eef1f6;}}
</style>
""", unsafe_allow_html=True)

# ---------------------- GLOBALS ----------------------
VERSION = "1.6.0"
CHANGELOG = [
    "Added Opportunity Heatmap, Gap Radar, AI Promo Generator",
    "Restored AI Playbook, Suggested Actions, Insights export",
    "Added DOCX export for follow-ups + campaigns",
    "Startup self-test & feature registry guardrails",
    "Improved ARS explainability and health tester",
]

FEATURES = {
    "google_search": True,
    "map_markers": True,
    "heatmap": True,
    "charts": True,
    "ai_market_summary": True,
    "ai_suggested_actions": True,
    "owner_playbook": True,
    "insights_export_md": True,
    "ars_planner": True,
    "ars_explainability": True,
    "ai_polish": True,
    "export_txt": True,
    "export_docx": True,   # hidden if DOCX_AVAILABLE == False
    "setup_checklist": True,
}

# ---------------------- SECRETS ----------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

GOOGLE_PLACES_API_KEY = get_secret("GOOGLE_PLACES_API_KEY", "")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
ARS_URL = get_secret("ARS_URL", "http://localhost:8080/ars/plan")
ARS_SECRET = get_secret("ARS_SECRET", "ars_secret_2c5d6a3b7a9f4d0c8e1f5a7b3c9d2e4f").encode()

client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None
def llm_ok() -> bool: return client is not None

def llm(prompt: str, system: str, temp: float = 0.35) -> str:
    if not llm_ok(): return ""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=temp,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(AI unavailable: {e})"

# ---------------------- UTILITIES ----------------------
def strip_code_fences(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if s.startswith("```"):
        s = s[3:]
        s = re.sub(r'^(json|JSON|python|txt)\s*\n', '', s, count=1)
        if "```" in s: s = s.split("```", 1)[0]
    return s.strip()

def steps_to_markdown(steps: List[Dict]) -> str:
    lines = []
    for step in steps:
        dtv = step.get("send_dt","")
        ch = (step.get("channel","") or "").title()
        subject = step.get("subject","")
        body = (step.get("body","") or "").strip()
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
        if isinstance(obj, list): return steps_to_markdown(obj)
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

# ---------------------- GOOGLE PLACES (New) ----------------------
def gp_headers(field_mask: str):
    if not GOOGLE_PLACES_API_KEY:
        raise RuntimeError("GOOGLE_PLACES_API_KEY not set.")
    return {"Content-Type":"application/json","X-Goog-Api-Key":GOOGLE_PLACES_API_KEY,"X-Goog-FieldMask":field_mask}

def geocode_location(city: str, state: str, zip_code: str):
    parts = []
    if city.strip(): parts.append(city.strip())
    if state.strip(): parts.append(state.strip())
    address = ", ".join(parts)
    if zip_code.strip(): address = f"{address} {zip_code.strip()}" if address else zip_code.strip()
    if not address: return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address":address, "key":GOOGLE_PLACES_API_KEY}, timeout=20)
    r.raise_for_status(); data = r.json()
    if data.get("status") != "OK": return None
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])

def search_competitors(term: str, city: str, state: str, zip_code: str, limit: int = 12) -> pd.DataFrame:
    err_msg = ""
    # Text search
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
    rt.raise_for_status(); jt = rt.json()
    status = jt.get("status","OK"); 
    if "error" in jt: err_msg = jt["error"].get("message","")
    places = jt.get("places", []) or []

    # Nearby fallback
    if not places:
        geo = geocode_location(city, state, zip_code)
        if geo:
            lat,lng = geo
            nb_url = "https://places.googleapis.com/v1/places:searchNearby"
            nb_body = {
                "maxResultCount": limit,
                "locationRestriction": {"circle":{"center":{"latitude":lat,"longitude":lng},"radius":15000.0}},
                "rankPreference":"RELEVANCE","includedTypes":["bakery","cafe"],
                "keyword": term or "donut doughnut"
            }
            rn = requests.post(nb_url, headers=gp_headers(fields), json=nb_body, timeout=20)
            rn.raise_for_status(); jn = rn.json()
            if "error" in jn and not err_msg: err_msg = jn["error"].get("message","")
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
            "Address": p.get("formattedAddress",""),
            "PriceLevel": p.get("priceLevel", None),
            "Lat": locd.get("latitude", None),
            "Lng": locd.get("longitude", None),
            "PlaceID": p.get("id", p.get("name","").split("/")[-1]),
        })
    df = pd.DataFrame(out); df._search_status = status; df._error_message = err_msg
    return df

def get_place_details(place_id: str) -> Dict:
    res = place_id if place_id.startswith("places/") else f"places/{place_id}"
    url = f"https://places.googleapis.com/v1/{res}"
    fields = ",".join([
        "id","displayName","formattedAddress","location","rating","userRatingCount",
        "priceLevel","nationalPhoneNumber","websiteUri","regularOpeningHours.weekdayDescriptions"
    ])
    r = requests.get(url, headers=gp_headers(fields), timeout=20)
    r.raise_for_status(); d = r.json()
    return {
        "name": (d.get("displayName") or {}).get("text",""),
        "rating": float(d.get("rating",0) or 0),
        "user_ratings_total": int(d.get("userRatingCount",0) or 0),
        "formatted_address": d.get("formattedAddress",""),
        "formatted_phone_number": d.get("nationalPhoneNumber",""),
        "price_level": d.get("priceLevel", None),
        "website": d.get("websiteUri",""),
        "opening_hours": {"weekday_text": (d.get("regularOpeningHours") or {}).get("weekdayDescriptions", [])},
        "reviews": d.get("reviews", []),
    }

# ---------------------- ARS ----------------------
def sign_payload(body_bytes: bytes) -> str:
    return hmac.new(ARS_SECRET, body_bytes, hashlib.sha256).hexdigest()

def _normalize_ars_steps(obj):
    arm, score, steps = None, None, []
    if isinstance(obj, dict):
        arm = obj.get("arm"); score = obj.get("score")
        raw = obj.get("steps", obj.get("plan", obj.get("data", [])))
        if isinstance(raw, list): steps = raw
        elif isinstance(raw, dict) and "steps" in raw and isinstance(raw["steps"], list): steps = raw["steps"]
    elif isinstance(obj, list): steps = obj
    normed = []
    for s in steps:
        if not isinstance(s, dict): continue
        send_dt = s.get("send_dt") or s.get("date") or s.get("when") or ""
        channel = (s.get("channel") or "").lower() or "email"
        subject = s.get("subject","") if channel=="email" else ""
        body = s.get("body","")
        if channel=="email" and not subject: subject="Quick follow-up"
        normed.append({"send_dt":str(send_dt), "channel":channel, "subject":subject, "body":body})
    return arm, score, normed

def plan_with_ars(lead: Dict, context: Dict, cohort="donut_shop") -> Dict:
    payload = {"cohort":cohort, "lead":lead, "context":context}
    body = json.dumps(payload).encode(); sig = sign_payload(body)
    r = requests.post(ARS_URL, headers={"x-signature":sig,"Content-Type":"application/json"}, data=body, timeout=45)
    text = r.text
    if not r.ok: raise RuntimeError(f"ARS HTTP {r.status_code}: {text[:300]}")
    try: data = r.json()
    except Exception: data = json.loads(text)
    arm, score, steps = _normalize_ars_steps(data)
    return {"arm":arm, "score":score, "steps":steps, "_raw":data}

# ---------------------- ANALYTICS / SCORING ----------------------
def opportunity_score(row):
    rating = float(row["Rating"] or 0); reviews = int(row["Reviews"] or 0)
    base = max(0.0, 5.0 - rating)              # lower rating => bigger opportunity
    big_player = 1.0 if (reviews >= 300 and rating < 4.2) else 0.0
    sleeper    = 1.0 if (reviews < 60 and rating >= 4.5) else 0.0
    return round(base + 1.5*big_player + 0.8*sleeper, 2)

def render_folium_map(df: pd.DataFrame, heatmap: bool = True):
    dfc = df.dropna(subset=["Lat","Lng"]).copy()
    if dfc.empty: st.info("No coordinates to plot."); return
    m = folium.Map(location=[dfc["Lat"].mean(), dfc["Lng"].mean()], zoom_start=12, control_scale=True)
    # markers
    for _, r in dfc.iterrows():
        popup = folium.Popup(
            f"<b>{r['Name']}</b><br>Rating: {r['Rating']} ⭐ | Reviews: {r['Reviews']}<br>{r['Address']}",
            max_width=350
        )
        icon_color = "green" if r["Rating"] >= 4.5 else ("orange" if r["Rating"] >= 4.0 else "red")
        folium.Marker([r["Lat"], r["Lng"]], tooltip=r["Name"], popup=popup,
                      icon=folium.Icon(color=icon_color, icon="info-sign")).add_to(m)
    # lightweight heat effect (cluster via semi-transparent circles)
    if heatmap:
        for _, r in dfc.iterrows():
            radius = 80 + min(220, 2 * r["Reviews"])
            folium.Circle(
                location=[r["Lat"], r["Lng"]],
                radius=radius,
                color="#ff0066" if r["Rating"] < 4.0 else "#00c29a",
                fill=True, fill_opacity=0.08, opacity=0.15
            ).add_to(m)
    st_folium(m, width=None, height=540)

# ---------------------- CONTENT BUILDERS ----------------------
def build_insights_md(term, city, state, df_view, opp_view, playbook_text=""):
    lines = [f"# Pulse Insights — {term} in {city}, {state}", ""]
    if not df_view.empty:
        lines += [f"- Shown: **{len(df_view)}**",
                  f"- Avg rating: **{df_view['Rating'].mean():.2f}**",
                  f"- Median reviews: **{int(df_view['Reviews'].median())}**", ""]
    lines += ["## Top Opportunities", ""]
    if opp_view is not None and not opp_view.empty:
        for _, r in opp_view.iterrows():
            act = str(r.get("Suggested Action","")).strip()
            lines += [f"- **{r['Name']}** — Opportunity {r['Opportunity']:.2f} "
                      f"(Rating {r['Rating']}, Reviews {r['Reviews']})"
                      + (f" — {act}" if act else "")]
    else:
        lines += ["(No opportunities identified in current filter.)"]
    if playbook_text:
        lines += ["", "## Owner Playbook", "", playbook_text]
    lines += ["", f"_Pulse v{VERSION}_"]
    return "\n".join(lines)

# ---------------------- STARTUP SELF-TEST ----------------------
def startup_self_test():
    issues = []
    if not FEATURES["google_search"]: issues.append("Google search feature flag off.")
    if not GOOGLE_PLACES_API_KEY: issues.append("Missing GOOGLE_PLACES_API_KEY.")
    if FEATURES["export_docx"] and not DOCX_AVAILABLE:
        issues.append("python-docx not installed; DOCX export will be hidden.")
    return issues

# ---------------------- SESSION DEFAULTS ----------------------
if "search_df" not in st.session_state: st.session_state.search_df = None
if "search_inputs" not in st.session_state:
    st.session_state.search_inputs = {"term":"doughnut shop","city":"Fulshear","state":"TX","zip_code":"77441","limit":12}

# ---------------------- HEADER ----------------------
left, right = st.columns([3,1])
with left:
    st.write(f"**Pulse v{VERSION}** — Win your neighborhood with AI.")
with right:
    if st.button("View changelog"):
        st.info("\n".join(f"- {c}" for c in CHANGELOG))

issues = startup_self_test()
if issues:
    with st.expander("⚙️ Setup checklist & warnings"):
        for i in issues: st.warning(i)
        st.caption("This checklist prevents silent regressions.")

# ---------------------- TABS ----------------------
tab1, tab2 = st.tabs(["⭐ Competitor Watch", "📬 Lead Follow-Up (ARS)"])

# ===================== TAB 1 =====================
with tab1:
    st.subheader("Find similar businesses via Google Places")
    st.caption("Facts from Google. AI turns it into moves you can take this week.")

    with st.form("search_form"):
        inputs = st.session_state.search_inputs
        c1,c2,c3 = st.columns([1.3,1,1])
        with c1: term = st.text_input("Category/term", inputs.get("term","doughnut shop"))
        with c2:
            city = st.text_input("City", inputs.get("city","Fulshear"))
            state = st.text_input("State (2-letter)", inputs.get("state","TX"), max_chars=2)
        with c3: zip_code = st.text_input("ZIP (optional)", inputs.get("zip_code","77441"))
        limit = st.slider("How many places?", 3, 25, inputs.get("limit",12))
        submitted = st.form_submit_button("Search")

    if st.button("Clear results"):
        st.session_state.search_df = None
        st.session_state.search_inputs = {"term":"doughnut shop","city":"Fulshear","state":"TX","zip_code":"77441","limit":12}
        st.experimental_rerun()

    if submitted:
        try:
            df = search_competitors(term, city, state, zip_code, limit=limit)
            st.session_state.search_inputs = {"term":term,"city":city,"state":state,"zip_code":zip_code,"limit":limit}
            if df.empty:
                msg = getattr(df, "_search_status","UNKNOWN"); em = getattr(df, "_error_message","")
                (st.error if em else st.warning)(f"No results. Google status: {msg} {'| '+em if em else ''}")
                st.session_state.search_df = None
            else:
                st.session_state.search_df = df
        except Exception as e:
            st.error(f"Google Places error: {e}"); st.session_state.search_df = None

    df = st.session_state.search_df
    if df is None:
        st.info("Enter a search and press **Search** to see competitors.")
        st.stop()

    # Filters
    f1,f2,f3 = st.columns(3)
    with f1: min_reviews = st.slider("Min reviews", 0, int(df["Reviews"].max() or 0), 10, step=5)
    with f2: min_rating  = st.slider("Min rating", 0.0, 5.0, 3.5, step=0.1)
    with f3: top_n       = st.slider("Show top N by reviews", 3, min(20, len(df)), min(10, len(df)))

    dff = df[(df["Reviews"]>=min_reviews) & (df["Rating"]>=min_rating)].copy()
    dff = dff.sort_values(["Reviews","Rating"], ascending=[False,False]).head(top_n)

    # KPIs
    k1,k2,k3 = st.columns(3)
    with k1: st.markdown(f"<div class='kpi'><b>Shown</b><br>{len(dff)}</div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi'><b>Avg rating</b><br>{dff['Rating'].mean():.2f if len(dff) else '–'}</div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi'><b>Median reviews</b><br>{int(dff['Reviews'].median() if len(dff) else 0)}</div>", unsafe_allow_html=True)

    # Map + Heat
    st.markdown("<div class='section-title'>Map & Heat</div>", unsafe_allow_html=True)
    st.caption("Hover pins for name/rating/reviews. Colored circles show density (hot = tougher, cool = gaps).")
    render_folium_map(dff, heatmap=True)

    # Charts
    st.markdown("<div class='section-title'>Top 10 by reviews — Local awareness leaderboard</div>", unsafe_allow_html=True)
    if not dff.empty:
        bar = (alt.Chart(dff.sort_values(["Reviews","Rating"], ascending=[False,False]).head(10))
               .mark_bar().encode(x=alt.X("Name:N", sort='-y'), y="Reviews:Q",
                                  tooltip=["Name","Rating","Reviews"]).properties(height=320))
        st.altair_chart(bar, use_container_width=True)
        st.caption("Use: copy what top players do *well*; target their weak hours/locations with promos.")

    st.markdown("<div class='section-title'>Rating vs. Review Volume — Who’s loved vs. who’s loud</div>", unsafe_allow_html=True)
    if not dff.empty:
        scatter = (alt.Chart(dff).mark_circle(size=80)
                   .encode(x=alt.X("Reviews:Q", title="Review count"), y=alt.Y("Rating:Q"),
                           tooltip=["Name","Rating","Reviews","Address"]).interactive().properties(height=320))
        st.altair_chart(scatter, use_container_width=True)
        st.caption("Top-right = winners; bottom-right = big but weak (ripe to steal). Top-left = sleepers.")

    # AI Market Summary
    if llm_ok() and not dff.empty:
        sample = dff.head(12)[["Name","Rating","Reviews","Address"]].to_dict(orient="records")
        prompt = ("Summarize local competition in 4 sentences and list 2 quick tests. "
                  "Keep it practical for a donut/coffee shop.\n"
                  f"Data:\n{json.dumps(sample)}")
        st.markdown("<div class='section-title'>AI Market Summary</div>", unsafe_allow_html=True)
        st.info(llm(prompt, system="You are a practical SMB strategist."))

    # Opportunity Finder + AI Actions
    st.markdown("<div class='section-title'>🔎 Opportunity Finder</div>", unsafe_allow_html=True)
    opp = None
    if not dff.empty:
        dff["Opportunity"] = dff.apply(opportunity_score, axis=1)
        opp = dff.sort_values("Opportunity", ascending=False)[["Name","Rating","Reviews","Opportunity","Address"]].head(5).copy()
        if llm_ok():
            acts = []
            for row in opp.to_dict(orient="records"):
                p = ("Suggest one high-ROI, low-lift action to win share from this competitor in 7 days. "
                     "1–2 sentences, concrete.\n"
                     f"Competitor: {json.dumps(row)}")
                acts.append(llm(p, system="You are a scrappy local growth marketer."))
            opp["Suggested Action"] = acts
        else:
            opp["Suggested Action"] = "Add OPENAI_API_KEY to see tailored actions."
        st.dataframe(opp, use_container_width=True)

    # AI Promo Generator (5 campaigns)
    promos_text = ""
    if llm_ok() and not dff.empty:
        p_prompt = ("Create 5 micro-campaign ideas tailored to these competitors and morning commute patterns. "
                    "Each: a title + 1 sentence, include timing (e.g., 7-9am) and channel suggestion (SMS/email/in-store).\n"
                    f"Data:\n{json.dumps(dff.head(12).to_dict(orient='records'))}")
        st.markdown("<div class='section-title'>🎯 Steal-Share Plays (AI)</div>", unsafe_allow_html=True)
        promos_text = llm(p_prompt, system="You are a local growth hacker writing concise play ideas.")
        st.info(promos_text)

    # Owner Playbook
    playbook_text = ""
    if llm_ok() and not dff.empty:
        pb = ("Write a 5-bullet, 14-day playbook for this shop based on the competition list. "
              "Prioritize quick wins, morning rush, pre-orders, offices.\n"
              f"Data:\n{json.dumps(dff.head(12).to_dict(orient='records'))}")
        st.markdown("<div class='section-title'>📓 Owner Playbook (AI)</div>", unsafe_allow_html=True)
        playbook_text = llm(pb, system="You are a practical small-business coach.")
        st.info(playbook_text)

    # Export insights kit
    term = st.session_state.search_inputs.get("term","doughnut shop")
    city = st.session_state.search_inputs.get("city","")
    state = st.session_state.search_inputs.get("state","")
    insights_md = build_insights_md(term, city, state, dff, opp, playbook_text=playbook_text)
    st.download_button("⬇️ Download Insights.md", insights_md.encode("utf-8"),
                       "insights.md", "text/markdown")

# ===================== TAB 2 =====================
with tab2:
    st.subheader("Adaptive Follow-Ups (ARS) + AI Polish")
    st.caption("ARS picks channels/dates; AI polishes copy. Owners get ready-to-send messages.")

    # Diagnostics
    st.sidebar.header("⚙️ Diagnostics")
    if st.sidebar.button("Test ARS connection"):
        try:
            lead = {"name":"Test","contact":"test@example.com"}
            context = {"today": str(dt.date.today()), "hour_local": dt.datetime.now().hour, "weekend": False}
            result = plan_with_ars(lead, context, cohort="diagnostic")
            st.sidebar.success(f"ARS OK — {len(result.get('steps', []))} steps")
        except Exception as e:
            st.sidebar.error(f"ARS error: {e}")

    # Inputs
    ca, cb = st.columns(2)
    with ca:
        lead_name = st.text_input("Lead name", "Jane Smith")
        contact   = st.text_input("Contact (email or phone)", "jane@example.com")
        pref      = st.selectbox("Preferred channel", ["email","sms"], index=0)
        notes     = st.text_area("Notes/context", "Interested in a dozen + coffee for office pickup")
    with cb:
        avg_sent7 = st.number_input("Avg sentiment (7d)", value=0.10, step=0.05, format="%.2f")
        wait_iss  = st.number_input("Complaint: wait time (0..1)", value=0.00, step=0.05, format="%.2f")
        recency   = st.number_input("Recency (days since last touch)", value=2, step=1)
        prior_rr  = st.number_input("Prior reply rate (0..1)", value=0.12, step=0.01, format="%.2f")

    if st.button("Generate Follow-Up Plan"):
        lead = {"name":lead_name, "contact":contact, "channel_pref":pref, "notes":notes, "last_interaction":str(dt.date.today())}
        context = {"today":str(dt.date.today()), "hour_local":dt.datetime.now().hour,
                   "weekend": dt.date.today().weekday()>=5, "holiday_flag": False,
                   "avg_sentiment_7d": float(avg_sent7), "complaint_wait_time": float(wait_iss),
                   "recency_days": int(recency), "prior_reply_rate": float(prior_rr)}
        try:
            result = plan_with_ars(lead, context, cohort="donut_shop")
            steps = result.get("steps", [])
            st.success(f"Chosen arm: {result.get('arm','?')} • Score: {result.get('score','?')}")
            st.markdown("#### Planned Steps (before AI polish)")
            for s in steps:
                subj = s.get("subject",""); subj_str = f" — {subj}" if subj else ""
                st.markdown(f"📅 **{s.get('send_dt','')}** — *{s.get('channel','')}*{subj_str}")
                st.write(s.get("body","")); st.markdown("---")

            # Why this plan
            reasons = []
            if pref == "sms": reasons.append("Lead prefers SMS, so we include SMS early.")
            hr = dt.datetime.now().hour
            if 8 <= hr <= 11: reasons.append("Morning window aligns with coffee + office orders.")
            if float(avg_sent7) >= 0.1: reasons.append("Positive sentiment → light, upbeat tone.")
            if float(wait_iss) > 0: reasons.append("Wait-time complaints → include pre-order link.")
            if not reasons: reasons = ["Balanced plan to learn what works fastest."]
            st.markdown("#### Why this plan works")
            st.write("\n".join(f"- {r}" for r in reasons))

            # AI polish
            polished_text = steps_to_markdown(steps)
            if llm_ok() and steps:
                prompt = (
                    "Polish these steps for a friendly donut shop brand. "
                    "Keep SAME dates/channels; concise, warm, action-oriented. "
                    "Return PLAIN TEXT (no JSON, no code fences). "
                    f"Steps JSON:\n{json.dumps(steps)}"
                )
                raw = llm(prompt, system="You write high-converting SMB follow-ups.")
                polished_text = coerce_polished_to_markdown(raw, steps)
                st.markdown("#### AI-Polished Copy")
                st.markdown(polished_text)

            # Exports
            st.download_button("⬇️ Download polished plan (TXT)",
                               build_txt_bytes("Polished Follow-Up Plan", polished_text),
                               "polished_followup.txt","text/plain")
            if DOCX_AVAILABLE:
                st.download_button("⬇️ Download polished plan (DOCX)",
                                   build_docx_bytes("Polished Follow-Up Plan", polished_text),
                                   "polished_followup.docx",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            else:
                st.caption("Install `python-docx` to enable DOCX export.")

            # Bonus: download the 5 AI promos as docx/txt if generated on tab 1
            if llm_ok():
                campaigns_txt = f"Steal-Share Plays\n\n{promos_text or 'Run a Competitor Watch first to generate promos.'}"
                st.download_button("⬇️ Download campaigns (TXT)", campaigns_txt.encode("utf-8"),
                                   "campaigns.txt","text/plain")
                if DOCX_AVAILABLE:
                    st.download_button("⬇️ Download campaigns (DOCX)",
                                       build_docx_bytes("Steal-Share Plays", campaigns_txt),
                                       "campaigns.docx",
                                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        except Exception as e:
            st.error(f"ARS error: {e}")

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.caption(f"Pulse v{VERSION} • Features OK: " + ", ".join([k for k,v in FEATURES.items() if v]) +
           (" • DOCX enabled" if DOCX_AVAILABLE else " • DOCX not installed"))
