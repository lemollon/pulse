# app.py — Pulse (Google Places API v1 + ARS + AI summaries)
import os
import json
import hmac
import hashlib
import requests
import datetime as dt
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Pulse — Google Places + ARS", page_icon="💼", layout="wide")
st.title("💼 Pulse — Competitor Insights (Google Places) + ARS Follow-Up")

# --------------------------------------------------------------------
# Secrets / Keys helpers
# --------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

GOOGLE_PLACES_API_KEY = get_secret("GOOGLE_PLACES_API_KEY", "")
ARS_URL    = get_secret("ARS_URL", "http://localhost:8080/ars/plan")
ARS_SECRET = get_secret("ARS_SECRET", "ars_secret_2c5d6a3b7a9f4d0c8e1f5a7b3c9d2e4f").encode()
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")

if not GOOGLE_PLACES_API_KEY:
    st.warning("Missing GOOGLE_PLACES_API_KEY. Add it in Streamlit Secrets to enable Google Places search.")

# ---------- OpenAI (optional) ----------
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

def llm_available() -> bool:
    return client is not None

def llm(prompt: str, system: str = "You are a concise business analyst for local SMBs.") -> str:
    if not llm_available():
        return ""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI helper unavailable: {e})"

# --------------------------------------------------------------------
# Google Places (New, v1): Text Search -> fallback to Nearby via Geocode
# Docs: https://developers.google.com/maps/documentation/places/web-service/overview
# --------------------------------------------------------------------
def gp_headers(field_mask: str):
    if not GOOGLE_PLACES_API_KEY:
        raise RuntimeError("GOOGLE_PLACES_API_KEY not set.")
    return {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": field_mask,   # which fields to return
    }

def geocode_location(city: str, state: str, zip_code: str):
    """Geocode to get a lat/lng for Nearby fallback (uses Geocoding API)."""
    parts = []
    if city.strip(): parts.append(city.strip())
    if state.strip(): parts.append(state.strip())
    address = ", ".join(parts)
    if zip_code.strip():
        address = f"{address} {zip_code.strip()}" if address else zip_code.strip()
    if not address:
        return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_PLACES_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return None
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])

def search_competitors(term: str, city: str, state: str, zip_code: str, limit: int = 12):
    """
    Places API (New):
      1) /v1/places:searchText
      2) Fallback: /v1/places:searchNearby around geocoded city/ZIP (15km)
    Returns a DataFrame; attaches _search_status and _error_message for debugging.
    """
    err_msg = ""

    # -------- Text Search --------
    text_url = "https://places.googleapis.com/v1/places:searchText"
    loc_parts, q_parts = [], []
    if city.strip(): loc_parts.append(city.strip())
    if state.strip(): loc_parts.append(state.strip())
    loc = ", ".join(loc_parts)
    if zip_code.strip():
        loc = f"{loc} {zip_code.strip()}" if loc else zip_code.strip()
    if term.strip(): q_parts.append(term.strip())
    query = f"{' '.join(q_parts)} in {loc}" if loc else (' '.join(q_parts) or "donut shop")

    text_body = {"textQuery": query, "maxResultCount": limit}
    text_fields = ",".join([
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
    ])
    rt = requests.post(text_url, headers=gp_headers(text_fields), json=text_body, timeout=20)
    rt.raise_for_status()
    jt = rt.json()
    status = jt.get("status", "OK")  # v1 usually omits 'status' when OK
    if "error" in jt:
        err_msg = jt["error"].get("message", "")
    places = jt.get("places", []) or []

    # -------- Fallback: Nearby --------
    if not places:
        geo = geocode_location(city, state, zip_code)
        if geo:
            lat, lng = geo
            nearby_url = "https://places.googleapis.com/v1/places:searchNearby"
            nearby_body = {
                "maxResultCount": limit,
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": lat, "longitude": lng},
                        "radius": 15000.0  # meters (15km)
                    }
                },
                "rankPreference": "RELEVANCE",
                "includedTypes": ["bakery", "cafe"],  # sane defaults
                "keyword": term or "donut doughnut"
            }
            nf = ",".join([
                "places.id",
                "places.displayName",
                "places.formattedAddress",
                "places.location",
                "places.rating",
                "places.userRatingCount",
                "places.priceLevel",
            ])
            rn = requests.post(nearby_url, headers=gp_headers(nf), json=nearby_body, timeout=20)
            rn.raise_for_status()
            jn = rn.json()
            if "error" in jn and not err_msg:
                err_msg = jn["error"].get("message", "")
            status = f"{status} -> Nearby:{'OK' if 'error' not in jn else 'ERROR'}"
            places = jn.get("places", []) or []

    # -------- Normalize to DataFrame --------
    out = []
    for p in places[:limit]:
        name = ((p.get("displayName") or {}).get("text")) or p.get("name","").split("/")[-1]
        loc = p.get("location") or {}
        out.append({
            "Name": name,
            "Rating": float(p.get("rating", 0) or 0),
            "Reviews": int(p.get("userRatingCount", 0) or 0),
            "Address": p.get("formattedAddress", ""),
            "PriceLevel": p.get("priceLevel", None),
            "Lat": loc.get("latitude", None),
            "Lng": loc.get("longitude", None),
            "PlaceID": p.get("id", p.get("name","").split("/")[-1]),
        })
    df = pd.DataFrame(out)
    df._search_status = status
    df._error_message = err_msg
    return df

def get_place_details(place_id: str):
    """Places API (New) details call. GET /v1/places/{place_id}"""
    resource = place_id if place_id.startswith("places/") else f"places/{place_id}"
    url = f"https://places.googleapis.com/v1/{resource}"
    fields = ",".join([
        "id","displayName","formattedAddress","location","rating","userRatingCount",
        "priceLevel","nationalPhoneNumber","websiteUri","regularOpeningHours.weekdayDescriptions"
        # "reviews",  # often restricted; include if your access allows
    ])
    headers = gp_headers(fields)
    headers["X-Goog-Api-Key"] = GOOGLE_PLACES_API_KEY
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    name = (data.get("displayName") or {}).get("text", "")
    opening = (data.get("regularOpeningHours") or {}).get("weekdayDescriptions", [])
    details = {
        "name": name,
        "rating": float(data.get("rating", 0) or 0),
        "user_ratings_total": int(data.get("userRatingCount", 0) or 0),
        "formatted_address": data.get("formattedAddress", ""),
        "formatted_phone_number": data.get("nationalPhoneNumber", ""),
        "price_level": data.get("priceLevel", None),
        "website": data.get("websiteUri", ""),
        "opening_hours": {"weekday_text": opening},
        "reviews": data.get("reviews", []),
    }
    return details

# --------------------------------------------------------------------
# ARS client (inline)
# --------------------------------------------------------------------
def sign_payload(body_bytes: bytes) -> str:
    return hmac.new(ARS_SECRET, body_bytes, hashlib.sha256).hexdigest()

def plan_with_ars(lead: dict, context: dict, cohort: str = "donut_shop") -> dict:
    payload = {"cohort": cohort, "lead": lead, "context": context}
    body = json.dumps(payload).encode()
    sig = sign_payload(body)
    r = requests.post(ARS_URL, headers={"x-signature": sig, "Content-Type": "application/json"}, data=body, timeout=20)
    r.raise_for_status()
    return r.json()

# --------------------------------------------------------------------
# Simple helpers
# --------------------------------------------------------------------
def opportunity_score(row):
    rating  = float(row["Rating"] or 0)
    reviews = int(row["Reviews"] or 0)
    base = max(0.0, 5.0 - rating)                      # lower rating => higher base opportunity
    big_player = 1.0 if (reviews >= 300 and rating < 4.2) else 0.0
    sleeper    = 1.0 if (reviews < 60 and rating >= 4.5) else 0.0
    return round(base + 1.5*big_player + 0.8*sleeper, 2)

def render_streamlit_map(df):
    coords = df[["Lat","Lng"]].dropna().rename(columns={"Lat":"lat","Lng":"lon"})
    if coords.empty:
        st.info("No coordinates available to plot.")
        return
    st.map(coords, zoom=11)

def build_insights_md(query_term, query_city, query_state, df_view, opp_view):
    lines = []
    lines += [f"# Pulse Insights — {query_term} in {query_city}, {query_state}", ""]
    if not df_view.empty:
        lines += [f"- Shown: **{len(df_view)}**",
                  f"- Avg rating: **{df_view['Rating'].mean():.2f}**",
                  f"- Median reviews: **{int(df_view['Reviews'].median())}**",
                  ""]
    lines += ["## Top Opportunities", ""]
    if opp_view is not None and not opp_view.empty:
        for _, r in opp_view.iterrows():
            act = r.get("Suggested Action", "")
            if not isinstance(act, str): act = ""
            lines += [f"- **{r['Name']}** — Opportunity {r['Opportunity']:.2f} "
                      f"(Rating {r['Rating']}, Reviews {r['Reviews']})"
                      f"{' — ' + act if act else ''}"]
    else:
        lines += ["(No opportunities identified in current filter.)"]
    lines += ["", "— Generated by Pulse"]
    return "\n".join(lines)

# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tab1, tab2 = st.tabs(["⭐ Competitor Watch (Google)", "📬 Lead Follow-Up (ARS)"])

# ========================= TAB 1 ===========================
with tab1:
    st.subheader("Find similar businesses via Google Places")
    st.info("**How this page uses AI:** We use Google Places data for facts (ratings, reviews, locations). "
            "If an OpenAI key is provided, AI summarizes the market and suggests quick wins based on the data you’re viewing.")

    col1, col2, col3 = st.columns([1.2,1,1])
    with col1:
        term = st.text_input("Category/term", "doughnut shop")
    with col2:
        city = st.text_input("City", "Fulshear")
        state = st.text_input("State (2-letter)", "TX", max_chars=2)
    with col3:
        zip_code = st.text_input("ZIP (optional)", "77441")
    limit = st.slider("How many competitors?", 3, 25, 12)

    if st.button("Search"):
        try:
            df = search_competitors(term, city, state, zip_code, limit=limit)
            if df.empty:
                msg = getattr(df, "_search_status", "UNKNOWN")
                em  = getattr(df, "_error_message", "")
                if em:
                    st.error(f"No results. Google said: {msg}. Error: {em}")
                else:
                    st.warning(f"No results. Try broader terms or nearby locations. (Google status: {msg})")
                st.stop()

            # Filters
            f1, f2, f3 = st.columns(3)
            with f1:
                min_reviews = st.slider("Min reviews", 0, int(df["Reviews"].max() or 0), 10, step=5)
            with f2:
                min_rating  = st.slider("Min rating", 0.0, 5.0, 3.5, step=0.1)
            with f3:
                top_n       = st.slider("Show top N by reviews", 3, min(20, len(df)), min(10, len(df)))

            dff = df[(df["Reviews"] >= min_reviews) & (df["Rating"] >= min_rating)].copy()
            dff = dff.sort_values(["Reviews","Rating"], ascending=[False, False]).head(top_n)

            # KPIs
            k1,k2,k3 = st.columns(3)
            k1.metric("Shown", len(dff))
            k2.metric("Avg rating", f"{dff['Rating'].mean():.2f}" if len(dff) else "–")
            k3.metric("Median reviews", int(dff["Reviews"].median() if len(dff) else 0))

            # Map
            st.markdown("### Map of competitors")
            st.caption("**What this is:** A simple map of similar businesses returned by Google near your search area.\n\n"
                       "**How to use it:** Zoom to see spatial clusters (e.g., highway corridors, office parks). "
                       "Consider flyering or ads around clusters.")
            render_streamlit_map(dff)

            # Charts — Bar
            st.markdown("### Top 10 by reviews — *Local awareness leaderboard*")
            st.caption("**What this is:** Bars sorted by total public review count (a proxy for foot traffic/awareness). \n"
                       "**How to use it:** The top bars are the loudest competitors; copy what works or target their weak hours.")
            if not dff.empty:
                bar = (
                    alt.Chart(dff.sort_values(["Reviews","Rating"], ascending=[False,False]).head(10))
                    .mark_bar()
                    .encode(x=alt.X("Name:N", sort='-y', title="Business"),
                            y=alt.Y("Reviews:Q", title="Review count"),
                            tooltip=["Name","Rating","Reviews"])
                    .properties(height=350)
                )
                st.altair_chart(bar, use_container_width=True)

            # Charts — Scatter
            st.markdown("### Rating vs. Review Volume — *Who’s loved vs. who’s loud*")
            st.caption("**What this is:** Each dot is a business. X = total reviews (awareness), Y = average rating (satisfaction).\n"
                       "**How to use it:** High-reviews + low-rating = big player with weaknesses → opportunity to win customers with better service/promos.")
            if not dff.empty:
                scatter = (
                    alt.Chart(dff)
                    .mark_circle(size=80)
                    .encode(x=alt.X("Reviews:Q", title="Review count"),
                            y=alt.Y("Rating:Q", title="Rating"),
                            tooltip=["Name","Rating","Reviews","Address"])
                    .interactive()
                    .properties(height=350)
                )
                st.altair_chart(scatter, use_container_width=True)

            # AI Market Summary
            if llm_available() and not dff.empty:
                sample_rows = dff.head(12)[["Name","Rating","Reviews","Address"]].to_dict(orient="records")
                prompt = (
                    "You are analyzing a local market for a small business owner.\n"
                    "Given this list of competitors (name, rating, reviews, address), summarize "
                    "1) what the market looks like, and 2) two quick wins they can test this week. "
                    f"Competitors JSON:\n{json.dumps(sample_rows)}"
                )
                st.markdown("### AI Market Summary")
                st.info(llm(prompt))
            else:
                st.caption("Tip: add `OPENAI_API_KEY` in Secrets to get AI summaries and suggested actions.")

            # Opportunity Finder
            st.markdown("### 🔎 Opportunity Finder")
            st.caption(
                "**What this is:** A ranked list of where you can steal share quickly.\n"
                "- **Opportunity** = (5 − rating) + bonus if they’re big but weak, "
                "or if there’s a sleeper with great rating but low visibility.\n"
                "**How to use it:** Start at the top row; run the suggested action for 7 days and measure results."
            )
            opp = None
            if not dff.empty:
                dff["Opportunity"] = dff.apply(opportunity_score, axis=1)
                opp = dff.sort_values("Opportunity", ascending=False)[
                    ["Name","Rating","Reviews","Opportunity","Address"]
                ].head(5).copy()

                # LLM suggested actions (optional)
                if llm_available():
                    actions = []
                    for row in opp.to_dict(orient="records"):
                        prompt = (
                            "Suggest one high-ROI, low-lift action a donut shop could take to win customers "
                            "from this competitor within 7 days. Keep it to 1–2 sentences, concrete and testable.\n"
                            f"Competitor: {row}"
                        )
                        actions.append(llm(prompt, system="You are a scrappy local growth marketer."))
                    opp["Suggested Action"] = actions
                else:
                    opp["Suggested Action"] = "Add OPENAI_API_KEY to see tailored actions."

                st.dataframe(opp, use_container_width=True)

            # Details + (optional) AI review summary
            st.markdown("---")
            st.subheader("Place details")
            name_choice = st.selectbox("Select a business", list(dff["Name"]))
            chosen = dff[dff["Name"] == name_choice].iloc[0]
            try:
                details = get_place_details(chosen["PlaceID"])
                st.write(f"**{details.get('name','')}**")
                st.write(details.get("formatted_address",""))
                phone = details.get("formatted_phone_number","")
                if phone: st.write(phone)
                st.write(f"Rating: **{details.get('rating',0)}** ({details.get('user_ratings_total',0)} reviews)")
                site = details.get("website","")
                if site: st.write(site)

                if (details.get("opening_hours") or {}).get("weekday_text"):
                    with st.expander("Opening hours"):
                        for line in details["opening_hours"]["weekday_text"]:
                            st.write(line)

                revs = details.get("reviews", []) or []
                if llm_available() and revs:
                    texts = [r.get("text","") for r in revs if r.get("text")]
                    if texts:
                        prompt = (
                            "From these customer review snippets, extract:\n"
                            "1) Top 3 things customers love\n2) Top 3 friction points\n"
                            "Be brief and specific.\n\n"
                            f"Reviews:\n{json.dumps(texts[:12])}"
                        )
                        st.markdown("#### AI summary — what customers love vs. where they struggle")
                        st.info(llm(prompt))
            except Exception as e:
                st.error(f"Details error: {e}")

            # Export Insights.md
            insights_md = build_insights_md(term, city, state, dff, opp)
            st.download_button("⬇️ Download Insights.md", insights_md.encode("utf-8"),
                               "insights.md", "text/markdown")

        except Exception as e:
            st.error(f"Google Places error: {e}")

    st.caption("Requires **Places API (New)** and **Geocoding API** enabled. "
               "In your API key: Application restrictions = None; API restrictions = Places API (New) + Geocoding API.")

# ========================= TAB 2 ===========================
with tab2:
    st.subheader("Generate a 3-step follow-up plan using your private ARS backend")
    with st.expander("What is ARS and why it helps"):
        st.write(
            "- **ARS = Adaptive Revenue Sequencer.** It picks a 3-step follow-up plan "
            "(channels, timing, copy) that maximizes replies.\n"
            "- Under the hood: constraint-aware scoring + a **multi-armed bandit (UCB)** to learn over time.\n"
            "- Result: better response rates without guesswork."
        )
    st.info("**How this page uses AI:** ARS scores features (timing, channel preference, sentiment) "
            "and uses a bandit algorithm to explore what works. As you feed outcomes later (replied/booked), "
            "it will adapt sequences to your market.")

    colA, colB = st.columns(2)
    with colA:
        lead_name = st.text_input("Lead name", "Jane Smith")
        contact   = st.text_input("Contact (email or phone)", "jane@example.com")
        pref      = st.selectbox("Preferred channel", ["email", "sms"], index=0)
        notes     = st.text_area("Notes/context", "Interested in a dozen + coffee for office pickup")
    with colB:
        avg_sent7 = st.number_input("Avg sentiment (7d)", value=0.10, step=0.05, format="%.2f")
        wait_iss  = st.number_input("Complaint: wait time (0..1)", value=0.00, step=0.05, format="%.2f")
        recency   = st.number_input("Recency (days since last touch)", value=2, step=1)
        prior_rr  = st.number_input("Prior reply rate (0..1)", value=0.12, step=0.01, format="%.2f")

    if st.button("Generate Follow-Up Plan"):
        lead = {
            "name": lead_name,
            "contact": contact,
            "channel_pref": pref,
            "notes": notes,
            "last_interaction": str(dt.date.today()),
        }
        context = {
            "today": str(dt.date.today()),
            "hour_local": dt.datetime.now().hour,
            "weekend": dt.date.today().weekday() >= 5,
            "holiday_flag": False,
            "avg_sentiment_7d": float(avg_sent7),
            "complaint_wait_time": float(wait_iss),
            "recency_days": int(recency),
            "prior_reply_rate": float(prior_rr),
        }
        try:
            data = plan_with_ars(lead, context, cohort="donut_shop")
            st.success(f"Chosen arm: {data.get('arm','?')} • Score: {data.get('score','?')}")
            st.markdown("### Planned Steps")
            for step in data.get("steps", []):
                st.markdown(
                    f"📅 **{step.get('send_dt','')}** — *{step.get('channel','')}* "
                    f"{'— ' + step.get('subject','') if step.get('subject') else ''}"
                )
                st.write(step.get("body",""))
                st.markdown("---")

            # Explain *why* this plan (simple heuristics)
            reasons = []
            if pref == "sms":
                reasons.append("Lead prefers SMS, so we start or include SMS early.")
            hour_now = dt.datetime.now().hour
            if 8 <= hour_now <= 11:
                reasons.append("Morning window is strong for food orders and office pickups.")
            if float(avg_sent7) >= 0.1:
                reasons.append("Recent sentiment is positive; light, friendly tone increases replies.")
            if float(wait_iss) > 0:
                reasons.append("Some wait-time complaints—include pre-order/pickup link.")
            if not reasons:
                reasons = ["Balanced plan chosen to learn what works fastest."]
            st.markdown("#### Why we chose this plan")
            st.write("\n".join([f"- {r}" for r in reasons]))

            st.download_button("⬇️ Export follow-ups (CSV)",
                               pd.DataFrame(data.get("steps", [])).to_csv(index=False).encode(),
                               "followups.csv", "text/csv")
        except Exception as e:
            st.error(f"ARS error: {e}")

st.markdown("---")
st.caption("Pulse © — Google Places (New) competitor insights + private ARS follow-ups.")
