# app.py
import os
import json
import hmac
import hashlib
import requests
import datetime as dt
import pandas as pd
import altair as alt
import streamlit as st
import pydeck as pdk

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

if not GOOGLE_PLACES_API_KEY:
    st.warning("Missing GOOGLE_PLACES_API_KEY. Add it in Streamlit Secrets to enable Google Places search.")

# --------------------------------------------------------------------
# Google Places (inline client)
# --------------------------------------------------------------------
def gp_base_params():
    if not GOOGLE_PLACES_API_KEY:
        raise RuntimeError("GOOGLE_PLACES_API_KEY not set.")
    return {"key": GOOGLE_PLACES_API_KEY}

def build_query(term: str, city: str, state: str, zip_code: str) -> str:
    parts = []
    if city.strip(): parts.append(city.strip())
    if state.strip(): parts.append(state.strip())
    loc = ", ".join(parts)
    if zip_code.strip():
        loc = f"{loc} {zip_code.strip()}" if loc else zip_code.strip()
    return f"{term} in {loc}" if loc else term

def search_competitors(term: str, city: str, state: str, zip_code: str, limit: int = 12):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    q = build_query(term, city, state, zip_code)
    params = gp_base_params() | {"query": q}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    results = r.json().get("results", [])
    out = []
    for p in results[:limit]:
        loc = (p.get("geometry") or {}).get("location") or {}
        out.append({
            "Name": p.get("name", ""),
            "Rating": float(p.get("rating", 0) or 0),
            "Reviews": int(p.get("user_ratings_total", 0) or 0),
            "Address": p.get("formatted_address", ""),
            "PriceLevel": p.get("price_level", None),
            "Lat": loc.get("lat", None),
            "Lng": loc.get("lng", None),
            "PlaceID": p.get("place_id", "")
        })
    return pd.DataFrame(out)

def get_place_details(place_id: str):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = ",".join([
        "name","rating","user_ratings_total","formatted_address",
        "formatted_phone_number","price_level","website","opening_hours","reviews"
    ])
    params = gp_base_params() | {"place_id": place_id, "fields": fields}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("result", {}) or {}

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
# UI helpers
# --------------------------------------------------------------------
POSITIVE_KWS = ["friendly","fresh","clean","fast","delicious","tasty","warm","helpful"]
NEGATIVE_KWS = ["wait","slow","cold","stale","price","expensive","dirty","rude","parking","line","crowded"]

def keyword_counts(texts, vocab):
    counts = {w: 0 for w in vocab}
    for t in texts:
        low = t.lower()
        for w in vocab:
            counts[w] += low.count(w)
    return pd.DataFrame([{"keyword":k, "count":v} for k,v in counts.items()])

def opportunity_score(row):
    rating  = float(row["Rating"] or 0)
    reviews = int(row["Reviews"] or 0)
    base = max(0.0, 5.0 - rating)
    big_player = 1.0 if (reviews >= 300 and rating < 4.2) else 0.0
    sleeper    = 1.0 if (reviews < 60 and rating >= 4.5) else 0.0
    return round(base + 1.5*big_player + 0.8*sleeper, 2)

def map_layer(df):
    def color_for_rating(r):
        if r >= 4.6: return [0,128,0]
        if r >= 4.2: return [76,175,80]
        if r >= 3.8: return [255,193,7]
        return [244,67,54]

    plot_df = df[["Name","Rating","Reviews","Lat","Lng"]].dropna().copy()
    if plot_df.empty:
        return None
    plot_df["color"] = plot_df["Rating"].apply(color_for_rating)
    plot_df["size"] = plot_df["Reviews"].clip(lower=1).apply(lambda x: min(60, 10 + (x**0.5)*3))
    layer = pdk.Layer("ScatterplotLayer", data=plot_df, get_position='[Lng, Lat]',
                      get_fill_color='color', get_radius='size', pickable=True)
    view_state = pdk.ViewState(latitude=float(plot_df["Lat"].mean()), longitude=float(plot_df["Lng"].mean()), zoom=11)
    return pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", layers=[layer], initial_view_state=view_state,
                    tooltip={"text":"{Name}\nRating: {Rating}★  Reviews: {Reviews}"})

# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tab1, tab2 = st.tabs(["⭐ Competitor Watch (Google)", "📬 Lead Follow-Up (ARS)"])

with tab1:
    st.subheader("Find similar businesses via Google Places")
    col1, col2, col3 = st.columns([1.2,1,1])
    with col1:
        term = st.text_input("Category/term", "doughnut shop")
    with col2:
        city = st.text_input("City", "Austin")
        state = st.text_input("State (2-letter)", "TX", max_chars=2)
    with col3:
        zip_code = st.text_input("ZIP (optional)", "")
    limit = st.slider("How many competitors?", 3, 25, 12)

    if st.button("Search"):
        try:
            df = search_competitors(term, city, state, zip_code, limit=limit)
            if df.empty:
                st.warning("No results. Try broader terms or nearby locations.")
            else:
                f1, f2, f3 = st.columns(3)
                with f1:
                    min_reviews = st.slider("Min reviews", 0, int(df["Reviews"].max() or 0), 10, step=5)
                with f2:
                    min_rating  = st.slider("Min rating", 0.0, 5.0, 3.5, step=0.1)
                with f3:
                    top_n = st.slider("Show top N by reviews", 3, min(20, len(df)), min(10, len(df)))

                dff = df[(df["Reviews"] >= min_reviews) & (df["Rating"] >= min_rating)].copy()
                dff = dff.sort_values(["Reviews","Rating"], ascending=[False, False]).head(top_n)

                k1,k2,k3 = st.columns(3)
                k1.metric("Shown", len(dff))
                k2.metric("Avg rating", f"{dff['Rating'].mean():.2f}" if len(dff) else "–")
                k3.metric("Median reviews", int(dff["Reviews"].median() if len(dff) else 0))

                st.markdown("### Map of competitors")
                deck = map_layer(dff)
                if deck: st.pydeck_chart(deck)
                else: st.info("No coordinates available to plot.")

                st.markdown("### Visualizations")
                c1, c2 = st.columns(2)
                if not dff.empty:
                    bar = (
                        alt.Chart(dff.sort_values(["Reviews","Rating"], ascending=[False,False]).head(10))
                        .mark_bar()
                        .encode(x=alt.X("Name:N", sort='-y', title="Business"),
                                y=alt.Y("Rating:Q", title="Rating"),
                                tooltip=["Name","Rating","Reviews"])
                        .properties(height=350)
                    )
                    c1.altair_chart(bar, use_container_width=True)

                    scatter = (
                        alt.Chart(dff)
                        .mark_circle(size=80)
                        .encode(x=alt.X("Reviews:Q", title="Review count"),
                                y=alt.Y("Rating:Q", title="Rating"),
                                tooltip=["Name","Rating","Reviews","Address"])
                        .interactive()
                        .properties(height=350)
                    )
                    c2.altair_chart(scatter, use_container_width=True)

                st.markdown("### 🔎 Opportunity Finder")
                if not dff.empty:
                    dff["Opportunity"] = dff.apply(opportunity_score, axis=1)
                    opp = dff.sort_values("Opportunity", ascending=False)[["Name","Rating","Reviews","Opportunity","Address"]].head(5)
                    st.dataframe(opp, use_container_width=True)

                st.markdown("### Competitor list (filtered)")
                st.dataframe(dff, use_container_width=True)
                st.download_button("⬇️ Export filtered CSV", dff.to_csv(index=False).encode(),
                                   "competitors_filtered.csv", "text/csv")

                st.markdown("---")
                st.subheader("Place details & recent reviews")
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

                    st.markdown("#### Recent reviews")
                    revs = details.get("reviews", []) or []
                    if not revs:
                        st.info("No reviews available from Google for this place.")
                    else:
                        for r in revs:
                            author = r.get("author_name","Anonymous")
                            rating = r.get("rating","?")
                            when   = r.get("relative_time_description","")
                            st.markdown(f"**{author}** — {rating}★ — {when}")
                            st.write(r.get("text",""))
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Details error: {e}")
        except Exception as e:
            st.error(f"Google Places error: {e}")

    st.caption("Set GOOGLE_PLACES_API_KEY in Secrets. Text Search + Place Details are used.")


with tab2:
    st.subheader("Generate a 3-step follow-up plan using your private ARS backend")
    st.caption("Make sure your ARS FastAPI server is live; set ARS_URL and ARS_SECRET in Secrets.")

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
            st.success(f"Chosen arm: {data['arm']} • Score: {data['score']}")
            st.markdown("### Planned Steps")
            for step in data["steps"]:
                st.markdown(
                    f"📅 **{step['send_dt']}** — *{step['channel']}* "
                    f"{'— ' + step.get('subject','') if step.get('subject') else ''}"
                )
                st.write(step["body"])
                st.markdown("---")
            st.download_button("⬇️ Export follow-ups (CSV)",
                               pd.DataFrame(data["steps"]).to_csv(index=False).encode(),
                               "followups.csv", "text/csv")
        except Exception as e:
            st.error(f"ARS error: {e}")

st.markdown("---")
st.caption("Pulse © — Google Places competitor insights + private ARS follow-ups.")
