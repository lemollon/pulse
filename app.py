# app.py  — Pulse (Google Places API v1 + ARS)
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
    """
    Places API (New) details call.
    GET /v1/places/{place_id}
    """
    if place_id.startswith("places/"):
        resource = place_id
    else:
        resource = f"places/{place_id}"
    url = f"https://places.googleapis.com/v1/{resource}"
    fields = ",".join([
        "id",
        "displayName",
        "formattedAddress",
        "location",
        "rating",
        "userRatingCount",
        "priceLevel",
        "nationalPhoneNumber",
        "websiteUri",
        "regularOpeningHours.weekdayDescriptions",
        # "reviews",  # often restricted; uncomment if your project has access
    ])
    headers = gp_headers(fields)
    headers["X-Goog-Api-Key"] = GOOGLE_PLACES_API_KEY  # also acceptable for GET
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    name = (data.get("displayName") or {}).get("text", "")
    loc = data.get("location") or {}
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
    if plot_df.empty: return None
    plot_df["color"] = plot_df["Rating"].apply(color_for_rating)
    plot_df["size"]  = plot_df["Reviews"].clip(lower=1).apply(lambda x: min(60, 10 + (x**0.5)*3))
    layer = pdk.Layer("ScatterplotLayer", data=plot_df, get_position='[Lng, Lat]',
                      get_fill_color='color', get_radius='size', pickable=True)
    view_state = pdk.ViewState(latitude=float(plot_df["Lat"].mean()),
                               longitude=float(plot_df["Lng"].mean()), zoom=11)
    return pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", layers=[layer],
                    initial_view_state=view_state,
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
            deck = map_layer(dff)
            if deck: st.pydeck_chart(deck)
            else: st.info("No coordinates available to plot.")

            # Charts
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

            # Opportunity Finder
            st.markdown("### 🔎 Opportunity Finder")
            if not dff.empty:
                dff["Opportunity"] = dff.apply(opportunity_score, axis=1)
                opp = dff.sort_values("Opportunity", ascending=False)[["Name","Rating","Reviews","Opportunity","Address"]].head(5)
                st.dataframe(opp, use_container_width=True)

            # Table + export
            st.markdown("### Competitor list (filtered)")
            st.dataframe(dff, use_container_width=True)
            st.download_button("⬇️ Export filtered CSV", dff.to_csv(index=False).encode(),
                               "competitors_filtered.csv", "text/csv")

            # Details + Reviews
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

                # reviews often restricted in v1 standard access, so we display if present
                revs = details.get("reviews", []) or []
                if revs:
                    st.markdown("#### Recent reviews")
                    for r in revs:
                        author = r.get("authorName","Anonymous")
                        rating = r.get("rating","?")
                        when   = r.get("relativePublishTimeDescription","")
                        st.markdown(f"**{author}** — {rating}★ — {when}")
                        st.write(r.get("text",""))
                        st.markdown("---")
            except Exception as e:
                st.error(f"Details error: {e}")

        except Exception as e:
            st.error(f"Google Places error: {e}")

    st.caption("Requires Places API (New) and Geocoding API enabled. Key must be restricted to those APIs.")

# --------------------------------------------------------------------
# TAB 2 — ARS Lead Follow-Up (unchanged)
# --------------------------------------------------------------------
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
st.caption("Pulse © — Google Places (New) competitor insights + private ARS follow-ups.")
