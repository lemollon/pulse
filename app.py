# app.py
import os
import datetime as dt
import pandas as pd
import altair as alt
import streamlit as st
import pydeck as pdk  # comes with Streamlit

# Google client
from google_places_client import search_competitors, get_place_details
# ARS client (unchanged)
from ars_client import plan_with_ars

st.set_page_config(page_title="Pulse — Google Places + ARS", page_icon="💼", layout="wide")
st.title("💼 Pulse — Competitor Insights (Google Places) + ARS Follow-Up")

tab1, tab2 = st.tabs(["⭐ Competitor Watch (Google)", "📬 Lead Follow-Up (ARS)"])

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
POSITIVE_KWS = ["friendly", "fresh", "clean", "fast", "delicious", "tasty", "warm", "helpful"]
NEGATIVE_KWS = ["wait", "slow", "cold", "stale", "price", "expensive", "dirty", "rude", "parking", "line", "crowded"]

def keyword_counts(texts, vocab):
    counts = {w: 0 for w in vocab}
    for t in texts:
        low = t.lower()
        for w in vocab:
            if w in low:
                counts[w] += low.count(w)
    # return as DataFrame
    return pd.DataFrame([{"keyword": k, "count": v} for k, v in counts.items()])

def opportunity_score(row):
    """
    Heuristic:
      - Penalize low ratings
      - Boost when review count is high (signal it's a big player with issues),
        or very low (signal a sleeper to out-market).
    """
    rating = float(row["Rating"] or 0)
    reviews = int(row["Reviews"] or 0)
    # base: invert rating (lower rating = higher opportunity to win)
    base = max(0.0, 5.0 - rating)
    # big player with issues: many reviews but rating < 4.2
    big_player_flag = 1.0 if (reviews >= 300 and rating < 4.2) else 0.0
    # sleeper: great rating but low awareness
    sleeper_flag = 1.0 if (reviews < 60 and rating >= 4.5) else 0.0
    return round(base + 1.5 * big_player_flag + 0.8 * sleeper_flag, 2)

def map_layer(df):
    # color by rating bucket
    def color_for_rating(r):
        if r >= 4.6: return [0, 128, 0]       # deep green
        if r >= 4.2: return [76, 175, 80]     # green
        if r >= 3.8: return [255, 193, 7]     # amber
        return [244, 67, 54]                  # red

    plot_df = df[["Name", "Rating", "Reviews", "Lat", "Lng"]].dropna().copy()
    plot_df["color"] = plot_df["Rating"].apply(color_for_rating)
    # scale size by review count
    plot_df["size"] = plot_df["Reviews"].clip(lower=1).apply(lambda x: min(60, 10 + (x ** 0.5) * 3))

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position='[Lng, Lat]',
        get_fill_color='color',
        get_radius='size',
        pickable=True
    )
    view_state = pdk.ViewState(
        latitude=float(plot_df["Lat"].mean()) if len(plot_df) else 30.2672,
        longitude=float(plot_df["Lng"].mean()) if len(plot_df) else -97.7431,
        zoom=11
    )
    tooltip = {"text": "{Name}\nRating: {Rating}★  Reviews: {Reviews}"}
    return pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", layers=[layer], initial_view_state=view_state, tooltip=tooltip)


# -------------------------------------------------------
# TAB 1 — Google Places Competitor Watch
# -------------------------------------------------------
with tab1:
    st.subheader("Find similar businesses via Google Places")

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        term = st.text_input("Category/term", value="doughnut shop",
                             help="e.g., 'doughnut shop', 'coffee', 'bakery'")
    with col2:
        city = st.text_input("City", value="Austin")
        state = st.text_input("State (2-letter)", value="TX", max_chars=2)
    with col3:
        zip_code = st.text_input("ZIP (optional)", value="")
    limit = st.slider("How many competitors?", min_value=3, max_value=25, value=12, step=1)

    # search button
    submitted = st.button("Search")

    if submitted:
        try:
            places = search_competitors(term, city, state, zip_code, limit=limit)
            if not places:
                st.warning("No results. Try a broader term, nearby location, or raise the limit.")
            else:
                rows = []
                for p in places:
                    rows.append({
                        "Name": p["name"],
                        "Rating": p["rating"],
                        "Reviews": p["user_ratings_total"],
                        "Address": p["formatted_address"],
                        "PriceLevel": p["price_level"],
                        "Lat": p["lat"],
                        "Lng": p["lng"],
                        "Place ID": p["place_id"],
                    })
                df = pd.DataFrame(rows)

                # ---------------- Filters & KPIs ----------------
                filt1, filt2, filt3 = st.columns([1,1,1])
                with filt1:
                    min_reviews = st.slider("Min reviews", 0, int(df["Reviews"].max() or 0), 10, step=5)
                with filt2:
                    min_rating = st.slider("Min rating", 0.0, 5.0, 3.5, step=0.1)
                with filt3:
                    show_top_n = st.slider("Show top N by reviews", 3, min(20, len(df)), min(10, len(df)))

                df_f = df[(df["Reviews"] >= min_reviews) & (df["Rating"] >= min_rating)].copy()
                df_f = df_f.sort_values(["Reviews", "Rating"], ascending=[False, False]).head(show_top_n)

                k1, k2, k3 = st.columns(3)
                k1.metric("Shown", len(df_f))
                k2.metric("Avg rating", f"{df_f['Rating'].mean():.2f}" if len(df_f) else "–")
                k3.metric("Median reviews", int(df_f["Reviews"].median() if len(df_f) else 0))

                # ---------------- Map ----------------
                st.markdown("### Map of competitors")
                st.pydeck_chart(map_layer(df_f))

                # ---------------- Visuals ----------------
                st.markdown("### Visualizations")
                c1, c2 = st.columns(2)

                # Bar: Ratings by business (top filtered)
                df_bar = df_f.copy()
                bar = (
                    alt.Chart(df_bar)
                    .mark_bar()
                    .encode(
                        x=alt.X("Name:N", sort='-y', title="Business"),
                        y=alt.Y("Rating:Q", title="Rating"),
                        tooltip=["Name", "Rating", "Reviews"]
                    )
                    .properties(height=350)
                )
                c1.altair_chart(bar, use_container_width=True)

                # Scatter: Reviews vs Rating
                scatter = (
                    alt.Chart(df_f)
                    .mark_circle(size=80)
                    .encode(
                        x=alt.X("Reviews:Q", title="Review count"),
                        y=alt.Y("Rating:Q", title="Rating"),
                        tooltip=["Name", "Rating", "Reviews", "Address"]
                    )
                    .interactive()
                    .properties(height=350)
                )
                c2.altair_chart(scatter, use_container_width=True)

                # ---------------- Opportunity Finder ----------------
                st.markdown("### 🔎 Opportunity Finder")
                df_f["Opportunity"] = df_f.apply(opportunity_score, axis=1)
                opp = df_f.sort_values("Opportunity", ascending=False)[["Name","Rating","Reviews","Opportunity","Address"]].head(5)
                st.write("These are the competitors you can most easily beat (or out-market) based on rating vs. review volume:")
                st.dataframe(opp, use_container_width=True)

                # Optional: lightweight review analysis over top-K by reviews
                with st.expander("Review theme analysis (top 5 by reviews)"):
                    top_ids = df_f.sort_values("Reviews", ascending=False)["Place ID"].head(5).tolist()
                    texts = []
                    for pid in top_ids:
                        try:
                            det = get_place_details(pid)
                            for r in det.get("reviews", []):
                                if r.get("text"): texts.append(r["text"])
                        except Exception:
                            pass
                    if not texts:
                        st.info("No review text available from Google for these places.")
                    else:
                        pos_df = keyword_counts(texts, POSITIVE_KWS)
                        neg_df = keyword_counts(texts, NEGATIVE_KWS)
                        c3, c4 = st.columns(2)
                        c3.markdown("**What customers love**")
                        c3.altair_chart(
                            alt.Chart(pos_df.sort_values("count", ascending=False)).mark_bar().encode(
                                x=alt.X("keyword:N", sort='-y', title="Keyword"),
                                y=alt.Y("count:Q", title="Mentions"),
                                tooltip=["keyword","count"]
                            ).properties(height=250),
                            use_container_width=True
                        )
                        c4.markdown("**What customers complain about**")
                        c4.altair_chart(
                            alt.Chart(neg_df.sort_values("count", ascending=False)).mark_bar().encode(
                                x=alt.X("keyword:N", sort='-y', title="Keyword"),
                                y=alt.Y("count:Q", title="Mentions"),
                                tooltip=["keyword","count"]
                            ).properties(height=250),
                            use_container_width=True
                        )

                        # Action suggestions (heuristics)
                        actions = []
                        if (neg_df.set_index("keyword")["count"].get("wait", 0) +
                            neg_df.set_index("keyword")["count"].get("line", 0) +
                            neg_df.set_index("keyword")["count"].get("crowded", 0)) > 0:
                            actions.append("Add a 'skip the line' pre-order link on your Google profile & website.")
                        if neg_df.set_index("keyword")["count"].get("cold", 0) > 0:
                            actions.append("Advertise 'hot & fresh every 30 minutes' and time your batches for rush hours.")
                        if neg_df.set_index("keyword")["count"].get("price", 0) + neg_df.set_index("keyword")["count"].get("expensive", 0) > 0:
                            actions.append("Run a weekday dozen promo before 10am to capture office orders.")
                        if not actions:
                            actions.append("Lean into your strengths (highlight 'fresh', 'friendly', 'clean' in your posts and signage).")

                        st.markdown("#### Suggested actions")
                        for a in actions:
                            st.markdown(f"- {a}")

                # ---------------- Table + exports ----------------
                st.markdown("### Competitor list (filtered)")
                st.dataframe(df_f, use_container_width=True)

                colx, coly, colz = st.columns(3)
                colx.download_button("⬇️ Export filtered CSV", df_f.to_csv(index=False).encode(), "competitors_filtered.csv", "text/csv")
                # Export an insights note
                insights_md = [
                    "# Pulse — Competitor Snapshot",
                    f"- Query: **{term}** in **{city}, {state} {zip_code}**",
                    f"- Shown: **{len(df_f)}** | Avg rating: **{df_f['Rating'].mean():.2f if len(df_f) else '–'}** | Median reviews: **{int(df_f['Reviews'].median() if len(df_f) else 0)}**",
                    "## Top Opportunities",
                ]
                for _, r in opp.iterrows():
                    insights_md.append(f"- **{r['Name']}** — {r['Rating']}★ ({r['Reviews']} reviews) — Opportunity: **{r['Opportunity']}**")
                coly.download_button("⬇️ Export insights.md", "\n".join(insights_md).encode(), "insights.md", "text/markdown")

        except Exception as e:
            st.error(f"Google Places error: {e}")

    st.caption("Add your key in Secrets as GOOGLE_PLACES_API_KEY. Text Search is used to find competitors.")


# -------------------------------------------------------
# TAB 2 — ARS Lead Follow-Up (unchanged)
# -------------------------------------------------------
with tab2:
    st.subheader("Generate a 3-step follow-up plan using your private ARS backend")
    st.caption("Ensure your ARS FastAPI server is running and ARS_URL/ARS_SECRET are set in Secrets or env.")

    colA, colB = st.columns(2)
    with colA:
        lead_name = st.text_input("Lead name", "Jane Smith")
        contact   = st.text_input("Contact (email or phone)", "jane@example.com")
        pref      = st.selectbox("Preferred channel", ["email", "sms"], index=0)
        notes     = st.text_area("Notes/context", "Interested in a dozen + coffee for office pickup")
    with colB:
        avg_sent7   = st.number_input("Avg sentiment (7d)", value=0.10, step=0.05, format="%.2f")
        wait_issue  = st.number_input("Complaint: wait time (0..1)", value=0.00, step=0.05, format="%.2f")
        recency     = st.number_input("Recency (days since last touch)", value=2, step=1)
        prior_rr    = st.number_input("Prior reply rate (0..1)", value=0.12, step=0.01, format="%.2f")

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
            "complaint_wait_time": float(wait_issue),
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

            df_steps = pd.DataFrame(data["steps"])
            st.download_button("⬇️ Export follow-ups (CSV)", df_steps.to_csv(index=False).encode(),
                               "followups.csv", "text/csv")

        except Exception as e:
            st.error(f"ARS error: {e}")

st.markdown("---")
st.caption("Pulse © — Competitor insights via Google Places + private ARS follow-ups.")
