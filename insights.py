"""Owner insights: a log of every conversation turn plus the dashboard that summarises it.

The log lives in the visitor's session (it resets when the app restarts). To keep it permanently,
send these rows to a free external store such as Google Sheets or Supabase.
"""
import time

import pandas as pd
import streamlit as st

ROUTE_LABEL = {"auto": "Auto-sent", "review": "Needs review", "escalate": "Escalated"}


def new_entry(entry_id: int, message: str, result: dict) -> dict:
    return {
        "id": entry_id,
        "time": time.strftime("%H:%M:%S"),
        "message": message,
        "route": ROUTE_LABEL[result["route"]],
        "intent": result.get("intent", "other"),
        "language": result.get("language", "unknown"),
        "confidence": result.get("confidence", "n/a"),
        "answerable": result.get("answerable", "n/a"),
        "verified": bool(result["verification"]["ok"]),
        "seconds": result.get("seconds", 0.0),
        "reason": result.get("reason", ""),
        "owner_action": "-" if result["route"] == "auto" else "pending",
    }


def mark_action(log: list, entry_id: int, action: str):
    for e in log:
        if e["id"] == entry_id:
            e["owner_action"] = action


def render_insights(log: list, learned_count: int, uploaded_examples: int):
    st.markdown("### 📊 Owner insights")
    if not log:
        st.info("No conversations yet. Ask a few questions in the Chat tab and this dashboard fills up.")
        return
    df = pd.DataFrame(log)
    total = len(df)
    auto = int((df["route"] == "Auto-sent").sum())
    review = int((df["route"] == "Needs review").sum())
    esc = int((df["route"] == "Escalated").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions handled", total)
    c2.metric("Answered automatically", f"{round(100 * auto / total)}%", f"{auto} of {total}")
    c3.metric("Needed the owner", review + esc, f"{review} review · {esc} escalated", delta_color="off")
    c4.metric("Avg. reply time", f"{df['seconds'].mean():.1f}s")

    minutes = st.number_input("Assumption: minutes it takes you to answer one message by hand", 0.5, 30.0, 2.0, 0.5)
    st.caption(f"Estimated time saved: about {auto * minutes:.0f} minutes ({auto} auto-answered x {minutes:g} min). This is only an estimate.")

    left, right = st.columns(2)
    with left:
        st.markdown("**How messages were handled**")
        st.bar_chart(df["route"].value_counts())
    with right:
        st.markdown("**What customers ask about**")
        st.bar_chart(df["intent"].value_counts())

    left, right = st.columns(2)
    with left:
        st.markdown("**Languages used**")
        st.bar_chart(df["language"].value_counts())
    with right:
        st.markdown("**Safety net and learning**")
        st.metric("Replies stopped by verification", int((~df["verified"]).sum()))
        st.metric("Owner replies learned from", learned_count, f"+{uploaded_examples} uploaded examples", delta_color="off")

    gaps = df[(df["confidence"] == "none") | (df["answerable"] == "NO")]
    st.markdown("**🕳️ Knowledge gaps** - questions the shop data could not answer (add this info to your files)")
    if gaps.empty:
        st.success("No gaps so far.")
    else:
        st.dataframe(gaps[["time", "message"]].rename(columns={"time": "Time", "message": "Question"}),
                     width="stretch", hide_index=True)

    st.markdown("**Recent activity**")
    shown = df[["time", "message", "route", "intent", "language", "confidence", "owner_action", "reason"]].tail(15).iloc[::-1]
    st.dataframe(shown, width="stretch", hide_index=True)
    st.download_button("⬇ Download full log (CSV)", df.to_csv(index=False), "shopmind_log.csv", "text/csv")
