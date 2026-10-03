"""Test lab: run a batch of questions through the whole pipeline and see what happens."""
import time

import pandas as pd
import streamlit as st

from data_loader import TEST_EXPECT_ALIASES, TEST_Q_ALIASES, guess_column, read_upload

SAMPLE_QUESTIONS = [
    "How much is the denim jacket?",
    "Do you have the hoodie in size XL?",
    "What is your return policy?",
    "How long does delivery take?",
    "I want a refund right now, this is terrible",
    "Do you sell shoes?",
    "Can I get a discount if I buy 3 t-shirts?",
    "What are your opening hours?",
]
LABEL = {"auto": "Auto-sent", "review": "Needs review", "escalate": "Escalated"}


def expected_label(value) -> str:
    """Understands expected values like auto, owner, review, escalate."""
    v = str(value).strip().lower()
    if "escalat" in v:
        return "Escalated"
    if "review" in v:
        return "Needs review"
    if any(w in v for w in ("owner", "human", "yes", "true", "1")):
        return "Owner (any)"
    return "Auto-sent"


def matches(expected: str, actual: str) -> bool:
    if expected == "Owner (any)":
        return actual in ("Needs review", "Escalated")
    return expected == actual


def run_batch(questions, run_one, progress=None, delay=1.0):
    """Runs each question; stops early if the free AI limit is hit. Returns a list of result rows."""
    rows = []
    for i, q in enumerate(questions, start=1):
        try:
            r = run_one(q)
            rows.append({"Question": q, "Route": LABEL[r["route"]], "Draft reply": r["reply"], "Why": r["reason"],
                         "Search query": r["query"], "Confidence": r["confidence"], "Seconds": r["seconds"],
                         "Facts found": r["facts"]})
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            rows.append({"Question": q, "Route": "Error", "Draft reply": "", "Why": msg[:120],
                         "Search query": "", "Confidence": "", "Seconds": 0.0, "Facts found": ""})
            if "429" in msg or "rate" in msg.lower():
                break
        if progress:
            progress(i / len(questions))
        time.sleep(delay)
    return rows


def render_test_lab(run_one, api_key):
    st.markdown("### 🧪 Test lab")
    st.caption("Run a batch of questions through the whole pipeline. Each question uses up to 4 AI calls "
               "(complaints use none), so start small to protect your free Groq quota.")

    up = st.file_uploader("Test questions CSV (optional)", type=["csv"], key="up_tests")
    questions, expected = SAMPLE_QUESTIONS, None
    if up is not None:
        try:
            df = read_upload(up)
            q_col = guess_column(df.columns, TEST_Q_ALIASES) or df.columns[0]
            keep = df[q_col].astype(str).str.strip() != ""
            questions = [q.strip() for q in df.loc[keep, q_col].astype(str)]
            e_col = guess_column([c for c in df.columns if c != q_col], TEST_EXPECT_ALIASES)
            if e_col:
                expected = [expected_label(v) for v in df.loc[keep, e_col]]
            st.success(f"Loaded {len(questions)} questions from column '{q_col}'"
                       + (f" with expected results from '{e_col}'." if expected else "."))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the test file: {e}")
            questions, expected = SAMPLE_QUESTIONS, None
    else:
        st.info("Using built-in sample questions. Upload your own test file to use yours.")

    if not questions:
        st.warning("No questions found.")
        return
    cap = min(len(questions), 20)
    n = st.slider("How many questions to run", 1, cap, min(5, cap)) if cap > 1 else 1

    if st.button("▶ Run test", type="primary", disabled=not api_key):
        bar = st.progress(0.0, text="Running the agents...")
        rows = run_batch(questions[:n], run_one, progress=lambda f: bar.progress(f, text="Running the agents..."))
        if expected:
            for row, exp in zip(rows, expected):
                if row["Route"] != "Error":
                    row["Expected"] = exp
                    row["Match"] = "✅" if matches(exp, row["Route"]) else "❌"
        bar.empty()
        st.session_state.test_results = rows
    if not api_key:
        st.caption("Add your Groq API key in the sidebar to run tests.")

    rows = st.session_state.get("test_results")
    if rows:
        df = pd.DataFrame(rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Auto-sent", int((df["Route"] == "Auto-sent").sum()))
        c2.metric("Needs review", int((df["Route"] == "Needs review").sum()))
        c3.metric("Escalated", int((df["Route"] == "Escalated").sum()))
        if "Match" in df.columns:
            c4.metric("Matches expectation", f"{int((df['Match'] == '✅').sum())}/{len(df)}")
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("⬇ Download results (CSV)", df.to_csv(index=False), "shopmind_test_results.csv", "text/csv")
