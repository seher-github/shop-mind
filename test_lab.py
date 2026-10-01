"""Test lab: run a batch of questions through the agents and see what happens."""
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
OWNER_WORDS = ("owner", "escalat", "review", "human", "yes", "true", "1")


def expects_owner(value) -> bool:
    return any(w in str(value).strip().lower() for w in OWNER_WORDS)


def run_batch(questions, run_one, progress=None, delay=1.0):
    """Runs each question; stops early if the free AI limit is hit. Returns a list of result rows."""
    rows = []
    for i, q in enumerate(questions, start=1):
        try:
            r = run_one(q)
            rows.append({"Question": q, "Route": "Owner" if r["needs_owner"] else "Auto-sent",
                         "Draft reply": r["reply"], "Why": r["reason"]})
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            rows.append({"Question": q, "Route": "Error", "Draft reply": "", "Why": msg[:120]})
            if "429" in msg or "rate" in msg.lower():
                break
        if progress:
            progress(i / len(questions))
        time.sleep(delay)
    return rows


def render_test_lab(run_one, api_key):
    st.markdown("### 🧪 Test lab")
    st.caption("Run a batch of questions through the three agents and check which ones are answered automatically "
               "and which go to the owner. Uses your free Groq quota, so start small.")

    up = st.file_uploader("Test questions CSV (optional)", type=["csv"], key="up_tests")
    questions, expected = SAMPLE_QUESTIONS, None
    if up is not None:
        try:
            df = read_upload(up)
            q_col = guess_column(df.columns, TEST_Q_ALIASES) or df.columns[0]
            questions = [q.strip() for q in df[q_col].astype(str) if q.strip()]
            e_col = guess_column([c for c in df.columns if c != q_col], TEST_EXPECT_ALIASES)
            if e_col:
                expected = [expects_owner(v) for v in df.loc[df[q_col].astype(str).str.strip() != "", e_col]]
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
                if row["Route"] in ("Owner", "Auto-sent"):
                    row["Expected"] = "Owner" if exp else "Auto-sent"
                    row["Match"] = "✅" if row["Expected"] == row["Route"] else "❌"
        bar.empty()
        st.session_state.test_results = rows
    if not api_key:
        st.caption("Add your Groq API key in the sidebar to run tests.")

    rows = st.session_state.get("test_results")
    if rows:
        df = pd.DataFrame(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Auto-sent", int((df["Route"] == "Auto-sent").sum()))
        c2.metric("Sent to owner", int((df["Route"] == "Owner").sum()))
        if "Match" in df.columns:
            c3.metric("Matches expectation", f"{int((df['Match'] == '✅').sum())}/{len(df)}")
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("⬇ Download results (CSV)", df.to_csv(index=False), "shopmind_test_results.csv", "text/csv")
