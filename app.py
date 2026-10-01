"""ShopMind - Streamlit screen. Run on Streamlit Community Cloud with app.py as the main file."""
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import streamlit as st  # noqa: E402

from shopmind_crew import answer_customer  # noqa: E402
from tools.shop_data_tool import load_shop  # noqa: E402

st.set_page_config(page_title="ShopMind", page_icon="🧵", layout="centered")
shop = load_shop()


def get_api_key() -> str:
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        key = st.sidebar.text_input("Groq API key", type="password", help="Only needed if no secret is set.")
    return key


def history_text(messages: list, limit: int = 6) -> str:
    lines = []
    for m in messages[-limit:]:
        who = "Customer" if m["role"] == "user" else "Shop"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # chat shown to the customer
if "inbox" not in st.session_state:
    st.session_state.inbox = []      # messages waiting for the owner
if "next_id" not in st.session_state:
    st.session_state.next_id = 1


def send_owner_reply(item_id: int):
    item = next(i for i in st.session_state.inbox if i["id"] == item_id)
    text = st.session_state.get(f"draft_{item_id}", item["draft"]).strip()
    if text:
        st.session_state.messages.append({"role": "assistant", "content": text, "meta": {"by": "owner"}})
    st.session_state.inbox = [i for i in st.session_state.inbox if i["id"] != item_id]


def dismiss(item_id: int):
    st.session_state.inbox = [i for i in st.session_state.inbox if i["id"] != item_id]


# ---------- sidebar ----------
api_key = get_api_key()
st.sidebar.title(f"🧵 {shop['shop_name']}")
view = st.sidebar.radio("View", ["Customer chat", f"Owner inbox ({len(st.session_state.inbox)})"])
if st.sidebar.button("Clear chat"):
    st.session_state.messages, st.session_state.inbox = [], []
    st.rerun()

# ---------- customer chat ----------
if view == "Customer chat":
    st.title("ShopMind")
    st.caption(f"Chat with {shop['shop_name']}. Try: 'Do you have the denim jacket in size M?' or 'What is your return policy?'")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            meta = m.get("meta", {})
            if meta.get("by") == "owner":
                st.caption("✍️ Sent by the shop owner")
            elif meta.get("reason"):
                label = "Sent to owner" if meta["needs_owner"] else "Auto-sent"
                with st.expander(f"Behind the scenes: {label}"):
                    st.markdown(f"**Facts found:**\n\n{meta['facts']}")
                    st.markdown(f"**Draft reply:** {meta['reply']}")
                    st.markdown(f"**Review decision:** {label} - {meta['reason']}")

    prompt = st.chat_input("Ask about prices, sizes, stock, delivery, returns...")
    if prompt:
        if not api_key:
            st.error("Add your Groq API key in the sidebar (or in Streamlit Secrets) first.")
            st.stop()
        history = history_text(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking the shop data..."):
                try:
                    result = answer_customer(prompt, history, api_key)
                except Exception as e:  # keep the app alive on any error
                    err = str(e).lower()
                    if "429" in err or "rate" in err:
                        st.warning("The free AI limit was reached. Please wait a minute and try again.")
                    else:
                        st.error(f"Something went wrong: {e}")
                    st.session_state.messages.pop()  # remove the unanswered question
                    st.stop()
            st.write(result["customer_text"])
        if result["needs_owner"]:
            st.session_state.inbox.append(
                {"id": st.session_state.next_id, "question": prompt, "draft": result["reply"], "reason": result["reason"]}
            )
            st.session_state.next_id += 1
        st.session_state.messages.append({"role": "assistant", "content": result["customer_text"], "meta": result})
        st.rerun()

# ---------- owner inbox ----------
else:
    st.title("Owner inbox")
    if not st.session_state.inbox:
        st.info("Nothing waiting. Messages that need you will appear here.")
    for item in st.session_state.inbox:
        with st.container(border=True):
            st.markdown(f"**Customer asked:** {item['question']}")
            st.caption(f"Why it needs you: {item['reason']}")
            st.text_area("Edit the draft reply, then send", value=item["draft"], key=f"draft_{item['id']}")
            c1, c2 = st.columns(2)
            c1.button("Send reply", key=f"send_{item['id']}", on_click=send_owner_reply, args=(item["id"],))
            c2.button("Dismiss", key=f"dismiss_{item['id']}", on_click=dismiss, args=(item["id"],))
