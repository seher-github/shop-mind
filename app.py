"""ShopMind - Streamlit app: chat, owner inbox, test lab, and bring-your-own-data upload."""
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from data_loader import (  # noqa: E402
    PRODUCT_FIELDS, PRODUCT_LABELS, auto_mapping, build_shop, corrections_to_csv, parse_corrections,
    parse_policies, parse_products, pick_examples, read_upload, template_files,
)
from shopmind_crew import answer_customer  # noqa: E402
from test_lab import render_test_lab  # noqa: E402
from tools.shop_data_tool import load_demo_shop  # noqa: E402
from ui_styles import APP_CSS, hero_html, stats_html, steps_html  # noqa: E402

st.set_page_config(page_title="ShopMind", page_icon="🧵", layout="wide")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
PAGES = ["💬 Chat", "📥 Owner inbox", "🧪 Test lab", "📦 Shop data"]

# ---------- session state ----------
for key, default in [("messages", []), ("inbox", []), ("learned", []), ("next_id", 1), ("data_sig", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_api_key() -> str:
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:  # noqa: BLE001
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        key = st.text_input("Groq API key", type="password", help="Only needed if no secret is set.")
    return key


def safe_read(file):
    """Returns (DataFrame or None, error text or None)."""
    if file is None:
        return None, None
    if len(file.getvalue()) > MAX_UPLOAD_BYTES:
        return None, f"{file.name} is larger than 5 MB."
    try:
        return read_upload(file), None
    except Exception as e:  # noqa: BLE001
        return None, f"Could not read {file.name}: {e}"


def history_text(messages: list, limit: int = 6) -> str:
    lines = []
    for m in messages[-limit:]:
        who = "Customer" if m["role"] == "user" else "Shop"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand">🧵 <b>ShopMind</b></div>', unsafe_allow_html=True)
    st.markdown("##### 🗂️ Shop data")
    mode = st.radio("Data source", ["Demo shop", "My own data"], label_visibility="collapsed")

    f_products = f_policies = f_corr = None
    df_products, own_mapping = None, {}
    own_name, own_currency, own_contact = "My Shop", "USD", "the shop owner"
    errors = []

    if mode == "My own data":
        st.caption("Upload CSV files to test ShopMind with your own business. Only the products file is required.")
        f_products = st.file_uploader("Products CSV (required)", type=["csv"], key="up_products")
        f_policies = st.file_uploader("Policies CSV (optional)", type=["csv"], key="up_policies")
        f_corr = st.file_uploader("Owner corrections CSV (optional)", type=["csv"], key="up_corr")
        own_name = st.text_input("Shop name", "My Shop")
        own_currency = st.text_input("Currency", "USD")
        own_contact = st.text_input("Owner contact shown to customers", "the shop owner")

        df_products, err = safe_read(f_products)
        if err:
            errors.append(err)
        if df_products is not None:
            auto = auto_mapping(df_products)
            sig = f"{f_products.name}_{f_products.size}"
            with st.expander("🔧 Column mapping (auto-detected)"):
                options = ["(none)"] + list(df_products.columns)
                for field in PRODUCT_FIELDS:
                    idx = options.index(auto[field]) if auto[field] in options else 0
                    chosen = st.selectbox(PRODUCT_LABELS[field], options, index=idx, key=f"map_{field}_{sig}")
                    own_mapping[field] = "" if chosen == "(none)" else chosen
        st.caption("🔒 Files are used only for your current session and aren't saved by this app. "
                   "Their text is sent to the AI to answer questions, so please don't upload customers' personal details.")

    st.divider()
    api_key = get_api_key()
    if st.button("🧹 Clear chat & inbox", width="stretch"):
        st.session_state.messages, st.session_state.inbox, st.session_state.learned = [], [], []
        st.rerun()

# ---------- decide which data is active ----------
shop = load_demo_shop()
corrections, notices = [], []
source_label = "Demo shop"

if mode == "My own data":
    if df_products is None:
        notices.append("Upload a products CSV in the sidebar to switch to your own data. Showing the demo shop meanwhile.")
    else:
        products, warns = parse_products(df_products, own_mapping)
        notices += warns
        if products:
            policies = {}
            df_pol, err = safe_read(f_policies)
            if err:
                errors.append(err)
            if df_pol is not None:
                policies = parse_policies(df_pol)
            df_corr, err = safe_read(f_corr)
            if err:
                errors.append(err)
            if df_corr is not None:
                corrections, w = parse_corrections(df_corr)
                notices += w
            shop = build_shop(own_name, own_currency, own_contact, products, policies)
            source_label = "Your data"
        else:
            notices.append("No products could be read yet. Check the column mapping in the sidebar.")

# Fresh start whenever the data source changes.
data_sig = (mode, source_label, getattr(f_products, "name", None), getattr(f_products, "size", None),
            getattr(f_policies, "name", None), getattr(f_corr, "name", None))
if st.session_state.data_sig != data_sig:
    st.session_state.data_sig = data_sig
    st.session_state.messages, st.session_state.inbox, st.session_state.learned = [], [], []

pool = corrections + st.session_state.learned


def run_one(question: str, history: str = "") -> dict:
    return answer_customer(question, history, api_key, shop, pick_examples(pool, question))


# ---------- header ----------
st.markdown(hero_html(shop["shop_name"], source_label), unsafe_allow_html=True)
for e in errors:
    st.error(e)
for n in notices:
    st.warning(n)
st.markdown(
    stats_html([
        ("🛍️", len(shop["products"]), "Products loaded"),
        ("📜", len(shop["policies"]), "Policies loaded"),
        ("🎓", len(pool), "Owner examples"),
        ("📥", len(st.session_state.inbox), "Waiting for owner"),
    ]),
    unsafe_allow_html=True,
)

page = st.segmented_control("Navigate", PAGES, default=PAGES[0], label_visibility="collapsed", key="nav") or PAGES[0]


# ---------- callbacks ----------
def send_owner_reply(item_id: int):
    item = next(i for i in st.session_state.inbox if i["id"] == item_id)
    text = st.session_state.get(f"draft_{item_id}", item["draft"]).strip()
    if text:
        st.session_state.messages.append({"role": "assistant", "content": text, "meta": {"by": "owner"}})
        if text != item["draft"].strip():  # the owner edited it -> learn from it
            st.session_state.learned.append({"question": item["question"], "reply": text})
    st.session_state.inbox = [i for i in st.session_state.inbox if i["id"] != item_id]


def dismiss(item_id: int):
    st.session_state.inbox = [i for i in st.session_state.inbox if i["id"] != item_id]


def queue_prompt(text: str):
    st.session_state.queued = text


# ---------- pages ----------
if page == PAGES[0]:
    if not st.session_state.messages:
        st.markdown('<div class="welcome">👋 Ask anything a customer would ask - or tap a question to start.</div>',
                    unsafe_allow_html=True)
        first = shop["products"][0]["name"]
        samples = [f"How much is the {first}?", "What is your return policy?", "How long does delivery take?", "Do you have anything in size M?"]
        cols = st.columns(len(samples))
        for col, s in zip(cols, samples):
            col.button(s, key=f"chip_{s}", on_click=queue_prompt, args=(s,), width="stretch")

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🛍️" if m["role"] == "user" else "🧵"):
            st.write(m["content"])
            meta = m.get("meta", {})
            if meta.get("by") == "owner":
                st.caption("✍️ Sent by the shop owner")
            elif meta.get("reason"):
                with st.expander("Behind the scenes"):
                    st.markdown(steps_html(meta["needs_owner"]), unsafe_allow_html=True)
                    st.markdown(f"**Facts found:**\n\n{meta['facts']}")
                    st.markdown(f"**Draft reply:** {meta['reply']}")
                    st.markdown(f"**Review:** {meta['reason']}")

    prompt = st.chat_input("Ask about prices, sizes, stock, delivery, returns...")
    if not prompt and st.session_state.get("queued"):
        prompt = st.session_state.pop("queued")
    if prompt:
        if not api_key:
            st.error("Add your Groq API key in the sidebar (or in Streamlit Secrets) first.")
            st.stop()
        history = history_text(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🛍️"):
            st.write(prompt)
        with st.chat_message("assistant", avatar="🧵"):
            with st.spinner("The agents are checking the shop data..."):
                try:
                    result = run_one(prompt, history)
                except Exception as e:  # noqa: BLE001  keep the app alive on any error
                    err = str(e).lower()
                    if "429" in err or "rate" in err:
                        st.warning("The free AI limit was reached. Please wait a minute and try again.")
                    else:
                        st.error(f"Something went wrong: {e}")
                    st.session_state.messages.pop()
                    st.stop()
            st.write(result["customer_text"])
        if result["needs_owner"]:
            st.session_state.inbox.append(
                {"id": st.session_state.next_id, "question": prompt, "draft": result["reply"], "reason": result["reason"]}
            )
            st.session_state.next_id += 1
        st.session_state.messages.append({"role": "assistant", "content": result["customer_text"], "meta": result})
        st.rerun()

elif page == PAGES[1]:
    st.markdown("### 📥 Owner inbox")
    if not st.session_state.inbox:
        st.info("Nothing waiting. Messages that need you will appear here.")
    for item in st.session_state.inbox:
        with st.container(border=True):
            st.markdown(f"**Customer asked:** {item['question']}")
            st.caption(f"Why it needs you: {item['reason']}")
            st.text_area("Edit the draft reply, then send", value=item["draft"], key=f"draft_{item['id']}")
            c1, c2 = st.columns(2)
            c1.button("Send reply", key=f"send_{item['id']}", type="primary", on_click=send_owner_reply, args=(item["id"],))
            c2.button("Dismiss", key=f"dismiss_{item['id']}", on_click=dismiss, args=(item["id"],))
    if st.session_state.learned:
        st.success(f"🎓 ShopMind has learned from {len(st.session_state.learned)} reply edit(s) you made this session.")
        st.download_button("⬇ Download learned replies (CSV)", corrections_to_csv(st.session_state.learned),
                           "owner_corrections_learned.csv", "text/csv")

elif page == PAGES[2]:
    render_test_lab(run_one, api_key)

else:
    st.markdown("### 📦 Shop data in use")
    rows = []
    for p in shop["products"]:
        stock = []
        for color, sizes in p["stock_by_color"].items():
            for s, q in sizes.items():
                label = "avail." if q < 0 else q
                stock.append(f"{(color + ' ') if color else ''}{s}: {label}")
        rows.append({"Product": p["name"], "Category": p.get("category", ""),
                     "Price": p["price"] if p.get("price") is not None else "-",
                     "Colours": ", ".join(c for c in p.get("colors", []) if c), "Stock": "; ".join(stock)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander(f"📜 Policies ({len(shop['policies'])})"):
        if shop["policies"]:
            for topic, text in shop["policies"].items():
                st.markdown(f"**{topic}** - {text}")
        else:
            st.write("No policies loaded. The assistant will pass policy questions to the owner.")
    with st.expander(f"🎓 Owner examples ({len(pool)})"):
        for c in pool[:10]:
            st.markdown(f"**Customer:** {c['question']}  \n**Owner:** {c['reply']}")
        if not pool:
            st.write("None yet. Upload a corrections file, or edit drafts in the Owner inbox and ShopMind will learn.")
    st.markdown("##### 📄 Not sure how your CSV should look? Download a template")
    cols = st.columns(4)
    for col, (fname, content) in zip(cols, template_files().items()):
        col.download_button(fname, content, fname, "text/csv", key=f"tpl_{fname}", width="stretch")
