"""Reads the CSV files a shop owner uploads and turns them into ShopMind's data format.

Column names are auto-detected, so many different CSV layouts work.
"""
import io
import re

import pandas as pd

MAX_ROWS = 5000

# ---------- column detection ----------
PRODUCT_FIELDS = {
    "name": ["product_name", "name", "title", "product", "item_name", "item"],
    "price": ["price", "unit_price", "selling_price", "sale_price", "retail_price", "mrp", "amount"],
    "category": ["category", "product_type", "type", "department", "collection", "group"],
    "color": ["color", "colour", "colors", "colours"],
    "size": ["size", "sizes", "available_sizes"],
    "stock": ["stock", "stock_quantity", "quantity", "qty", "inventory", "units", "on_hand", "in_stock"],
    "notes": ["description", "notes", "details", "material", "fabric", "fit"],
}
PRODUCT_LABELS = {
    "name": "Product name (required)", "price": "Price", "category": "Category",
    "color": "Colour", "size": "Size", "stock": "Stock / quantity", "notes": "Description / notes",
}
TOPIC_ALIASES = ["policy_name", "policy_type", "policy", "topic", "title", "name", "category", "section", "type"]
TEXT_ALIASES = ["policy_text", "policy_details", "policy_detail", "details", "detail", "description", "text", "content", "answer", "rule", "terms", "info", "summary"]
QUALIFIER_ALIASES = ["scope", "applies_to", "applicable_to", "city", "region", "location", "collection", "applies"]
GENERIC_SCOPES = {"general", "all", "all orders", "all items", "all products", "everything", "any"}
Q_ALIASES = ["customer_message", "customer_question", "customer_query", "question", "message", "query", "customer", "input", "user_message"]
R_ALIASES = [
    "owner_reply", "owner_correction", "owner_edit", "owner_edited_reply", "corrected_reply", "corrected_response",
    "correct_reply", "correct_response", "final_reply", "owner_response", "owner_answer", "human_reply",
    "human_response", "edited_reply", "correction", "correct_answer", "reply", "response", "answer",
]
TEST_Q_ALIASES = ["question", "customer_message", "message", "query", "customer_question", "input", "prompt", "text"]
TEST_EXPECT_ALIASES = ["expected_route", "expected_action", "expected_decision", "should_escalate", "needs_owner", "escalate", "route", "expected"]


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def guess_column(columns, aliases, avoid=()) -> str:
    normed = {c: _norm(c) for c in columns}

    def allowed(c):
        return not any(a in normed[c].split("_") for a in avoid)

    for a in aliases:                       # exact matches first
        for c, n in normed.items():
            if n == a and allowed(c):
                return c
    for a in aliases:                       # then "contains"
        for c, n in normed.items():
            if a in n and allowed(c):
                return c
    return ""


def auto_mapping(df: pd.DataFrame) -> dict:
    mapping, used = {}, set()
    for field, aliases in PRODUCT_FIELDS.items():
        free = [c for c in df.columns if c not in used]
        col = guess_column(free, aliases)
        mapping[field] = col
        if col:
            used.add(col)
    return mapping


# ---------- reading files ----------
def read_upload(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, sep=None, engine="python", encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not read this file. Please save it as a CSV (UTF-8).")
    df = df.fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df.head(MAX_ROWS)


# ---------- small parsers ----------
def to_float(s):
    m = re.search(r"\d[\d,]*\.?\d*", str(s))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


OUT_WORDS = {"out of stock", "out", "no", "sold out", "false", "unavailable", "n", "0"}
IN_WORDS = {"in stock", "yes", "available", "true", "y"}


def to_qty(s):
    """number -> that number, out-words -> 0, in-words -> -1 (available, quantity unknown), blank -> None."""
    t = str(s).strip().lower()
    if t == "":
        return None
    if t in OUT_WORDS:
        return 0
    if t in IN_WORDS:
        return -1
    f = to_float(t)
    return None if f is None else int(max(0, round(f)))


def _split(s):
    return [x.strip() for x in re.split(r"[,;|]+", str(s)) if x.strip()]


def _merge_qty(prev, new):
    if prev is None:
        return new
    if prev >= 0 and new >= 0:
        return prev + new
    return 0 if (prev == 0 and new == 0) else -1


# ---------- products ----------
def parse_products(df: pd.DataFrame, mapping: dict):
    warnings = []
    c = {f: (mapping.get(f) if mapping.get(f) in df.columns else "") for f in PRODUCT_FIELDS}
    if not c["name"]:
        return [], ["Pick which column holds the product name."]

    products, skipped = {}, 0
    for _, row in df.iterrows():
        name = str(row[c["name"]]).strip()
        if not name:
            skipped += 1
            continue
        p = products.setdefault(
            name.lower(),
            {"name": name, "category": "", "price": None, "colors": [], "stock_by_color": {}, "notes": ""},
        )
        if c["category"] and not p["category"]:
            p["category"] = str(row[c["category"]]).strip()
        if c["price"] and p["price"] is None:
            p["price"] = to_float(row[c["price"]])
        if c["notes"] and not p["notes"]:
            p["notes"] = str(row[c["notes"]]).strip()[:200]

        colors = _split(row[c["color"]]) if c["color"] else []
        sizes = _split(row[c["size"]]) if c["size"] else []
        qty = to_qty(row[c["stock"]]) if c["stock"] else None
        if qty is None:
            qty = -1  # listed, but no quantity given
        if len(sizes) > 1 or len(colors) > 1:
            qty = -1 if qty != 0 else 0  # one number can't be split across variants

        for color in colors or [""]:
            if color and color not in p["colors"]:
                p["colors"].append(color)
            bucket = p["stock_by_color"].setdefault(color, {})
            for size in sizes or ["One size"]:
                bucket[size] = _merge_qty(bucket.get(size), qty)

    result = list(products.values())
    if skipped:
        warnings.append(f"{skipped} row(s) without a product name were skipped.")
    missing_price = sum(1 for p in result if p["price"] is None)
    if missing_price:
        warnings.append(f"{missing_price} product(s) have no price (the assistant will say the price isn't listed).")
    if len(df) >= MAX_ROWS:
        warnings.append(f"Only the first {MAX_ROWS} rows were used.")
    return result, warnings


# ---------- policies ----------
def parse_policies(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    cols = list(df.columns)
    topic_c = guess_column(cols, TOPIC_ALIASES)
    text_c = guess_column([x for x in cols if x != topic_c], TEXT_ALIASES)
    policies = {}

    def add(topic, text):
        topic, text = str(topic).strip(), str(text).strip()
        if topic and text:
            policies[topic] = f"{policies[topic]} {text}" if topic in policies else text

    if topic_c and text_c:
        for _, r in df.iterrows():
            add(r[topic_c], r[text_c])
    elif len(df) <= 3 and len(cols) >= 3:       # wide layout: each column is a topic
        for col in cols:
            add(col, " ".join(v for v in df[col].astype(str) if v.strip()))
    elif len(cols) >= 2:
        for _, r in df.iterrows():
            add(r[cols[0]], r[cols[1]])
    else:
        for i, v in enumerate(df[cols[0]], start=1):
            add(f"policy {i}", v)
    return policies


# ---------- owner corrections ----------
def parse_corrections(df: pd.DataFrame):
    cols = list(df.columns)
    bad_q = ("ai", "bot", "original", "draft")
    bad_r = ("ai", "bot", "original", "draft", "wrong", "incorrect")
    q_c = guess_column(cols, Q_ALIASES, avoid=bad_q)
    r_c = guess_column([x for x in cols if x != q_c], R_ALIASES, avoid=bad_r)
    if not q_c or not r_c:
        return [], ["Couldn't find a customer-message column and an owner-reply column in the corrections file."]
    items = []
    for _, r in df.iterrows():
        q, a = str(r[q_c]).strip(), str(r[r_c]).strip()
        if q and a:
            items.append({"question": q, "reply": a})
    return items, []


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def pick_examples(corrections: list, message: str, k: int = 3) -> str:
    """Pick the k past owner replies most similar to this message, formatted for the prompt."""
    if not corrections:
        return ""
    target = _words(message)
    scored = [(len(target & _words(c["question"])), c) for c in corrections]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    lines = []
    for _, c in scored[:k]:
        lines.append(f"Customer: {c['question'][:200]}\nOwner's reply: {c['reply'][:300]}")
    return "\n\n".join(lines)


# ---------- building the shop ----------
def build_shop(name: str, currency: str, contact: str, products: list, policies: dict) -> dict:
    return {
        "shop_name": name.strip() or "My Shop",
        "currency": currency.strip(),
        "owner_contact": contact.strip() or "the shop owner",
        "products": products,
        "policies": policies,
    }


def corrections_to_csv(items: list) -> str:
    return pd.DataFrame(
        [{"customer_message": i["question"], "owner_reply": i["reply"]} for i in items],
        columns=["customer_message", "owner_reply"],
    ).to_csv(index=False)


def template_files() -> dict:
    return {
        "products_template.csv": (
            "product_name,category,price,color,size,stock,description\n"
            "Classic Denim Jacket,jackets,59.99,blue,M,3,Regular fit denim\n"
            "Classic Denim Jacket,jackets,59.99,blue,L,0,Regular fit denim\n"
            "Cozy Hoodie,hoodies,39.50,grey,S,10,Fleece lined\n"
        ),
        "policies_template.csv": (
            "policy,details\n"
            "delivery,Standard delivery takes 3-5 working days and costs 4.00.\n"
            "returns,Unworn items can be returned within 14 days for a refund.\n"
            "hours,Open Monday to Saturday 10:00 to 19:00.\n"
        ),
        "owner_corrections_template.csv": (
            "customer_message,owner_reply\n"
            "Do you deliver on Sundays?,Hi! We deliver Monday to Saturday - your order will arrive on the next working day.\n"
            "Can I return a worn hoodie?,Sorry - we can only accept unworn items with tags within 14 days.\n"
        ),
        "test_questions_template.csv": (
            "question,expected_route\n"
            "How much is the denim jacket?,auto\n"
            "I want a refund right now,owner\n"
            "What is your return policy?,auto\n"
        ),
    }
