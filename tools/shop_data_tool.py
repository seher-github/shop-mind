"""Shop data lookup: plain Python search + a CrewAI tool wrapper.

The search itself uses no AI, so it costs nothing and never invents facts.
"""
import json
import re
from pathlib import Path

from crewai.tools import tool

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "shop.json"

STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "you", "your", "have", "has",
    "in", "on", "of", "for", "to", "and", "or", "i", "me", "my", "we", "can",
    "what", "how", "much", "many", "any", "there", "it", "this", "that", "with",
    "size", "sizes", "please", "hi", "hello", "want", "need", "get", "buy",
}

# Which words in a question point to which policy.
POLICY_KEYWORDS = {
    "delivery": {"deliver", "delivery", "shipping", "ship", "shipped", "arrive", "courier", "day", "express", "international"},
    "returns": {"return", "refund", "money", "back", "sale"},
    "exchange": {"exchange", "swap", "change", "different", "replace"},
    "payment": {"pay", "payment", "cash", "card", "cod", "credit", "debit"},
    "hours": {"open", "close", "closed", "hour", "time", "sunday", "saturday", "monday"},
}


def load_shop() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    cleaned = set()
    for w in words:
        if w in STOPWORDS:
            continue
        # crude plural handling: "hoodies" -> "hoodie", "shirts" -> "shirt"
        if w.endswith("ies") and len(w) > 4:
            cleaned.add(w[:-3] + "y")
            cleaned.add(w[:-1])
        elif w.endswith("s") and len(w) > 3:
            cleaned.add(w[:-1])
        cleaned.add(w)
    return cleaned


def _product_text(p: dict) -> set:
    return _tokens(" ".join([p["name"], p["category"], " ".join(p["colors"])]))


def _format_product(p: dict, currency: str) -> str:
    stock_parts = []
    for size, qty in p["stock"].items():
        if qty == 0:
            stock_parts.append(f"{size}: OUT OF STOCK")
        elif qty <= 3:
            stock_parts.append(f"{size}: only {qty} left")
        else:
            stock_parts.append(f"{size}: in stock ({qty})")
    return (
        f"PRODUCT: {p['name']} | category: {p['category']} | price: {p['price']:.2f} {currency} | "
        f"colours: {', '.join(p['colors'])} | stock by size -> {'; '.join(stock_parts)} | notes: {p['notes']}"
    )


def search_shop_data(query: str) -> str:
    shop = load_shop()
    tokens = _tokens(query)
    currency = shop["currency"]
    results = []

    scored = []
    for p in shop["products"]:
        score = len(tokens & _product_text(p))
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    for _, p in scored[:3]:
        results.append(_format_product(p, currency))

    for topic, keys in POLICY_KEYWORDS.items():
        if tokens & keys:
            results.append(f"POLICY ({topic}): {shop['policies'][topic]}")

    if not results:
        catalog = ", ".join(f"{p['name']} ({p['category']})" for p in shop["products"])
        return (
            "NO_MATCH: nothing in the shop data matches this search. "
            f"The shop only sells: {catalog}. "
            f"Policy topics available: {', '.join(shop['policies'])}."
        )
    return "\n".join(results)


@tool("Shop Data Lookup")
def shop_data_lookup(query: str) -> str:
    """Search the shop's real product data (price, colours, stock by size) and
    policies (delivery, returns, exchange, payment, opening hours).
    Input: a short search phrase such as 'blue denim jacket' or 'return policy'."""
    return search_shop_data(query)
